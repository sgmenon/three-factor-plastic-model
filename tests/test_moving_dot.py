from experiments.moving_dot import (
    DotState,
    MovingDotPredictor,
    advance,
    evaluation_metrics,
    frame_from_state,
    persistence_rollout_position_accuracy,
    rollout_position_accuracy,
    run_online_experiment,
    sample_trajectory,
)
from experiments.sparse_connections import SparseConnectionLayer
import torch


def test_dot_bounces_at_frame_boundaries() -> None:
    state = DotState(
        row=0,
        column=5,
        row_velocity=-1,
        column_velocity=1,
    )

    next_state = advance(state, grid_size=6)

    assert next_state.row == 1
    assert next_state.column == 4
    assert next_state.row_velocity == 1
    assert next_state.column_velocity == -1


def test_training_trajectory_contains_consecutive_frames() -> None:
    frames = sample_trajectory(
        torch.Generator().manual_seed(7),
        grid_size=6,
        frame_count=10,
    )

    assert len(frames) == 10
    positions = [int(frame.flatten().argmax()) for frame in frames]
    for previous, current in zip(positions, positions[1:]):
        previous_row, previous_column = divmod(previous, 6)
        current_row, current_column = divmod(current, 6)
        assert abs(current_row - previous_row) <= 1
        assert abs(current_column - previous_column) <= 1
        assert previous != current


def test_sparse_layer_matches_dense_connection_equation() -> None:
    layer = SparseConnectionLayer(
        source_count=4,
        target_count=3,
        generator=torch.Generator().manual_seed(17),
        bias_low=0.02,
        bias_high=0.12,
    )
    source = torch.tensor([0.0, 0.4, 0.8, 1.0])
    dense_weight = layer.weight.reshape(3, 4)
    dense_bias = layer.bias.reshape(3, 4)
    expected = torch.relu(
        dense_weight * source.unsqueeze(0) - dense_bias
    ).sum(dim=1)

    actual, _ = layer.forward(source)

    assert torch.allclose(actual, expected)


def test_stage_0_learns_next_frame_without_autograd() -> None:
    model, history = run_online_experiment(observations=10_000)
    mse, persistence_mse, one_step_accuracy = evaluation_metrics(model)

    assert mse < persistence_mse * 0.5
    assert one_step_accuracy > 0.6
    assert history[-1][1] < history[0][1]

    for value in (
        model.hidden_weight,
        model.hidden_bias,
        model.output_weight,
        model.output_bias,
    ):
        assert value.grad is None
        assert not value.requires_grad


def test_stage_0_rollout_beats_persistence() -> None:
    model, _ = run_online_experiment(observations=10_000)

    model_accuracy = rollout_position_accuracy(model)
    persistence_accuracy = persistence_rollout_position_accuracy(
        grid_size=model.grid_size
    )

    assert model_accuracy > persistence_accuracy


def test_stationary_target_gets_no_reinforcement() -> None:
    control = MovingDotPredictor(reinforcement_learning_rate=0.0)
    reinforced = MovingDotPredictor(reinforcement_learning_rate=1e-2)
    previous = frame_from_state(DotState(2, 2, 1, 0), grid_size=6)
    current = frame_from_state(DotState(3, 2, 1, 0), grid_size=6)
    stationary_delta = torch.zeros(6, 6)

    control.correct(control.forward(previous, current), stationary_delta)
    reinforced.correct(
        reinforced.forward(previous, current),
        stationary_delta,
    )

    assert torch.equal(control.hidden_bias, reinforced.hidden_bias)
    assert torch.equal(control.output_bias, reinforced.output_bias)


def test_low_error_nontrivial_output_opens_active_connections() -> None:
    control = MovingDotPredictor(
        reinforcement_learning_rate=0.0,
        meaningful_change_threshold=1e-8,
    )
    reinforced = MovingDotPredictor(
        reinforcement_learning_rate=1e-2,
        meaningful_change_threshold=1e-8,
    )
    previous = frame_from_state(DotState(2, 2, 1, 0), grid_size=6)
    current = frame_from_state(DotState(3, 2, 1, 0), grid_size=6)
    control_record = control.forward(previous, current)
    reinforced_record = reinforced.forward(previous, current)
    exactly_predicted_delta = control_record.predicted_delta

    control.correct(control_record, exactly_predicted_delta)
    reinforced.correct(reinforced_record, exactly_predicted_delta)

    assert torch.any(reinforced.output_bias < control.output_bias)
    assert torch.any(reinforced.hidden_bias < control.hidden_bias)


def test_periodic_pruning_removes_and_reactivates_connections() -> None:
    model, _ = run_online_experiment(
        observations=1_000,
        pruning_interval=1_000,
        reactivation_fraction=0.05,
    )

    assert model.pruning_events == 1
    assert model.total_pruned > 0
    assert model.total_reactivated > 0
    assert model.connection_presence_fraction() < 1.0
    present_edges = (
        model.hidden_layer.edge_count + model.output_layer.edge_count
    )
    possible_edges = (
        model.hidden_layer.possible_edge_count
        + model.output_layer.possible_edge_count
    )
    assert present_edges < possible_edges
    for layer in (model.hidden_layer, model.output_layer):
        pair_keys = (
            layer.target_index * layer.source_count + layer.source_index
        )
        assert pair_keys.unique().numel() == layer.edge_count


def test_redundant_connections_merge_without_changing_history_output() -> None:
    source_history = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ]
    )
    layer = SparseConnectionLayer(
        source_count=3,
        target_count=1,
        generator=torch.Generator().manual_seed(19),
        bias_low=0.0,
        bias_high=0.1,
    )
    layer.weight = torch.tensor([0.5, 0.5, 0.2])
    layer.bias = torch.tensor([0.1, 0.1, 0.0])
    output_before = torch.stack(
        [layer.forward(source)[0][0] for source in source_history]
    )

    merged = MovingDotPredictor._merge_redundant_layer(
        layer,
        source_history,
        similarity_threshold=0.995,
        error_tolerance=1e-6,
        maximum_merge_fraction=0.5,
    )

    output_after = torch.stack(
        [layer.forward(source)[0][0] for source in source_history]
    )
    assert merged == 1
    assert layer.edge_count == 2
    assert torch.allclose(output_after, output_before)
