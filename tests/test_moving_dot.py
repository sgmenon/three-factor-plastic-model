from experiments.moving_dot import (
    DotState,
    MovingDotPredictor,
    advance,
    evaluation_metrics,
    frame_from_state,
    persistence_rollout_position_accuracy,
    rollout_position_accuracy,
    run_online_experiment,
    sample_transition,
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
    assert one_step_accuracy > 0.3
    assert history[-1][1] < history[0][1]

    for value in (
        model.hidden_weight,
        model.hidden_bias,
        model.output_weight,
        model.output_bias,
    ):
        assert value.grad is None
        assert not value.requires_grad


def test_video_model_uses_input_memory_not_hidden_state() -> None:
    model = MovingDotPredictor()
    first = frame_from_state(DotState(2, 2, 1, 0), grid_size=6)
    second = frame_from_state(DotState(3, 2, 1, 0), grid_size=6)

    model.prime_state(first)
    record = model.forward(second)

    assert model.input_size == 2 * model.pixel_count
    assert record.model_input.shape == (2 * model.pixel_count,)
    assert torch.equal(
        record.model_input[: model.pixel_count],
        second.flatten(),
    )
    assert torch.equal(
        record.model_input[model.pixel_count :],
        first.flatten(),
    )
    assert torch.equal(model.previous_frame, second.flatten())


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

    control.correct(
        control.record_from_pair(previous, current),
        stationary_delta,
    )
    reinforced.correct(
        reinforced.record_from_pair(previous, current),
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
    control_record = control.record_from_pair(previous, current)
    reinforced_record = reinforced.record_from_pair(previous, current)
    exactly_predicted_delta = control_record.predicted_delta

    control.correct(control_record, exactly_predicted_delta)
    reinforced.correct(reinforced_record, exactly_predicted_delta)

    assert torch.any(reinforced.output_bias < control.output_bias)
    assert torch.any(reinforced.hidden_bias < control.hidden_bias)


def test_advantage_reinforcement_opens_above_average_connection() -> None:
    model = MovingDotPredictor(
        reinforcement_learning_rate=1e-2,
        reinforcement_interval=10,
    )
    model.output_layer.bias[:2] = 0.5
    bias_before = model.output_layer.bias[:2].clone()
    model.output_success_sum[0] = 5.0
    model.output_success_count[0] = 10.0
    model.output_reinforcement_sum[:2] = torch.tensor([9.0, 1.0])
    model.output_reinforcement_count[:2] = 10.0
    model.reinforcement_observations = 10

    model._apply_advantage_reinforcement()

    assert model.output_layer.bias[0] < bias_before[0]
    assert model.output_layer.bias[1] > bias_before[1]


def test_hidden_layer_can_be_frozen_as_an_ablation() -> None:
    model = MovingDotPredictor(update_hidden=False)
    previous = frame_from_state(DotState(2, 2, 1, 0), grid_size=6)
    current = frame_from_state(DotState(3, 2, 1, 0), grid_size=6)
    next_frame = frame_from_state(DotState(4, 2, 1, 0), grid_size=6)
    hidden_weight_before = model.hidden_weight.clone()
    hidden_bias_before = model.hidden_bias.clone()
    output_weight_before = model.output_weight.clone()

    model.correct(
        model.record_from_pair(previous, current),
        next_frame - current,
    )

    assert torch.equal(model.hidden_weight, hidden_weight_before)
    assert torch.equal(model.hidden_bias, hidden_bias_before)
    assert not torch.equal(model.output_weight, output_weight_before)


def test_negative_pruning_score_can_remove_active_connection() -> None:
    model = MovingDotPredictor()
    initial_edges = model.present_connection_count()
    model.history_observations = 1
    model.hidden_edge_activity.fill_(1)
    model.output_edge_activity.fill_(1)
    model.hidden_pruning_score[0] = -2
    model.output_pruning_score[0] = -2

    model.prune_inactive(
        minimum_pruning_score=-1,
        reactivation_fraction=0.0,
    )

    assert model.present_connection_count() == initial_edges - 2
    assert model.total_score_pruned == 2


def test_pruning_preserves_surviving_pi_integrals() -> None:
    model = MovingDotPredictor()
    hidden_integral = torch.arange(
        model.hidden_layer.edge_count,
        dtype=torch.float32,
    )
    output_integral = torch.arange(
        model.output_layer.edge_count,
        dtype=torch.float32,
    )
    model.hidden_bias_controller.integral = hidden_integral.clone()
    model.output_bias_controller.integral = output_integral.clone()
    model.history_observations = 1
    model.hidden_edge_activity.fill_(1)
    model.output_edge_activity.fill_(1)
    model.hidden_edge_activity[0] = 0
    model.output_edge_activity[0] = 0

    model.prune_inactive(reactivation_fraction=0.0)

    assert torch.equal(
        model.hidden_bias_controller.integral,
        hidden_integral[1:],
    )
    assert torch.equal(
        model.output_bias_controller.integral,
        output_integral[1:],
    )


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


def test_identical_hidden_neurons_can_merge() -> None:
    model = MovingDotPredictor(
        hidden_size=2,
        update_hidden=False,
    )
    first_hidden = model.hidden_layer.target_index == 0
    second_hidden = model.hidden_layer.target_index == 1
    model.hidden_layer.weight[second_hidden] = (
        model.hidden_layer.weight[first_hidden]
    )
    model.hidden_layer.bias[second_hidden] = (
        model.hidden_layer.bias[first_hidden]
    )
    for target in range(model.output_size):
        edges = (
            model.output_layer.target_index == target
        ).nonzero().flatten()
        model.output_layer.weight[edges[1]] = (
            model.output_layer.weight[edges[0]]
        )
        model.output_layer.bias[edges[1]] = (
            model.output_layer.bias[edges[0]]
        )

    probe_before = model._current_structure_probe()
    for record in probe_before:
        model._record_pruning_history(record)
    predictions_before = torch.stack(
        [record.predicted_delta for record in probe_before]
    )
    model.prune_inactive(
        reactivation_fraction=0.0,
        merge_redundant_neurons=True,
        merge_similarity_threshold=0.999,
        merge_error_tolerance=1e-5,
    )
    predictions_after = torch.stack(
        [
            record.predicted_delta
            for record in model._current_structure_probe()
        ]
    )

    assert model.hidden_size == 1
    assert model.total_neurons_merged == 1
    assert torch.allclose(
        predictions_after,
        predictions_before,
        atol=1e-5,
    )
