import torch

from experiments.deep_moving_dot import DeepMovingDotPredictor
from experiments.moving_dot import DotState, frame_from_state, sample_transition
from experiments.sparse_connections import SparseConnectionLayer


def test_hidden_depth_is_configurable_and_layers_share_initial_ranges() -> None:
    model = DeepMovingDotPredictor(hidden_layer_count=3)

    assert model.hidden_layer_count == 3
    assert len(model.layers) == 4
    assert model.input_size == model.pixel_count
    assert model.first_raw_size == 48
    assert model.layers[0].target_count == 48
    assert model.layers[1].source_count == 96
    for layer in model.layers[:-1]:
        assert torch.all((layer.weight >= 0.08) & (layer.weight <= 0.20))
        assert torch.all((layer.bias >= 0.005) & (layer.bias <= 0.02))


def test_only_first_hidden_layer_has_temporal_residual_state() -> None:
    model = DeepMovingDotPredictor(hidden_layer_count=2)
    previous = frame_from_state(DotState(2, 2, 1, 0), grid_size=6)
    current = frame_from_state(DotState(3, 2, 1, 0), grid_size=6)

    model.prime_state(previous)
    previous_raw = model.previous_first_raw.clone()
    record = model.forward(current)
    current_raw, _ = model.layers[0].forward(current.flatten())
    change = current_raw - previous_raw
    expected_first = torch.cat([torch.relu(change), torch.relu(-change)])
    expected_second, _ = model.layers[1].forward(expected_first)

    assert torch.allclose(record.hidden[0], expected_first)
    assert torch.allclose(record.hidden[1], expected_second)
    assert torch.allclose(model.previous_first_raw, current_raw)


def test_backward_allocates_target_and_inverts_active_edges() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
    )
    model.layers[0] = SparseConnectionLayer(
        2,
        1,
        generator=torch.Generator().manual_seed(2),
        weight_low=1.0,
        weight_high=1.0,
        bias_low=0.0,
        bias_high=0.0,
    )
    layer = model.layers[0]
    layer.weight = torch.tensor([1.0, 2.0])
    layer.bias = torch.tensor([0.5, 0.5])

    desired_source = model.backward(
        0,
        torch.tensor([5.0]),
        torch.tensor([1.0, 3.0]),
    )
    reconstructed, _ = layer.forward(desired_source)

    assert torch.allclose(
        desired_source,
        torch.tensor([1.75, 2.125]),
    )
    assert torch.allclose(reconstructed, torch.tensor([5.0]))


def test_backward_deactivates_low_gain_and_reallocates_target() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
    )
    model.layers[0] = SparseConnectionLayer(
        2,
        1,
        generator=torch.Generator().manual_seed(3),
        weight_low=1.0,
        weight_high=1.0,
        bias_low=0.0,
        bias_high=0.0,
    )
    layer = model.layers[0]
    layer.weight = torch.tensor([1e-6, 2.0])
    layer.bias = torch.tensor([0.5, 0.5])

    desired_source = model.backward(
        0,
        torch.tensor([5.0]),
        torch.tensor([1.0, 3.0]),
    )
    reconstructed, _ = layer.forward(desired_source)

    assert torch.equal(desired_source[:1], torch.tensor([0.0]))
    assert torch.allclose(desired_source[1:], torch.tensor([2.75]))
    assert layer.weight[0].item() == 0.0
    assert torch.allclose(reconstructed, torch.tensor([5.0]))


def test_output_target_is_canonical_positive_negative_encoding() -> None:
    target_delta = torch.tensor([[-1.0, 0.0], [0.5, 0.0]])

    desired = DeepMovingDotPredictor.output_target(target_delta)

    assert torch.equal(desired[:4], torch.tensor([0.0, 0.0, 0.5, 0.0]))
    assert torch.equal(desired[4:], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_negative_quota_penalizes_all_but_least_bad_n() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
        negative_tolerance=10,
    )
    model.layers[0] = SparseConnectionLayer(
        100,
        1,
        generator=torch.Generator().manual_seed(4),
        weight_low=1.0,
        weight_high=1.0,
        bias_low=0.0,
        bias_high=0.0,
    )
    current_feedback = torch.cat(
        [
            torch.tensor([3.0, 2.0, 1.0]),
            torch.zeros(10),
            -torch.arange(1.0, 88.0),
        ]
    )

    adjustment, penalized = model._competition_bias_adjustment(
        0,
        current_feedback,
        current_feedback,
    )

    assert penalized == 77
    assert int((adjustment > 0).sum()) == 77
    assert int((adjustment < 0).sum()) == 3
    assert int((adjustment == 0).sum()) == 20


def test_pi_history_does_not_act_when_current_feedback_is_zero() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
    )
    model.competition_integral[0].fill_(5.0)

    pi_value = model._competition_pi(
        0,
        torch.zeros_like(model.competition_integral[0]),
    )

    assert torch.count_nonzero(pi_value) == 0
    assert torch.all(model.competition_integral[0] == 5.0)


def test_bias_penalty_can_close_edge_before_weight_update() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
        negative_tolerance=0,
        competition_proportional_gain=1.0,
        competition_integral_gain=0.0,
        severe_penalty_multiplier=4.0,
    )
    layer = model.layers[0]
    layer.weight.fill_(1.0)
    layer.bias.fill_(0.5)
    old_weight = layer.weight.clone()

    model._update_layer(
        0,
        torch.tensor([1.0]),
        torch.tensor([-1.0]),
        torch.tensor([True]),
        torch.tensor([0.5]),
    )

    assert layer.bias.item() > 1.0
    assert torch.equal(layer.weight, old_weight)


def test_correction_updates_every_layer_without_autograd() -> None:
    model = DeepMovingDotPredictor(hidden_layer_count=2)
    generator = torch.Generator().manual_seed(31)
    before = [layer.weight.clone() for layer in model.layers]

    for _ in range(100):
        previous, current, next_frame = sample_transition(generator, 6)
        model.correct(
            model.record_from_pair(previous, current),
            next_frame - current,
        )

    for layer, old_weight in zip(model.layers, before):
        assert torch.any(layer.weight != old_weight)
        assert layer.weight.grad is None
        assert not layer.weight.requires_grad


def test_one_hidden_layer_does_not_use_inverse_replay() -> None:
    model = DeepMovingDotPredictor(hidden_layer_count=1)
    generator = torch.Generator().manual_seed(37)
    previous, current, next_frame = sample_transition(generator, 6)

    def fail_if_called(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("one hidden layer must not replay an inverse")

    model.backward = fail_if_called  # type: ignore[method-assign]
    model.correct(
        model.record_from_pair(previous, current),
        next_frame - current,
    )


def test_frozen_hidden_control_updates_only_output_layer() -> None:
    model = DeepMovingDotPredictor(
        hidden_layer_count=2,
        update_hidden=False,
    )
    generator = torch.Generator().manual_seed(41)
    before = [layer.weight.clone() for layer in model.layers]

    for _ in range(10):
        previous, current, next_frame = sample_transition(generator, 6)
        model.correct(
            model.record_from_pair(previous, current),
            next_frame - current,
        )

    assert torch.equal(model.layers[0].weight, before[0])
    assert torch.equal(model.layers[1].weight, before[1])
    assert torch.any(model.layers[2].weight != before[2])


def test_inactivity_pruning_physically_filters_edges_and_state() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
    )
    model.activity_observations = 10
    for counts in model.activity_counts:
        counts.zero_()
        counts[0] = 10

    pruned = model.prune_inactive(minimum_activity_rate=0.1)

    assert pruned == 3
    assert [layer.edge_count for layer in model.layers] == [1, 1]
    assert [state.numel() for state in model.competition_integral] == [1, 1]


def test_pruning_removes_active_edges_with_negligible_weight() -> None:
    model = DeepMovingDotPredictor(
        grid_size=1,
        hidden_size=2,
        hidden_layer_count=1,
    )
    model.activity_observations = 10
    for layer, counts in zip(model.layers, model.activity_counts):
        counts.fill_(10)
        layer.weight[0] = 1e-6

    pruned = model.prune_inactive(
        minimum_activity_rate=0.1,
        minimum_weight=1e-4,
    )

    assert pruned == 2
    assert [layer.edge_count for layer in model.layers] == [0, 3]
