import torch

from experiments.deep_moving_dot import DeepMovingDotPredictor
from experiments.moving_dot import (
    DotState,
    frame_from_state,
    sample_transition,
)


def test_broadcaster_and_uniform_controls_have_equal_edge_budgets() -> None:
    broadcaster = DeepMovingDotPredictor(
        long_range_mode="broadcaster",
    )
    uniform = DeepMovingDotPredictor(long_range_mode="uniform")
    local = DeepMovingDotPredictor(long_range_mode="local")

    assert broadcaster.long_edge_count() == 480
    assert uniform.long_edge_count() == 480
    assert local.long_edge_count() == 0
    assert broadcaster.present_connection_count() == (
        uniform.present_connection_count()
    )


def test_broadcaster_edges_are_concentrated_in_four_sources() -> None:
    model = DeepMovingDotPredictor(long_range_mode="broadcaster")
    sources = torch.cat(
        [
            model.long_to_hidden.source_index,
            model.long_to_output.source_index,
        ]
    )

    assert torch.equal(sources.unique(), torch.arange(4))
    for source in range(4):
        assert int((sources == source).sum()) == 120


def test_deep_model_keeps_hidden_stages_stateless() -> None:
    model = DeepMovingDotPredictor(long_range_mode="local")
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
    assert not hasattr(model, "hidden_state")


def test_deep_correction_updates_every_adjacent_layer() -> None:
    model = DeepMovingDotPredictor(long_range_mode="broadcaster")
    generator = torch.Generator().manual_seed(31)
    before = [layer.weight.clone() for layer in model.layers]

    for _ in range(20):
        previous, current, next_frame = sample_transition(
            generator,
            grid_size=6,
        )
        model.correct(
            model.record_from_pair(previous, current),
            next_frame - current,
        )

    for layer, old_weight in zip(model.layers, before):
        assert torch.any(layer.weight != old_weight)
        assert layer.weight.grad is None
        assert not layer.weight.requires_grad


def test_long_range_pruning_physically_removes_inactive_edges() -> None:
    model = DeepMovingDotPredictor(long_range_mode="broadcaster")
    model.long_history_observations = 1
    model.long_hidden_activity.zero_()
    model.long_output_activity.zero_()
    model.long_hidden_activity[0] = 1
    model.long_output_activity[0] = 1

    pruned = model.prune_inactive_long_edges()

    assert pruned == 478
    assert model.long_edge_count() == 2
    assert model.total_long_pruned == 478
