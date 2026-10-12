import torch

from experiments.two_head_fork import TwoHeadFork, run_online_experiment


def test_equal_opposite_corrections_cancel_at_backbone() -> None:
    model = TwoHeadFork()
    record = model.forward(torch.tensor(1.0))

    old_backbone_weight = model.backbone_weight.clone()
    old_backbone_bias = model.backbone_bias.clone()
    old_head_1_weight = model.head_1_weight.clone()
    old_head_2_weight = model.head_2_weight.clone()

    model.correct(
        record,
        target_1=record.head_1_prediction + 0.2,
        target_2=record.head_2_prediction - 0.2,
    )

    assert torch.isclose(model.backbone_weight, old_backbone_weight)
    assert torch.isclose(model.backbone_bias, old_backbone_bias)
    assert model.head_1_weight > old_head_1_weight
    assert model.head_2_weight < old_head_2_weight


def test_two_heads_learn_different_targets_without_autograd() -> None:
    model, losses = run_online_experiment()

    assert losses[-1] < losses[0] * 0.01
    assert model.head_1_weight > model.head_2_weight

    for value in (
        model.backbone_weight,
        model.head_1_weight,
        model.head_2_weight,
    ):
        assert value.grad is None
        assert not value.requires_grad
