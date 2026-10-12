import torch

from experiments.scalar_chain import ScalarChain, run_online_experiment


def test_local_rule_reduces_error_without_autograd() -> None:
    model, losses = run_online_experiment(observations=10_000)

    assert losses[-1] < losses[0] * 0.01
    assert model.w1.grad is None
    assert model.w2.grad is None
    assert not model.w1.requires_grad
    assert not model.w2.requires_grad


def test_positive_correction_opens_gate_and_increases_weight() -> None:
    model = ScalarChain()
    x = torch.tensor(1.0)
    record = model.forward(x)
    target = record.prediction + 0.2

    old_weight = model.w2.clone()
    old_bias = model.b2.clone()
    model.correct(record, target)

    assert model.w2 > old_weight
    assert model.b2 < old_bias


def test_inactive_connection_receives_no_correction() -> None:
    model = ScalarChain()
    model.b2.fill_(10.0)
    record = model.forward(torch.tensor(1.0))

    old_weight = model.w2.clone()
    old_bias = model.b2.clone()
    model.correct(record, torch.tensor(1.0))

    assert model.w2 == old_weight
    assert model.b2 == old_bias
