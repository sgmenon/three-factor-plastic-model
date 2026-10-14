"""Configurable deep moving-dot experiment with active-path reverse targets.

The forward graph contains only adjacent positive connection layers. Hidden
layer one carries a temporal residual; later hidden layers are stateless.
Learning uses no autograd:

1. update the output from the externally measured pixel error;
2. allocate that correction directly to the final hidden layer;
3. invert through updated active connections only for earlier hidden layers.

Within each neuron, active incoming connections accumulate local feedback.
Positive connections receive an extra bias reward. Of the currently negative
connections, all but the least-bad N receive a stronger bias penalty.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from experiments.adaptive_correction import PICorrection
from experiments.moving_dot import (
    advance,
    frame_from_state,
    random_state,
    sample_trajectory,
    sample_transition,
)
from experiments.sparse_connections import SparseConnectionLayer


@dataclass(frozen=True)
class DeepVideoRecord:
    current_input: Tensor
    previous_first_raw: Tensor
    raw_hidden: tuple[Tensor, ...]
    hidden: tuple[Tensor, ...]
    output: Tensor
    predicted_delta: Tensor
    edge_active: tuple[Tensor, ...]
    edge_contribution: tuple[Tensor, ...]


class DeepMovingDotPredictor:
    """Symmetric hidden stack trained by tied reverse target generation."""

    def __init__(
        self,
        *,
        grid_size: int = 6,
        hidden_size: int = 96,
        hidden_layer_count: int = 2,
        seed: int = 3,
        temporal_residual_gain: float = 1.0,
        weight_learning_rate: float = 5e-2,
        bias_proportional_gain: float = 1.5e-3,
        bias_integral_gain: float = 1.5e-4,
        bias_integral_decay: float = 0.99,
        bias_integral_limit: float = 20.0,
        competition_proportional_gain: float = 5e-2,
        competition_integral_gain: float = 5e-3,
        competition_integral_decay: float = 1.0,
        severe_penalty_multiplier: float = 4.0,
        negative_tolerance: int = 10,
        minimum_weight: float = 1e-4,
        update_hidden: bool = True,
    ) -> None:
        if hidden_layer_count < 1:
            raise ValueError("At least one hidden layer is required")
        if hidden_size < 1:
            raise ValueError("Hidden size must be positive")
        if hidden_size % 2:
            raise ValueError(
                "Hidden size must be even for paired temporal features"
            )
        if temporal_residual_gain <= 0:
            raise ValueError("Temporal residual gain must be positive")
        if negative_tolerance < 0:
            raise ValueError("Negative tolerance cannot be negative")
        if not 0 <= competition_integral_decay <= 1:
            raise ValueError(
                "Competition integral decay must be between zero and one"
            )

        generator = torch.Generator().manual_seed(seed)
        self.grid_size = grid_size
        self.pixel_count = grid_size * grid_size
        self.input_size = self.pixel_count
        self.hidden_size = hidden_size
        self.first_raw_size = hidden_size // 2
        self.hidden_layer_count = hidden_layer_count
        self.output_size = 2 * self.pixel_count
        self.temporal_residual_gain = temporal_residual_gain
        self.weight_learning_rate = weight_learning_rate
        self.bias_proportional_gain = bias_proportional_gain
        self.bias_integral_gain = bias_integral_gain
        self.bias_integral_decay = bias_integral_decay
        self.bias_integral_limit = bias_integral_limit
        self.competition_proportional_gain = (
            competition_proportional_gain
        )
        self.competition_integral_gain = competition_integral_gain
        self.competition_integral_decay = competition_integral_decay
        self.severe_penalty_multiplier = severe_penalty_multiplier
        self.negative_tolerance = negative_tolerance
        self.minimum_weight = minimum_weight
        self.update_hidden = update_hidden

        hidden_arguments = {
            "weight_low": 0.08,
            "weight_high": 0.20,
            "bias_low": 0.005,
            "bias_high": 0.02,
        }
        self.layers: list[SparseConnectionLayer] = [
            SparseConnectionLayer(
                self.input_size,
                self.first_raw_size,
                generator=generator,
                **hidden_arguments,
            )
        ]
        source_count = hidden_size
        for _ in range(1, hidden_layer_count):
            self.layers.append(
                SparseConnectionLayer(
                    source_count,
                    hidden_size,
                    generator=generator,
                    **hidden_arguments,
                )
            )
            source_count = hidden_size
        self.layers.append(
            SparseConnectionLayer(
                hidden_size,
                self.output_size,
                generator=generator,
                weight_low=0.08,
                weight_high=0.20,
                bias_low=0.002,
                bias_high=0.014,
            )
        )

        controller_arguments = {
            "proportional_gain": bias_proportional_gain,
            "integral_gain": bias_integral_gain,
            "integral_decay": bias_integral_decay,
            "integral_limit": bias_integral_limit,
        }
        self.bias_controllers = [
            PICorrection(layer.edge_count, **controller_arguments)
            for layer in self.layers
        ]
        self.competition_integral = [
            torch.zeros(layer.edge_count) for layer in self.layers
        ]
        self.activity_counts = [
            torch.zeros(layer.edge_count, dtype=torch.int64)
            for layer in self.layers
        ]
        self.activity_observations = 0
        self.total_pruned = 0
        self.total_severely_penalized = 0
        self.last_severely_penalized = tuple(
            0 for _ in self.layers
        )
        self.previous_first_raw = torch.zeros(self.first_raw_size)

    @property
    def output_layer_index(self) -> int:
        return len(self.layers) - 1

    def reset_state(self) -> None:
        self.previous_first_raw = torch.zeros(self.first_raw_size)

    def prime_state(self, first_frame: Tensor) -> None:
        self.reset_state()
        raw, _ = self.layers[0].forward(first_frame.flatten())
        self.previous_first_raw = raw.clone()

    def forward(self, current_frame: Tensor) -> DeepVideoRecord:
        current_input = current_frame.flatten()
        previous_first_raw = self.previous_first_raw.clone()
        source = current_input
        raw_hidden: list[Tensor] = []
        hidden: list[Tensor] = []
        edge_active: list[Tensor] = []
        edge_contribution: list[Tensor] = []

        for index, layer in enumerate(self.layers[:-1]):
            raw, active = layer.forward(source)
            contribution = layer.edge_contribution(source)
            raw_hidden.append(raw)
            edge_active.append(active)
            edge_contribution.append(contribution)
            if index == 0:
                temporal_change = self.temporal_residual_gain * (
                    raw - previous_first_raw
                )
                state = torch.cat(
                    [
                        torch.relu(temporal_change),
                        torch.relu(-temporal_change),
                    ]
                )
                self.previous_first_raw = raw.clone()
            else:
                state = raw
            hidden.append(state)
            source = state

        output_layer = self.layers[-1]
        output, output_active = output_layer.forward(source)
        output_contribution = output_layer.edge_contribution(source)
        edge_active.append(output_active)
        edge_contribution.append(output_contribution)
        predicted_delta = (
            output[: self.pixel_count] - output[self.pixel_count :]
        )
        return DeepVideoRecord(
            current_input=current_input,
            previous_first_raw=previous_first_raw,
            raw_hidden=tuple(raw_hidden),
            hidden=tuple(hidden),
            output=output,
            predicted_delta=predicted_delta.reshape(
                self.grid_size,
                self.grid_size,
            ),
            edge_active=tuple(edge_active),
            edge_contribution=tuple(edge_contribution),
        )

    def record_from_pair(
        self,
        previous_frame: Tensor,
        current_frame: Tensor,
    ) -> DeepVideoRecord:
        saved_state = self.previous_first_raw.clone()
        self.prime_state(previous_frame)
        record = self.forward(current_frame)
        self.previous_first_raw = saved_state
        return record

    @staticmethod
    def output_target(target_delta: Tensor) -> Tensor:
        flat_target = target_delta.flatten()
        return torch.cat(
            [torch.relu(flat_target), torch.relu(-flat_target)]
        )

    def backward(
        self,
        layer_index: int,
        desired_target: Tensor,
        recorded_contribution: Tensor,
    ) -> Tensor:
        """Approximate the source state that would produce desired_target."""
        layer = self.layers[layer_index]
        low_gain = layer.weight <= self.minimum_weight
        layer.weight[low_gain] = 0.0
        invertible = (recorded_contribution > 0) & (~low_gain)
        invertible_contribution = (
            recorded_contribution * invertible
        )
        target_contribution = torch.zeros(
            layer.target_count,
            dtype=desired_target.dtype,
            device=desired_target.device,
        )
        target_contribution.scatter_add_(
            0,
            layer.target_index,
            invertible_contribution,
        )
        allocation = (
            invertible_contribution
            / target_contribution[layer.target_index].clamp_min(1e-12)
        )
        desired_edge_contribution = (
            allocation * desired_target[layer.target_index]
        )
        source_request = torch.zeros_like(recorded_contribution)
        source_request[invertible] = (
            desired_edge_contribution[invertible] + layer.bias[invertible]
        ) / layer.weight[invertible]

        desired_source_sum = torch.zeros(
            layer.source_count,
            dtype=desired_target.dtype,
            device=desired_target.device,
        )
        desired_source_weight = torch.zeros_like(desired_source_sum)
        desired_source_sum.scatter_add_(
            0,
            layer.source_index,
            source_request * allocation,
        )
        desired_source_weight.scatter_add_(
            0,
            layer.source_index,
            allocation,
        )
        return (
            desired_source_sum
            / desired_source_weight.clamp_min(1e-12)
        )

    def _competition_pi(
        self,
        layer_index: int,
        current_feedback: Tensor,
    ) -> Tensor:
        active_feedback = current_feedback != 0
        accumulated = (
            self.competition_integral_decay
            * self.competition_integral[layer_index]
            + current_feedback
        )
        self.competition_integral[layer_index] = torch.where(
            active_feedback,
            accumulated,
            self.competition_integral[layer_index],
        )
        pi_value = (
            self.competition_proportional_gain * current_feedback
            + self.competition_integral_gain
            * self.competition_integral[layer_index]
        )
        return torch.where(
            active_feedback,
            pi_value,
            torch.zeros_like(pi_value),
        )

    def _competition_bias_adjustment(
        self,
        layer_index: int,
        current_feedback: Tensor,
        pi_feedback: Tensor,
    ) -> tuple[Tensor, int]:
        """Return an additive bias adjustment and severe-penalty count."""
        layer = self.layers[layer_index]
        adjustment = torch.zeros_like(current_feedback)
        positive = current_feedback > 0
        adjustment[positive] = -pi_feedback[positive].abs()

        severely_penalized = 0
        for target in range(layer.target_count):
            negative = torch.nonzero(
                (layer.target_index == target) & (current_feedback < 0),
                as_tuple=False,
            ).flatten()
            excess = negative.numel() - self.negative_tolerance
            if excess <= 0:
                continue
            order = torch.argsort(pi_feedback[negative])
            worst = negative[order[:excess]]
            effective = worst[pi_feedback[worst] != 0]
            adjustment[effective] = (
                self.severe_penalty_multiplier
                * pi_feedback[effective].abs()
            )
            severely_penalized += effective.numel()
        return adjustment, severely_penalized

    def _recorded_source(
        self,
        record: DeepVideoRecord,
        layer_index: int,
    ) -> Tensor:
        if layer_index == 0:
            return record.current_input
        return record.hidden[layer_index - 1]

    def _recorded_target(
        self,
        record: DeepVideoRecord,
        layer_index: int,
    ) -> Tensor:
        if layer_index == self.output_layer_index:
            return record.output
        if layer_index == 0:
            return record.raw_hidden[0]
        return record.hidden[layer_index]

    def _first_raw_target(
        self,
        desired_first_hidden: Tensor,
        previous_first_raw: Tensor,
    ) -> Tensor:
        gain = self.temporal_residual_gain
        positive = desired_first_hidden[: self.first_raw_size]
        negative = desired_first_hidden[self.first_raw_size :]
        return (
            previous_first_raw + (positive - negative) / gain
        ).clamp_min(0.0)

    def _update_layer(
        self,
        layer_index: int,
        source: Tensor,
        target_correction: Tensor,
        recorded_active: Tensor,
        recorded_contribution: Tensor,
    ) -> int:
        layer = self.layers[layer_index]
        edge_correction = (
            target_correction[layer.target_index] * recorded_active
        )
        ordinary_bias_update = self.bias_controllers[layer_index].update(
            edge_correction
        )
        current_feedback = (
            target_correction[layer.target_index]
            * recorded_contribution
        )
        pi_feedback = self._competition_pi(
            layer_index,
            current_feedback,
        )
        competition_adjustment, penalized = (
            self._competition_bias_adjustment(
                layer_index,
                current_feedback,
                pi_feedback,
            )
        )

        layer.bias.sub_(ordinary_bias_update)
        layer.bias.add_(competition_adjustment)
        layer.bias.clamp_(min=0.0)

        final_active = (
            layer.edge_contribution(source) > 0
        ) & recorded_active
        layer.weight.add_(
            self.weight_learning_rate
            * target_correction[layer.target_index]
            * source[layer.source_index]
            * final_active
        )
        return penalized

    def correct(
        self,
        record: DeepVideoRecord,
        target_delta: Tensor,
    ) -> Tensor:
        """Use direct output error, then invert only between hidden layers."""
        flat_target = target_delta.flatten()
        error = flat_target - record.predicted_delta.flatten()
        penalized_by_layer = [0 for _ in self.layers]

        output_correction = torch.cat([error, -error])
        output_correction *= record.output > 0
        output_index = self.output_layer_index
        penalized_by_layer[output_index] = self._update_layer(
            output_index,
            record.hidden[-1],
            output_correction,
            record.edge_active[output_index],
            record.edge_contribution[output_index],
        )

        output_layer = self.layers[output_index]
        output_edge_correction = (
            output_correction[output_layer.target_index]
            * record.edge_active[output_index]
        )
        hidden_correction_sum = torch.zeros(self.hidden_size)
        hidden_correction_sum.scatter_add_(
            0,
            output_layer.source_index,
            output_edge_correction
            * record.edge_contribution[output_index],
        )
        outgoing_contribution = torch.zeros(self.hidden_size)
        outgoing_contribution.scatter_add_(
            0,
            output_layer.source_index,
            record.edge_contribution[output_index],
        )
        hidden_correction = (
            hidden_correction_sum
            / outgoing_contribution.clamp_min(1e-12)
        )
        hidden_correction *= record.hidden[-1] > 0
        desired_target = record.hidden[-1] + hidden_correction

        for layer_index in reversed(range(self.hidden_layer_count)):
            recorded_target = self._recorded_target(
                record,
                layer_index,
            )
            if layer_index == 0:
                desired_for_update = self._first_raw_target(
                    desired_target,
                    record.previous_first_raw,
                )
            else:
                desired_for_update = desired_target
            target_correction = desired_for_update - recorded_target

            if self.update_hidden:
                penalized_by_layer[layer_index] = self._update_layer(
                    layer_index,
                    self._recorded_source(record, layer_index),
                    target_correction,
                    record.edge_active[layer_index],
                    record.edge_contribution[layer_index],
                )

            if layer_index > 0:
                desired_target = self.backward(
                    layer_index,
                    desired_target,
                    record.edge_contribution[layer_index],
                )

        for index, active in enumerate(record.edge_active):
            self.activity_counts[index] += active.to(torch.int64)
        self.activity_observations += 1
        self.last_severely_penalized = tuple(penalized_by_layer)
        self.total_severely_penalized += sum(penalized_by_layer)
        return error.square().mean()

    def prune_inactive(
        self,
        *,
        minimum_activity_rate: float = 0.001,
        minimum_weight: float | None = None,
    ) -> int:
        """Physically remove inactive edges and edges with negligible gain."""
        if self.activity_observations == 0:
            return 0
        if minimum_weight is None:
            minimum_weight = self.minimum_weight
        pruned = 0
        for index, layer in enumerate(self.layers):
            activity_rate = (
                self.activity_counts[index] / self.activity_observations
            )
            keep = (
                (activity_rate >= minimum_activity_rate)
                & (layer.weight > minimum_weight)
            )
            pruned += int((~keep).sum().item())
            layer.filter_edges(keep)
            self.bias_controllers[index].filter_connections(keep)
            self.competition_integral[index] = (
                self.competition_integral[index][keep]
            )
            self.activity_counts[index] = torch.zeros(
                layer.edge_count,
                dtype=torch.int64,
            )
        self.activity_observations = 0
        self.total_pruned += pruned
        return pruned

    def present_connection_count(self) -> int:
        return sum(layer.edge_count for layer in self.layers)

    def learned_parameter_count(self) -> int:
        return 2 * self.present_connection_count()

    def training_state_count(self) -> int:
        return 2 * self.present_connection_count()

    def predict_next(self, current_frame: Tensor) -> Tensor:
        delta = self.forward(current_frame).predicted_delta
        return (current_frame + delta).clamp(0.0, 1.0)


def evaluation_metrics(
    model: DeepMovingDotPredictor,
    *,
    seed: int = 10_003,
    examples: int = 512,
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    squared_error = 0.0
    persistence_error = 0.0
    correct_positions = 0
    for _ in range(examples):
        previous, current, next_frame = sample_transition(
            generator,
            model.grid_size,
        )
        record = model.record_from_pair(previous, current)
        target_delta = next_frame - current
        squared_error += (
            target_delta - record.predicted_delta
        ).square().mean().item()
        persistence_error += target_delta.square().mean().item()
        prediction = current + record.predicted_delta
        correct_positions += int(
            prediction.flatten().argmax() == next_frame.flatten().argmax()
        )
    return (
        squared_error / examples,
        persistence_error / examples,
        correct_positions / examples,
    )


def evaluation_active_edge_fractions(
    model: DeepMovingDotPredictor,
    *,
    seed: int = 10_003,
    examples: int = 512,
) -> tuple[float, ...]:
    generator = torch.Generator().manual_seed(seed)
    active = [0 for _ in model.layers]
    possible = [0 for _ in model.layers]
    for _ in range(examples):
        previous, current, _ = sample_transition(
            generator,
            model.grid_size,
        )
        record = model.record_from_pair(previous, current)
        for index, edge_active in enumerate(record.edge_active):
            active[index] += int(edge_active.sum().item())
            possible[index] += edge_active.numel()
    return tuple(
        count / total if total else 0.0
        for count, total in zip(active, possible)
    )


def rollout_position_accuracy(
    model: DeepMovingDotPredictor,
    *,
    seed: int = 20_003,
    rollouts: int = 64,
    horizon: int = 8,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    correct = 0
    for _ in range(rollouts):
        previous_state = random_state(generator, model.grid_size)
        current_state = advance(previous_state, model.grid_size)
        previous = frame_from_state(previous_state, model.grid_size)
        current = frame_from_state(current_state, model.grid_size)
        model.prime_state(previous)
        for _ in range(horizon):
            current_state = advance(current_state, model.grid_size)
            actual = frame_from_state(current_state, model.grid_size)
            predicted = model.predict_next(current)
            correct += int(
                predicted.flatten().argmax() == actual.flatten().argmax()
            )
            current = predicted
    return correct / (rollouts * horizon)


def run_online_experiment(
    *,
    seed: int = 11,
    observations: int = 10_000,
    hidden_size: int = 96,
    hidden_layer_count: int = 2,
    negative_tolerance: int = 10,
    competition_proportional_gain: float = 5e-2,
    competition_integral_gain: float = 5e-3,
    severe_penalty_multiplier: float = 4.0,
    update_hidden: bool = True,
    pruning_interval: int | None = None,
    minimum_activity_rate: float = 0.001,
    minimum_weight: float = 1e-4,
    trajectory_frame_count: int = 10,
) -> tuple[
    DeepMovingDotPredictor,
    list[tuple[int, float, float, int]],
]:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = DeepMovingDotPredictor(
        hidden_size=hidden_size,
        hidden_layer_count=hidden_layer_count,
        seed=seed,
        negative_tolerance=negative_tolerance,
        competition_proportional_gain=competition_proportional_gain,
        competition_integral_gain=competition_integral_gain,
        severe_penalty_multiplier=severe_penalty_multiplier,
        minimum_weight=minimum_weight,
        update_hidden=update_hidden,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    initial_mse, _, initial_accuracy = evaluation_metrics(model)
    history = [
        (
            0,
            initial_mse,
            initial_accuracy,
            model.present_connection_count(),
            model.total_pruned,
        )
    ]
    trajectory: list[Tensor] = []
    trajectory_offset = 0
    for observation in range(1, observations + 1):
        if trajectory_offset + 2 >= len(trajectory):
            trajectory = sample_trajectory(
                generator,
                model.grid_size,
                trajectory_frame_count,
            )
            trajectory_offset = 0
            model.prime_state(trajectory[0])
        current = trajectory[trajectory_offset + 1]
        next_frame = trajectory[trajectory_offset + 2]
        trajectory_offset += 1
        record = model.forward(current)
        model.correct(record, next_frame - current)
        if (
            pruning_interval is not None
            and observation % pruning_interval == 0
        ):
            model.prune_inactive(
                minimum_activity_rate=minimum_activity_rate,
                minimum_weight=minimum_weight,
            )
        if observation % 2_000 == 0:
            mse, _, accuracy = evaluation_metrics(model)
            history.append(
                (
                    observation,
                    mse,
                    accuracy,
                    model.present_connection_count(),
                    model.total_pruned,
                )
            )
    return model, history


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--observations", type=int, default=10_000)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--negative-tolerance", type=int, default=10)
    parser.add_argument("--competition-kp", type=float, default=5e-2)
    parser.add_argument("--competition-ki", type=float, default=5e-3)
    parser.add_argument("--severe-multiplier", type=float, default=4.0)
    parser.add_argument("--pruning-interval", type=int, default=1000)
    parser.add_argument("--minimum-activity-rate", type=float, default=0.001)
    parser.add_argument("--minimum-weight", type=float, default=1e-4)
    parser.add_argument("--freeze-hidden", action="store_true")
    args = parser.parse_args()

    started = perf_counter()
    model, history = run_online_experiment(
        observations=args.observations,
        hidden_size=args.hidden_size,
        hidden_layer_count=args.hidden_layers,
        negative_tolerance=args.negative_tolerance,
        competition_proportional_gain=args.competition_kp,
        competition_integral_gain=args.competition_ki,
        severe_penalty_multiplier=args.severe_multiplier,
        update_hidden=not args.freeze_hidden,
        pruning_interval=args.pruning_interval,
        minimum_activity_rate=args.minimum_activity_rate,
        minimum_weight=args.minimum_weight,
    )
    elapsed = perf_counter() - started
    mse, persistence_mse, accuracy = evaluation_metrics(model)
    rollout = rollout_position_accuracy(model)
    active_fractions = evaluation_active_edge_fractions(model)
    print(f"Hidden layers:         {model.hidden_layer_count}")
    print(f"Hidden updates:        {model.update_hidden}")
    print(f"Autograd used:         no")
    print(f"Final MSE:             {mse:.6f}")
    print(f"Persistence MSE:       {persistence_mse:.6f}")
    print(f"One-step position:     {accuracy:.1%}")
    print(f"8-step rollout:        {rollout:.1%}")
    print(f"Connections present:   {model.present_connection_count()}")
    print(f"Connections pruned:    {model.total_pruned}")
    print(f"Learned parameters:    {model.learned_parameter_count()}")
    print(f"PI training state:     {model.training_state_count()}")
    print(f"Severe penalties:      {model.total_severely_penalized}")
    print(f"Training seconds:      {elapsed:.2f}")
    print(
        "Active edge fractions: "
        + ", ".join(f"{fraction:.1%}" for fraction in active_fractions)
    )
    print("Progress every 2,000 observations:")
    for observation, checkpoint_mse, checkpoint_accuracy, connections, pruned in history:
        print(
            f"  {observation:>6}: MSE={checkpoint_mse:.6f}, "
            f"accuracy={checkpoint_accuracy:.1%}, "
            f"connections={connections}",
            f"pruned={pruned}",
        )


if __name__ == "__main__":
    main()
