"""Three-hidden-layer moving-dot experiment with sparse long-range fanout."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

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

LongRangeMode = Literal["local", "uniform", "broadcaster"]


@dataclass(frozen=True)
class DeepVideoRecord:
    model_input: Tensor
    hidden: tuple[Tensor, Tensor, Tensor]
    output: Tensor
    predicted_delta: Tensor
    adjacent_edge_active: tuple[Tensor, Tensor, Tensor, Tensor]
    long_to_hidden_active: Tensor
    long_to_output_active: Tensor


class DeepMovingDotPredictor:
    """Three local layers plus optional first-to-later-layer projections."""

    def __init__(
        self,
        *,
        grid_size: int = 6,
        hidden_size: int = 48,
        seed: int = 3,
        long_range_mode: LongRangeMode = "local",
        broadcaster_count: int = 4,
        long_edge_budget: int = 480,
        weight_learning_rate: float = 1e-1,
        bias_proportional_gain: float = 2e-3,
        bias_integral_gain: float = 2e-4,
        bias_integral_decay: float = 0.99,
        bias_integral_limit: float = 20.0,
    ) -> None:
        if long_range_mode not in {"local", "uniform", "broadcaster"}:
            raise ValueError(f"Unsupported long-range mode: {long_range_mode}")
        if broadcaster_count > hidden_size:
            raise ValueError("Broadcaster count cannot exceed hidden size")

        generator = torch.Generator().manual_seed(seed)
        self.grid_size = grid_size
        self.pixel_count = grid_size * grid_size
        self.input_size = self.pixel_count * 2
        self.hidden_size = hidden_size
        self.output_size = self.pixel_count * 2
        self.long_range_mode = long_range_mode
        self.broadcaster_count = broadcaster_count
        self.weight_learning_rate = weight_learning_rate
        self.bias_proportional_gain = bias_proportional_gain
        self.bias_integral_gain = bias_integral_gain
        self.bias_integral_decay = bias_integral_decay
        self.bias_integral_limit = bias_integral_limit
        self.previous_frame = torch.zeros(self.pixel_count)

        self.layers = [
            SparseConnectionLayer(
                self.input_size,
                hidden_size,
                generator=generator,
                bias_low=0.08,
                bias_high=0.18,
            ),
            SparseConnectionLayer(
                hidden_size,
                hidden_size,
                generator=generator,
                weight_low=0.005,
                weight_high=0.02,
                bias_low=0.0005,
                bias_high=0.005,
            ),
            SparseConnectionLayer(
                hidden_size,
                hidden_size,
                generator=generator,
                weight_low=0.25,
                weight_high=1.0,
                bias_low=0.00005,
                bias_high=0.00025,
            ),
            SparseConnectionLayer(
                hidden_size,
                self.output_size,
                generator=generator,
                weight_low=0.05,
                weight_high=0.2,
                bias_low=0.002,
                bias_high=0.014,
            ),
        ]
        self.long_to_hidden = SparseConnectionLayer(
            hidden_size,
            hidden_size,
            generator=generator,
            weight_low=0.02,
            weight_high=0.08,
            bias_low=0.001,
            bias_high=0.006,
        )
        self.long_to_output = SparseConnectionLayer(
            hidden_size,
            self.output_size,
            generator=generator,
            weight_low=0.05,
            weight_high=0.2,
            bias_low=0.002,
            bias_high=0.014,
        )
        self._select_long_range_edges(
            generator,
            long_edge_budget=long_edge_budget,
        )
        self._reset_bias_controllers()
        self.total_long_pruned = 0
        self._reset_long_history()

    def _reset_bias_controllers(self) -> None:
        arguments = {
            "proportional_gain": self.bias_proportional_gain,
            "integral_gain": self.bias_integral_gain,
            "integral_decay": self.bias_integral_decay,
            "integral_limit": self.bias_integral_limit,
        }
        self.bias_controllers = [
            PICorrection(layer.edge_count, **arguments)
            for layer in (
                *self.layers,
                self.long_to_hidden,
                self.long_to_output,
            )
        ]

    def _select_long_range_edges(
        self,
        generator: torch.Generator,
        *,
        long_edge_budget: int,
    ) -> None:
        total_possible = (
            self.long_to_hidden.edge_count + self.long_to_output.edge_count
        )
        if self.long_range_mode == "local":
            selected = torch.zeros(total_possible, dtype=torch.bool)
        elif self.long_range_mode == "broadcaster":
            available = self.broadcaster_count * (
                self.hidden_size + self.output_size
            )
            if long_edge_budget != available:
                raise ValueError(
                    "Broadcaster budget must equal broadcaster_count times "
                    "all long-range targets"
                )
            hidden_selected = (
                self.long_to_hidden.source_index < self.broadcaster_count
            )
            output_selected = (
                self.long_to_output.source_index < self.broadcaster_count
            )
            selected = torch.cat([hidden_selected, output_selected])
        else:
            if long_edge_budget > total_possible:
                raise ValueError("Long-edge budget exceeds possible edges")
            selected = torch.zeros(total_possible, dtype=torch.bool)
            order = torch.randperm(total_possible, generator=generator)
            selected[order[:long_edge_budget]] = True

        hidden_count = self.long_to_hidden.edge_count
        self.long_to_hidden.filter_edges(selected[:hidden_count])
        self.long_to_output.filter_edges(selected[hidden_count:])

    def reset_state(self) -> None:
        self.previous_frame = torch.zeros(self.pixel_count)

    def prime_state(self, first_frame: Tensor) -> None:
        self.reset_state()
        self.previous_frame = first_frame.flatten().clone()

    def forward(self, current_frame: Tensor) -> DeepVideoRecord:
        current_input = current_frame.flatten()
        model_input = torch.cat([current_input, self.previous_frame])
        self.previous_frame = current_input.clone()
        hidden_1, active_1 = self.layers[0].forward(model_input)
        hidden_2, active_2 = self.layers[1].forward(hidden_1)
        hidden_3, active_3 = self.layers[2].forward(hidden_2)
        long_hidden, long_hidden_active = self.long_to_hidden.forward(
            hidden_1
        )
        hidden_3 += long_hidden

        output, output_active = self.layers[3].forward(hidden_3)
        long_output, long_output_active = self.long_to_output.forward(
            hidden_1
        )
        output += long_output
        predicted_delta = (
            output[: self.pixel_count] - output[self.pixel_count :]
        )
        return DeepVideoRecord(
            model_input=model_input,
            hidden=(hidden_1, hidden_2, hidden_3),
            output=output,
            predicted_delta=predicted_delta.reshape(
                self.grid_size,
                self.grid_size,
            ),
            adjacent_edge_active=(
                active_1,
                active_2,
                active_3,
                output_active,
            ),
            long_to_hidden_active=long_hidden_active,
            long_to_output_active=long_output_active,
        )

    def record_from_pair(
        self,
        previous_frame: Tensor,
        current_frame: Tensor,
    ) -> DeepVideoRecord:
        saved_state = self.previous_frame.clone()
        self.prime_state(previous_frame)
        record = self.forward(current_frame)
        self.previous_frame = saved_state
        return record

    @staticmethod
    def _aggregate_source_correction(
        layer: SparseConnectionLayer,
        edge_correction: Tensor,
        edge_active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        correction = torch.zeros(layer.source_count)
        correction.scatter_add_(0, layer.source_index, edge_correction)
        active_count = torch.zeros(layer.source_count)
        active_count.scatter_add_(
            0,
            layer.source_index,
            edge_active.to(torch.float32),
        )
        return correction, active_count

    @staticmethod
    def _edge_correction(
        layer: SparseConnectionLayer,
        target_correction: Tensor,
        edge_active: Tensor,
    ) -> Tensor:
        return target_correction[layer.target_index] * edge_active

    def _update_layer(
        self,
        layer: SparseConnectionLayer,
        source: Tensor,
        edge_correction: Tensor,
        bias_update: Tensor,
    ) -> None:
        layer.bias.sub_(bias_update)
        layer.weight.add_(
            self.weight_learning_rate
            * edge_correction
            * source[layer.source_index]
        )
        layer.bias.clamp_(min=0.0)

    def correct(
        self,
        record: DeepVideoRecord,
        target_delta: Tensor,
    ) -> Tensor:
        flat_target = target_delta.flatten()
        error = flat_target - record.predicted_delta.flatten()
        output_correction = torch.cat([error, -error])
        output_correction *= record.output > 0

        output_edge_correction = self._edge_correction(
            self.layers[3],
            output_correction,
            record.adjacent_edge_active[3],
        )
        long_output_correction = self._edge_correction(
            self.long_to_output,
            output_correction,
            record.long_to_output_active,
        )

        hidden_3_sum, hidden_3_count = self._aggregate_source_correction(
            self.layers[3],
            output_edge_correction,
            record.adjacent_edge_active[3],
        )
        hidden_3_correction = (
            hidden_3_sum / hidden_3_count.clamp_min(1.0)
        )
        hidden_3_correction *= record.hidden[2] > 0
        hidden_3_edge_correction = self._edge_correction(
            self.layers[2],
            hidden_3_correction,
            record.adjacent_edge_active[2],
        )
        long_hidden_correction = self._edge_correction(
            self.long_to_hidden,
            hidden_3_correction,
            record.long_to_hidden_active,
        )

        hidden_2_sum, hidden_2_count = self._aggregate_source_correction(
            self.layers[2],
            hidden_3_edge_correction,
            record.adjacent_edge_active[2],
        )
        hidden_2_correction = (
            hidden_2_sum / hidden_2_count.clamp_min(1.0)
        )
        hidden_2_correction *= record.hidden[1] > 0
        hidden_2_edge_correction = self._edge_correction(
            self.layers[1],
            hidden_2_correction,
            record.adjacent_edge_active[1],
        )

        hidden_1_adjacent_sum, hidden_1_adjacent_count = (
            self._aggregate_source_correction(
                self.layers[1],
                hidden_2_edge_correction,
                record.adjacent_edge_active[1],
            )
        )
        hidden_1_long_sum, hidden_1_long_count = (
            self._aggregate_source_correction(
                self.long_to_hidden,
                long_hidden_correction,
                record.long_to_hidden_active,
            )
        )
        hidden_1_output_sum, hidden_1_output_count = (
            self._aggregate_source_correction(
                self.long_to_output,
                long_output_correction,
                record.long_to_output_active,
            )
        )
        hidden_1_correction = (
            hidden_1_adjacent_sum
            + hidden_1_long_sum
            + hidden_1_output_sum
        ) / (
            hidden_1_adjacent_count
            + hidden_1_long_count
            + hidden_1_output_count
        ).clamp_min(1.0)
        hidden_1_correction *= record.hidden[0] > 0
        hidden_1_edge_correction = self._edge_correction(
            self.layers[0],
            hidden_1_correction,
            record.adjacent_edge_active[0],
        )

        self._update_layer(
            self.layers[3],
            record.hidden[2],
            output_edge_correction,
            self.bias_controllers[3].update(output_edge_correction),
        )
        self._update_layer(
            self.long_to_output,
            record.hidden[0],
            long_output_correction,
            self.bias_controllers[5].update(long_output_correction),
        )
        self._update_layer(
            self.layers[2],
            record.hidden[1],
            hidden_3_edge_correction,
            self.bias_controllers[2].update(hidden_3_edge_correction),
        )
        self._update_layer(
            self.long_to_hidden,
            record.hidden[0],
            long_hidden_correction,
            self.bias_controllers[4].update(long_hidden_correction),
        )
        self._update_layer(
            self.layers[1],
            record.hidden[0],
            hidden_2_edge_correction,
            self.bias_controllers[1].update(hidden_2_edge_correction),
        )
        self._update_layer(
            self.layers[0],
            record.model_input,
            hidden_1_edge_correction,
            self.bias_controllers[0].update(hidden_1_edge_correction),
        )

        self.long_history_observations += 1
        self.long_hidden_activity += record.long_to_hidden_active
        self.long_output_activity += record.long_to_output_active
        return error.square().mean()

    def _reset_long_history(self) -> None:
        self.long_history_observations = 0
        self.long_hidden_activity = torch.zeros(
            self.long_to_hidden.edge_count,
            dtype=torch.int32,
        )
        self.long_output_activity = torch.zeros(
            self.long_to_output.edge_count,
            dtype=torch.int32,
        )

    def prune_inactive_long_edges(
        self,
        *,
        minimum_activity_rate: float = 0.0,
    ) -> int:
        if self.long_history_observations == 0:
            return 0
        minimum_activations = int(
            self.long_history_observations * minimum_activity_rate
        )
        hidden_keep = self.long_hidden_activity > minimum_activations
        output_keep = self.long_output_activity > minimum_activations
        pruned = int((~hidden_keep).sum() + (~output_keep).sum())
        self.long_to_hidden.filter_edges(hidden_keep)
        self.long_to_output.filter_edges(output_keep)
        self.total_long_pruned += pruned
        self._reset_long_history()
        self._reset_bias_controllers()
        return pruned

    def long_edge_count(self) -> int:
        return (
            self.long_to_hidden.edge_count + self.long_to_output.edge_count
        )

    def present_connection_count(self) -> int:
        return sum(layer.edge_count for layer in self.layers) + (
            self.long_edge_count()
        )

    def learned_parameter_count(self) -> int:
        return 2 * self.present_connection_count()

    def predict_next(
        self,
        current_frame: Tensor,
    ) -> Tensor:
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
            previous, current = current, predicted
    return correct / (rollouts * horizon)


def run_online_experiment(
    *,
    seed: int = 11,
    observations: int = 10_000,
    hidden_size: int = 48,
    long_range_mode: LongRangeMode = "local",
    broadcaster_count: int = 4,
    long_edge_budget: int = 480,
    long_pruning_interval: int | None = None,
    trajectory_frame_count: int = 10,
) -> tuple[DeepMovingDotPredictor, list[tuple[int, float, float, int]]]:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = DeepMovingDotPredictor(
        hidden_size=hidden_size,
        seed=seed,
        long_range_mode=long_range_mode,
        broadcaster_count=broadcaster_count,
        long_edge_budget=long_edge_budget,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    initial_mse, _, initial_accuracy = evaluation_metrics(model)
    history = [(0, initial_mse, initial_accuracy, model.long_edge_count())]
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
        previous = trajectory[trajectory_offset]
        current = trajectory[trajectory_offset + 1]
        next_frame = trajectory[trajectory_offset + 2]
        trajectory_offset += 1
        record = model.forward(current)
        model.correct(record, next_frame - current)
        if (
            long_pruning_interval is not None
            and observation % long_pruning_interval == 0
        ):
            model.prune_inactive_long_edges()
        if observation % 2_000 == 0:
            mse, _, accuracy = evaluation_metrics(model)
            history.append(
                (
                    observation,
                    mse,
                    accuracy,
                    model.long_edge_count(),
                )
            )
    return model, history


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("local", "uniform", "broadcaster"),
        default="broadcaster",
    )
    parser.add_argument("--observations", type=int, default=10_000)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--broadcasters", type=int, default=4)
    parser.add_argument("--long-edge-budget", type=int, default=480)
    parser.add_argument("--long-pruning-interval", type=int)
    args = parser.parse_args()

    started = perf_counter()
    model, history = run_online_experiment(
        observations=args.observations,
        hidden_size=args.hidden_size,
        long_range_mode=args.mode,
        broadcaster_count=args.broadcasters,
        long_edge_budget=args.long_edge_budget,
        long_pruning_interval=args.long_pruning_interval,
    )
    elapsed = perf_counter() - started
    mse, persistence_mse, accuracy = evaluation_metrics(model)
    rollout = rollout_position_accuracy(model)
    print(f"Mode:                 {args.mode}")
    print(f"Autograd used:        no")
    print(f"Final MSE:            {mse:.6f}")
    print(f"Persistence MSE:      {persistence_mse:.6f}")
    print(f"One-step position:    {accuracy:.1%}")
    print(f"8-step rollout:       {rollout:.1%}")
    print(f"Connections present:  {model.present_connection_count()}")
    print(f"Long edges present:   {model.long_edge_count()}")
    print(f"Long edges pruned:    {model.total_long_pruned}")
    print(f"Learned parameters:   {model.learned_parameter_count()}")
    print(f"Training seconds:     {elapsed:.2f}")
    print("Progress every 2,000 observations:")
    for observation, checkpoint_mse, checkpoint_accuracy, long_edges in history:
        print(
            f"  {observation:>6}: MSE={checkpoint_mse:.6f}, "
            f"accuracy={checkpoint_accuracy:.1%}, "
            f"long_edges={long_edges}"
        )


if __name__ == "__main__":
    main()
