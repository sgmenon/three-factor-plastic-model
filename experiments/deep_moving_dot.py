"""Configurable deep moving-dot experiment with active-path reverse targets.

The forward graph contains only adjacent connection layers. Hidden layer one
carries a temporal residual; later hidden layers are stateless. A configurable
minority of hidden neurons can have fixed inhibitory outgoing signs.
Learning uses no autograd:

1. update the output from the externally measured pixel error;
2. allocate that correction directly to the final hidden layer;
3. invert through updated active connections only for earlier hidden layers.

Each active connection receives ordinary local PI bias correction. Structural
Gaussian locality replaces the removed ranked negative-feedback competition.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil
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


@dataclass(frozen=True)
class NeuronClassConnectivity:
    layer: int
    inhibitory_neurons: int
    excitatory_neurons: int
    inhibitory_mean_fan_in: float
    excitatory_mean_fan_in: float
    inhibitory_mean_fan_out: float
    excitatory_mean_fan_out: float
    top_inhibitory_neuron: int | None
    top_inhibitory_fan_in: int
    top_inhibitory_fan_out: int


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
        inhibitory_fraction: float = 0.0,
        local_connectivity_fraction: float | None = None,
        local_connectivity_sigma: float | None = None,
        local_connectivity_scope: str = "all",
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
        if not 0 <= inhibitory_fraction < 1:
            raise ValueError("Inhibitory fraction must be in [0, 1)")
        if (
            local_connectivity_fraction is not None
            and not 0 < local_connectivity_fraction <= 1
        ):
            raise ValueError(
                "Local connectivity fraction must be in (0, 1]"
            )
        if (
            local_connectivity_sigma is not None
            and local_connectivity_sigma <= 0
        ):
            raise ValueError("Local connectivity sigma must be positive")
        if local_connectivity_scope not in {"all", "hidden"}:
            raise ValueError(
                "Local connectivity scope must be 'all' or 'hidden'"
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
        self.inhibitory_fraction = inhibitory_fraction
        self.local_connectivity_fraction = local_connectivity_fraction
        self.local_connectivity_sigma = (
            local_connectivity_fraction
            if local_connectivity_sigma is None
            else local_connectivity_sigma
        )
        self.local_connectivity_scope = local_connectivity_scope
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
        if local_connectivity_fraction is not None:
            topology_generator = torch.Generator().manual_seed(
                seed + 2_000_003
            )
            for layer_index, layer in enumerate(self.layers):
                if (
                    local_connectivity_scope == "hidden"
                    and (
                        layer_index == 0
                        or layer_index == self.output_layer_index
                    )
                ):
                    continue
                layer.filter_edges(
                    self._local_connection_mask(
                        layer_index,
                        local_connectivity_fraction,
                        self.local_connectivity_sigma,
                        topology_generator,
                    )
                )

        type_generator = torch.Generator().manual_seed(seed + 1_000_003)
        hidden_target_sizes = [
            self.first_raw_size,
            *([hidden_size] * (hidden_layer_count - 1)),
        ]
        inhibitory_masks: list[Tensor] = []
        for neuron_count in hidden_target_sizes:
            inhibitory_count = round(inhibitory_fraction * neuron_count)
            mask = torch.zeros(neuron_count, dtype=torch.bool)
            if inhibitory_count:
                selected = torch.randperm(
                    neuron_count,
                    generator=type_generator,
                )[:inhibitory_count]
                mask[selected] = True
            inhibitory_masks.append(mask)
        self.hidden_inhibitory = tuple(inhibitory_masks)
        self.hidden_output_sign = (
            torch.cat(
                [
                    torch.where(
                        self.hidden_inhibitory[0],
                        -torch.ones(self.first_raw_size),
                        torch.ones(self.first_raw_size),
                    ),
                ]
                * 2
            ),
            *(
                torch.where(
                    mask,
                    -torch.ones(mask.numel()),
                    torch.ones(mask.numel()),
                )
                for mask in self.hidden_inhibitory[1:]
            ),
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
        self.activity_counts = [
            torch.zeros(layer.edge_count, dtype=torch.int64)
            for layer in self.layers
        ]
        self.activity_observations = 0
        self.total_pruned = 0
        self.previous_first_raw = torch.zeros(self.first_raw_size)

    @property
    def output_layer_index(self) -> int:
        return len(self.layers) - 1

    def _local_connection_mask(
        self,
        layer_index: int,
        fraction: float,
        sigma: float,
        generator: torch.Generator,
    ) -> Tensor:
        """Sample a fixed edge budget around proportional Gaussian centers."""
        layer = self.layers[layer_index]
        pair_mask = torch.zeros(
            layer.target_count,
            layer.source_count,
            dtype=torch.bool,
        )
        if layer_index == 1:
            source_position = (
                torch.arange(layer.source_count) % self.first_raw_size
            )
            source_position = (
                source_position.to(torch.float32) + 0.5
            ) / self.first_raw_size
        else:
            source_position = (
                torch.arange(layer.source_count, dtype=torch.float32) + 0.5
            ) / layer.source_count

        if layer_index == self.output_layer_index:
            group_size = self.pixel_count
            total_neighbors = max(
                1,
                ceil(fraction * layer.target_count),
            )
            first_group = (total_neighbors + 1) // 2
            neighbors_per_group = [first_group, total_neighbors - first_group]
            target_position = (
                torch.arange(group_size, dtype=torch.float32) + 0.5
            ) / group_size
            for source in range(layer.source_count):
                distance = (target_position - source_position[source]).abs()
                probability = torch.exp(
                    -(distance.square()) / (2 * sigma**2)
                )
                for group, neighbor_count in enumerate(
                    neighbors_per_group
                ):
                    if neighbor_count == 0:
                        continue
                    selected = torch.multinomial(
                        probability,
                        min(neighbor_count, group_size),
                        replacement=False,
                        generator=generator,
                    )
                    pair_mask[
                        group * group_size + selected,
                        source,
                    ] = True
        else:
            neighbor_count = max(
                1,
                ceil(fraction * layer.target_count),
            )
            target_position = (
                torch.arange(
                    layer.target_count,
                    dtype=torch.float32,
                )
                + 0.5
            ) / layer.target_count
            for source in range(layer.source_count):
                distance = (target_position - source_position[source]).abs()
                probability = torch.exp(
                    -(distance.square()) / (2 * sigma**2)
                )
                selected = torch.multinomial(
                    probability,
                    min(neighbor_count, layer.target_count),
                    replacement=False,
                    generator=generator,
                )
                pair_mask[selected, source] = True
        return pair_mask[layer.target_index, layer.source_index]

    def reset_state(self) -> None:
        self.previous_first_raw = torch.zeros(self.first_raw_size)

    def _edge_sign(self, layer_index: int) -> Tensor:
        layer = self.layers[layer_index]
        if layer_index == 0:
            return torch.ones(layer.edge_count)
        return self.hidden_output_sign[layer_index - 1][
            layer.source_index
        ]

    def _forward_layer(
        self,
        layer_index: int,
        source: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        layer = self.layers[layer_index]
        contribution = layer.edge_contribution(source)
        target = torch.zeros(
            layer.target_count,
            dtype=source.dtype,
            device=source.device,
        )
        target.scatter_add_(
            0,
            layer.target_index,
            self._edge_sign(layer_index) * contribution,
        )
        return torch.relu(target), contribution > 0, contribution

    def prime_state(self, first_frame: Tensor) -> None:
        self.reset_state()
        raw, _, _ = self._forward_layer(0, first_frame.flatten())
        self.previous_first_raw = raw.clone()

    def forward(self, current_frame: Tensor) -> DeepVideoRecord:
        current_input = current_frame.flatten()
        previous_first_raw = self.previous_first_raw.clone()
        source = current_input
        raw_hidden: list[Tensor] = []
        hidden: list[Tensor] = []
        edge_active: list[Tensor] = []
        edge_contribution: list[Tensor] = []

        for index, _layer in enumerate(self.layers[:-1]):
            raw, active, contribution = self._forward_layer(index, source)
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

        output, output_active, output_contribution = self._forward_layer(
            self.output_layer_index,
            source,
        )
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
        edge_sign = self._edge_sign(layer_index)
        current_target = torch.zeros_like(desired_target)
        current_target.scatter_add_(
            0,
            layer.target_index,
            edge_sign * invertible_contribution,
        )
        current_target.relu_()
        target_delta = desired_target - current_target
        if torch.all(edge_sign > 0):
            # Preserve the original positive-only calculation exactly.
            desired_edge_contribution = (
                allocation * desired_target[layer.target_index]
            )
        else:
            desired_edge_contribution = (
                invertible_contribution
                + edge_sign
                * allocation
                * target_delta[layer.target_index]
            )
            desired_edge_contribution.clamp_(min=0.0)
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
    ) -> None:
        layer = self.layers[layer_index]
        edge_sign = self._edge_sign(layer_index)
        edge_correction = (
            edge_sign
            * target_correction[layer.target_index]
            * recorded_active
        )
        ordinary_bias_update = self.bias_controllers[layer_index].update(
            edge_correction
        )

        layer.bias.sub_(ordinary_bias_update)
        layer.bias.clamp_(min=0.0)

        final_active = (
            layer.edge_contribution(source) > 0
        ) & recorded_active
        layer.weight.add_(
            self.weight_learning_rate
            * edge_sign
            * target_correction[layer.target_index]
            * source[layer.source_index]
            * final_active
        )

    def correct(
        self,
        record: DeepVideoRecord,
        target_delta: Tensor,
    ) -> Tensor:
        """Use direct output error, then invert only between hidden layers."""
        flat_target = target_delta.flatten()
        error = flat_target - record.predicted_delta.flatten()

        output_correction = torch.cat([error, -error])
        output_index = self.output_layer_index
        self._update_layer(
            output_index,
            record.hidden[-1],
            output_correction,
            record.edge_active[output_index],
        )

        output_layer = self.layers[output_index]
        output_edge_correction = (
            output_correction[output_layer.target_index]
            * self._edge_sign(output_index)
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
                self._update_layer(
                    layer_index,
                    self._recorded_source(record, layer_index),
                    target_correction,
                    record.edge_active[layer_index],
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
        return self.present_connection_count()

    def neuron_class_connectivity(
        self,
    ) -> tuple[NeuronClassConnectivity, ...]:
        """Summarize post-pruning fan-in and fan-out by hidden neuron type."""
        summaries: list[NeuronClassConnectivity] = []
        for layer_index, inhibitory in enumerate(self.hidden_inhibitory):
            neuron_count = inhibitory.numel()
            incoming = torch.bincount(
                self.layers[layer_index].target_index,
                minlength=neuron_count,
            )
            outgoing_source = self.layers[layer_index + 1].source_index
            if layer_index == 0:
                outgoing_source = outgoing_source % self.first_raw_size
            outgoing = torch.bincount(
                outgoing_source,
                minlength=neuron_count,
            )
            excitatory = ~inhibitory

            def mean_for(values: Tensor, mask: Tensor) -> float:
                if not torch.any(mask):
                    return 0.0
                return values[mask].to(torch.float32).mean().item()

            inhibitory_indices = torch.nonzero(
                inhibitory,
                as_tuple=False,
            ).flatten()
            if inhibitory_indices.numel():
                total_connectivity = incoming + outgoing
                top_offset = torch.argmax(
                    total_connectivity[inhibitory_indices]
                )
                top = int(inhibitory_indices[top_offset].item())
                top_fan_in = int(incoming[top].item())
                top_fan_out = int(outgoing[top].item())
            else:
                top = None
                top_fan_in = 0
                top_fan_out = 0

            summaries.append(
                NeuronClassConnectivity(
                    layer=layer_index + 1,
                    inhibitory_neurons=int(inhibitory.sum().item()),
                    excitatory_neurons=int(excitatory.sum().item()),
                    inhibitory_mean_fan_in=mean_for(
                        incoming,
                        inhibitory,
                    ),
                    excitatory_mean_fan_in=mean_for(
                        incoming,
                        excitatory,
                    ),
                    inhibitory_mean_fan_out=mean_for(
                        outgoing,
                        inhibitory,
                    ),
                    excitatory_mean_fan_out=mean_for(
                        outgoing,
                        excitatory,
                    ),
                    top_inhibitory_neuron=top,
                    top_inhibitory_fan_in=top_fan_in,
                    top_inhibitory_fan_out=top_fan_out,
                )
            )
        return tuple(summaries)

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
    inhibitory_fraction: float = 0.0,
    local_connectivity_fraction: float | None = None,
    local_connectivity_sigma: float | None = None,
    local_connectivity_scope: str = "all",
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
        inhibitory_fraction=inhibitory_fraction,
        local_connectivity_fraction=local_connectivity_fraction,
        local_connectivity_sigma=local_connectivity_sigma,
        local_connectivity_scope=local_connectivity_scope,
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
    parser.add_argument("--inhibitory-fraction", type=float, default=0.06)
    parser.add_argument("--local-connectivity", type=float, default=None)
    parser.add_argument("--local-connectivity-sigma", type=float, default=None)
    parser.add_argument(
        "--local-connectivity-scope",
        choices=["all", "hidden"],
        default="all",
    )
    parser.add_argument("--pruning-interval", type=int, default=1000)
    parser.add_argument("--no-pruning", action="store_true")
    parser.add_argument("--minimum-activity-rate", type=float, default=0.001)
    parser.add_argument("--minimum-weight", type=float, default=1e-3)
    parser.add_argument("--freeze-hidden", action="store_true")
    args = parser.parse_args()

    started = perf_counter()
    model, history = run_online_experiment(
        observations=args.observations,
        hidden_size=args.hidden_size,
        hidden_layer_count=args.hidden_layers,
        inhibitory_fraction=args.inhibitory_fraction,
        local_connectivity_fraction=args.local_connectivity,
        local_connectivity_sigma=args.local_connectivity_sigma,
        local_connectivity_scope=args.local_connectivity_scope,
        update_hidden=not args.freeze_hidden,
        pruning_interval=(
            None if args.no_pruning else args.pruning_interval
        ),
        minimum_activity_rate=args.minimum_activity_rate,
        minimum_weight=args.minimum_weight,
    )
    elapsed = perf_counter() - started
    mse, persistence_mse, accuracy = evaluation_metrics(model)
    rollout = rollout_position_accuracy(model)
    active_fractions = evaluation_active_edge_fractions(model)
    print(f"Hidden layers:         {model.hidden_layer_count}")
    print(f"Inhibitory fraction:   {model.inhibitory_fraction:.1%}")
    print(
        "Initial topology:      "
        + (
            "dense"
            if model.local_connectivity_fraction is None
            else (
                f"{model.local_connectivity_fraction:.1%} Gaussian "
                f"(sigma={model.local_connectivity_sigma:.3f}, "
                f"scope={model.local_connectivity_scope})"
            )
        )
    )
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
    print(f"Training seconds:      {elapsed:.2f}")
    print(
        "Active edge fractions: "
        + ", ".join(f"{fraction:.1%}" for fraction in active_fractions)
    )
    for summary in model.neuron_class_connectivity():
        print(
            f"Hidden {summary.layer} connectivity: "
            f"I={summary.inhibitory_neurons} "
            f"(in={summary.inhibitory_mean_fan_in:.1f}, "
            f"out={summary.inhibitory_mean_fan_out:.1f}), "
            f"E={summary.excitatory_neurons} "
            f"(in={summary.excitatory_mean_fan_in:.1f}, "
            f"out={summary.excitatory_mean_fan_out:.1f}), "
            f"top I={summary.top_inhibitory_neuron} "
            f"(in={summary.top_inhibitory_fan_in}, "
            f"out={summary.top_inhibitory_fan_out})"
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
