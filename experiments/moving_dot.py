"""Stage 0: online next-frame prediction for a bouncing dot.

This experiment uses the current 6x6 frame plus residual hidden state to
predict the signed delta to the next frame. The first frame of each trajectory
is passed twice to seed that state.

All learning updates are written directly. PyTorch autograd and optimizers are
not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from argparse import ArgumentParser

from experiments.adaptive_correction import PICorrection
from experiments.sparse_connections import SparseConnectionLayer


DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class DotState:
    row: int
    column: int
    row_velocity: int
    column_velocity: int


def frame_from_state(state: DotState, grid_size: int) -> Tensor:
    frame = torch.zeros(grid_size, grid_size)
    frame[state.row, state.column] = 1.0
    return frame


def advance(state: DotState, grid_size: int) -> DotState:
    row_velocity = state.row_velocity
    column_velocity = state.column_velocity
    next_row = state.row + row_velocity
    next_column = state.column + column_velocity

    if next_row < 0 or next_row >= grid_size:
        row_velocity *= -1
        next_row = state.row + row_velocity
    if next_column < 0 or next_column >= grid_size:
        column_velocity *= -1
        next_column = state.column + column_velocity

    return DotState(
        row=next_row,
        column=next_column,
        row_velocity=row_velocity,
        column_velocity=column_velocity,
    )


def random_state(generator: torch.Generator, grid_size: int) -> DotState:
    direction_index = int(
        torch.randint(len(DIRECTIONS), (), generator=generator).item()
    )
    row_velocity, column_velocity = DIRECTIONS[direction_index]
    return DotState(
        row=int(torch.randint(grid_size, (), generator=generator).item()),
        column=int(
            torch.randint(grid_size, (), generator=generator).item()
        ),
        row_velocity=row_velocity,
        column_velocity=column_velocity,
    )


@dataclass(frozen=True)
class VideoRecord:
    model_input: Tensor
    hidden: Tensor
    predicted_delta: Tensor
    hidden_edge_active: Tensor
    hidden_active: Tensor
    output_edge_active: Tensor
    output_edge_contribution: Tensor
    output_active: Tensor


class MovingDotPredictor:
    """One hidden layer with connection-specific ReLU gates."""

    def __init__(
        self,
        *,
        grid_size: int = 6,
        hidden_size: int = 48,
        seed: int = 3,
        weight_learning_rate: float = 5e-2,
        bias_proportional_gain: float = 1.5e-3,
        bias_integral_gain: float = 1.5e-4,
        bias_integral_decay: float = 0.99,
        bias_integral_limit: float = 20.0,
        reinforcement_learning_rate: float = 0.0,
        reinforcement_interval: int | None = None,
        reward_error_tolerance: float = 0.1,
        meaningful_change_threshold: float = 0.1,
        update_hidden: bool = True,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)

        self.grid_size = grid_size
        self.pixel_count = grid_size * grid_size
        self.input_size = self.pixel_count * 2
        self.hidden_size = hidden_size
        self.output_size = self.pixel_count * 2

        self.hidden_layer = SparseConnectionLayer(
            self.input_size,
            hidden_size,
            generator=generator,
            bias_low=0.08,
            bias_high=0.18,
        )
        self.output_layer = SparseConnectionLayer(
            hidden_size,
            self.output_size,
            generator=generator,
            bias_low=0.002,
            bias_high=0.014,
        )
        self.weight_learning_rate = weight_learning_rate
        self.bias_proportional_gain = bias_proportional_gain
        self.bias_integral_gain = bias_integral_gain
        self.bias_integral_decay = bias_integral_decay
        self.bias_integral_limit = bias_integral_limit
        self._reset_bias_controllers()
        self.reinforcement_learning_rate = reinforcement_learning_rate
        self.reinforcement_interval = reinforcement_interval
        self.reward_error_tolerance = reward_error_tolerance
        self.meaningful_change_threshold = meaningful_change_threshold
        self.update_hidden = update_hidden
        self.previous_frame = torch.zeros(self.pixel_count)
        self.pruning_generator = torch.Generator().manual_seed(seed + 30_000)
        self.total_pruned = 0
        self.total_score_pruned = 0
        self.total_neurons_merged = 0
        self.total_reactivated = 0
        self.pruning_events = 0
        self._reset_pruning_history()
        self._reset_reinforcement_history()

    def _reset_bias_controllers(self) -> None:
        arguments = {
            "proportional_gain": self.bias_proportional_gain,
            "integral_gain": self.bias_integral_gain,
            "integral_decay": self.bias_integral_decay,
            "integral_limit": self.bias_integral_limit,
        }
        self.hidden_bias_controller = PICorrection(
            self.hidden_layer.edge_count,
            **arguments,
        )
        self.output_bias_controller = PICorrection(
            self.output_layer.edge_count,
            **arguments,
        )

    @property
    def hidden_weight(self) -> Tensor:
        return self.hidden_layer.weight

    @property
    def hidden_bias(self) -> Tensor:
        return self.hidden_layer.bias

    @property
    def output_weight(self) -> Tensor:
        return self.output_layer.weight

    @property
    def output_bias(self) -> Tensor:
        return self.output_layer.bias

    def reset_state(self) -> None:
        self.previous_frame = torch.zeros(self.pixel_count)

    def prime_state(self, first_frame: Tensor) -> None:
        self.reset_state()
        self.previous_frame = first_frame.flatten().clone()

    def forward(self, current_frame: Tensor) -> VideoRecord:
        current_input = current_frame.flatten()
        model_input = torch.cat([current_input, self.previous_frame])
        hidden, hidden_edge_active = self.hidden_layer.forward(
            model_input
        )
        self.previous_frame = current_input.clone()
        output, output_edge_active = self.output_layer.forward(hidden)
        output_edge_contribution = self.output_layer.edge_contribution(hidden)

        positive_delta = output[: self.pixel_count]
        negative_delta = output[self.pixel_count :]
        predicted_delta = positive_delta - negative_delta

        return VideoRecord(
            model_input=model_input,
            hidden=hidden,
            predicted_delta=predicted_delta.reshape(
                self.grid_size,
                self.grid_size,
            ),
            hidden_edge_active=hidden_edge_active,
            hidden_active=hidden > 0,
            output_edge_active=output_edge_active,
            output_edge_contribution=output_edge_contribution,
            output_active=output > 0,
        )

    def record_from_pair(
        self,
        previous_frame: Tensor,
        current_frame: Tensor,
    ) -> VideoRecord:
        """Evaluate a frame pair without changing the live temporal state."""
        saved_state = self.previous_frame.clone()
        self.prime_state(previous_frame)
        record = self.forward(current_frame)
        self.previous_frame = saved_state
        return record

    def correct(self, record: VideoRecord, target_delta: Tensor) -> Tensor:
        self._record_pruning_history(record)
        flat_target_delta = target_delta.flatten()
        error = flat_target_delta - record.predicted_delta.flatten()

        meaningful_change = (
            flat_target_delta.abs() > self.meaningful_change_threshold
        )
        low_error_reward = (
            1.0 - error.abs() / self.reward_error_tolerance
        ).clamp(min=0.0)
        pixel_reward = meaningful_change * low_error_reward

        output_correction = torch.cat([error, -error])
        output_correction *= record.output_active
        output_reward = torch.cat([pixel_reward, pixel_reward])
        output_reward *= record.output_active

        output_targets = self.output_layer.target_index
        output_sources = self.output_layer.source_index
        output_edge_correction = (
            output_correction[output_targets] * record.output_edge_active
        )
        output_edge_reward = (
            output_reward[output_targets] * record.output_edge_active
        )
        self.output_pruning_score += torch.sign(output_edge_correction)
        immediate_reinforcement_rate = (
            self.reinforcement_learning_rate
            if self.reinforcement_interval is None
            else 0.0
        )
        output_bias_update = self.output_bias_controller.update(
            output_edge_correction
        )
        self.output_layer.bias.sub_(
            output_bias_update
            + immediate_reinforcement_rate * output_edge_reward
        )
        self.output_layer.weight.add_(
            self.weight_learning_rate
            * output_edge_correction
            * record.hidden[output_sources]
        )

        # Every hidden neuron receives the mean correction from the output
        # connections that were active during this prediction.
        outgoing_activity = torch.zeros(self.hidden_size)
        outgoing_activity.scatter_add_(
            0,
            output_sources,
            record.output_edge_contribution,
        )
        hidden_correction = torch.zeros(self.hidden_size)
        hidden_correction.scatter_add_(
            0,
            output_sources,
            output_edge_correction * record.output_edge_contribution,
        )
        hidden_correction /= outgoing_activity.clamp_min(1e-12)
        hidden_correction *= record.hidden_active
        hidden_reward = torch.zeros(self.hidden_size)
        hidden_reward.scatter_add_(
            0,
            output_sources,
            output_edge_reward * record.output_edge_contribution,
        )
        hidden_reward /= outgoing_activity.clamp_min(1e-12)
        hidden_reward *= record.hidden_active

        hidden_targets = self.hidden_layer.target_index
        hidden_sources = self.hidden_layer.source_index
        hidden_edge_correction = (
            hidden_correction[hidden_targets] * record.hidden_edge_active
        )
        hidden_edge_reward = (
            hidden_reward[hidden_targets] * record.hidden_edge_active
        )
        self.hidden_pruning_score += torch.sign(hidden_edge_correction)
        if self.update_hidden:
            hidden_bias_update = self.hidden_bias_controller.update(
                hidden_edge_correction
            )
            self.hidden_layer.bias.sub_(
                hidden_bias_update
                + immediate_reinforcement_rate * hidden_edge_reward
            )
            self.hidden_layer.weight.add_(
                self.weight_learning_rate
                * hidden_edge_correction
                * record.model_input[hidden_sources]
            )
        if self.reinforcement_interval is not None:
            self._record_advantage_reinforcement(
                record,
                meaningful_change,
                low_error_reward,
            )
            if (
                self.reinforcement_observations
                >= self.reinforcement_interval
            ):
                self._apply_advantage_reinforcement()

        self.hidden_bias.clamp_(min=0.0)
        self.output_bias.clamp_(min=0.0)

        return error.square().mean()

    def _reset_reinforcement_history(self) -> None:
        self.reinforcement_observations = 0
        self.hidden_reinforcement_sum = torch.zeros(
            self.hidden_layer.edge_count
        )
        self.hidden_reinforcement_count = torch.zeros(
            self.hidden_layer.edge_count
        )
        self.output_reinforcement_sum = torch.zeros(
            self.output_layer.edge_count
        )
        self.output_reinforcement_count = torch.zeros(
            self.output_layer.edge_count
        )
        self.hidden_success_sum = torch.zeros(self.hidden_size)
        self.hidden_success_count = torch.zeros(self.hidden_size)
        self.output_success_sum = torch.zeros(self.output_size)
        self.output_success_count = torch.zeros(self.output_size)

    def _record_advantage_reinforcement(
        self,
        record: VideoRecord,
        meaningful_change: Tensor,
        low_error_reward: Tensor,
    ) -> None:
        self.reinforcement_observations += 1
        output_meaningful = torch.cat(
            [meaningful_change, meaningful_change]
        )
        output_success = torch.cat(
            [low_error_reward, low_error_reward]
        )
        output_eligible = output_meaningful & record.output_active
        self.output_success_sum += output_success * output_eligible
        self.output_success_count += output_eligible

        output_targets = self.output_layer.target_index
        output_sources = self.output_layer.source_index
        output_edge_eligible = (
            record.output_edge_active & output_eligible[output_targets]
        )
        self.output_reinforcement_sum += (
            output_success[output_targets] * output_edge_eligible
        )
        self.output_reinforcement_count += output_edge_eligible

        hidden_evidence_count = torch.zeros(self.hidden_size)
        hidden_evidence_count.scatter_add_(
            0,
            output_sources,
            output_edge_eligible.to(torch.float32),
        )
        hidden_success_sum = torch.zeros(self.hidden_size)
        hidden_success_sum.scatter_add_(
            0,
            output_sources,
            output_success[output_targets] * output_edge_eligible,
        )
        hidden_eligible = (
            (hidden_evidence_count > 0) & record.hidden_active
        )
        hidden_success = (
            hidden_success_sum / hidden_evidence_count.clamp_min(1.0)
        )
        self.hidden_success_sum += hidden_success * hidden_eligible
        self.hidden_success_count += hidden_eligible

        hidden_targets = self.hidden_layer.target_index
        hidden_edge_eligible = (
            record.hidden_edge_active & hidden_eligible[hidden_targets]
        )
        self.hidden_reinforcement_sum += (
            hidden_success[hidden_targets] * hidden_edge_eligible
        )
        self.hidden_reinforcement_count += hidden_edge_eligible

    def _apply_advantage_reinforcement(self) -> None:
        if self.reinforcement_observations == 0:
            return

        self._apply_layer_advantage(
            self.hidden_layer,
            self.hidden_reinforcement_sum,
            self.hidden_reinforcement_count,
            self.hidden_success_sum,
            self.hidden_success_count,
        )
        self._apply_layer_advantage(
            self.output_layer,
            self.output_reinforcement_sum,
            self.output_reinforcement_count,
            self.output_success_sum,
            self.output_success_count,
        )
        self._reset_reinforcement_history()

    def _apply_layer_advantage(
        self,
        layer: SparseConnectionLayer,
        edge_success_sum: Tensor,
        edge_success_count: Tensor,
        target_success_sum: Tensor,
        target_success_count: Tensor,
    ) -> None:
        target_baseline = (
            target_success_sum / target_success_count.clamp_min(1.0)
        )
        valid = (
            (edge_success_count > 0)
            & (target_success_count[layer.target_index] > 0)
        )
        centered_success = (
            edge_success_sum
            - edge_success_count * target_baseline[layer.target_index]
        )
        layer.bias[valid] -= (
            self.reinforcement_learning_rate * centered_success[valid]
        )
        layer.bias.clamp_(min=0.0)

    def _reset_pruning_history(self) -> None:
        self.history_observations = 0
        self.hidden_edge_activity = torch.zeros_like(
            self.hidden_weight,
            dtype=torch.int32,
        )
        self.hidden_pruning_score = torch.zeros_like(self.hidden_weight)
        self.output_edge_activity = torch.zeros_like(
            self.output_weight,
            dtype=torch.int32,
        )
        self.output_pruning_score = torch.zeros_like(self.output_weight)
        self.hidden_source_activity = torch.zeros(
            self.input_size,
            dtype=torch.int32,
        )
        self.output_source_activity = torch.zeros(
            self.hidden_size,
            dtype=torch.int32,
        )
        self.hidden_target_activity = torch.zeros(
            self.hidden_size,
            dtype=torch.int32,
        )
        self.output_target_activity = torch.zeros(
            self.output_size,
            dtype=torch.int32,
        )

    def _record_pruning_history(self, record: VideoRecord) -> None:
        self.history_observations += 1
        self.hidden_edge_activity += record.hidden_edge_active
        self.output_edge_activity += record.output_edge_active
        self.hidden_source_activity += record.model_input != 0
        self.output_source_activity += record.hidden != 0
        self.hidden_target_activity += record.hidden_active
        self.output_target_activity += record.output_active

    def prune_inactive(
        self,
        *,
        minimum_activity_rate: float = 0.0,
        minimum_pruning_score: float | None = None,
        reactivation_fraction: float = 0.05,
        merge_redundant_neurons: bool = False,
        merge_similarity_threshold: float = 0.95,
        merge_error_tolerance: float = 0.10,
    ) -> tuple[int, int]:
        """Prune unused edges, merge redundant edges, and reopen a few."""
        if self.history_observations == 0:
            return 0, 0

        if (
            self.reinforcement_interval is not None
            and self.reinforcement_observations > 0
        ):
            self._apply_advantage_reinforcement()

        minimum_activations = int(
            self.history_observations * minimum_activity_rate
        )
        hidden_keep = self.hidden_edge_activity > minimum_activations
        output_keep = self.output_edge_activity > minimum_activations
        hidden_score_pruned = torch.zeros_like(hidden_keep)
        output_score_pruned = torch.zeros_like(output_keep)
        if minimum_pruning_score is not None:
            hidden_score_pruned = (
                hidden_keep
                & (self.hidden_pruning_score < minimum_pruning_score)
            )
            output_score_pruned = (
                output_keep
                & (self.output_pruning_score < minimum_pruning_score)
            )
            hidden_keep &= ~hidden_score_pruned
            output_keep &= ~output_score_pruned
        score_pruned = int(
            hidden_score_pruned.sum() + output_score_pruned.sum()
        )
        pruned = int((~hidden_keep).sum() + (~output_keep).sum())
        self.hidden_layer.filter_edges(hidden_keep)
        self.output_layer.filter_edges(output_keep)
        self.hidden_bias_controller.filter_connections(hidden_keep)
        self.output_bias_controller.filter_connections(output_keep)

        neurons_merged = 0
        if merge_redundant_neurons:
            neurons_merged = self._merge_one_redundant_hidden_neuron(
                similarity_threshold=merge_similarity_threshold,
                error_tolerance=merge_error_tolerance,
            )

        hidden_reactivation_candidates = (
            ~self.hidden_layer.existing_pair_mask()
            & (self.hidden_target_activity.unsqueeze(1) > 0)
            & (self.hidden_source_activity.unsqueeze(0) > 0)
        )
        output_reactivation_candidates = (
            ~self.output_layer.existing_pair_mask()
            & (self.output_target_activity.unsqueeze(1) > 0)
            & (self.output_source_activity.unsqueeze(0) > 0)
        )
        requested_reactivations = round(pruned * reactivation_fraction)
        hidden_candidate_count = int(
            hidden_reactivation_candidates.sum().item()
        )
        output_candidate_count = int(
            output_reactivation_candidates.sum().item()
        )
        total_candidates = hidden_candidate_count + output_candidate_count
        requested_reactivations = min(
            requested_reactivations,
            total_candidates,
        )

        hidden_requested = 0
        if total_candidates:
            hidden_requested = round(
                requested_reactivations
                * hidden_candidate_count
                / total_candidates
            )
        hidden_requested = min(hidden_requested, hidden_candidate_count)
        output_requested = min(
            requested_reactivations - hidden_requested,
            round(
                requested_reactivations
                * output_candidate_count
                / total_candidates
            )
            if total_candidates
            else 0,
            output_candidate_count,
        )
        hidden_reactivated = self._reactivate_edges(
            self.hidden_layer,
            hidden_reactivation_candidates,
            hidden_requested,
            bias_low=0.08,
            bias_high=0.18,
        )
        output_reactivated = self._reactivate_edges(
            self.output_layer,
            output_reactivation_candidates,
            output_requested,
            bias_low=0.002,
            bias_high=0.014,
        )
        reactivated = hidden_reactivated + output_reactivated
        if neurons_merged:
            # Merging reconstructs and renumbers several edges, so their old
            # integral traces no longer have a one-to-one interpretation.
            self._reset_bias_controllers()
        else:
            self.hidden_bias_controller.append_connections(
                hidden_reactivated
            )
            self.output_bias_controller.append_connections(
                output_reactivated
            )

        self.total_pruned += pruned
        self.total_score_pruned += score_pruned
        self.total_neurons_merged += neurons_merged
        self.total_reactivated += reactivated
        self.pruning_events += 1
        self._reset_pruning_history()
        self._reset_reinforcement_history()
        return pruned, reactivated

    def _merge_one_redundant_hidden_neuron(
        self,
        *,
        similarity_threshold: float,
        error_tolerance: float,
    ) -> int:
        """Absorb one contained hidden neuron using the current state."""
        if self.hidden_size < 2:
            return 0

        records = self._current_structure_probe()
        observation_count = len(records)
        incoming_signatures = torch.zeros(
            self.hidden_size,
            observation_count * self.input_size,
            dtype=torch.bool,
        )
        hidden_targets = self.hidden_layer.target_index
        hidden_sources = self.hidden_layer.source_index
        outgoing_signatures = torch.zeros(
            self.hidden_size,
            observation_count * self.output_size,
            dtype=torch.bool,
        )
        output_targets = self.output_layer.target_index
        output_sources = self.output_layer.source_index
        for observation, record in enumerate(records):
            active_edges = record.hidden_edge_active
            active_indices = active_edges.nonzero().flatten()
            incoming_signatures[
                hidden_targets[active_indices],
                observation * self.input_size
                + hidden_sources[active_indices],
            ] = True
            active_output_indices = (
                record.output_edge_active.nonzero().flatten()
            )
            outgoing_signatures[
                output_sources[active_output_indices],
                observation * self.output_size
                + output_targets[active_output_indices],
            ] = True

        hidden_history = torch.stack([record.hidden for record in records])
        candidates: list[tuple[float, int, int]] = []
        for removed in range(self.hidden_size):
            for keeper in range(self.hidden_size):
                if removed == keeper:
                    continue
                removed_incoming = incoming_signatures[removed].sum()
                removed_outgoing = outgoing_signatures[removed].sum()
                if removed_incoming < 10 or removed_outgoing < 10:
                    continue
                incoming_containment = float(
                    (
                        incoming_signatures[removed]
                        & incoming_signatures[keeper]
                    ).sum()
                    / removed_incoming
                )
                outgoing_containment = float(
                    (
                        outgoing_signatures[removed]
                        & outgoing_signatures[keeper]
                    ).sum()
                    / removed_outgoing
                )
                if (
                    incoming_containment < similarity_threshold
                    or outgoing_containment < similarity_threshold
                ):
                    continue

                removed_output = hidden_history[:, removed]
                keeper_output = hidden_history[:, keeper]
                removed_active = removed_output != 0
                removed_output = removed_output[removed_active]
                keeper_output = keeper_output[removed_active]
                denominator = (
                    removed_output.square().sum().sqrt()
                    * keeper_output.square().sum().sqrt()
                )
                if denominator <= 0:
                    continue
                output_similarity = float(
                    (removed_output @ keeper_output) / denominator
                )
                if output_similarity >= similarity_threshold:
                    candidates.append(
                        (
                            min(
                                incoming_containment,
                                outgoing_containment,
                                output_similarity,
                            ),
                            keeper,
                            removed,
                        )
                    )

        for _, keeper, removed in sorted(candidates, reverse=True):
            if self._fold_hidden_neuron_outputs(
                keeper,
                removed,
                hidden_history,
                records,
                error_tolerance=error_tolerance,
            ):
                self._remove_hidden_neuron(removed)
                return 1
        return 0

    def _current_structure_probe(self) -> list[VideoRecord]:
        records = []
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                for row_velocity, column_velocity in DIRECTIONS:
                    previous_state = DotState(
                        row,
                        column,
                        row_velocity,
                        column_velocity,
                    )
                    current_state = advance(previous_state, self.grid_size)
                    records.append(
                        self.record_from_pair(
                            frame_from_state(
                                previous_state,
                                self.grid_size,
                            ),
                            frame_from_state(
                                current_state,
                                self.grid_size,
                            ),
                        )
                    )
        return records

    def _fold_hidden_neuron_outputs(
        self,
        keeper: int,
        removed: int,
        hidden_history: Tensor,
        records: list[VideoRecord],
        *,
        error_tolerance: float,
    ) -> bool:
        return self._fold_relu_hidden_neurons(
            keeper,
            removed,
            hidden_history,
            records,
            error_tolerance=error_tolerance,
        )

    def _fold_relu_hidden_neurons(
        self,
        keeper: int,
        removed: int,
        hidden_history: Tensor,
        records: list[VideoRecord],
        *,
        error_tolerance: float,
    ) -> bool:
        model_inputs = torch.stack(
            [record.model_input for record in records]
        )
        keeper_inputs = (
            self.hidden_layer.target_index == keeper
        ).nonzero().flatten()
        removed_inputs = (
            self.hidden_layer.target_index == removed
        ).nonzero().flatten()
        keeper_by_source = {
            int(self.hidden_layer.source_index[edge]): int(edge)
            for edge in keeper_inputs
        }
        removed_by_source = {
            int(self.hidden_layer.source_index[edge]): int(edge)
            for edge in removed_inputs
        }

        input_proposals: list[
            tuple[int | None, int, Tensor, Tensor]
        ] = []
        reconstructed_keeper_values = torch.zeros(model_inputs.shape[0])
        for source in keeper_by_source.keys() | removed_by_source.keys():
            keeper_edge = keeper_by_source.get(source)
            removed_edge = removed_by_source.get(source)
            source_values = model_inputs[:, source]
            before = torch.zeros_like(source_values)
            if keeper_edge is not None:
                before += torch.relu(
                    source_values * self.hidden_layer.weight[keeper_edge]
                    - self.hidden_layer.bias[keeper_edge]
                )
            if removed_edge is not None:
                before += torch.relu(
                    source_values * self.hidden_layer.weight[removed_edge]
                    - self.hidden_layer.bias[removed_edge]
                )
            fitted = self._fit_relu_edge(source_values, before)
            new_weight, new_bias, after = fitted
            if self._relative_reconstruction_error(before, after) > (
                error_tolerance
            ):
                return False
            reconstructed_keeper_values += after
            input_proposals.append(
                (keeper_edge, source, new_weight, new_bias)
            )

        keeper_values = hidden_history[:, keeper]
        removed_values = hidden_history[:, removed]
        reconstructed_keeper_values = keeper_values + removed_values
        keeper_outputs = (
            self.output_layer.source_index == keeper
        ).nonzero().flatten()
        removed_outputs = (
            self.output_layer.source_index == removed
        ).nonzero().flatten()
        keeper_by_target = {
            int(self.output_layer.target_index[edge]): int(edge)
            for edge in keeper_outputs
        }
        removed_by_target = {
            int(self.output_layer.target_index[edge]): int(edge)
            for edge in removed_outputs
        }

        output_proposals: list[
            tuple[int | None, int, Tensor, Tensor]
        ] = []
        for target in keeper_by_target.keys() | removed_by_target.keys():
            keeper_edge = keeper_by_target.get(target)
            removed_edge = removed_by_target.get(target)
            before = torch.zeros_like(keeper_values)
            if keeper_edge is not None:
                before += torch.relu(
                    keeper_values * self.output_layer.weight[keeper_edge]
                    - self.output_layer.bias[keeper_edge]
                )
            if removed_edge is not None:
                before += torch.relu(
                    removed_values * self.output_layer.weight[removed_edge]
                    - self.output_layer.bias[removed_edge]
                )
            fitted = self._fit_relu_edge(
                reconstructed_keeper_values,
                before,
            )
            new_weight, new_bias, after = fitted
            if self._relative_reconstruction_error(before, after) > (
                error_tolerance
            ):
                return False
            output_proposals.append(
                (keeper_edge, target, new_weight, new_bias)
            )

        for edge, source, weight, bias in input_proposals:
            if edge is None:
                self.hidden_layer.append_edges(
                    torch.tensor([source]),
                    torch.tensor([keeper]),
                    weight.reshape(1),
                    bias.reshape(1),
                )
            else:
                self.hidden_layer.weight[edge] = weight
                self.hidden_layer.bias[edge] = bias
        for edge, target, weight, bias in output_proposals:
            if edge is None:
                self.output_layer.append_edges(
                    torch.tensor([keeper]),
                    torch.tensor([target]),
                    weight.reshape(1),
                    bias.reshape(1),
                )
            else:
                self.output_layer.weight[edge] = weight
                self.output_layer.bias[edge] = bias
        return True

    @staticmethod
    def _fit_relu_edge(
        source: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not torch.any(target > 0):
            zero = torch.tensor(0.0)
            return zero, torch.tensor(1.0), torch.zeros_like(target)
        design = torch.stack(
            [source, -torch.ones_like(source)],
            dim=1,
        )
        candidates = []
        for rows in (torch.ones_like(target, dtype=torch.bool), target > 0):
            weight, bias = torch.linalg.lstsq(
                design[rows],
                target[rows],
            ).solution
            weight = weight.clamp_min(0.0)
            bias = bias.clamp_min(0.0)
            prediction = torch.relu(source * weight - bias)
            error = (prediction - target).square().mean()
            candidates.append((error, weight, bias, prediction))
        _, weight, bias, prediction = min(
            candidates,
            key=lambda candidate: float(candidate[0]),
        )
        return weight, bias, prediction

    @staticmethod
    def _relative_reconstruction_error(
        before: Tensor,
        after: Tensor,
    ) -> Tensor:
        denominator = before.square().mean().sqrt()
        if denominator <= 1e-12:
            return (after - before).square().mean().sqrt()
        return (
            (after - before).square().mean().sqrt() / denominator
        )

    def _remove_hidden_neuron(self, removed: int) -> None:
        hidden_keep = self.hidden_layer.target_index != removed
        self.hidden_layer.filter_edges(hidden_keep)
        self.hidden_layer.target_index[
            self.hidden_layer.target_index > removed
        ] -= 1
        self.hidden_layer.target_count -= 1

        output_keep = self.output_layer.source_index != removed
        self.output_layer.filter_edges(output_keep)
        self.output_layer.source_index[
            self.output_layer.source_index > removed
        ] -= 1
        self.output_layer.source_count -= 1

        self.hidden_target_activity = torch.cat(
            [
                self.hidden_target_activity[:removed],
                self.hidden_target_activity[removed + 1 :],
            ]
        )
        self.output_source_activity = torch.cat(
            [
                self.output_source_activity[:removed],
                self.output_source_activity[removed + 1 :],
            ]
        )
        self.hidden_size -= 1

    def _reactivate_edges(
        self,
        layer: SparseConnectionLayer,
        candidates: Tensor,
        requested: int,
        *,
        bias_low: float,
        bias_high: float,
    ) -> int:
        if requested <= 0:
            return 0

        candidate_indices = candidates.flatten().nonzero().flatten()
        order = torch.randperm(
            candidate_indices.numel(),
            generator=self.pruning_generator,
        )
        selected = candidate_indices[order[:requested]]
        target_index = torch.div(
            selected,
            layer.source_count,
            rounding_mode="floor",
        )
        source_index = selected % layer.source_count
        layer.append_edges(
            source_index,
            target_index,
            0.08
            + 0.12
            * torch.rand(
                requested,
                generator=self.pruning_generator,
            ),
            bias_low
            + (bias_high - bias_low)
            * torch.rand(
                requested,
                generator=self.pruning_generator,
            ),
        )
        return requested

    def connection_presence_fraction(self) -> float:
        present = self.present_connection_count()
        possible = (
            self.hidden_layer.possible_edge_count
            + self.output_layer.possible_edge_count
        )
        return present / possible

    def present_connection_count(self) -> int:
        present = (
            self.hidden_layer.edge_count + self.output_layer.edge_count
        )
        return present

    def learned_parameter_count(self) -> int:
        # Every physical connection owns one learned weight and one bias.
        return 2 * self.present_connection_count()

    def predict_next(
        self,
        current_frame: Tensor,
    ) -> Tensor:
        delta = self.forward(current_frame).predicted_delta
        return (current_frame + delta).clamp(0.0, 1.0)


def sample_transition(
    generator: torch.Generator,
    grid_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    previous_state = random_state(generator, grid_size)
    current_state = advance(previous_state, grid_size)
    next_state = advance(current_state, grid_size)
    return (
        frame_from_state(previous_state, grid_size),
        frame_from_state(current_state, grid_size),
        frame_from_state(next_state, grid_size),
    )


def sample_trajectory(
    generator: torch.Generator,
    grid_size: int,
    frame_count: int = 10,
) -> list[Tensor]:
    """Generate consecutive frames from one randomly chosen initial state."""
    if frame_count < 3:
        raise ValueError("A training trajectory needs at least three frames")

    state = random_state(generator, grid_size)
    frames = [frame_from_state(state, grid_size)]
    for _ in range(frame_count - 1):
        state = advance(state, grid_size)
        frames.append(frame_from_state(state, grid_size))
    return frames


def evaluation_metrics(
    model: MovingDotPredictor,
    *,
    seed: int = 10_003,
    examples: int = 512,
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    squared_error = 0.0
    persistence_error = 0.0
    correct_positions = 0

    for _ in range(examples):
        previous_frame, current_frame, next_frame = sample_transition(
            generator,
            model.grid_size,
        )
        record = model.record_from_pair(previous_frame, current_frame)
        target_delta = next_frame - current_frame
        squared_error += (
            target_delta - record.predicted_delta
        ).square().mean().item()
        persistence_error += target_delta.square().mean().item()

        predicted_next = (
            current_frame + record.predicted_delta
        ).flatten().argmax()
        actual_next = next_frame.flatten().argmax()
        correct_positions += int(predicted_next == actual_next)

    return (
        squared_error / examples,
        persistence_error / examples,
        correct_positions / examples,
    )


def evaluation_active_edge_fraction(
    model: MovingDotPredictor,
    *,
    seed: int = 10_003,
    examples: int = 512,
) -> float:
    """Fraction of possible edges active on typical evaluation inputs."""
    generator = torch.Generator().manual_seed(seed)
    active_edges = 0
    possible_edges = 0

    for _ in range(examples):
        previous_frame, current_frame, _ = sample_transition(
            generator,
            model.grid_size,
        )
        record = model.record_from_pair(previous_frame, current_frame)
        active_edges += int(record.hidden_edge_active.sum().item())
        active_edges += int(record.output_edge_active.sum().item())
        possible_edges += model.hidden_layer.possible_edge_count
        possible_edges += model.output_layer.possible_edge_count

    return active_edges / possible_edges


def rollout_position_accuracy(
    model: MovingDotPredictor,
    *,
    seed: int = 20_003,
    rollouts: int = 64,
    horizon: int = 8,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    correct_positions = 0
    total_positions = rollouts * horizon

    for _ in range(rollouts):
        previous_state = random_state(generator, model.grid_size)
        current_state = advance(previous_state, model.grid_size)
        previous_frame = frame_from_state(previous_state, model.grid_size)
        current_frame = frame_from_state(current_state, model.grid_size)
        model.prime_state(previous_frame)

        for _ in range(horizon):
            current_state = advance(current_state, model.grid_size)
            actual_next = frame_from_state(current_state, model.grid_size)
            predicted_next = model.predict_next(current_frame)

            correct_positions += int(
                predicted_next.flatten().argmax()
                == actual_next.flatten().argmax()
            )
            previous_frame, current_frame = current_frame, predicted_next

    return correct_positions / total_positions


def plot_final_prediction(
    model: MovingDotPredictor,
    *,
    output_path: str = "moving_dot_final_prediction.png",
    seed: int = 20_003,
    horizon: int = 8,
) -> str:
    """Save an enlarged final one-step prediction using real input frames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generator = torch.Generator().manual_seed(seed)
    previous_state = random_state(generator, model.grid_size)
    current_state = advance(previous_state, model.grid_size)
    previous_frame = frame_from_state(previous_state, model.grid_size)
    current_frame = frame_from_state(current_state, model.grid_size)
    model.prime_state(previous_frame)

    actual_frame = current_frame
    planned_frame = current_frame
    for _ in range(horizon):
        current_state = advance(current_state, model.grid_size)
        actual_frame = frame_from_state(current_state, model.grid_size)
        planned_frame = model.predict_next(current_frame)
        previous_frame, current_frame = current_frame, actual_frame

    actual_image = actual_frame.detach().cpu().numpy()
    planned_image = planned_frame.detach().cpu().numpy()
    error_image = abs(actual_image - planned_image)
    actual_position = divmod(int(actual_frame.flatten().argmax()), model.grid_size)
    planned_position = divmod(
        int(planned_frame.flatten().argmax()),
        model.grid_size,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        dpi=160,
        constrained_layout=True,
    )
    panels = (
        (
            actual_image,
            f"Actual frame {horizon}\nposition={actual_position}",
            "magma",
        ),
        (
            planned_image,
            f"Planned frame {horizon}\nposition={planned_position}",
            "magma",
        ),
        (
            error_image,
            f"Absolute error\nmean={error_image.mean():.4f}",
            "viridis",
        ),
    )
    for axis, (image, title, colour_map) in zip(axes, panels):
        plot = axis.imshow(
            image,
            cmap=colour_map,
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title, fontsize=15)
        axis.set_xticks(range(model.grid_size))
        axis.set_yticks(range(model.grid_size))
        axis.set_xticks(
            [value - 0.5 for value in range(1, model.grid_size)],
            minor=True,
        )
        axis.set_yticks(
            [value - 0.5 for value in range(1, model.grid_size)],
            minor=True,
        )
        axis.grid(which="minor", color="white", linewidth=1.2, alpha=0.55)
        axis.tick_params(which="minor", length=0)
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        "Final teacher-forced prediction: actual vs planned",
        fontsize=18,
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_prediction_animation(
    model: MovingDotPredictor,
    *,
    output_path: str = "moving_dot_prediction.gif",
    seed: int = 20_0013,
    frame_count: int = 32,
    frames_per_second: int = 2,
) -> str:
    """Animate teacher-forced actual, planned, and error frames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    generator = torch.Generator().manual_seed(seed)
    previous_state = random_state(generator, model.grid_size)
    current_state = advance(previous_state, model.grid_size)
    previous_frame = frame_from_state(previous_state, model.grid_size)
    current_frame = frame_from_state(current_state, model.grid_size)
    model.prime_state(previous_frame)
    frames: list[tuple[Tensor, Tensor]] = []

    for _ in range(frame_count):
        current_state = advance(current_state, model.grid_size)
        actual_frame = frame_from_state(current_state, model.grid_size)
        planned_frame = model.predict_next(current_frame)
        frames.append((actual_frame.cpu(), planned_frame.cpu()))
        previous_frame, current_frame = current_frame, actual_frame

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        dpi=120,
        constrained_layout=True,
    )
    initial_actual, initial_planned = frames[0]
    images = (
        axes[0].imshow(
            initial_actual,
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        ),
        axes[1].imshow(
            initial_planned,
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        ),
        axes[2].imshow(
            (initial_actual - initial_planned).abs(),
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        ),
    )
    for axis, image in zip(axes, images):
        axis.set_xticks(range(model.grid_size))
        axis.set_yticks(range(model.grid_size))
        axis.set_xticks(
            [value - 0.5 for value in range(1, model.grid_size)],
            minor=True,
        )
        axis.set_yticks(
            [value - 0.5 for value in range(1, model.grid_size)],
            minor=True,
        )
        axis.grid(which="minor", color="white", linewidth=1.2, alpha=0.55)
        axis.tick_params(which="minor", length=0)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def update(frame_index: int) -> tuple:
        actual, planned = frames[frame_index]
        error = (actual - planned).abs()
        images[0].set_data(actual)
        images[1].set_data(planned)
        images[2].set_data(error)
        actual_position = divmod(
            int(actual.flatten().argmax()),
            model.grid_size,
        )
        planned_position = divmod(
            int(planned.flatten().argmax()),
            model.grid_size,
        )
        axes[0].set_title(
            f"Actual\nposition={actual_position}",
            fontsize=15,
        )
        axes[1].set_title(
            f"Planned\nposition={planned_position}",
            fontsize=15,
        )
        axes[2].set_title(
            f"Absolute error\nmean={error.mean():.4f}",
            fontsize=15,
        )
        figure.suptitle(
            f"Teacher-forced one-step prediction — frame {frame_index + 1}",
            fontsize=18,
        )
        return images

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1_000 / frames_per_second,
        blit=False,
    )
    animation.save(
        output_path,
        writer=PillowWriter(fps=frames_per_second),
    )
    plt.close(figure)
    return output_path


def persistence_rollout_position_accuracy(
    *,
    grid_size: int = 6,
    seed: int = 20_003,
    rollouts: int = 64,
    horizon: int = 8,
) -> float:
    """Accuracy when every predicted frame is simply the current frame."""
    generator = torch.Generator().manual_seed(seed)
    correct_positions = 0
    total_positions = rollouts * horizon

    for _ in range(rollouts):
        current_state = advance(
            random_state(generator, grid_size),
            grid_size,
        )
        predicted_frame = frame_from_state(current_state, grid_size)

        for _ in range(horizon):
            current_state = advance(current_state, grid_size)
            actual_next = frame_from_state(current_state, grid_size)
            correct_positions += int(
                predicted_frame.flatten().argmax()
                == actual_next.flatten().argmax()
            )

    return correct_positions / total_positions


def run_online_experiment(
    *,
    seed: int = 11,
    observations: int = 100_000,
    hidden_size: int = 48,
    weight_learning_rate: float = 5e-2,
    bias_proportional_gain: float = 1.5e-3,
    bias_integral_gain: float = 1.5e-4,
    bias_integral_decay: float = 0.99,
    bias_integral_limit: float = 20.0,
    reinforcement_learning_rate: float = 0.0,
    reinforcement_interval: int | None = None,
    update_hidden: bool = True,
    pruning_interval: int | None = None,
    minimum_activity_rate: float = 0.0,
    minimum_pruning_score: float | None = None,
    reactivation_fraction: float = 0.05,
    merge_redundant_neurons: bool = False,
    merge_similarity_threshold: float = 0.95,
    merge_error_tolerance: float = 0.10,
    trajectory_frame_count: int = 10,
) -> tuple[
    MovingDotPredictor,
    list[tuple[int, float, float, int, int, int, int, int]],
]:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    model = MovingDotPredictor(
        hidden_size=hidden_size,
        seed=seed,
        weight_learning_rate=weight_learning_rate,
        bias_proportional_gain=bias_proportional_gain,
        bias_integral_gain=bias_integral_gain,
        bias_integral_decay=bias_integral_decay,
        bias_integral_limit=bias_integral_limit,
        reinforcement_learning_rate=reinforcement_learning_rate,
        reinforcement_interval=reinforcement_interval,
        update_hidden=update_hidden,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    history: list[tuple[int, float, float, int, int, int, int, int]] = []

    initial_mse, _, initial_accuracy = evaluation_metrics(model)
    history.append(
        (
            0,
            initial_mse,
            initial_accuracy,
            model.present_connection_count(),
            model.learned_parameter_count(),
            model.total_pruned,
            model.total_neurons_merged,
            model.total_reactivated,
        )
    )

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
        previous_frame = trajectory[trajectory_offset]
        current_frame = trajectory[trajectory_offset + 1]
        next_frame = trajectory[trajectory_offset + 2]
        trajectory_offset += 1
        record = model.forward(current_frame)
        model.correct(record, next_frame - current_frame)

        if (
            pruning_interval is not None
            and observation % pruning_interval == 0
        ):
            model.prune_inactive(
                minimum_activity_rate=minimum_activity_rate,
                minimum_pruning_score=minimum_pruning_score,
                reactivation_fraction=reactivation_fraction,
                merge_redundant_neurons=merge_redundant_neurons,
                merge_similarity_threshold=merge_similarity_threshold,
                merge_error_tolerance=merge_error_tolerance,
            )

        if observation % 2_000 == 0:
            mse, _, accuracy = evaluation_metrics(model)
            history.append(
                (
                    observation,
                    mse,
                    accuracy,
                    model.present_connection_count(),
                    model.learned_parameter_count(),
                    model.total_pruned,
                    model.total_neurons_merged,
                    model.total_reactivated,
                )
            )

    return model, history


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--observations", type=int, default=10_000)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--pruning-interval", type=int, default=1_000)
    parser.add_argument("--merge-redundant-neurons", action="store_true")
    parser.add_argument(
        "--weight-learning-rate",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "--bias-proportional-gain",
        type=float,
        default=1.5e-3,
    )
    parser.add_argument(
        "--bias-integral-gain",
        type=float,
        default=1.5e-4,
    )
    parser.add_argument("--bias-integral-decay", type=float, default=0.99)
    parser.add_argument("--bias-integral-limit", type=float, default=20.0)
    parser.add_argument("--frame-count", type=int, default=10)
    args = parser.parse_args()
    baseline, _ = run_online_experiment(
        observations=args.observations,
        hidden_size=args.hidden_size,
        weight_learning_rate=args.weight_learning_rate,
        bias_proportional_gain=args.bias_proportional_gain,
        bias_integral_gain=args.bias_integral_gain,
        bias_integral_decay=args.bias_integral_decay,
        bias_integral_limit=args.bias_integral_limit,
        trajectory_frame_count=args.frame_count,
    )
    model, history = run_online_experiment(
        observations=args.observations,
        hidden_size=args.hidden_size,
        weight_learning_rate=args.weight_learning_rate,
        bias_proportional_gain=args.bias_proportional_gain,
        bias_integral_gain=args.bias_integral_gain,
        bias_integral_decay=args.bias_integral_decay,
        bias_integral_limit=args.bias_integral_limit,
        pruning_interval=args.pruning_interval,
        merge_redundant_neurons=args.merge_redundant_neurons,
        trajectory_frame_count=args.frame_count,
    )

    baseline_mse, persistence_mse, baseline_accuracy = evaluation_metrics(
        baseline
    )
    baseline_rollout = rollout_position_accuracy(baseline)
    baseline_active_edges = evaluation_active_edge_fraction(baseline)

    mse, _, one_step_accuracy = evaluation_metrics(model)
    rollout_accuracy = rollout_position_accuracy(model)
    persistence_rollout = persistence_rollout_position_accuracy(
        grid_size=model.grid_size
    )
    active_edge_fraction = evaluation_active_edge_fraction(model)
    plot_path = plot_final_prediction(model)
    animation_path = save_prediction_animation(model)

    print("Autograd used: no")
    print(f"Persistence delta MSE: {persistence_mse:.6f}")
    print(f"Initial model MSE:     {history[0][1]:.6f}")
    print()
    print("Without pruning:")
    print(f"  Final model MSE:     {baseline_mse:.6f}")
    print(f"  One-step position:   {baseline_accuracy:.1%}")
    print(f"  8-step rollout:      {baseline_rollout:.1%}")
    print(f"  Active edges/input:  {baseline_active_edges:.1%}")
    print(f"  Connections present: {baseline.connection_presence_fraction():.1%}")
    print(f"  Learned parameters:  {baseline.learned_parameter_count()}")
    print()
    print("Prune and test neuron merging every 1,000 observations:")
    print(f"  Final model MSE:     {mse:.6f}")
    print(f"  One-step position:   {one_step_accuracy:.1%}")
    print(f"  8-step rollout:      {rollout_accuracy:.1%}")
    print(f"  Active edges/input:  {active_edge_fraction:.1%}")
    print(f"  Connections present: {model.connection_presence_fraction():.1%}")
    print(f"  Learned parameters:  {model.learned_parameter_count()}")
    print(f"  Total pruned:        {model.total_pruned}")
    print(f"  Score-pruned:        {model.total_score_pruned}")
    print(f"  Neurons merged:      {model.total_neurons_merged}")
    print(f"  Total reactivated:   {model.total_reactivated}")
    print(f"Persistence rollout:   {persistence_rollout:.1%}")
    print(f"Final one-step plot:   {plot_path}")
    print(f"Prediction animation:  {animation_path}")
    print("Progress every 2,000 observations:")
    for (
        observations,
        checkpoint_mse,
        accuracy,
        connections,
        parameters,
        pruned,
        neuron_merges,
        reactivated,
    ) in history:
        print(
            f"  {observations:>6}: "
            f"MSE={checkpoint_mse:.6f}, accuracy={accuracy:.1%}, "
            f"connections={connections}, parameters={parameters}, "
            f"pruned={pruned}, neuron_merges={neuron_merges}, "
            f"reactivated={reactivated}"
        )


if __name__ == "__main__":
    main()
