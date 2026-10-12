"""Stage 0: online next-frame prediction for a bouncing dot.

This experiment uses two consecutive 6x6 frames to predict the signed delta
to the next frame. The previous frame is explicit state for this first video
test; a later experiment can replace it with learned recurrent state.

All learning updates are written directly. PyTorch autograd and optimizers are
not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

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
        bias_learning_rate: float = 1e-2,
        reinforcement_learning_rate: float = 1e-4,
        reward_error_tolerance: float = 0.1,
        meaningful_change_threshold: float = 0.1,
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
        self.bias_learning_rate = bias_learning_rate
        self.reinforcement_learning_rate = reinforcement_learning_rate
        self.reward_error_tolerance = reward_error_tolerance
        self.meaningful_change_threshold = meaningful_change_threshold
        self.pruning_generator = torch.Generator().manual_seed(seed + 30_000)
        self.total_pruned = 0
        self.total_merged = 0
        self.total_reactivated = 0
        self.pruning_events = 0
        self._reset_pruning_history()

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

    def forward(self, previous_frame: Tensor, current_frame: Tensor) -> VideoRecord:
        model_input = torch.cat(
            [previous_frame.flatten(), current_frame.flatten()]
        )

        hidden, hidden_edge_active = self.hidden_layer.forward(model_input)
        output, output_edge_active = self.output_layer.forward(hidden)

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
            output_active=output > 0,
        )

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

        # Raising the positive channel or lowering the negative channel both
        # increase the signed prediction.
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
        self.output_layer.bias.sub_(
            self.bias_learning_rate * output_edge_correction
            + self.reinforcement_learning_rate * output_edge_reward
        )
        self.output_layer.weight.add_(
            self.weight_learning_rate
            * output_edge_correction
            * record.hidden[output_sources]
        )

        # Every hidden neuron receives the mean correction from the output
        # connections that were active during this prediction.
        active_output_count = torch.zeros(
            self.hidden_size,
            dtype=torch.int64,
        )
        active_output_count.scatter_add_(
            0,
            output_sources,
            record.output_edge_active.to(torch.int64),
        )
        hidden_correction = torch.zeros(self.hidden_size)
        hidden_correction.scatter_add_(
            0,
            output_sources,
            output_edge_correction,
        )
        hidden_correction /= active_output_count.clamp_min(1)
        hidden_correction *= record.hidden_active
        hidden_reward = torch.zeros(self.hidden_size)
        hidden_reward.scatter_add_(0, output_sources, output_edge_reward)
        hidden_reward /= active_output_count.clamp_min(1)
        hidden_reward *= record.hidden_active

        hidden_targets = self.hidden_layer.target_index
        hidden_sources = self.hidden_layer.source_index
        hidden_edge_correction = (
            hidden_correction[hidden_targets] * record.hidden_edge_active
        )
        hidden_edge_reward = (
            hidden_reward[hidden_targets] * record.hidden_edge_active
        )
        self.hidden_layer.bias.sub_(
            self.bias_learning_rate * hidden_edge_correction
            + self.reinforcement_learning_rate * hidden_edge_reward
        )
        self.hidden_layer.weight.add_(
            self.weight_learning_rate
            * hidden_edge_correction
            * record.model_input[hidden_sources]
        )

        self.hidden_bias.clamp_(min=0.0)
        self.output_bias.clamp_(min=0.0)

        return error.square().mean()

    def _reset_pruning_history(self) -> None:
        self.history_observations = 0
        self.hidden_source_history: list[Tensor] = []
        self.output_source_history: list[Tensor] = []
        self.hidden_edge_activity = torch.zeros_like(
            self.hidden_weight,
            dtype=torch.int32,
        )
        self.output_edge_activity = torch.zeros_like(
            self.output_weight,
            dtype=torch.int32,
        )
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
        self.hidden_source_history.append(record.model_input.clone())
        self.output_source_history.append(record.hidden.clone())
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
        reactivation_fraction: float = 0.05,
        merge_redundant: bool = False,
        merge_similarity_threshold: float = 0.95,
        merge_error_tolerance: float = 0.10,
        maximum_merge_fraction: float = 0.05,
    ) -> tuple[int, int]:
        """Prune unused edges, merge redundant edges, and reopen a few."""
        if self.history_observations == 0:
            return 0, 0

        minimum_activations = int(
            self.history_observations * minimum_activity_rate
        )
        hidden_keep = self.hidden_edge_activity > minimum_activations
        output_keep = self.output_edge_activity > minimum_activations
        pruned = int((~hidden_keep).sum() + (~output_keep).sum())
        self.hidden_layer.filter_edges(hidden_keep)
        self.output_layer.filter_edges(output_keep)

        merged = 0
        if merge_redundant:
            merged += self._merge_redundant_layer(
                self.hidden_layer,
                torch.stack(self.hidden_source_history),
                similarity_threshold=merge_similarity_threshold,
                error_tolerance=merge_error_tolerance,
                maximum_merge_fraction=maximum_merge_fraction,
            )
            merged += self._merge_redundant_layer(
                self.output_layer,
                torch.stack(self.output_source_history),
                similarity_threshold=merge_similarity_threshold,
                error_tolerance=merge_error_tolerance,
                maximum_merge_fraction=maximum_merge_fraction,
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

        self.total_pruned += pruned
        self.total_merged += merged
        self.total_reactivated += reactivated
        self.pruning_events += 1
        self._reset_pruning_history()
        return pruned, reactivated

    @staticmethod
    def _merge_redundant_layer(
        layer: SparseConnectionLayer,
        source_history: Tensor,
        *,
        similarity_threshold: float,
        error_tolerance: float,
        maximum_merge_fraction: float,
    ) -> int:
        """Greedily merge pairs whose combined historical output is preserved."""
        merged = 0
        remove = torch.zeros(layer.edge_count, dtype=torch.bool)

        for target in range(layer.target_count):
            edge_indices = (layer.target_index == target).nonzero().flatten()
            merge_limit = int(
                edge_indices.numel() * maximum_merge_fraction
            )
            if merge_limit == 0:
                continue

            source_indices = layer.source_index[edge_indices]
            source_values = source_history[:, source_indices]
            contributions = torch.relu(
                source_values * layer.weight[edge_indices]
                - layer.bias[edge_indices]
            )
            contribution_norms = contributions.square().sum(dim=0).sqrt()
            nonzero = contribution_norms > 0
            edge_indices = edge_indices[nonzero]
            source_indices = source_indices[nonzero]
            contributions = contributions[:, nonzero]
            contribution_norms = contribution_norms[nonzero]
            if edge_indices.numel() < 2:
                continue

            similarity = contributions.T @ contributions
            similarity /= (
                contribution_norms.unsqueeze(1)
                * contribution_norms.unsqueeze(0)
            )
            similarity.fill_diagonal_(-1.0)
            candidate_pairs = (
                similarity >= similarity_threshold
            ).triu(diagonal=1).nonzero()
            if candidate_pairs.numel() == 0:
                continue

            candidate_scores = similarity[
                candidate_pairs[:, 0],
                candidate_pairs[:, 1],
            ]
            order = torch.argsort(candidate_scores, descending=True)
            already_used: set[int] = set()
            target_merged = 0

            for candidate_index in order:
                if target_merged >= merge_limit:
                    break
                left_local, right_local = candidate_pairs[candidate_index]
                left = int(edge_indices[left_local])
                right = int(edge_indices[right_local])
                if left in already_used or right in already_used:
                    continue

                best = None
                for keeper, removed in ((left, right), (right, left)):
                    keeper_values = source_history[
                        :, layer.source_index[keeper]
                    ]
                    removed_values = source_history[
                        :, layer.source_index[removed]
                    ]
                    before = (
                        torch.relu(
                            keeper_values * layer.weight[keeper]
                            - layer.bias[keeper]
                        )
                        + torch.relu(
                            removed_values * layer.weight[removed]
                            - layer.bias[removed]
                        )
                    )
                    design = torch.stack(
                        [keeper_values, -torch.ones_like(keeper_values)],
                        dim=1,
                    )
                    fit_rows = before > 0
                    new_weight, new_bias = torch.linalg.lstsq(
                        design[fit_rows],
                        before[fit_rows],
                    ).solution
                    new_weight = new_weight.clamp_min(0.0)
                    new_bias = new_bias.clamp_min(0.0)
                    after = torch.relu(
                        keeper_values * new_weight - new_bias
                    )
                    relative_error = (
                        (after - before).square().mean().sqrt()
                        / before.square().mean().sqrt().clamp_min(1e-12)
                    )
                    proposal = (
                        float(relative_error),
                        keeper,
                        removed,
                        new_weight,
                        new_bias,
                    )
                    if best is None or proposal[0] < best[0]:
                        best = proposal

                if best is None or best[0] > error_tolerance:
                    continue

                _, keeper, removed, new_weight, new_bias = best
                layer.weight[keeper] = new_weight
                layer.bias[keeper] = new_bias
                remove[removed] = True
                already_used.update((keeper, removed))
                merged += 1
                target_merged += 1

        layer.filter_edges(~remove)
        return merged

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
        return self.hidden_layer.edge_count + self.output_layer.edge_count

    def learned_parameter_count(self) -> int:
        # Every physical connection owns one learned weight and one bias.
        return 2 * self.present_connection_count()

    def predict_next(
        self,
        previous_frame: Tensor,
        current_frame: Tensor,
    ) -> Tensor:
        delta = self.forward(previous_frame, current_frame).predicted_delta
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
        record = model.forward(previous_frame, current_frame)
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
        record = model.forward(previous_frame, current_frame)
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

        for _ in range(horizon):
            current_state = advance(current_state, model.grid_size)
            actual_next = frame_from_state(current_state, model.grid_size)
            predicted_next = model.predict_next(
                previous_frame,
                current_frame,
            )

            correct_positions += int(
                predicted_next.flatten().argmax()
                == actual_next.flatten().argmax()
            )
            previous_frame, current_frame = current_frame, predicted_next

    return correct_positions / total_positions


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
    reinforcement_learning_rate: float = 1e-4,
    pruning_interval: int | None = None,
    minimum_activity_rate: float = 0.0,
    reactivation_fraction: float = 0.05,
    merge_redundant: bool = False,
    merge_similarity_threshold: float = 0.95,
    merge_error_tolerance: float = 0.10,
    maximum_merge_fraction: float = 0.05,
    trajectory_frame_count: int = 10,
) -> tuple[
    MovingDotPredictor,
    list[tuple[int, float, float, int, int, int, int, int]],
]:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    model = MovingDotPredictor(
        seed=seed,
        reinforcement_learning_rate=reinforcement_learning_rate,
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
            model.total_merged,
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
        previous_frame = trajectory[trajectory_offset]
        current_frame = trajectory[trajectory_offset + 1]
        next_frame = trajectory[trajectory_offset + 2]
        trajectory_offset += 1
        record = model.forward(previous_frame, current_frame)
        model.correct(record, next_frame - current_frame)

        if (
            pruning_interval is not None
            and observation % pruning_interval == 0
        ):
            model.prune_inactive(
                minimum_activity_rate=minimum_activity_rate,
                reactivation_fraction=reactivation_fraction,
                merge_redundant=merge_redundant,
                merge_similarity_threshold=merge_similarity_threshold,
                merge_error_tolerance=merge_error_tolerance,
                maximum_merge_fraction=maximum_merge_fraction,
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
                    model.total_merged,
                    model.total_reactivated,
                )
            )

    return model, history


def main() -> None:
    baseline, _ = run_online_experiment()
    model, history = run_online_experiment(
        pruning_interval=1_000,
        merge_redundant=True,
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
    print("Prune and merge every 1,000 observations:")
    print(f"  Final model MSE:     {mse:.6f}")
    print(f"  One-step position:   {one_step_accuracy:.1%}")
    print(f"  8-step rollout:      {rollout_accuracy:.1%}")
    print(f"  Active edges/input:  {active_edge_fraction:.1%}")
    print(f"  Connections present: {model.connection_presence_fraction():.1%}")
    print(f"  Learned parameters:  {model.learned_parameter_count()}")
    print(f"  Total pruned:        {model.total_pruned}")
    print(f"  Total merged:        {model.total_merged}")
    print(f"  Total reactivated:   {model.total_reactivated}")
    print(f"Persistence rollout:   {persistence_rollout:.1%}")
    print("Progress every 2,000 observations:")
    for (
        observations,
        checkpoint_mse,
        accuracy,
        connections,
        parameters,
        pruned,
        merged,
        reactivated,
    ) in history:
        print(
            f"  {observations:>6}: "
            f"MSE={checkpoint_mse:.6f}, accuracy={accuracy:.1%}, "
            f"connections={connections}, parameters={parameters}, "
            f"pruned={pruned}, merged={merged}, reactivated={reactivated}"
        )


if __name__ == "__main__":
    main()
