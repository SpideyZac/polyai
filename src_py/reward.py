"""Modular reward system for PolyTrackEnv."""

# pylint: disable=no-name-in-module

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from simulation_worker import CarStatePy


class RewardComponent(ABC):
    """Base class for a single reward term."""

    @abstractmethod
    def compute(
        self, data: CarStatePy, prev_data: CarStatePy | None, info: dict[str, Any]
    ) -> float:
        """Return the raw (unweighted) reward for this component."""

    def reset(self) -> None:
        """Called on env reset; override if the component is stateful."""


@dataclass
class WeightedComponent:
    """A reward component paired with a scalar weight."""

    component: RewardComponent
    weight: float = 1.0


@dataclass
class RewardSystem:
    """Combines multiple weighted reward components into a single scalar."""

    components: list[WeightedComponent] = field(default_factory=list)

    def add(self, component: RewardComponent, weight: float = 1.0) -> "RewardSystem":
        """Register a component; returns self for chaining."""
        self.components.append(WeightedComponent(component, weight))
        return self

    def compute(
        self, data: CarStatePy, prev_data: CarStatePy | None, info: dict[str, Any]
    ) -> float:
        """Sum all weighted component rewards."""
        return sum(
            wc.weight * wc.component.compute(data, prev_data, info)
            for wc in self.components
        )

    def reset(self) -> None:
        """Forward reset to all stateful components."""
        for wc in self.components:
            wc.component.reset()


class SpeedReward(RewardComponent):
    """Reward proportional to the car's forward speed."""

    def compute(
        self, data: CarStatePy, prev_data: CarStatePy | None, info: dict[str, Any]
    ) -> float:
        return data.get_speed_kmh()


class FinishReward(RewardComponent):
    """Reward for finishing the track."""

    def compute(
        self, data: CarStatePy, prev_data: CarStatePy | None, info: dict[str, Any]
    ) -> float:
        return float(data.get_is_finished())


class ApproachCheckpointReward(RewardComponent):
    """Reward for approaching the next checkpoint."""

    def compute(
        self, data: CarStatePy, prev_data: CarStatePy | None, info: dict[str, Any]
    ) -> float:
        if prev_data is None:
            return 0.0
        last_pos = prev_data.get_position()
        curr_pos = data.get_position()
        cp_pos = data.get_next_checkpoint_position()
        prev_dist_sq = (last_pos[0] - cp_pos[0]) ** 2 + (last_pos[1] - cp_pos[1]) ** 2
        curr_dist_sq = (curr_pos[0] - cp_pos[0]) ** 2 + (curr_pos[1] - cp_pos[1]) ** 2
        return prev_dist_sq - curr_dist_sq
