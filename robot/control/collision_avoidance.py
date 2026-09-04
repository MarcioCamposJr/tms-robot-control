import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CollisionStage(Enum):
    CLEAR = "clear"
    APPROACH = "approach"
    WORKING = "working"
    STOP = "stop"


@dataclass(frozen=True)
class RepulsionConfig:
    strength: float
    safety_margin: float
    working_distance: float
    stop_distance: float
    max_offset: float = 2.0
    max_delta_time: float = 0.05

    def __post_init__(self):
        values = (
            self.strength,
            self.safety_margin,
            self.working_distance,
            self.stop_distance,
            self.max_offset,
            self.max_delta_time,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Repulsion configuration values must be finite")
        if self.strength < 0:
            raise ValueError("Repulsion strength must not be negative")
        if self.max_offset <= 0 or self.max_delta_time <= 0:
            raise ValueError("Repulsion limits must be positive")
        if not 0 <= self.stop_distance < self.working_distance < self.safety_margin:
            raise ValueError(
                "Expected stop_distance < working_distance < safety_margin"
            )


@dataclass(frozen=True)
class RepulsionResult:
    stage: CollisionStage
    magnitude: float


@dataclass(frozen=True)
class RepulsionCommand:
    stage: CollisionStage
    magnitude: float
    offset: np.ndarray
    stop_requested: bool


class RepulsionField:
    """Compute the repulsion magnitude from the distance between two coils."""

    def __init__(self, config: RepulsionConfig):
        self.config = config

    def compute(self, distance: float) -> RepulsionResult:
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Coil distance must be a finite, non-negative value")

        if distance <= self.config.stop_distance:
            return RepulsionResult(CollisionStage.STOP, 0.0)

        if distance >= self.config.safety_margin:
            return RepulsionResult(CollisionStage.CLEAR, 0.0)

        if distance <= self.config.working_distance:
            normalized = distance / self.config.working_distance
            magnitude = self.config.strength * math.exp(2 * (1 - normalized))
            return RepulsionResult(CollisionStage.WORKING, magnitude)

        normalized = (self.config.safety_margin - distance) / (
            self.config.safety_margin - self.config.working_distance
        )
        magnitude = self.config.strength * normalized**2
        return RepulsionResult(CollisionStage.APPROACH, magnitude)

    def compute_offset(
        self, distance: float, direction, delta_time: float
    ) -> RepulsionCommand:
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be a finite, non-negative value")

        result = self.compute(distance)
        zero_offset = np.zeros(3, dtype=float)

        if result.stage is CollisionStage.STOP:
            return RepulsionCommand(result.stage, result.magnitude, zero_offset, True)
        if result.stage is CollisionStage.CLEAR:
            return RepulsionCommand(result.stage, result.magnitude, zero_offset, False)

        direction = np.asarray(direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Repulsion direction must contain three finite values")

        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-9:
            raise ValueError("Repulsion direction must not be a zero vector")

        effective_delta_time = min(delta_time, self.config.max_delta_time)
        offset = result.magnitude * (direction / direction_norm) * effective_delta_time

        offset_norm = np.linalg.norm(offset)
        if offset_norm > self.config.max_offset:
            offset *= self.config.max_offset / offset_norm

        return RepulsionCommand(result.stage, result.magnitude, offset, False)
