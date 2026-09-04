import math
from dataclasses import dataclass
from enum import Enum


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

    def __post_init__(self):
        values = (
            self.strength,
            self.safety_margin,
            self.working_distance,
            self.stop_distance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Repulsion configuration values must be finite")
        if self.strength < 0:
            raise ValueError("Repulsion strength must not be negative")
        if not 0 <= self.stop_distance < self.working_distance < self.safety_margin:
            raise ValueError(
                "Expected stop_distance < working_distance < safety_margin"
            )


@dataclass(frozen=True)
class RepulsionResult:
    stage: CollisionStage
    magnitude: float


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
