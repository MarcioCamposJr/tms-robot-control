"""Convert raw tracker frames into measurements for collision avoidance."""

import time

from robot.control.coil_geometry import (
    CoilCollisionCalculator,
    box_from_tracker_to_robot,
    constrain_direction_away_from_head,
    direction_from_tracker_to_robot,
)
from robot.control.collision_avoidance import ClosingSpeedEstimator, CollisionStage


class CoilCollisionTracker:
    """Own coil registrations and motion history, without commanding a robot."""

    def __init__(self, controller, clock=time.monotonic):
        self.controller = controller
        self.calculator = None
        self.speed_estimator = ClosingSpeedEstimator(
            smoothing=controller.config.velocity_smoothing,
            max_closing_speed=controller.config.max_closing_speed,
            clock=clock,
        )

    def set_registrations(self, registrations, coil_index, **geometry_config):
        """Validate all registration data before replacing the current geometry."""
        if (
            not isinstance(coil_index, int)
            or isinstance(coil_index, bool)
            or coil_index < 0
        ):
            raise ValueError("Own coil tracker object ID must be a non-negative integer")
        calculator = CoilCollisionCalculator(registrations, **geometry_config)
        if coil_index not in calculator.object_ids:
            raise ValueError("Own coil is not present in the collision registrations")
        self.calculator = calculator
        self.speed_estimator.reset()

    def update_from_tracker_poses(
        self, poses, visibilities, coil_index, tracker_to_robot, get_head_center
    ):
        """Return (stage, distance), or None when no new measurement is available.

        get_head_center is evaluated only when repulsion needs a head constraint.
        It must return the current frame's head center in robot coordinates.
        Invalid frames raise an error for the caller to stop the robot.
        """
        if self.calculator is None:
            return None
        try:
            if any(not bool(visibilities[i]) for i in self.calculator.object_ids):
                # Keep the previous timestamp so the controller's watchdog expires.
                self.speed_estimator.reset()
                return None
            measurement = self.calculator.measure(poses)
            closing_speed = self.speed_estimator.update(
                measurement.box_a.center, measurement.box_b.center
            )
            stage = self.controller.field.compute(
                measurement.distance, closing_speed
            ).stage
            direction = measurement.direction_for(coil_index)
            if stage not in (CollisionStage.STOP, CollisionStage.CLEAR):
                direction = direction_from_tracker_to_robot(direction, tracker_to_robot)
                head_center = get_head_center()
                coil_box = box_from_tracker_to_robot(
                    measurement.box_for(coil_index), tracker_to_robot
                )
                direction = constrain_direction_away_from_head(
                    direction, coil_box, head_center
                )
            stage = self.controller.update_measurement(
                measurement.distance, direction, closing_speed
            )
            return stage, measurement.distance
        except (IndexError, TypeError, ValueError, RuntimeError):
            self.speed_estimator.reset()
            raise
