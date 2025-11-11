import numpy as np
from robot.constants import REPULSION_CONFIG


class RepulsionField:
    def __init__(self):
        self.cfg = REPULSION_CONFIG
        self.safety_margin = None
        self.distance_coils = None
        self._ema_offset = np.zeros(3)

    def compute_offset(self, distance, current_velocity, dt):
        """
        Calculates a "braking" offset that opposes the current velocity.

        Args:
            distance (float): Scalar distance to the other object (in mm).
            current_velocity (np.ndarray): The robot's current velocity vector.
            dt (float): Delta time to scale the offset.

        Returns:
            tuple: (offset_xyz, stop_now)
        """
        stop_now = False
        raw_offset = np.zeros(3)

        # Emergency stop condition
        if distance is not None and distance < self.cfg.stop_distance:
            stop_now = True
            # Return an offset that completely cancels the current velocity
            return -current_velocity, stop_now

        # Repulsion (braking) is only active within the safety margin
        if (
            self.safety_margin is not None
            and distance is not None
            and distance < self.safety_margin
        ):
            velocity_norm = np.linalg.norm(current_velocity)
            if velocity_norm > 1e-6:  # Avoid division by zero if the robot is stationary
                # The braking force increases with the inverse square of the distance
                brake_magnitude = self.cfg.strength * (
                    (self.safety_margin / (distance + 1e-9)) ** 2 - 1
                )

                # The direction of the "brake" is opposite to the velocity
                brake_direction = -current_velocity / velocity_norm

                # The offset is the braking force applied in the correct direction
                raw_offset = brake_magnitude * brake_direction * dt

        # EMA smoothing to prevent jerky movements
        self._ema_offset = (
            self.cfg.ema * self._ema_offset + (1 - self.cfg.ema) * raw_offset
        )

        return self._ema_offset, stop_now

    def update_safety_margin(self, new_margin):
        self.safety_margin = new_margin

    def update_distance_coils(self, new_distance):
        self.distance_coils = new_distance

