import numpy as np
from robot.constants import REPULSION_CONFIG


class RepulsionField:
    def __init__(self):
        self.cfg = REPULSION_CONFIG
        self.safety_margin = self.cfg['stop_distance']
        self.distance_coils = None
        self._ema_offset = np.zeros(3, dtype=float)
        self.brake_direction = np.zeros(3, dtype=float)

    def compute_offset(self, distance, dt):
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
        raw_offset = np.zeros(3, dtype=float)

        # Emergency stop condition
        if distance is not None and distance < self.cfg['stop_distance']*0.5:
            stop_now = True
            # Return an offset that completely cancels the current velocity
            return self.brake_direction, stop_now

        # Repulsion (braking) is only active within the safety margin
        if (
            self.safety_margin is not None
            and distance is not None
            and distance < self.safety_margin
        ):
            # The braking force increases with the inverse square of the distance
            brake_magnitude = self.cfg['strength'] * (
                (self.safety_margin / (distance + 1e-9)) ** 2 - 1
            )

            # The offset is the braking force applied in the correct direction
            raw_offset = brake_magnitude * self.brake_direction * dt

            print('brake_magnitude', brake_magnitude, 'brake_direction',self.brake_direction, 'raw_offset', raw_offset, "_ema_offset", self._ema_offset, "dt", dt)
        # EMA smoothing to prevent jerky movements
        self._ema_offset = (
            self.cfg['ema'] * self._ema_offset + (1 - self.cfg['ema']) * raw_offset
        )
        print("_ema_offset", self._ema_offset)
        return self._ema_offset, stop_now

    def update_safety_margin(self, new_margin):
        self.safety_margin = new_margin

    def update_distance_coils(self, new_distance):
        self.distance_coils = new_distance

    def update_opposite_coil_vector(self, poses, visibilities, robot_id):
        if len(poses)>3 and all(visibilities[1:]):
                opposite_coil_vector = (np.array(poses[2], dtype=float) - np.array(poses[3], dtype=float))[:3]
                opposite_subject_vector = (np.array(poses[2], dtype=float) - np.array(poses[1], dtype=float))[:3]
                opposite_coil_vector_norm = np.linalg.norm(opposite_coil_vector)
                opposite_subject_vector_norm = np.linalg.norm(opposite_subject_vector)
                if opposite_coil_vector_norm > 1e-9 and opposite_subject_vector_norm > 1e-9:
                    self.brake_direction = (opposite_coil_vector/opposite_coil_vector_norm) + (opposite_subject_vector/opposite_subject_vector_norm)

                    if robot_id == "robot_2":
                        self.brake_direction[2] = - self.brake_direction[2]

