import numpy as np
from robot.constants import REPULSION_CONFIG


class RepulsionField:
    def __init__(self):
        self.cfg = REPULSION_CONFIG.copy()  # Make a copy to allow instance-specific changes
        self.safety_margin = self.cfg['safety_margin']
        self.distance_coils = None
        self._ema_offset = np.zeros(3, dtype=float)
        self.brake_direction = np.zeros(3, dtype=float)
    
    def update_config(self, config_updates):
        """
        Update configuration parameters dynamically during runtime.
        Called by RobotControl when receiving config update messages from the server.
        
        Args:
            config_updates (dict): Dictionary with configuration keys to update.
                                  Valid keys: 'strength', 'safety_margin', 'ema', 'stop_distance'
        """
        if not isinstance(config_updates, dict):
            print(f"RepulsionField: Invalid config update (expected dict, got {type(config_updates)})")
            return
        
        for key, value in config_updates.items():
            if key in self.cfg:
                old_value = self.cfg[key]
                self.cfg[key] = value
                print(f"RepulsionField: Updated {key} from {old_value} to {value}")
                
                # Update safety_margin separately since it's also an instance variable
                if key == 'safety_margin':
                    self.safety_margin = value
            else:
                print(f"RepulsionField: Unknown config key '{key}' (ignored)")
        
        print(f"RepulsionField: Current config: {self.cfg}")

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
        if distance is not None and distance < self.cfg['stop_distance']:
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
                (self.safety_margin / (distance + 1e-6)) ** 3 
            )

            # The offset is the braking force applied in the correct direction
            raw_offset = brake_magnitude * self.brake_direction * dt

            print('brake_magnitude', brake_magnitude, 'brake_direction',self.brake_direction, 'raw_offset', raw_offset, "_ema_offset", self._ema_offset, "dt", dt)
        # EMA smoothing to prevent jerky movements
        self._ema_offset = (
            self.cfg['ema'] * self._ema_offset + (1 - self.cfg['ema']) * raw_offset
        )
        return self._ema_offset, stop_now

    def update_safety_margin(self, new_margin):
        self.safety_margin = new_margin

    def update_distance_coils(self, new_distance):
        self.distance_coils = new_distance

    def update_opposite_coil_vector(self, brake_direction):
        self.brake_direction = brake_direction

