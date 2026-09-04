import unittest

import numpy as np

from robot.constants import COLLISION_AVOIDANCE_CONFIG
from robot.control.collision_avoidance import (
    CollisionAvoidanceController,
    RepulsionConfig,
)
from robot.control.robot_control import RobotControl


class DummyRobotPoseStorage:
    def __init__(self, pose):
        self.pose = pose

    def GetRobotPose(self):
        return self.pose


class RobotCollisionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.control = RobotControl.__new__(RobotControl)
        self.control.collision_avoidance = CollisionAvoidanceController(
            RepulsionConfig(**COLLISION_AVOIDANCE_CONFIG)
        )
        self.control.robot = None
        self.control._collision_safety_active = False
        self.control._collision_warning = None

    def test_transforms_direction_from_base_to_tool_coordinates(self):
        self.control.robot_pose_storage = DummyRobotPoseStorage(
            [0, 0, 0, 0, 0, 90]
        )

        direction = self.control._collision_direction_in_tool_space([1, 0, 0])

        np.testing.assert_allclose(direction, [0, -1, 0], atol=1e-12)

    def test_stop_measurement_does_not_require_robot_pose(self):
        success = self.control.on_update_coil_distance({"distance": 1})

        self.assertTrue(success)
        self.assertTrue(self.control.collision_avoidance.stop_latched)
        self.assertTrue(self.control._collision_safety_active)

    def test_invalid_measurement_latches_stop(self):
        success = self.control.on_update_coil_distance({"distance": "invalid"})

        self.assertFalse(success)
        self.assertTrue(self.control.collision_avoidance.stop_latched)

    def test_updates_collision_config(self):
        success = self.control.on_update_collision_config(
            {"config_updates": {"strength": 30}}
        )

        self.assertTrue(success)
        self.assertEqual(self.control.collision_avoidance.config.strength, 30)

    def test_rejects_invalid_collision_config(self):
        original_config = self.control.collision_avoidance.config

        success = self.control.on_update_collision_config(
            {"config_updates": {"working_distance": 100}}
        )

        self.assertFalse(success)
        self.assertIs(self.control.collision_avoidance.config, original_config)


if __name__ == "__main__":
    unittest.main()
