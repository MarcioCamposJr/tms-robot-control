import unittest

import numpy as np

from robot.constants import COLLISION_AVOIDANCE_CONFIG
from robot.control import coordinates
from robot.control.collision_avoidance import (
    ClosingSpeedEstimator,
    CollisionAvoidanceController,
    RepulsionConfig,
)
from robot.control.robot_control import RobotControl, RobotObjective


class DummyRobotPoseStorage:
    def __init__(self, pose):
        self.pose = pose

    def GetRobotPose(self):
        return self.pose


class DummyRobot:
    def __init__(self):
        self.dynamic_calls = []

    def dynamic_motion(self, target, speed_ratio):
        self.dynamic_calls.append((target, speed_ratio))
        return True


class DummyMovementAlgorithm:
    def __init__(self):
        self.reset_count = 0

    def reset_state(self):
        self.reset_count += 1


class DummyRobotStateController:
    def __init__(self):
        self.start_count = 0

    def set_state_to_start_moving(self):
        self.start_count += 1


class RobotCollisionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.control = RobotControl.__new__(RobotControl)
        self.control.collision_avoidance = CollisionAvoidanceController(
            RepulsionConfig(**COLLISION_AVOIDANCE_CONFIG)
        )
        self.control._reset_collision_speed_estimator()
        self.control.robot = None
        self.control._collision_safety_active = False
        self.control._collision_repulsion_active = False
        self.control._collision_warning = None
        self.control._last_collision_command_time = 0.0
        self.control.coil_collision_calculator = None
        self.control.coil_index = 2
        self.control.matrix_tracker_to_robot = (np.eye(4), np.eye(4), np.eye(4))
        self.control.robot_pose_storage = DummyRobotPoseStorage([0, 0, 0, 0, 0, 0])
        self.control.tracker = coordinates.Tracker()
        self.control.head_center = [0, -100, 0]

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

    def test_sets_two_collision_registrations_for_own_coil(self):
        registrations = {
            "robotized": self._make_registration(2),
            "other": self._make_registration(3),
        }

        success = self.control.on_set_collision_registrations(
            {"coil_idx": 2, "registrations": registrations}
        )

        self.assertTrue(success)
        self.assertEqual(self.control.coil_index, 2)
        self.assertEqual(self.control.coil_collision_calculator.object_ids, (2, 3))

        poses = np.zeros((4, 6))
        poses[3, 0] = 12
        measurement = self.control.coil_collision_calculator.measure(poses)
        self.assertAlmostEqual(measurement.distance, 7)

    def test_rejects_registrations_without_own_coil_atomically(self):
        original_calculator = object()
        self.control.coil_collision_calculator = original_calculator

        success = self.control.on_set_collision_registrations(
            {
                "coil_idx": 4,
                "registrations": {
                    "first": self._make_registration(2),
                    "second": self._make_registration(3),
                },
            }
        )

        self.assertFalse(success)
        self.assertIs(self.control.coil_collision_calculator, original_calculator)

    def test_calculates_own_repulsion_from_raw_tracker_poses(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "manual": self._make_registration(3),
                },
            }
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = 12

        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )
        command = self.control.collision_avoidance.compute_command(0.01)

        self.assertGreater(command.magnitude, 0)
        self.assertLess(command.offset[0], 0)

    def test_selects_opposite_direction_for_other_robot(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 3,
                "registrations": {
                    "first": self._make_registration(2),
                    "robotized": self._make_registration(3),
                },
            }
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = 12

        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )
        command = self.control.collision_avoidance.compute_command(0.01)

        self.assertGreater(command.offset[0], 0)

    def test_invisible_manual_coil_does_not_refresh_measurement(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "manual": self._make_registration(3),
                },
            }
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = 12

        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, False]}
        )

        self.assertFalse(self.control.collision_avoidance.has_measurement)

    def test_closing_speed_activates_repulsion_before_static_margin(self):
        now = 10.0
        self.control.collision_speed_estimator = ClosingSpeedEstimator(
            smoothing=0.0,
            max_closing_speed=300,
            clock=lambda: now,
        )
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "manual": self._make_registration(3),
                },
            }
        )
        # Registration resets the estimator, so install the deterministic clock
        # after setting it.
        self.control.collision_speed_estimator = ClosingSpeedEstimator(
            smoothing=0.0,
            max_closing_speed=300,
            clock=lambda: now,
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = 40
        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        now += 0.1
        poses[3, 0] = 35
        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        measurement = self.control.collision_avoidance._measurement
        self.assertAlmostEqual(measurement.distance, 30)
        self.assertAlmostEqual(measurement.closing_speed, 50)
        self.assertEqual(
            self.control.collision_avoidance.compute_command(0.01).stage.value,
            "approach",
        )

    def test_local_overlap_latches_collision_stop(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "other": self._make_registration(3),
                },
            }
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = 3

        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        self.assertTrue(self.control.collision_avoidance.stop_latched)
        self.assertTrue(self.control._collision_safety_active)

    def test_moving_coils_apart_releases_collision_stop(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "other": self._make_registration(3),
                },
            }
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = 3
        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        poses[3, 0] = 13
        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        self.assertFalse(self.control.collision_avoidance.stop_latched)
        self.assertFalse(self.control._collision_safety_active)

    def test_invalid_local_pose_latches_collision_stop(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "other": self._make_registration(3),
                },
            }
        )
        poses = np.zeros((4, 6))
        poses[3, 0] = np.nan

        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        self.assertTrue(self.control.collision_avoidance.stop_latched)

    def test_stops_when_coil_separation_would_move_toward_head(self):
        self.control.on_set_collision_registrations(
            {
                "coil_idx": 2,
                "registrations": {
                    "robotized": self._make_registration(2),
                    "other": self._make_registration(3),
                },
            }
        )
        self.control.head_center = [-100, 0, 0]
        poses = np.zeros((4, 6))
        poses[3, 0] = 12

        self.control.on_update_tracker_poses(
            {"poses": poses, "visibilities": [True, True, True, True]}
        )

        self.assertTrue(self.control.collision_avoidance.stop_latched)
        self.assertIn("No repulsion direction", self.control._collision_warning)

    def test_repulsion_overrides_any_movement_algorithm_in_base_coordinates(self):
        self.control.robot = DummyRobot()
        self.control.movement_algorithm = DummyMovementAlgorithm()
        self.control.robot_state_controller = DummyRobotStateController()
        self.control.config = {"tuning_speed_ratio": 0.15}
        self.control.objective = RobotObjective.TRACK_TARGET
        self.control.robot_pose_storage = DummyRobotPoseStorage(
            [100, 200, 300, 10, 20, 30]
        )
        self.control.collision_avoidance.update_measurement(10, [-1, 0, 0])
        self.control._last_collision_command_time -= 0.05

        warning = self.control._handle_collision_repulsion()

        self.assertIn("working", warning)
        target, speed_ratio = self.control.robot.dynamic_calls[-1]
        self.assertLess(target[0], 100)
        self.assertEqual(target[1:], [200, 300, 10, 20, 30])
        self.assertEqual(speed_ratio, 0.15)
        self.assertEqual(self.control.movement_algorithm.reset_count, 1)

    def test_repulsion_does_not_move_without_automatic_objective(self):
        self.control.robot = DummyRobot()
        self.control.movement_algorithm = DummyMovementAlgorithm()
        self.control.robot_state_controller = DummyRobotStateController()
        self.control.config = {"tuning_speed_ratio": 0.15}
        self.control.objective = RobotObjective.NONE
        self.control.collision_avoidance.update_measurement(10, [-1, 0, 0])

        warning = self.control._handle_collision_repulsion()

        self.assertIsNone(warning)
        self.assertEqual(self.control.robot.dynamic_calls, [])

    @staticmethod
    def _make_registration(object_id):
        return {
            "obj_id": object_id,
            "fiducials": [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
            "orientations": [[0, 0, 0]] * 4,
        }


if __name__ == "__main__":
    unittest.main()
