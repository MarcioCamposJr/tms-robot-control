import math
import unittest

import numpy as np

from robot.constants import COLLISION_AVOIDANCE_CONFIG
from robot.control.collision_avoidance import (
    CollisionAvoidanceController,
    CollisionStage,
    RepulsionConfig,
    RepulsionField,
)


class RepulsionConfigTests(unittest.TestCase):
    def test_application_defaults_are_valid(self):
        config = RepulsionConfig(**COLLISION_AVOIDANCE_CONFIG)

        self.assertGreater(config.stop_distance, 0)

    def test_rejects_invalid_distance_order(self):
        with self.assertRaises(ValueError):
            RepulsionConfig(
                strength=25,
                safety_margin=12,
                working_distance=12,
                stop_distance=1,
            )

    def test_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            RepulsionConfig(
                strength=math.nan,
                safety_margin=22,
                working_distance=12,
                stop_distance=1,
            )


class RepulsionFieldTests(unittest.TestCase):
    def setUp(self):
        self.config = RepulsionConfig(
            strength=25,
            safety_margin=22,
            working_distance=12,
            stop_distance=1,
        )
        self.field = RepulsionField(self.config)

    def test_selects_stage_from_distance(self):
        cases = (
            (30, CollisionStage.CLEAR),
            (17, CollisionStage.APPROACH),
            (6, CollisionStage.WORKING),
            (1, CollisionStage.STOP),
        )

        for distance, expected_stage in cases:
            with self.subTest(distance=distance):
                result = self.field.compute(distance)
                self.assertEqual(result.stage, expected_stage)

    def test_magnitude_is_continuous_at_working_distance(self):
        at_boundary = self.field.compute(self.config.working_distance)
        just_above = self.field.compute(self.config.working_distance + 1e-8)

        self.assertAlmostEqual(at_boundary.magnitude, self.config.strength)
        self.assertAlmostEqual(at_boundary.magnitude, just_above.magnitude, places=6)

    def test_approach_magnitude_grows_as_distance_decreases(self):
        farther = self.field.compute(20)
        nearer = self.field.compute(15)

        self.assertGreater(nearer.magnitude, farther.magnitude)

    def test_working_magnitude_grows_as_distance_decreases(self):
        farther = self.field.compute(10)
        nearer = self.field.compute(5)

        self.assertGreater(nearer.magnitude, farther.magnitude)

    def test_rejects_invalid_measured_distance(self):
        for distance in (-1, math.nan, math.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.field.compute(distance)

    def test_normalizes_repulsion_direction(self):
        unit_direction = self.field.compute_offset(10, [1, 0, 0], 0.01)
        scaled_direction = self.field.compute_offset(10, [100, 0, 0], 0.01)

        np.testing.assert_allclose(unit_direction.offset, scaled_direction.offset)

    def test_rejects_invalid_repulsion_direction_in_active_zone(self):
        invalid_directions = ([0, 0, 0], [1, 0], [1, math.nan, 0])

        for direction in invalid_directions:
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                self.field.compute_offset(10, direction, 0.01)

    def test_limits_delta_time_and_offset(self):
        delayed = self.field.compute_offset(2, [1, 0, 0], 10)

        self.assertLessEqual(np.linalg.norm(delayed.offset), self.config.max_offset)

    def test_stop_stage_requests_stop_without_offset(self):
        result = self.field.compute_offset(1, [0, 0, 0], 0.01)

        self.assertTrue(result.stop_requested)
        np.testing.assert_array_equal(result.offset, np.zeros(3))


class CollisionAvoidanceControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.config = RepulsionConfig(
            strength=25,
            safety_margin=22,
            working_distance=12,
            stop_distance=1,
            measurement_timeout=0.25,
            stop_release_distance=2,
        )
        self.controller = CollisionAvoidanceController(
            self.config, clock=lambda: self.now
        )

    def test_does_not_stop_before_first_measurement(self):
        command = self.controller.compute_command(0.01)

        self.assertEqual(command.stage, CollisionStage.UNAVAILABLE)
        self.assertFalse(command.stop_requested)

    def test_requests_stop_when_measurement_becomes_stale(self):
        self.controller.update_measurement(10, [1, 0, 0])
        self.now += self.config.measurement_timeout + 0.01

        command = self.controller.compute_command(0.01)

        self.assertEqual(command.stage, CollisionStage.UNAVAILABLE)
        self.assertTrue(command.stop_requested)
        self.assertTrue(self.controller.measurement_is_stale())
        self.assertTrue(self.controller.stop_requested())

    def test_emergency_stop_remains_latched(self):
        self.controller.update_measurement(1, [0, 0, 0])
        self.controller.update_measurement(10, [1, 0, 0])

        command = self.controller.compute_command(0.01)

        self.assertEqual(command.stage, CollisionStage.STOP)
        self.assertTrue(command.stop_requested)
        self.assertTrue(self.controller.stop_latched)

    def test_can_latch_stop_after_invalid_external_input(self):
        self.controller.update_measurement(10, [1, 0, 0])

        self.controller.latch_stop()

        self.assertTrue(self.controller.stop_requested())

    def test_releases_stop_only_beyond_release_distance(self):
        self.controller.update_measurement(1, [0, 0, 0])
        self.controller.update_measurement(2, [1, 0, 0])
        self.assertFalse(self.controller.reset_stop())

        self.controller.update_measurement(4, [1, 0, 0])
        self.assertTrue(self.controller.reset_stop())
        self.assertFalse(self.controller.compute_command(0.01).stop_requested)

    def test_does_not_release_stop_from_stale_measurement(self):
        self.controller.update_measurement(1, [0, 0, 0])
        self.controller.update_measurement(4, [1, 0, 0])
        self.now += self.config.measurement_timeout + 0.01

        self.assertFalse(self.controller.reset_stop())

    def test_clear_stage_removes_smoothed_offset(self):
        self.controller.update_measurement(10, [1, 0, 0])
        self.assertGreater(np.linalg.norm(self.controller.compute_command(0.01).offset), 0)

        self.controller.update_measurement(30, [1, 0, 0])
        command = self.controller.compute_command(0.01)

        np.testing.assert_array_equal(command.offset, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
