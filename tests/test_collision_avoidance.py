import math
import unittest

from robot.control.collision_avoidance import (
    CollisionStage,
    RepulsionConfig,
    RepulsionField,
)


class RepulsionConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
