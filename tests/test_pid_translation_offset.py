import math
import unittest

from robot.control.PID import PIDControllerGroup


class PIDTranslationOffsetTests(unittest.TestCase):
    def setUp(self):
        self.pid_group = PIDControllerGroup()
        for pid in self.pid_group.translation_pids:
            pid.output = 1.0

    def test_adds_offset_to_translation_output(self):
        translations, _ = self.pid_group.get_outputs([0.5, 1.0, 1.5])

        self.assertEqual(translations, [-0.5, 0.0, 0.5])

    def test_limits_combined_translation_output(self):
        translations, _ = self.pid_group.get_outputs([10, -10, 0])

        self.assertEqual(translations, [2.0, -2.0, -1.0])

    def test_rejects_invalid_offset(self):
        with self.assertRaises(ValueError):
            self.pid_group.get_outputs([1, 2])
        with self.assertRaises(ValueError):
            self.pid_group.get_outputs([1, math.nan, 3])


if __name__ == "__main__":
    unittest.main()
