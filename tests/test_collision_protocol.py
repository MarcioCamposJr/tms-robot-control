import unittest

import robot.constants as const


class CollisionProtocolTests(unittest.TestCase):
    def test_collision_message_indices_match_topics(self):
        self.assertEqual(
            const.PUB_MESSAGES[const.FUNCTION_UPDATE_COIL_DISTANCE],
            "Neuronavigation to Robot: Dynamically update distance coils",
        )
        self.assertEqual(
            const.PUB_MESSAGES[const.FUNCTION_RESET_COLLISION_STOP],
            "Neuronavigation to Robot: Reset collision error",
        )
        self.assertEqual(
            const.PUB_MESSAGES[const.FUNCTION_UPDATE_COLLISION_CONFIG],
            "Neuronavigation to Robot: Update repulsion field config",
        )
        self.assertEqual(
            const.PUB_MESSAGES[const.FUNCTION_SET_COLLISION_REGISTRATIONS],
            "Neuronavigation to Robot: Set coil collision registrations",
        )


if __name__ == "__main__":
    unittest.main()
