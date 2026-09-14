import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot.control.coil_geometry import (
    CoilCollisionCalculator,
    OrientedBoundingBox,
    coil_box_from_registration,
    direction_from_tracker_to_robot,
    measure_obb_distance,
)


def make_box(center, half_sizes=(1, 1, 1), rotation=None):
    if rotation is None:
        rotation = np.eye(3)
    half_axes = (rotation @ np.diag(half_sizes)).T
    return OrientedBoundingBox(np.asarray(center, dtype=float), half_axes)


def make_registration(object_id):
    return {
        "obj_id": object_id,
        "fiducials": [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        "orientations": [[0, 0, 0]] * 4,
    }


class OrientedBoundingBoxTests(unittest.TestCase):
    def test_measures_surface_distance_and_opposite_directions(self):
        result = measure_obb_distance(make_box([0, 0, 0]), make_box([3, 0, 0]))

        self.assertAlmostEqual(result.distance, 1)
        np.testing.assert_allclose(result.closest_point_a, [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(result.closest_point_b, [2, 0, 0], atol=1e-7)
        np.testing.assert_allclose(result.brake_direction_a, [-1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(result.brake_direction_b, [1, 0, 0], atol=1e-7)

    def test_returns_zero_for_intersecting_boxes(self):
        result = measure_obb_distance(make_box([0, 0, 0]), make_box([1, 0, 0]))

        self.assertEqual(result.distance, 0)
        np.testing.assert_array_equal(result.brake_direction_a, np.zeros(3))
        np.testing.assert_array_equal(result.brake_direction_b, np.zeros(3))

    def test_supports_rotated_boxes(self):
        rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        result = measure_obb_distance(
            make_box([0, 0, 0], half_sizes=(2, 1, 1), rotation=rotation),
            make_box([5, 0, 0]),
        )

        expected = 5 - 2 * np.cos(np.radians(45)) - np.sin(np.radians(45)) - 1
        self.assertAlmostEqual(result.distance, expected)

    def test_rejects_degenerate_box(self):
        with self.assertRaises(ValueError):
            OrientedBoundingBox(np.zeros(3), np.zeros((3, 3)))


class CoilRegistrationTests(unittest.TestCase):
    def test_builds_marker_local_box(self):
        registration = {
            "fiducials": [
                [-20, 0, 0],
                [20, 0, 0],
                [0, 30, 0],
                [10, 20, 30],
            ],
            "orientations": [[0, 0, 0]] * 4,
        }

        box = coil_box_from_registration(registration, half_thickness=5)

        np.testing.assert_allclose(box.center, [-10, -20, -30])
        np.testing.assert_allclose(box.half_axes, [[20, 0, 0], [0, 30, 0], [0, 0, 5]])

    def test_expands_lateral_and_front_back_faces_independently(self):
        registration = {
            "fiducials": [
                [-20, 0, 0],
                [20, 0, 0],
                [0, 30, 0],
                [0, 0, 0],
            ],
            "orientations": [[0, 0, 0]] * 4,
        }

        box = coil_box_from_registration(
            registration,
            half_thickness=5,
            lateral_expansion=1.5,
            face_expansion=3,
        )

        np.testing.assert_allclose(
            box.half_axes, [[21.5, 0, 0], [0, 31.5, 0], [0, 0, 8]]
        )

    def test_rejects_invalid_expansions(self):
        registration = make_registration(2)

        for lateral, face in ((-1, 0), (0, -1), (np.nan, 0)):
            with self.subTest(lateral=lateral, face=face), self.assertRaises(ValueError):
                coil_box_from_registration(
                    registration,
                    lateral_expansion=lateral,
                    face_expansion=face,
                )

    def test_rejects_invalid_registration(self):
        for registration in (
            {},
            {"fiducials": [], "orientations": []},
            {"fiducials": [[0, 0, 0]] * 4, "orientations": [[0, 0, 0]] * 4},
        ):
            with self.subTest(registration=registration), self.assertRaises(ValueError):
                coil_box_from_registration(registration)


class CoilCollisionCalculatorTests(unittest.TestCase):
    def test_uses_registered_object_ids_and_selects_own_direction(self):
        calculator = CoilCollisionCalculator(
            {"coil-a": make_registration(4), "coil-b": make_registration(2)},
            half_thickness=1,
        )
        coordinates = np.zeros((5, 6))
        coordinates[2, 0] = 3

        result = calculator.measure(coordinates)

        self.assertEqual(calculator.object_ids, (4, 2))
        self.assertAlmostEqual(result.distance, 1)
        np.testing.assert_allclose(result.direction_for(4), [-1, 0, 0])
        np.testing.assert_allclose(result.direction_for(2), [1, 0, 0])

    def test_rejects_duplicate_object_ids(self):
        with self.assertRaises(ValueError):
            CoilCollisionCalculator(
                {"coil-a": make_registration(2), "coil-b": make_registration(2)}
            )

    def test_rejects_missing_tracker_pose(self):
        calculator = CoilCollisionCalculator(
            {"coil-a": make_registration(2), "coil-b": make_registration(4)}
        )

        with self.assertRaises(ValueError):
            calculator.measure(np.zeros((3, 6)))

    def test_rejects_direction_request_for_unregistered_object(self):
        calculator = CoilCollisionCalculator(
            {"coil-a": make_registration(0), "coil-b": make_registration(1)},
            half_thickness=1,
        )
        result = calculator.measure(np.array([[0] * 6, [3, 0, 0, 0, 0, 0]]))

        with self.assertRaises(ValueError):
            result.direction_for(2)


class DirectionTransformationTests(unittest.TestCase):
    def test_transforms_direction_using_robot_affine(self):
        affine = np.eye(4)
        affine[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()

        result = direction_from_tracker_to_robot([1, 0, 0], affine)

        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)

    def test_accepts_robot_registration_tuple(self):
        affine = np.eye(4)
        affine[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()

        result = direction_from_tracker_to_robot(
            [1, 0, 0], (np.eye(4), np.eye(4), affine)
        )

        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
