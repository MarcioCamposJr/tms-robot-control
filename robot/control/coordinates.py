import numpy as np

import robot.control.robot_processing as robot_process
import robot.transformations as tr


class RobotPoseStorage:
    """
    Class to set/send robot pose.
    The class is required to avoid acquisition conflict with different threads (coordinates and navigation)
    """

    def __init__(self):
        self.robot_pose = [None] * 6

    def SetRobotPose(self, robot_pose):
        robot_pose = np.array(robot_pose)
        self.robot_pose = robot_pose

    def GetRobotPose(self):
        return self.robot_pose


class Tracker:
    """
    Class to set/get tracker coordinates and do tracker-related transformations.
    Tracker coordinates are acquired in InVesalius.
    The class is required to avoid acquisition conflict with different threads
    """

    def __init__(self):
        self.poses = [None, None, None]
        self.m_tracker_to_robot = None

        self.probe_visible = False
        self.head_visible = False
        self.coil_visible = False

        self.head_pose = None

    def SetTrackerToRobotMatrix(self, m_tracker_to_robot):
        self.m_tracker_to_robot = m_tracker_to_robot

    def SetCoordinates(self, poses, visibilities):
        self.poses = poses

        self.probe_visible = visibilities[0]
        self.head_visible = visibilities[1]
        self.coil_visible = visibilities[2]

        self.head_pose = poses[1]
        self.coil_pose = poses[2]

        # XXX: The poses comes from neuronavigation using the 'rzyx' convention for interpreting Euler angles
        #   (That is, the axis order is z, y, x, using rotating frame). Convert it to the 'sxyz' convention used in
        #   this robot controller. The transformation below should not be the correct transformation, but it corresponds
        #   to how rotating frame is implemented in euler_matrix function in transformations.py: by swapping the x- and
        #   z-rotations.
        self.head_pose[3], self.head_pose[5] = self.head_pose[5], self.head_pose[3]
        self.coil_pose[3], self.coil_pose[5] = self.coil_pose[5], self.coil_pose[3]

    def get_head_pose(self):
        return self.head_pose

    def transform_matrix_to_robot_space(self, M):
        X, Y, affine = self.m_tracker_to_robot

        M_in_robot_space = Y @ M @ tr.inverse_matrix(X)
        M_affine_in_robot_space = affine @ M

        _, angles_as_deg = robot_process.transformation_matrix_to_coordinates(
            M_in_robot_space, axes="sxyz"
        )
        translation, _ = robot_process.transformation_matrix_to_coordinates(
            M_affine_in_robot_space, axes="sxyz"
        )

        pose_in_robot_space = list(translation) + list(angles_as_deg)

        return pose_in_robot_space

    def transform_pose_to_robot_space(self, pose):
        M = robot_process.coordinates_to_transformation_matrix(
            position=pose[:3],
            orientation=pose[3:],
            axes="sxyz",
        )
        pose_in_robot_space = self.transform_matrix_to_robot_space(M)

        if pose_in_robot_space is None:
            pose_in_robot_space = pose

        return pose_in_robot_space
