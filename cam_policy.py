import numpy as np
from webcam_tracking import handDetector
from pid import PID


class CamPolicy:
    """
    A policy that follows the position of a tracked hand with PID control.

    Args:
        obs (dict): observed envrionment.
        dt (float): control timestep.
    """
    def __init__(self, obs, dt=0.01):
        self.square_pos = obs['SquareNut_pos']
        self.ee_pos = obs['obs_robot0_eef_pos'] 


    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        # Compute action
        action = np.zeros(7)
        #action[:3] = delta
        #action[-1] = grasp
        return action


