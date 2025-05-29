import numpy as np
import cv2
from webcam_tracking import handDetector
from pid import PID
import time


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
        self.cap = cv2.VideoCapture(0)
        self.detector = handDetector()
        self.pid = PID(kp=2, ki=1, kd=0.1, target=self.eef_pos) 


    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """

        success, img = self.cap.read()
        img = self.detector.findHands(img)
        lmlist = self.detector.findPositions(img)
        # if len(lmlist) != 0:
        #     print(lmlist[4])
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # Move the eef according to the wrist landmark (lmlist[0])
        self.pid.reset(lmlist[0])

        # Compute grasp logic
        tipOfPointer = lmlist[8]
        tipOfThumb = lmlist[4]

        pinchDistance = np.linalg.norm(tipOfPointer - tipOfThumb)

        if (pinchDistance <= 0.5):
            grasp = 1
        else: 
            grasp = -1

        # Compute action
        action = np.zeros(7)
        delta = self.pid.update(current_pos=robot_eef_pos, dt=self.dt)
        action[:3] = delta
        action[-1] = grasp
        return action


