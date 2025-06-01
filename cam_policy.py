import numpy as np
import cv2
from webcam_tracking import handDetector
from pid import PID
import time
import os
import psutil
def running_in_mjpython():
   
    p = psutil.Process(os.getpid())
    # Get process executable path or name
    exe = p.exe().lower()   # full executable path
    name = p.name().lower() # process name

    # Check if 'mjpython' is in the executable path or process name
    if 'mjpython' in exe or 'mjpython' in name:
        return True
    return False


class CamPolicy:
    """
    A policy that follows the position of a tracked hand with PID control.

    Args:
        obs (dict): observed envrionment.
        dt (float): control timestep.
    """
    def __init__(self, obs, dt=0.01):
        self.square_pos = obs['SquareNut_pos']
        self.ee_pos = obs['robot0_eef_pos']
        
        self.cap = cv2.VideoCapture(0)
        self.detector = handDetector()
        self.pTime = time.time()
        self.pid = PID(kp=2, ki=1, kd=0.1, target=self.ee_pos) 
        self.dt = dt


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
        if not success or img is None:
            print("[ERROR] Failed to read image from webcam.")
            return np.zeros(7)
        lmlist = self.detector.findPositions(img)
        # if len(lmlist) != 0:
        #     print(lmlist[4])
            
        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if cTime != self.pTime else 0 #prevent div by 0
        self.pTime = cTime
        if not running_in_mjpython: #opencv2.show breaks in mjpython because of GUI issues
            try:
                cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

                cv2.imshow("Image", img)
                cv2.waitKey(1)
            except cv2.error as e:
                print(f"[OpenCV ERROR] {e}")
                return np.zeros(7)
        if not lmlist or len(lmlist) < 9:
            print("[WARNING] Hand landmarks not detected.")
            return np.zeros(7)
        # Move the eef according to the wrist landmark (lmlist[0])
        self.pid.reset(lmlist[0][1:4])

        # Compute grasp logic
        tipOfPointer = np.array(lmlist[8])
        tipOfThumb = np.array(lmlist[4])

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


