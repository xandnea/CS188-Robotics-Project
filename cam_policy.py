import numpy as np
import cv2
from webcam_tracking import handDetector
from pid import PID
import time
import os
import sys
import psutil
from scipy.spatial.transform import Rotation as R


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
        #self.square_pos = obs['SquareNut_pos']
        self.ee_pos = obs['robot0_eef_pos']

        self.cap = None
        self.webcam_available = self._initialize_webcam()
        if not self.webcam_available:
            print("[FATAL] No webcam available. Exiting...")
            sys.exit(1)
        self.detector = handDetector()
        self.pTime = time.time()
        self.pid = PID(kp=5, ki=2, kd=0.0, target=self.ee_pos) 
        self.dt = dt

        self.prev_depth_values = [0 for i in range(10)]
        self.target_rot_vec = np.zeros(3)
        self.initial_hand_size = None

    def _initialize_webcam(self):
        """Initialize webcam and verify it's working."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] Failed to open webcam.")
                return False
            # Try to read a frame to verify the webcam is working
            ret, _ = cap.read()
            if not ret:
                print("[ERROR] Could not read frame from webcam.")
                cap.release()
                return False
            self.cap = cap
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize webcam: {str(e)}")
            return False

    def compute_hand_size(self, lmlist):
        """
        Estimate hand size using the area of the triangle formed by the wrist and 2 outermost finger bases.
        lmlist: list of landmarks [[id, x, y, z], ...]
        Returns: float (size)
        """

        # area of triangle using y and z coords
        y1 = lmlist[0][2]
        z1 = lmlist[0][3]
        p1 = np.array([y1,z1])
        y2 = lmlist[5][2]
        z2 = lmlist[5][3]
        p2 = np.array([y2,z2])
        y3 = lmlist[17][2]
        z3 = lmlist[17][3]
        p3 = np.array([y3,z3])
        AB = p2 - p1
        AC = p3 - p1
        cross = np.cross(AB, AC)
        area = np.linalg.norm(cross) / 2
        return area
    
    def compute_depth(self):
        """
        Use a single (non palm) landmark to linearize depths and interpolate.

        Returns: float (depth)
        """

        # Cap eef depth and linearize
        min_eef_depth = -0.5
        max_eef_depth = 0.25
        steps = 100 # can change
        eef_depth = np.linspace(min_eef_depth, max_eef_depth, steps)

        # Cap landmark depth and linearize
        min_lm_depth = -0.5
        max_lm_depth = 2.5
        lm_depth = np.linspace(min_lm_depth, max_lm_depth, steps)

        # Average landmark depth over past n values 
        estimated_depth = np.mean(self.prev_depth_values)
        print(f"[INFO] Pre-Interpolated Thumb Depth: {estimated_depth}")

        # Interpolate the averaged landmark depth for a smoother calculation 
        depth = np.interp(estimated_depth, lm_depth, eef_depth)
        print(f"[INFO] Post-Interpolated Thumb Depth: {depth}")
        return depth
    
    def compute_rotation(self, lmlist):
        """
        Create axes based on landmarks of the hand and compute rotation.
        lmlist: list of landmarks [[id, x, y, z], ...]
        Returns: array (rotation vector)
        """

        # Create points
        wrist = np.array(lmlist[0][1:4])
        pinky_base = np.array(lmlist[17][1:4])
        thumb_base = np.array(lmlist[1][1:4])

        # Create axes, Z should point where the palm faces
        z_axis = np.cross(pinky_base - wrist, thumb_base - wrist)
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.array(thumb_base - wrist)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Compose rotation matrix
        rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

        rot_vec = R.from_matrix(rot_mat).as_rotvec()
        #print(f"[DEBUG] Axes: {x_axis, y_axis, z_axis}")
        #print(f"[DEBUG] Rotation Vector: {rot_vec}")
        return rot_vec

    
    def get_action(self, robot_eef_pos: np.ndarray, robot_eef_quat: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """

        # Webcam Initialization 
        success, img = self.cap.read()
        
        img = self.detector.findHands(img)
        if not success or img is None:
            print("[ERROR] Failed to read image from webcam.")
            return np.zeros(7)
        lmlist = self.detector.findPositions(img)
            
        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if cTime != self.pTime else 0 #prevent div by 0
        self.pTime = cTime
        if not running_in_mjpython(): #opencv2.show breaks in mjpython because of GUI issues
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

        # Old way of computing depth, can compare this to compute_depth once rotation is figured out 
            # #estimate hand size compared to original hand size to calc relative depth
            # #can't just use depth from camera because its relative to wrist location, not the world
            # current_hand_size = self.compute_hand_size(lmlist)

            # if self.initial_hand_size is None:
            #     self.initial_hand_size = current_hand_size

            # depth_scale = self.initial_hand_size / current_hand_size  

            # estimated_depth = depth_scale - 1 #because x should start at 0 but scaling would start at 1
            # estimated_depth = max(-0.5, min(estimated_depth, 0.25)) #clamp so robot doesnt bug out
            # # Move the eef according to the wrist landmark (lmlist[0])
            # new_pos = [estimated_depth] + lmlist[0][2:4]

        # Update previous depth values 
        self.prev_depth_values.pop(0)
        self.prev_depth_values.append(lmlist[4][1])

        # Compute depth 
        depth = self.compute_depth()
        new_pos = [depth] + lmlist[0][2:4]
        self.pid.reset(new_pos)

        # Compute rotation
        target_rot = R.from_rotvec(self.compute_rotation(lmlist))
        current_rot = R.from_quat(robot_eef_quat)
        delta_rot = target_rot * current_rot.inv()
        delta_rot = delta_rot.as_rotvec()

        # Compute grasp logic
        tipOfPointer = np.array(lmlist[8][2:4]) #just consider y and z, simplified 
        tipOfThumb = np.array(lmlist[4][2:4])
        pinchDistance = np.linalg.norm(tipOfPointer - tipOfThumb)
        if (pinchDistance <= 0.02):
            grasp = 1
        else: 
            grasp = -1

        # Compute action
        action = np.zeros(7)
        delta = self.pid.update(current_pos=robot_eef_pos, dt=self.dt)
        action[:3] = delta
        action[3:6] = delta_rot
        action[-1] = grasp
        return action


