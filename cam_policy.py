import numpy as np
import math
import cv2
from webcam_tracking import handDetector
from pid import PID
import time
import os
import sys
import psutil
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


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
        self.base_rot = None
        self.left_hand_was_fist = False

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
        min_eef_depth = -0.45
        max_eef_depth = 0.25
        steps = 100 # can change
        eef_depth = np.linspace(min_eef_depth, max_eef_depth, steps)

        # Cap landmark depth and linearize
        min_lm_depth = -0.5
        max_lm_depth = 1.25
        lm_depth = np.linspace(min_lm_depth, max_lm_depth, steps)

        # Average landmark depth over past n values
        estimated_depth = np.mean(self.prev_depth_values)
        #print(f"[INFO] Pre-Interpolated Pinky Depth: {estimated_depth}")

        # Interpolate the averaged landmark depth for a smoother calculation
        depth = np.interp(estimated_depth, lm_depth, eef_depth)
        #print(f"[INFO] Post-Interpolated Pinky Depth: {depth}")
        return depth

    def compute_rotation(self, lmlist):
        """
        Create axes based on landmarks of the hand and compute rotation.
        lmlist: list of landmarks [[id, x, y, z], ...]
        Returns: array (rotation vector)
        """

        # Create points
        wrist = np.array(lmlist[0][1:4])         # x (depth; always -1 for wrist), y, z 
        pinky_base = np.array(lmlist[17][1:4])
        thumb_base = np.array(lmlist[1][1:4])
        #wrist_normal = np.array(lmlist[21][1:4])

        # Create axes, Z should point where the palm faces
        def safe_norm(v, eps=1e-6):
            return v / (np.linalg.norm(v) + eps)

        v1 = thumb_base - wrist
        v2 = pinky_base - wrist
        z_axis = safe_norm(np.cross(v1, v2))
        x_axis = safe_norm(v1)
        y_axis = safe_norm(np.cross(z_axis, x_axis))

        # Compose rotation matrix
        rot_mat = np.stack([y_axis, z_axis, x_axis], axis=1)

        rot_vec = R.from_matrix(rot_mat).as_rotvec()
        print(f"[DEBUG] Axes: {x_axis, y_axis, z_axis}")
        print(f"[DEBUG] Rotation Vector: {rot_vec}")
        return rot_vec
    def vector_angle(self, v1, v2):
        """
        Compute angle (in degrees) between two vectors in 3D.
        """
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0
        cos_theta = max(min(dot / (norm1 * norm2), 1.0), -1.0)
        return math.degrees(math.acos(cos_theta))


    def is_hand_fist(self, lmlist):
        """
        Calculates whether a hand is in a fist pose (rough calculations)
        Note that this is most accurate when the palm faces the camera
        lmlist: list of landmarks [[id, x, y, z], ...]
        Returns: bool (true/false)
        """
        
        wrist = lmlist[0]
        firsts = [5,9,13,17]
        seconds = [6,10,14,18]
        thirds = [7,11,15,19]
        tips = [8,12,16,20]

        curled_fingers = 0
        finger_num = 0
        fingers = ["index", "middle", "ring", "pinky"]
        finger_map = dict(zip(range(0,4), fingers))
        for first_id, second_id, third_id, tip_id in zip(firsts,seconds,thirds, tips):
            
            first = lmlist[first_id]
            second = lmlist[second_id]
            third = lmlist[third_id]
            tip = lmlist[tip_id]

            wrist_first_vec = [first[i] - wrist[i] for i in range(1, 4)]
            first_second_vec = [second[i] - first[i] for i in range(1, 4)]
            angle_first_second = self.vector_angle(wrist_first_vec,first_second_vec) #unused
            #print("angle 1-2: ", angle_first_second)
            
            second_third_vec = [third[i] - second[i] for i in range(1, 4)]
            angle_second_third = self.vector_angle(first_second_vec,second_third_vec)
            #print("angle 2-3: ", angle_second_third)
            
            third_tip_vec = [tip[i] - third[i] for i in range(1, 4)]
            angle_third_tip = self.vector_angle(second_third_vec, third_tip_vec) #unused
            #print("angle 3-tip: ", angle_third_tip)
            
            #this is from trial and error, in curling 2nd to 3rd was the most reliably bent
            if angle_second_third>160:
                curled_fingers +=1
            finger_num+=1
        
        # If 2 or more fingers are curled, consider it a fist (diff fingers have diff angles)
        #this is mostly just based on whether the pinky and ring finger are curled bc those are the easiest to tell
        return curled_fingers >= 2


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

        img, handed_info = self.detector.findHands(img)

        if not success or img is None:
            print("[ERROR] Failed to read image from webcam.")
            return np.zeros(7)
        
        if handed_info == []:
            print("[ERROR] Failed to label handedness for hands.")
            return np.zeros(7)
        # Moved rotation calculation to here so it draws the axes before imshow
        target_rot = R.from_rotvec(np.zeros(3))
        lmlist_left = []
        lmlist_right = []
        for idx, label in handed_info:
            lmlist = self.detector.findPositions(img, handNo=idx)
            if label == 'Left':
                lmlist_right= lmlist #display image is flipped so detection is too
            elif label == 'Right':
                lmlist_left = lmlist
        left_hand_fist = False

        fps = int(1 / (time.time() - self.pTime)) if time.time() != self.pTime else 0
        self.pTime = time.time()
        if not running_in_mjpython(): #opencv2.show breaks in mjpython because of GUI issues
            try:
                cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.imshow("Image", img)
                cv2.waitKey(1)
            except cv2.error as e:
                print(f"[OpenCV ERROR] {e}")
                return np.zeros(7)
            
        if not lmlist_left or len(lmlist_left) < 9:
            print("[WARNING] Left hand not detected")
        else:
            left_hand_fist = self.is_hand_fist(lmlist_left)

        if not lmlist_right or len(lmlist_right) < 9:
            print("[WARNING] Right hand not detected")
            return np.zeros(7)


        
        #decide whether to compute rotation or translation based on whether
            # left hand is in a fist
        

        action = np.zeros(7)
        grasp = -1

        if left_hand_fist:

            current_rot = R.from_rotvec(self.compute_rotation(lmlist_right))

            if self.base_rot is None:
                self.base_rot = current_rot
                self.left_hand_was_fist = True
                print("[INFO] Base rotation captured.")
            

            #rotation relative to base rotation set each time you make a fist
            deviation_rot = current_rot * self.base_rot.inv()
            delta_rot = deviation_rot.as_rotvec()
            
            # Smooth & limit rotation
            rot_gain = 0.5
            max_rot_mag = 0.3
            delta_rot = np.clip(rot_gain * delta_rot, -max_rot_mag, max_rot_mag)
            action[3:6] = delta_rot
            

        else: 
            self.left_hand_was_fist= False
            self.base_rot = None
            # Update previous depth values
            # Update previous depth values 
            # self.prev_depth_values.pop(0)
            # self.prev_depth_values.append(lmlist[9][1]) # testing different landmarks for depth computation
            self.prev_depth_values.pop(0)
            self.prev_depth_values.append(lmlist_right[4][1])

            # Compute depth
            depth = self.compute_depth()
            print("depth: ", depth)
            new_pos = [depth] + lmlist_right[0][2:4]
            self.pid.reset(new_pos)

            # Compute grasp logic
            tipOfPointer = np.array(lmlist_right[8][2:4]) #just consider y and z, simplified
            tipOfThumb = np.array(lmlist_right[4][2:4])
            pinchDistance = np.linalg.norm(tipOfPointer - tipOfThumb)
            if (pinchDistance <= 0.04):
                grasp = 1
        
       

        # Compute action
        
        delta = self.pid.update(current_pos=robot_eef_pos, dt=self.dt)
        action[:3] = delta
        action[-1] = grasp
        return action
