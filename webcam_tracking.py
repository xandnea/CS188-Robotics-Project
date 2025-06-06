import cv2
import mediapipe as mp
import numpy as np
import time



class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        """
        Initialize a hand detector.

        Args:
            mode (boolean): _____________________________________.
            maxHands (integer): Max number of hands being tracked.
            detectionCon (float): Confidence in detection.
            trackCon (float): Confidence in tracking.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

       

    def findHands(self, img, draw = True):
        """
        Find hand and draw outline onto image.

        Args:
            img (array): Image to be converted and overlayed.
            draw (boolean): Draw the overlay or not.
        """
        # Convert from BGR to RGB and process 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If landmarks are found, draw landmarks on webcam
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    # Returns a 2D list of positions where each entry is [id, cx, cy, z]
    def findPositions(self, img, handNo = 0, draw = True):
        """
        Convert positions and overlay graphics.

        Args:
            img (array): Image to be converted and overlayed.
            handNo (integer): Which hand to find the positions of.
            draw (boolean): Draw the overlay or not.
        """
        lmlist = [] 
        coord_sys_adjust = 2500 #can trial and error this, converting units essentially
        
        # For [handNo] hand, find a specific position
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            palm_normal = [] # coming from wrist 
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                #in robosuite x is depth, the -50 and /50 are for scaling to robosuite
                adjusted_x = int((lm.z * -1000)) -50
                adjusted_x/=50

                # in robosuite, y: left/right (right = +y), center at 0
                rel_y = (w/2 - cx) / coord_sys_adjust

                # in robosuite, z: up/down (up = +z), center at 1
                rel_z = 1+( h / 2- cy) / coord_sys_adjust
                
                lmlist.append([id, adjusted_x, rel_y, rel_z, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            # Use normalized coordinates to compute normal and synthetic point
            wrist = np.array([myHand.landmark[0].x, myHand.landmark[0].y, myHand.landmark[0].z])
            thumb = np.array([myHand.landmark[1].x, myHand.landmark[1].y, myHand.landmark[1].z])
            pinky = np.array([myHand.landmark[17].x, myHand.landmark[17].y, myHand.landmark[17].z])

            v1 = thumb - wrist
            v2 = pinky - wrist
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)

            offset = 0.06  # distance from wrist
            palm_point = wrist + offset * normal

            # Project synthetic landmark 21 to robosuite-style and pixel space
            cx = int(palm_point[0] * w)
            cy = int(palm_point[1] * h)
            adjusted_x = (palm_point[2] * -1000 - 50) / 50
            rel_y = (w / 2 - cx) / coord_sys_adjust
            rel_z = 1 + (h / 2 - cy) / coord_sys_adjust

            lmlist.append([21, adjusted_x, rel_y, rel_z, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (int(wrist[0] * w), int(wrist[1] * h)), (cx, cy), (255, 255, 255), 2)

        return lmlist