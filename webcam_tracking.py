import cv2
import mediapipe as mp

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
        # For [handNo] hand, find a specific position
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # Convert from ___ space to ___ space ? 
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                adjusted_z = int((lm.z * -1000)) # arbitrary adjustment, can change
                adjusted_z = min(300, adjusted_z)
                adjusted_z = max(adjusted_z, 0)
                lmlist.append([id, cx, cy, adjusted_z])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist
    

# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = handDetector()

#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmlist = detector.findPositions(img)
#         if len(lmlist) != 0:
#             print(lmlist[4])
        
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

#         cv2.imshow("Image", img)
#         cv2.waitKey(1)

# if __name__ == "__main__":
#     main()