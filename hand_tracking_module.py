import math
import cv2
import mediapipe as mp
import time 

class handDetector():
    def __init__(self,static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # initialize mediapipe object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.static_image_mode, max_num_hands = self.max_num_hands, 
                                        min_detection_confidence = self.min_detection_confidence, min_tracking_confidence = self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] # thumb index middle ring pinkie

    def find_hands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks: # we can detect hand
            for handLms in self.results.multi_hand_landmarks: # handLms represent each hand landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNum = 0, draw = True):
        # self.hands.process(imgRGB) must be called before using this function
        self.lmList = []
        if self.results.multi_hand_landmarks: # we can detect hand, len() = number of detected hand(s)
            targetHand = self.results.multi_hand_landmarks[handNum]
            for idx, lm in enumerate(targetHand.landmark): # lm represent each landmark location(x, y, z) (There are 21 landmark)
                height, width, _ = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height) # convert normalize coordinate to pixel coordinate
                # print(f"cz = {lm.z}")
                self.lmList.append([cx, cy])
        return self.lmList

    def is_fingers_up(self):
        # only work when your palm is in straight position
        if len(self.lmList) == 0:
            print("Can't detect hand")
            return 
        fingers = [False] * 5
        # Thumb (only works on left thumb and you need to turn the palm to camera)
        if self.lmList[self.tipIds[0]][0] > self.lmList[self.tipIds[0] - 1][0]:
            fingers[0] = True
        
        # Other fingers
        for idx in range(1,5):
            if self.lmList[self.tipIds[idx]][1] < self.lmList[self.tipIds[idx] - 2][1]:
                fingers[idx] = True

        return fingers

    def finger_distance(self, finger1, finger2):
        first_coor = self.lmList[self.tipIds[finger1]]
        second_coor = self.lmList[self.tipIds[finger2]]
        return math.dist(first_coor, second_coor)
        
def main():
    HD = handDetector()
    #fps gen
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = HD.find_hands(img)
        lmlist = HD.find_position(img)
        if len(lmlist) >= 9:
            print(lmlist[8])

        cTime = time.time()
        fps = int(1/(cTime - pTime))
        pTime = cTime

        cv2.putText(img,str(fps),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": # เมื่อไฟล์ถูกรันจะรัน main() เอาไว้โชว์ว่า module นี้ทำอะไรได้/ใช้test module
    main()