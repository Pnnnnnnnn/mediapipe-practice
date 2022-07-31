import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#fps gen
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks: # we can detect hand
        for handLms in results.multi_hand_landmarks: # handLms represent each hand landmarks
            for idx, lm in enumerate(handLms.landmark): # lm represent each landmark location(x, y, z) (There are 21 landmark)
                height, width, _ = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height) # convert normalize coordinate to pixel coordinate
                print(f"landmark number = {idx} x = {cx} y ={cy}")

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = int(1/(cTime - pTime))
    pTime = cTime

    cv2.putText(img,str(fps),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()