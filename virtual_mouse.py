import cv2
import numpy as np
from hand_tracking_module import handDetector
import autopy
import time

# fps
pTime = 0
# opencv window size
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction (Padding control area) to be able to click the bottom part of screen
# user screen size
wScreen, hScreen = autopy.screen.size()
# decrease sensitiveness of virsual mouse by using smoothening value
smoothening = 5
prevLoc = [0, 0]
currentLoc = [0, 0]

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

HD = handDetector(max_num_hands = 1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = HD.find_hands(img)
    lmList = HD.find_position(img)

    # draw usable area
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255,0,255),2)

    if len(lmList) != 0:
        fingers_up = HD.is_fingers_up()
        x1, y1 = lmList[HD.tipIds[1]] # index finger coordinate
        x2, y2 = lmList[HD.tipIds[2]] # middle finger coordinate
        # convert corrdinate to screen scale
        screenX1, screenY1 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen)), np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

        # smoothening mouse moving
        dx = screenX1 - prevLoc[0]
        dy = screenY1 - prevLoc[1]

        currentLoc[0] = prevLoc[0] + dx / smoothening
        currentLoc[1] = prevLoc[1] + dy / smoothening

        prevLoc = currentLoc

        # index finger is up and middle finger is down
        if fingers_up[1] and not fingers_up[2]:
            # mouse moving mode
            cv2.circle(img,(x1,y1),20,(255,0,0),cv2.FILLED) # draw cursor
            autopy.mouse.move(currentLoc[0],currentLoc[1])
        # both index finger and middle finger are up
        elif fingers_up[1] and fingers_up[2]:
            print(f"distance = {HD.finger_distance(1, 2)}")
            if HD.finger_distance(1, 2) <= 20:
                # clicking mode
                cv2.circle(img,(x1,y1),20,(0,255,0),cv2.FILLED) # change cursor color
                autopy.mouse.click()

    # Frame Rate
    cTime = time.time()
    fps = int(1/(cTime - pTime))
    pTime = cTime

    cv2.putText(img,str(fps),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),2)
    # Display
    cv2.imshow("Result",img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()