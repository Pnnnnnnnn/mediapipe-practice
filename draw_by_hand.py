import cv2
import time
import numpy as np
from hand_tracking_module import handDetector

# frame size
width, height = 1280, 720

# drawing setting
brush_color = (0, 255, 255)
brush_thickness = 35
eraser = (0, 0, 0)
eraser_thickness = 100

# fps gen
pTime = 0
cTime = 0

prev_x, prev_y = -1, -1

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

canvas = np.zeros((height, width, 3), dtype = "uint8")

HD = handDetector(max_num_hands = 1)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = HD.find_hands(img)
    LmList = HD.find_position(img)

    if len(LmList) != 0:
        fingers_up = HD.is_fingers_up()
        middle_x, middle_y = LmList[HD.tipIds[2]] # middle finger coordinate
        index_x, index_y = LmList[HD.tipIds[1]] # index finger coordinate
        distance = HD.finger_distance(1, 2)
        print(f"distance = {distance}")
    
        if not fingers_up[3] and fingers_up[1] and fingers_up[2]:
            cur_x, cur_y = int((middle_x+index_x)/2), int((middle_y+index_y)/2) 
            if prev_x == prev_y == -1:
                # prevent drawing long line when detect hand again after lose detecting
                prev_x, prev_y = cur_x, cur_y
            # draw a pen tip
            cv2.line(img,(prev_x, prev_y), (cur_x, cur_y), (255, 0, 255), brush_thickness)
            if distance <= 25:
                # pen mode   
                # draw on canvas
                cv2.line(canvas,(prev_x, prev_y), (cur_x, cur_y), brush_color, brush_thickness)
                # update prev value
                prev_x, prev_y = cur_x, cur_y
            else:
                # just moving pen tip
                prev_x, prev_y = -1, -1 
        
        elif fingers_up[1] and fingers_up[2] and fingers_up[3]:
            # eraser mode
            ring_x, ring_y = LmList[HD.tipIds[3]] # ring finger coordinate
            cur_x, cur_y = int((middle_x + ring_x) / 2), int((middle_y + ring_y) / 2) 
            if prev_x == prev_y == -1:
                # prevent drawing long line when detect hand again after lose detecting
                prev_x, prev_y = cur_x, cur_y
            # draw a pen tip
            cv2.line(img,(prev_x, prev_y), (cur_x, cur_y), (255, 255, 255), eraser_thickness)
            # draw on canvas
            cv2.line(canvas,(prev_x, prev_y), (cur_x, cur_y), eraser, eraser_thickness)
            # update prev value
            prev_x, prev_y = cur_x, cur_y

    else:
        # can't detect hand
        prev_x, prev_y = -1, -1 

    # overlay the canvas to img
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # convert to gray before sent to threshold because we want to inverse the draw line to be black
    _, imgInverse = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV) # line is black, background is white
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR) # convert back to BGR to do element wise operation with img
    img = cv2.bitwise_and(img, imgInverse) # put the black draw line on the img
    img = cv2.bitwise_or(img, canvas) # add line color to img

    cTime = time.time()
    fps = int(1/(cTime - pTime))
    pTime = cTime

    cv2.putText(img,str(fps),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('c'):
        canvas = np.zeros((720,1280,3), dtype = "uint8")

cap.release()
cv2.destroyAllWindows()
