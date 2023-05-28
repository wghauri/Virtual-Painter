import cv2
import numpy as np
import os
import Hand_Detector as hd
import math

brush_thickness = 15
eraser_thickness = 50

folder_path = "Header"
mylist = os.listdir(folder_path)
overlay_list = []

for img_path in mylist:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(image)

header = overlay_list[1]

brush_control = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Brush.jpg')
eraser_control = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Eraser.jpg')
clear = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Clear.jpg')
brush_size_increase = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Brush_plus.jpg')
brush_size_decrease = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Brush_minus.jpg')
eraser_size_increase = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Eraser_plus.jpg')
eraser_size_decrease = cv2.imread('/Users/waqwaq/Python Projects/Virtual Painter Model/Control/Eraser_minus.jpg')

draw_color = (255,0,255)

detector = hd.detect_hands(detectionCon=0.85)

img_canvas = np.zeros((800,1450,3), np.uint8) # Have 0 - 255 values

xp,yp = 0,0
width = 1450
height = 800
dim = (width, height)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    img_canvas = cv2.resize(img_canvas, dim, interpolation=cv2.INTER_AREA)


    # Find Hand Landmarks
    frame = detector.findHands(frame,draw=False)
    lmlist = detector.findPosition(frame, draw=False)

    # Tip of fingers
    if len(lmlist) != 0:
        x1,y1 = lmlist[8][1:] # index
        x2,y2 = lmlist[12][1:] # middle
        x3,y3 = lmlist[4][1:] # thumb
        x4,y4 = lmlist[6][1:] # end of index

        # Check which fingers are up
        fingers = detector.fingers_up()

        # If selection mode (Two fingers are up)
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            # Checking for click
            if y1 < 135:
                if 0<x1<142:
                    header = overlay_list[9]
                    draw_color = (255,0,255)
                elif 142<x1<302:
                    header = overlay_list[7]
                    draw_color = (255,0,127)
                elif 302<x1<462:
                    header = overlay_list[8]
                    draw_color = (255,0,0)
                elif 462<x1<622:
                    header = overlay_list[3]
                    draw_color = (0,255,0)
                elif 622<x1<782:
                    header = overlay_list[4]
                    draw_color = (0,0,255)
                elif 782<x1<942:
                    header = overlay_list[6]
                    draw_color = (0,128,255)
                elif 942<x1<1102:
                    header = overlay_list[5]
                    draw_color = (0,255,255)
                elif 1102<x1<1262:
                    header = overlay_list[1]
                    draw_color = (255,255,255)
                elif 1262<x1<1422:
                    header = overlay_list[2]
                    draw_color = (0,0,0)
            cv2.rectangle(frame, (x1,y1-25), (x2,y2+25), draw_color, cv2.FILLED)
            
        # Distance between thumb and index finger
        length = math.hypot(x4 - x3, y4 - y3)
        
        # Checking for click
        if 135<y2<216:
            if length < 45:    
                if 541<x2<608:
                    brush_thickness += 5
                    brush_control = brush_size_increase
                    cv2.putText(frame,f'Brush Size: {brush_thickness}', (475,245),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                elif 608<x2<675:
                    brush_thickness -= 5
                    brush_control = brush_size_decrease
                    cv2.putText(frame,f'Brush Size: {brush_thickness}', (475,245),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                elif 841<x2<908:
                    eraser_thickness += 5
                    eraser_control = eraser_size_increase
                    cv2.putText(frame,f'Eraser Size: {eraser_thickness}', (775,245),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                elif 908<x2<975:
                    eraser_thickness -= 5
                    eraser_control = eraser_size_decrease
                    cv2.putText(frame,f'Eraser Size: {eraser_thickness}', (775,245),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

        if brush_thickness == 0:
            brush_thickness += 5

        if eraser_thickness == 0:
            eraser_thickness += 5
        
        # Checking for click
        if 135<y2<198:
            if length < 45:
                if 1263<x2<1403:
                    img_canvas = np.zeros((800,1450,3), np.uint8)

        # If drawing mode (Index finger is up)
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1,y1), 15, draw_color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp,yp = x1,y1

            if draw_color == (0,0,0):
                cv2.line(frame, (xp,yp), (x1,y1), draw_color,eraser_thickness)
                cv2.line(img_canvas, (xp,yp), (x1,y1), draw_color,eraser_thickness) 
            else:
                cv2.line(frame, (xp,yp), (x1,y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp,yp), (x1,y1), draw_color, brush_thickness)

            xp,yp = x1,y1

    img_gray = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    imgInv = cv2.resize(imgInv, dim, interpolation=cv2.INTER_AREA)
    frame = cv2.bitwise_and(frame,imgInv)
    frame = cv2.bitwise_or(frame,img_canvas)

    # Images
    frame[0:135,0:1450] = header
    frame[135:215,475:675] = brush_control
    frame[135:215,775:975] = eraser_control
    frame[140:203,1263:1403] = clear   
    cv2.imshow("Frame",frame)
    # cv2.imshow("Canvas",img_canvas)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)
