import cv2
from cvzone.HandTrackingModule import HandDetector

hand_detect = HandDetector(detectionCon=0.5,maxHands=2)
camera = cv2.VideoCapture(0)
fire_detect = cv2.CascadeClassifier('fire_detection.xml')
while True:
    ret,frame = camera.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fire = fire_detect.detectMultiScale(frame,1.2,5)
    for(x,y,w,h) in fire:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        print("Fire Detected!")
    cv2.imshow("camera",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
