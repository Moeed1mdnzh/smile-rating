import cv2,pickle
import numpy as np

cam = cv2.VideoCapture(1)
cascade = cv2.CascadeClassifier("face.xml")
model = pickle.load(open("smile_svm.sav", 'rb'))
det = False
rate = 0

while True:
    frame = cam.read()[1]
    frame = cv2.flip(frame,1)
    #frame = cv2.resize(frame,(320,240))
    clone = frame.copy()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =  cascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cropped = gray[y:y+h,x:x+w]
        cropped = cv2.resize(cropped,(64,64),interpolation=cv2.INTER_AREA)
        cropped = cropped.reshape(-1,64*64)
        cropped = cropped / 255.0
        pred = model.predict_proba(cropped)
        if pred[0][0] > pred[0][1]:
            rate = int(100-(pred[0][0]*100))
        else:
            rate = int(pred[0][1]*100)
        for i in range(0,91,10):
            if rate > i and rate < i+10:
                vector1,vector2 = np.sqrt((rate-i)**2),np.sqrt((rate-i+10)**2)
                if vector1 > vector2: rate = i
                elif vector1 < vector2: rate = i+10
                else: rate = i + 5
        pts = np.array([[9+(x+w//2)-15,y],[9+(x+w//2),y+15],[9+(x+w//2)+15,y]])
        cv2.polylines(frame,[pts],True,(100,100,255) if rate >= 100 else (230,230,255),3)
        cv2.fillPoly(frame,[pts],(20,20,255) if rate == 100 else (210,210,210))
        cv2.putText(frame,str(rate),((x+w//2)-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(180,180,255),2)
    cv2.imshow("WINDOW",frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
cam.release()

