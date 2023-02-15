import cv2
import math
import time

# loading video
vid = cv2.VideoCapture("bb3.mp4")

# loading tracker
tracker = cv2.TrackerCSRT_create()

# reading the first frame of the video
ret,img = vid.read()

# selecting the bounding box on the image
bbox = cv2.selectROI("Tracker Image",img,False)
 
# initializing the tracker on img and bbox
tracker.init(img,bbox)
print(bbox)

def drawbox(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3,1)
    cv2.putText(img,"Tracking...",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,150),2)

def goal_track(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    
while(True):
    ret,image = vid.read()

    success,bbox = tracker.update(image)
    
    if (success):
        drawbox(image,bbox)
    else:
        cv2.putText(image,"Lost",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,150),2)

    cv2.imshow("Result",image)
     
    if (cv2.waitKey(25) == 32):
        break

vid.release()
cv2.destroyAllWindows()
