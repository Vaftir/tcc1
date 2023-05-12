import cv2 
import numpy as np 
import imutils
   
img_rgb = cv2.imread('mainimage.jpg') 
   
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
   
template = cv2.imread('template',0) 
   
w, h = template.shape[::-1] 
found = None
maxVal = 116
maxLoc = 118

  
for scale in np.linspace(0.2, 1.0, 20)[::-1]: 
  
    
    
    resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale)) 
    r = img_gray.shape[1] / float(resized.shape[1]) 
    if resized.shape[0] < h or resized.shape[1] < w: 
            break
    found = (maxVal, maxLoc, r) 
   
(_, maxLoc, r) = found 
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r)) 
(endX, endY) = (int((maxLoc[0] + 256) * r), int((maxLoc[1] + 400) * r)) 
cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2) 
cv2.imshow("Image", img_rgb) 
cv2.waitKey(0) 

