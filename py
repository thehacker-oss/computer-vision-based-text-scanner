import cv2
import numpy as np
import pytesseract as pyt 
from tkinter.filedialog import askopenfilename

filename = askopenfilename()
image = cv2.imread(filename)

image_work = np.copy(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.Canny(smooth,100,300,)

image_contour , contours,heirarcy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

if(len(contours)!=0):
    area = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    
epsilon = 0.05*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
cv2.drawContours(image_work,[approx],-1,(255,0,0),3)

cv2.imshow("image",image_work)
cv2.waitKey(1000)

src =np.float332([approx[0,0],approx[1,0],approx[2,0],approx[3,0]])
dst= np.float32([[340,390],[340,0],[0,0],[0,390]])
M = cv2.getPerspectiveTransform(src,dst)
warped= cv2.warpPerspective(image,M,(340,390))
cv2.imshow("warped",warped)
cv2.waitKey(1000)

coords = np.column_stack(np.where(thresh>0))
rect = cv2.minAreaRect(coords)
angle = cv2.minAreaRect(coords)[-1]

if angle <-45:
    angle = -(90+angle)+90
else
    angle = -angle + 90
    
(h,w)=warped,shape[:2]
center = (w//2,h//2)

M= cv2.getRotationMatrix2D(center,angle,1.0)
warped = cv2.warpAffine(warped,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

gray_warped =cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ret,thresh_warped= cv2.threshold(gray_warped,190,255,cv2.THRESH_BINARY_INV)
cv2.imshow("thres",thresh_warped)
cv2.waitKey(1000)

cv2.destroyALLWindows()

text= pyt.image_to_string(thresh_warped)
print(text)

