import numpy as np
import cv2

img = cv2.imread('laptop.jpg',0)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()


im_bw = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]

im_bw = cv2.bitwise_not(im_bw)

_,contours,_ = cv2.findContours(im_bw, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )

cv2.imshow('binary',im_bw)
cv2.waitKey()
cv2.destroyAllWindows()
print(np.array(contours).shape)
#print(hierarchy)

max = contours[0]
for i in contours:
	if cv2.contourArea(i) >= cv2.contourArea(max):
		max = i

max = sorted(contours,key=lambda x: cv2.contourArea(x))

raw = np.ones((img.shape))

cv2.drawContours(raw,[max],0,(0,255,0),5)
cv2.imshow('img',raw)
cv2.waitKey()
cv2.destroyAllWindows()

print(contours.index(max))
