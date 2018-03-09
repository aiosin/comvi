import numpy as np
import cv2

img = cv2.imread('laptop.jpg',0)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()


im_bw = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]

#invert image to get useful contours
im_bw = cv2.bitwise_not(im_bw)

_,contours,_ = cv2.findContours(im_bw, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )

cv2.imshow('binary',im_bw)
cv2.waitKey()
cv2.destroyAllWindows()
print(np.array(contours).shape)


max = contours[0]
for i in contours:
	if cv2.contourArea(i) >= cv2.contourArea(max):
		max = i

max = sorted(contours,key=lambda x: cv2.contourArea(x))[-1]
print(cv2.contourArea(max))
print(max)
print(max[:,0,:])
raw = np.ones((img.shape))

cv2.drawContours(raw,[max],0,(0,1,0),1)
cv2.imshow('img',raw)
cv2.waitKey()
cv2.destroyAllWindows()


def findDescriptor(img):
    contour = []
    _, contour, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result



print(findDescriptor(img))