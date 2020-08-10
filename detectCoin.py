import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils

img = cv2.imread("/home/victor/Documentos/brazilian-coin-detector/photo_2020-08-10_12-17-58.jpg")
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

save_img = img.copy()

gray = cv2.cvtColor( shifted, cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
_,thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_OTSU)

gray = thresh

gray = cv2.GaussianBlur(gray, (9, 9),0)

avg_color_per_row = np.average(gray, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print(avg_color)

if(avg_color >= 125):
    gray = cv2.bitwise_not(gray)

res = cv2.bitwise_and(img,img,mask = gray)
gray = cv2.cvtColor( res, cv2.COLOR_BGR2GRAY)

_,thresh = cv2.threshold(gray ,1,255,cv2.THRESH_BINARY)


thr = thresh.copy()

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)


for label in np.unique(labels)[1:]:
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(c)

    x = int(x)
    y = int(y)
    r = int(r)

    r+=5

    crop_img = save_img[(y -r):(y+ r), (x -r):(x+r)]

    cv2.imwrite(str(label) + ".jpg", crop_img)

    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    cv2.putText(img, "#{}".format(label), (x - 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


cv2.imshow("teste", img)
cv2.imshow("thr", crop_img)
cv2.imshow("AND",res)
cv2.waitKey(0)
cv2.destroyAllWindows()