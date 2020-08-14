import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils

# imagePaths = list(imutils.paths.list_images('classification'))

img = cv2.imread("/home/victor/Documentos/brazilian-coin-detector/10_1477186332.jpg")
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

white = False

save_img = img.copy()

gray = cv2.cvtColor( shifted, cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
_,thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_OTSU)

gray = thresh

gray = cv2.GaussianBlur(gray, (9, 9),0)

avg_color_per_row = np.average(gray, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print(avg_color)

if(avg_color >= 100):
    gray = cv2.bitwise_not(gray)
    white = True

g = gray.copy()

res = cv2.bitwise_and(img, img,mask = gray)
gray = cv2.cvtColor( res, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)

g2 = gray.copy()

_,thresh = cv2.threshold(gray ,1,255,cv2.THRESH_BINARY)

# thresh = cv2.GaussianBlur(thresh, (9, 9),0)

thr = thresh.copy()

#Waterhed

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)

########################

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

    aux = save_img

    #adicionar limite caso a moeda esteja no canto.
    # aux = cv2.bitwise_and(save_img, save_img,mask = mask)
    
    # if(white):
    #     aux[mask == 0] = [255, 255, 255]
    # else:
    #     aux[mask == 0] = [0, 0, 0]

    limInfY = (y - r) 

    if  limInfY < 0:
        limInfY = 0
    
    limInfX = (x - r) 
    
    if limInfX < 0: 
        limInfX = 0

    limSupY = y + r
    
    if (limSupY > int(img.shape[0])):
        limSupY = int(img.shape[0])
    
    limSupX =  x + r
    
    if(limSupX > int(img.shape[1])):
        limSupX =  int(img.shape[1] )

    crop_img = aux[limInfY:limSupY, limInfX:limSupX]
    # print("entu")
    # print(label)
    cv2.imwrite("newDataset/" + str(label) + ".jpg", crop_img)

    cv2.circle(img, (x, y), r, (255, 0, 0), 3)
    cv2.putText(img, u'\u0024' +  "{}".format(label), (x - 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


cv2.imshow("img", img)
# cv2.imshow("thr", g2)
# cv2.imshow("AND", mask)
# cv2.imshow("img", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()