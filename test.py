import cv2
import imutils
import pytesseract
import numpy as np
import sys

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
original_image = cv2.imread('nomera.jpeg')





hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY )
thresh = cv2.inRange(hsv,30,100)
contours, new = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
img1 = original_image.copy()
c = cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
cv2.imshow('resultC', c) 


img1 =  thresh.copy()
cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
cv2.imshow("s", img1)

count = 12
idx = 1

# for c in contours:
#     # approximate the license plate contour
#     contour_perimeter = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * contour_perimeter, True)

#     # for i in range(20):
#     # Look for contours with 4 corners

#     if  len(approx) ==4:
#         screenCnt = approx

#         # find the coordinates of the license plate contour
#         x, y, w, h = cv2.boundingRect(c)
#         new_img = original_image [ y: y + h, x: x + w]

#         # stores the new image
#         cv2.imwrite('./'+str(idx)+"I"+str(len(approx))+'.png',new_img)
#         idx += 1
#         # break

for cnt in contours:
        rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        asd =original_image.copy()
        cv2.drawContours(asd,[box],0,(255,0,0),2) # рисуем прямоугольник
        cv2.imshow("s", asd)

# draws top 30 contours

cv2.waitKey(0)
cv2.destroyAllWindows()