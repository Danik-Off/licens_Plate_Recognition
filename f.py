import cv2
import numpy as np

if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow( "result" ) # создаем главное окно
cv2.namedWindow( "settings" ) # создаем окно настроек

img = cv2.imread('nomera1.jpeg')
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('min', 'settings', 30, 255, nothing)
cv2.createTrackbar('max', 'settings', 100, 255, nothing)
cv2.createTrackbar('sort', 'settings', 10, 50, nothing)


crange = [0,0,0, 0,0,0]

while True:
   
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
 
    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('min', 'settings')
    s1 = cv2.getTrackbarPos('max', 'settings')
    hs1 = cv2.getTrackbarPos('sort', 'settings')
    # формируем начальный и конечный цвет фильтра
    

    # накладываем фильтр на кадр в модели HSV
    thresh = cv2.inRange(hsv, h1, s1)
    contours, new = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:hs1]
    img1 =  img.copy()
    c = cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
    cv2.imshow('resultC', c) 
    cv2.imshow('result', thresh) 
 
    
    for cnt in contours:
        rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        asd =img.copy()
        cv2.drawContours(asd,[box],0,(255,0,0),2) # рисуем прямоугольник
        cv2.imshow("s", asd)


    idx = 0
    for c in contours:
    # approximate the license plate contour
        contour_perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * contour_perimeter, True)

    # for i in range(20):
    # Look for contours with 4 corners

        if  len(approx) ==4:
         screenCnt = approx

        # find the coordinates of the license plate contour
         x, y, w, h = cv2.boundingRect(c)
         new_img = img[ y: y + h, x: x + w]

        # stores the new image
         cv2.imshow("sd", new_img)
        #  cv2.imwrite('./'+str(idx)+"I"+str(len(approx))+'.png',new_img)
         idx += 1
        # break

        ch = cv2.waitKey(5)
        if ch == 27:
            break

   


cv2.destroyAllWindows()