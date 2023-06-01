import cv2
import pytesseract
import os
import imutils
import numpy as np

import f

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

if __name__ == '__main__':
    def nothing(*arg):
        pass


cv2.namedWindow( "settings" ) # создаем окно настроек
cv2.namedWindow( "main" ) # создаем окно настроек

# создаем  бегунков для настройки WW
cv2.createTrackbar('fastNlMeansDenoising', 'settings', 0, 255, nothing)


cv2.createTrackbar('alpha', 'settings', 0, 255, nothing)
cv2.createTrackbar('beta', 'settings', 0, 255, nothing)

blank_image = np.full((600, 1500, 3), (255, 255, 255) , dtype=np.uint8)

directory = 'photos_test'
files = os.listdir(directory)

for file in files:

    print(file)
    image = cv2.imread('photos_test/'+file)
    plate=f.detect_license_plate(image)
    
    plateResize = imutils.resize(plate, width=500 )
    gray = cv2.cvtColor(plateResize, cv2.COLOR_BGR2GRAY)

    gray = cv2.fastNlMeansDenoising(gray, h=10)

 
       
    fastNlMeansDenoising = cv2.getTrackbarPos('fastNlMeansDenoising', 'settings')

    _alpha = cv2.getTrackbarPos('alpha', 'settings')
    _beta = cv2.getTrackbarPos('beta', 'settings')


    # Фильтры для повышения качества
    sharpness = cv2.addWeighted(gray, 0, gray, -0.6, 143)
    sharpness = cv2.GaussianBlur( sharpness, (0, 0), 3)
    
    kernel = np.array([[-1, -1, -1],
                        [-1, 10, -1],
                        [-1, -1, -1]])
    sharpness= cv2.filter2D(sharpness, -1, kernel)

    _,tresh =  cv2.threshold(sharpness, 127,247, cv2.THRESH_BINARY)
    
    
    config = r'--oem 1  -c tessedit_char_whitelist=0123456789ABEKMHOPCTYX --psm 9 -l eng'  # Настройки OCRconfig = '--oem 1 --psm 6 -l eng'
    plate_text = pytesseract.image_to_string(  gray, config=config)
    
    print(plate_text)


    cv2.imshow('sharpness',  tresh)
    cv2.imshow('settings', blank_image)


    # ch = cv2.waitKey(5)
    # if ch == 27:
    #     break

    cv2.waitKey(0)
cv2.destroyAllWindows()