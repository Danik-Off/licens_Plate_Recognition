import cv2
import pytesseract
import numpy as np
import os
import imutils


if __name__ == '__main__':
    def nothing(*arg):
        pass
# Загрузка изображения
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

directory = 'photos_test'
files = os.listdir(directory)
print(files)

for file in files:

    print(file)
    image = cv2.imread('photos_test/'+file)
    image = imutils.resize(image, width=800 )
    # Предобработка изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Загрузка каскадного классификатора для обнаружения номеров автомобилей
    cascade_classifier = cv2.CascadeClassifier('hrpn.xml')

    # Обнаружение номеров автомобилей
    car_plates = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    color_yellow = (0,0,0)

    # Перебор найденных номеров автомобилей и улучшение распознавания
    for (x, y, w, h) in car_plates:
        # Обрезка области номера для дальнейшей обработки
        plate_image = image[y:y + h, x:x + w]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        blank_image = np.full((600, 1500, 3), (255, 255, 255) , dtype=np.uint8)
        

        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray,(3,3),1)
        grayP = imutils.resize(gray, width=1500 )
        gray = cv2.inRange(grayP,100,255)
        # gray= cv2.equalizeHist(grayP)
        gray = cv2.fastNlMeansDenoising(gray, h=15)
        _,gray =  cv2.threshold( gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        # gray = cv2.GaussianBlur(gray , (11, 11), 20)


        contours, new = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 500  # Минимальная площадь контура, которую вы хотите сохранить
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        max_contour_area = 28000
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) <max_contour_area]
        contours = [cnt for cnt in contours if  len(cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)) !=4]
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
        img1 = gray.copy()
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        

        mask = np.zeros_like(gray)


        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(blank_image, contours, -1, (0,0,0), thickness=cv2.FILLED)
        cv2.drawContours(blank_image, contours, -1, (0,0,0), thickness=10)
    # Вырежьте объекты изображения с помощью маски
        gray = cv2.bitwise_and(grayP, grayP, mask=mask)

        kernel = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)

        
        gray =    cv2.GaussianBlur(blank_image, (5, 5), 0)

    # Примените размытие обратно к оригинальному изображению
        # blurred_image = cv2.addWeighted(grayP, 1, blurred_edges, 0.5, 0)

        config = r'--oem 1  -c   tessedit_char_blacklist=``\'b --psm 9 -l rus'  # Настройки OCRconfig = '--oem 1 --psm 6 -l eng'
        plate_text = pytesseract.image_to_string(  gray, config=config)
        
        

    #
        print("Распознанный номер: ", plate_text)
        image1 = image.copy()
        cv2.putText(image1, plate_text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
        cv2.imshow('resud',  blank_image)
        cv2.imshow('Car Plate Image',  gray)

        cv2.waitKey(0)

        ch = cv2.waitKey(5)
        if ch == 27:
            break

cv2.waitKey(0)

cv2.destroyAllWindows()
