import cv2
import numpy as np
import imutils

def detect_license_plate(image):
    
 
    image = imutils.resize(image, width=800 )
    # Загрузка каскадного классификатора для обнаружения номеров автомобилей
    cascade_classifier = cv2.CascadeClassifier('hrpn.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Обнаружение номеров автомобилей
    car_plates = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    color_yellow = (0,0,0)


    # Перебор найденных номеров автомобилей и улучшение распознавания
    for (x, y, w, h) in car_plates:
        # Обрезка области номера для дальнейшей обработки
        plate_image = image[y:y + h, x:x + w]
        return plate_image;          

