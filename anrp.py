import cv2
import pytesseract
import numpy as np
from transliterate import translit



if __name__ == '__main__':
    def nothing(*arg):
        pass
# Загрузка изображения
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread('nomera1.jpeg')

cv2.namedWindow( "settings" ) # создаем окно настроек

img = cv2.imread('nomera1.jpeg')
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('min', 'settings', 30, 255, nothing)
cv2.createTrackbar('max', 'settings', 100, 255, nothing)

# Предобработка изображения
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Загрузка каскадного классификатора для обнаружения номеров автомобилей
cascade_classifier = cv2.CascadeClassifier('hrpn.xml')

# Обнаружение номеров автомобилей
car_plates = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
color_yellow = (0,255,255)
while  True:
# Перебор найденных номеров автомобилей и улучшение распознавания
    for (x, y, w, h) in car_plates:
        # Обрезка области номера для дальнейшей обработки
        plate_image = image[y:y + h, x:x + w]

        # Дополнительные действия с областью номера:
        # Например, применение фильтров, улучшение контраста, применение OCR и т. д.
        # Реализуйте соответствующий код здесь

        # Отрисовка рамки вокруг номера на исходном изображении
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, 'Car Plate', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
        # cv2.imwrite('./nnn.png',plate_image)

        config = r'--oem 3 --psm 6'
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        h1 = cv2.getTrackbarPos('min', 'settings')
        s1 = cv2.getTrackbarPos('max', 'settings')
        # gray =  thresh = cv2.inRange(gray, h1, s1)
  
        config = '-c tessedit_char_whitelist=0123456789 --psm 9'  # Настройки OCR
        plate_text = pytesseract.image_to_string(gray,lang='rus', config=config)
       
        print("Распознанный номер: ", plate_text)
        image1 = image.copy()
        cv2.putText(image1, translit(plate_text, language_code='ru', reversed=True), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
        cv2.imshow('result', image1)
        cv2.imshow('Car Plate Image',  gray)
        ch = cv2.waitKey(5)
        if ch == 27:
            break
  

cv2.destroyAllWindows()
