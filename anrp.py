import cv2
import pytesseract
import numpy as np
from transliterate import translit
import imutils


if __name__ == '__main__':
    def nothing(*arg):
        pass
# Загрузка изображения
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread('nomera1.jpeg')

cv2.namedWindow( "settings" ) # создаем окно настроек

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('createCLAHE', 'settings', 1, 50, nothing)
cv2.createTrackbar('fastNlMeansDenoising', 'settings', 32, 150, nothing)
cv2.createTrackbar('min', 'settings', 62, 255, nothing)
cv2.createTrackbar('max', 'settings', 83, 255, nothing)
cv2.createTrackbar('RageMin', 'settings', 94, 255, nothing)
cv2.createTrackbar('RageMax', 'settings', 161, 255, nothing)
# Предобработка изображения
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Загрузка каскадного классификатора для обнаружения номеров автомобилей
cascade_classifier = cv2.CascadeClassifier('hrpn.xml')

# Обнаружение номеров автомобилей
car_plates = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
color_yellow = (0,0,0)
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
      
    
        # cv2.imwrite('./nnn.png',plate_image)

        
        
        h1 = cv2.getTrackbarPos('min', 'settings')
        s1 = cv2.getTrackbarPos('max', 'settings')
        Rageh1 = cv2.getTrackbarPos('RageMin', 'settings')
        Rages1 = cv2.getTrackbarPos('RageMax', 'settings')
        den = cv2.getTrackbarPos('createCLAHE', 'settings')
        den1 = cv2.getTrackbarPos('fastNlMeansDenoising', 'settings')
    
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
       
        gray = imutils.resize(gray, width=1000 )
        gray = cv2.fastNlMeansDenoising(gray, h=den1)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.inRange(gray,Rageh1,Rages1)
        gray = cv2.equalizeHist(gray)
       

        clahe = cv2.createCLAHE(clipLimit=den)
        gray = clahe.apply(gray)

# Применение бинаризации
        _,  gray = cv2.threshold( gray, h1,s1, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)
  
        config = r'--oem 1  -c tessedit_char_whitelist=0123456789ABEKMHOPCTyX  --psm 9 -l eng'  # Настройки OCRconfig = '--oem 1 --psm 6 -l eng'
        plate_text = pytesseract.image_to_string(   gray, config=config)
       
        print("Распознанный номер: ", plate_text)
        image1 = image.copy()
        cv2.putText(image1, plate_text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
        cv2.imshow('result', image1)
        cv2.imshow('Car Plate Image',  gray)
        ch = cv2.waitKey(5)
        if ch == 27:
            break
  

cv2.destroyAllWindows()
