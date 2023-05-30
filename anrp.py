import cv2
import pytesseract
# Загрузка изображения
image = cv2.imread('scale_1200.jpeg')

# Предобработка изображения
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Загрузка каскадного классификатора для обнаружения номеров автомобилей
cascade_classifier = cv2.CascadeClassifier('hrpn.xml')

# Обнаружение номеров автомобилей
car_plates = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
    cv2.imshow('Car Image Plate', plate_image)
    cv2.imwrite('./nnn.png',plate_image)
  
    # 
# Отображение исходного изображения с обнаруженными номерами автомобилей


cv2.imshow('Car Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()