import cv2
import numpy as np
import pytesseract

def detect_license_plate(image_path):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Предварительная обработка изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Поиск номерного знака
    number_plate = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:  # Предполагаем, что номерной знак имеет 4 угла
            number_plate = approx
            break

    if number_plate is None:
        return None

    # Выделение номерного знака
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [number_plate], 0, 255, -1)
    masked_image = cv2.bitwise_and(gray, gray, mask=mask)

    # Использование Tesseract для распознавания текста на номерном знаке
    text = pytesseract.image_to_string(masked_image)

    return text

# Пример использования
image_path = "007-AN.jpg"
license_plate_text = detect_license_plate(image_path)
print("Распознанный номерной знак:", license_plate_text)