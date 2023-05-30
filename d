def imagetotext(img):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    results = model.predict(img)
    a = results[0].boxes.boxes
    a = a.cpu().numpy()
    px = pd.DataFrame(a).astype('float')
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
    if x1 != 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop = img[y1:y2, x1:x2]
        width = int(crop.shape[1] * 4)
        height = int(crop.shape[0] * 4)
        crop = cv2.resize(crop, (width, height))
        crop = cv2.GaussianBlur(crop, ksize=(3, 3), sigmaX=0, sigmaY=0)
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_OTSU)[1]
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', '', 'O',
                   'P', 'T', 'X', 'У')
        net = cv2.dnn.readNetFromONNX("models/model_OCR.onnx")
        cnts = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts, _ = contours.sort_contours(cnts[0])
        ntext = ''
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area > 500:
                crop2 = crop[y:y + h, x:x + w]
                crop2 = cv2.cvtColor(crop2, cv2.COLOR_GRAY2RGB)
                blob = cv2.dnn.blobFromImage(cv2.resize(crop2, (64, 64)), scalefactor=1.0 / 64, size=(64, 64), mean=(128, 128, 128), swapRB=True)
                net.setInput(blob)
                detection = net.forward()
                class_mark = np.argmax(detection)
                lpc = classes[int(class_mark)]
                ntext += lpc
    return ntext
def recognize(self):
        # Изображение
        ntext = ''
        if self.recognition_method.currentIndex() == 0:
            try:
                img = cv2.imread(self.fname)
                ntext = ocr.imagetotext(img)
            except Exception:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Не удалось загрузить изображение")
                msg.setInformativeText('Файл повреждён или имеет неизвестный формат')
                msg.setWindowTitle("Предупреждение")
                msg.exec_()
        # Видеозапись
        elif self.recognition_method.currentIndex() == 1:
            cap = cv2.VideoCapture(self.fname)
            skip = 0
            if not cap.isOpened():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Не удалось загрузить видео")
                msg.setInformativeText('Файл повреждён или имеет неизвестный формат')
                msg.setWindowTitle("Предупреждение")
                msg.exec_()
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        car = determine(frame)
                        height, width, channel = frame.shape
                        bpl = 3 * width
                        img = QImage(frame.data, width, height, bpl, QImage.Format_BGR888)
                        pixmap = QPixmap.fromImage(img).scaled(480, 360)
                        self.image_w.setPixmap(pixmap)
                        skip += 40
                        time.sleep(0.5)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
                        if car:
                            ntext = ocr.imagetotext(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
        # Потоковое видео
        elif self.recognition_method.currentIndex() == 2:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Камера не обнаружена")
                msg.setInformativeText('Подключите камеру или проверьте под-ключение камеры')
                msg.setWindowTitle("Предупреждение")
                msg.exec_()
            else:
                while True:
                    ret, frame = cap.read()
                    if ret:
                   car = determine(frame)
                        height, width, channel = frame.shape
                        bpl = 3 * width
                        img = QImage(frame.data, width, height, bpl, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(img).scaled(480, 360)
                        self.image_w.setPixmap(pixmap)
                        time.sleep(3)
                    if car:
                        ntext = ocr.imagetotext(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
        # Заполнение
        if len(ntext) > 5:
            self.recognized_number.setText(ntext)
            con = sqlite3.connect('database/acess_base.db')
            cur = con.cursor()
            data = cur.execute('''SELECT * FROM data WHERE number = ?''', (ntext, )).fetchall()
            if len(data) > 0:
                data = data[0]
                if str(data[3]) in ('0', 'True'):
                    self.accessdb.setText('Пропустить')
                else:
                    self.accessdb.setText('Не пропускать')
                self.name.setText(data[1])
                self.acess_time.setText(data[-1])
            else:
                self.atdb = AtDb(ntext)
                self.atdb.show()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Номер не распознан")
            msg.setWindowTitle("Информация")
            msg.exec_()

class AtDb(QMainWindow):
    ntext = ''
    def __init__(self, ntext):
        super().__init__()
        self.initUI(ntext)

    def initUI(self, ntext):
        uic.loadUi('UIC/atdb_window.ui', self)
        if ntext:
            print('nt')
            self.number.setText(ntext)
        self.add.clicked.connect(self.additem)

    def additem(self):
        con = sqlite3.connect('database/acess_base.db')
        cur = con.cursor()
        name = self.name.text()
        number = self.number.text()
        if self.acess.currentIndex() == 0:
            acess = True
        else:
            acess = False
        acess_time = self.acess_time.text()
        try:
            cur.execute('''INSERT INTO data(name, number, acess, acess_time) VALUES(?,?,?,?)''',
                        (name, number, acess, acess_time))
            con.commit()
        except Exception as e:
            print(e)
        self.close()

class UDb(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.initUI(data)

    def initUI(self, data):
        uic.loadUi('UIC/udb_window.ui', self)
        self.data = data

        self.update.clicked.connect(self.updateitem)

    def updateitem(self):
        con = sqlite3.connect('database/acess_base.db')
        cur = con.cursor()
        name = self.name.text()
        number = self.number.text()
        if self.acess.currentIndex() == 0:
            acess = True
        else:
            acess = False
        acess_time = self.acess_time.text()
        try:
            cur.execute('''UPDATE data SET name = ?, acess = ?, acess_time = ? WHERE number = ?''',
                        (name, acess, acess_time, number))
            con.commit()
        except Exception as e:
            print(e)
        self.close()

class DfDb(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        uic.loadUi('UIC/dfdb_window.ui', self)

        self.delete_2.clicked.connect(self.deleteitem)

    def deleteitem(self):
        con = sqlite3.connect('database/acess_base.db')
        cur = con.cursor()
        try:
            cur.execute('''DELETE from data WHERE number = ?''', (self.number.text(), ))
            con.commit()
        except Exception as e:
            print(e)
        self.close()
