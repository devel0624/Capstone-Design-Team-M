# Qt designer에 기본으로 import 목록
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

# 카메라 캡쳐를 위한 추가 import 목록
from PyQt5.QtGui import QPixmap, QImage
import cv2

import threading

# 모듈 import
from Gesture_Recog import Gesture_Cap


# tray 구현을 위한 모듈
import pystray
from PIL import Image
from pystray import MenuItem as item


def quit_window(icon, item):
       icon.stop()
       myWindow.destroy()

def show_window(icon, item):
       myWindow.show()
       icon.hide()

def hide_window():
       myWindow.hide()
       icon.run()
       
#UI파일 연결
#UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("untitled.ui")[0]

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
    def closeEvent(self, event):
        quit_msg = "Want to exit?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)
        

        if reply == QMessageBox.Yes:
            hide_window()
            event.accept()
        else:
            event.ignore()

def video_play():
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            cap.release() # 작업 완료 후 해제
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        h,w,c = img.shape
        qImg = QImage(img.data, w, h, w*c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        Cam_Label.setPixmap(pixmap)

if __name__ == "__main__" :

    seq = []
    action_seq = []
    last_action = None

    image=Image.open("image.jpg")
    menu=(item('열기', show_window),item('종료', quit_window) )
    icon=pystray.Icon("Gesture program", image, "Gesture program", menu)

    cap = cv2.VideoCapture(0)

    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 


    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 
    Cam_Label = myWindow.Cam_Label # 웹캠 label

    th1 = threading.Thread(target=video_play)
    th2 = threading.Thread(target=Gesture_Cap, args=(cap,seq,action_seq,None))

    th1.setDaemon(True)
    th2.setDaemon(True)

    #프로그램 화면을 보여주는 코드
    myWindow.show()
    th1.start()
    th2.start()


    sys.exit(app.exec_())
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드