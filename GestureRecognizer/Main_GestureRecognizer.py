from webbrowser import open

# Qt designer에 기본으로 import 목록
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

# 카메라 캡쳐를 위한 추가 import 목록
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2

import threading

# 모듈 import
from Gesture_Recog import Gesture_Cap
from Gesture_Recog import cap
from Help_Page import Help_Page 

# tray 구현을 위한 모듈




#UI파일 연결
#UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("Main_Window.ui")[0]

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.TrayIcon = QSystemTrayIcon(self)
        self.TrayIcon.setIcon(QIcon("resource/image.jpg"))

        self.TrayIcon.setToolTip('check out my tray icon')

        menu = QMenu()
        Menu_Show = menu.addAction('Show')
        Menu_Show.triggered.connect(self.show)
        Menu_Help = menu.addAction('Help')
        Menu_Help.triggered.connect(self.Tray_Help)
        Menu_Exit = menu.addAction('Exit')
        Menu_Exit.triggered.connect(app.quit)
    
        self.TrayIcon.setContextMenu(menu)
        self.TrayIcon.show()


        #### 버튼 ####
        self.Button_Help.clicked.connect(self.Help_Page)
        self.Button_You.clicked.connect(lambda: open('https://www.youtube.com/'))
        self.Button_Net.clicked.connect(lambda: open('https://www.netflix.com/browse'))

        ##### 버튼/윈도우 이미지 #####
        self.setWindowIcon(QIcon('resource/image.jpg'))
        self.Button_You.setIcon(QIcon('resource/image.jpg'))
        self.Button_Net.setIcon(QIcon('resource/netimg.png'))
        #self.Help.
        #self.Button_You.setStyleSheet("border-image : resource/image.png")




    def Help_Page(self):
        self.hide()
        self.Help = Help_Page()
        self.Help.exec_()
        self.show()

    def Tray_Help(self):
        self.Help = Help_Page()
        self.Help.exec_()
        self.show()
        self.hide()

    def closeEvent(self, event):
        quit_msg = "No 선택시 프로그램이 종료됩니다."
        reply = QMessageBox.question(self, '트레이 축소', quit_msg, QMessageBox.Yes, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            myWindow.hide()
            event.ignore()
            
        else:
            event.accept()
            

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
        Label_WebCam.setPixmap(pixmap)

if __name__ == "__main__" :

    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 
    Label_WebCam = myWindow.Label_WebCam # 웹캠 label
    
    

    th1 = threading.Thread(target=video_play)
    th2 = threading.Thread(target=Gesture_Cap)

    th1.setDaemon(True)
    th2.setDaemon(True)

    #프로그램 화면을 보여주는 코드
    myWindow.show()
    th1.start()
    th2.start()


    sys.exit(app.exec_())
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드