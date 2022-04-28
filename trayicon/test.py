# 모듈 호출

import tkinter as tk # Tkinter
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image # Pillow
import cv2 as cv # OpenCV
import os

import pystray
from pystray import MenuItem as item
from PIL import Image
import ctypes




def min():
    if messagebox.askyesno("Quit", "트레이로 최소화 하시겠습니까?"):
        win.withdraw()
        icon.run()
    else:
        win.destroy()

def tray_closing():
    if messagebox.askyesno("Quit", "프로그램을 종료하시겠습니까?"):
        icon.stop()
    else:
        pass

def video_play():
    ret, frame = cap.read() # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        cap.release() # 작업 완료 후 해제
        return
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = Image.fromarray(frame) # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img) # ImageTk 객체로 변환
    # OpenCV 동영상

    lbl1.imgtk = imgtk
    lbl1.configure(image=imgtk)
    lbl1.after(10, video_play)
 
def Max():
    win.deiconify()
    icon.stop()

image = Image.open("D:/My_Python/tray_icon/image.jpg")
menu = (item('Maxize', Max), item('close',tray_closing ))

icon = pystray.Icon("name", image, "title", menu)




# GUI 설계
win = tk.Tk() # 인스턴스 생성

win.title("인터페이스") # 제목 표시줄 추가
win.geometry("") # 지오메트리: 너비x높이+x좌표+y좌표
win.resizable(False, False) # x축, y축 크기 조정 비활성화

# 라벨 추가
lbl = tk.Label(win, text="Test label")
lbl.grid(row=0, column=0) # 라벨 행, 열 배치

# 프레임 추가
frm = tk.Frame(win, bg="black", width=720, height=480) # 프레임 너비, 높이 설정
frm.grid(row=1, column=0) # 격자 행, 열 배치

# 라벨1 추가
lbl1 = tk.Label(frm)
lbl1.grid()

cap = cv.VideoCapture(0) # VideoCapture 객체 정의

video_play()


win.protocol("WM_DELETE_WINDOW", min)
win.mainloop() #GUI 시작