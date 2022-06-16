from tkinter import *
from tkinter import messagebox
from cv2 import destroyWindow
from pystray import MenuItem as item
import pystray
from PIL import Image, ImageTk
import cv2 as cv # OpenCV
import os
import ctypes

ws = Tk()
ws.geometry('400x300')
ws.title('VIDEO TEST')
ws['bg']='#5d8a82'

f = ("Times bold", 14)

def quit_window(icon, item):
       icon.stop()
       ws.destroy()

def show_window(icon, item):
       icon.stop()
       ws.after(0,ws.deiconify())

def hide_window():
    if messagebox.askyesno("Quit", "트레이로 최소화 하시겠습니까?"):
        ws.withdraw()
        image=Image.open("C:/Users/gudrb/Desktop/그리;ㅁ/곰곰.png")
        menu=(item('종료', quit_window), item('열기', show_window))
        icon=pystray.Icon("Gesture program", image, "Gesture program", menu)
        icon.run()
    else:
        ws.destroy()


def nextPage():
    ws.destroy()
    import page2

    
#############################################

def video_play():
    ret, frame = cap.read() # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        cap.release() # 작업 완료 후 해제
        return
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = Image.fromarray(frame) # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img) # ImageTk 객체로 변환
    
    # OpenCV 동영상

cap = cv.VideoCapture(0)


#아직구현안함.
###############################################

Label(
    ws,
    text="VIDEO LABEL",
    padx=20,
    pady=20,
    bg='#5d8a82',
    font=f
    

).pack(expand=True, fill=BOTH)

Button(
    ws, 
    text="나가기", 
    font=f,
    command=hide_window
    ).pack(fill=X, expand=TRUE, side=LEFT)

Button(
    ws, 
    text="다음", 
    font=f,
    command=nextPage
    ).pack(fill=X, expand=TRUE, side=RIGHT)

ws.protocol('WM_DELETE_WINDOW', hide_window)
ws.mainloop()
