from tkinter import *
from tkinter import messagebox
from pystray import MenuItem as item
import pystray
from PIL import Image, ImageTk
import cv2 as cv # OpenCV
import os
import ctypes
import webbrowser

ws = Tk()
ws.geometry('400x300')
ws.title('SELECT PLATFORM')
ws['bg']='#ffbf00'


youtube = cv.imread("F:/Capstone-Design-Team-M-main/img/그림/youtube.png")
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
        icon=pystray.Icon("name", image, "Gesture Program", menu)
        icon.run()
    else:
        ws.destroy()

def nextPage():
    ws.destroy()
    import page3

def prevPage():
    ws.destroy()
    import page1

def Netflix_b():
    hide_window
    webbrowser.open("https://www.netflix.com/kr/")

def Youtube_b():
    hide_window
    webbrowser.open("https://www.youtube.com/")

Label(
    ws,
    text="NETFLIX YOUTUBE",
    padx=20,
    pady=20,
    bg='#ffbf00',
    font=f
).pack(expand=True, fill=BOTH)

Button(
    ws, 
    text="YOUTUBE", 
    font=f,
    command=Youtube_b
    #image=youtube
    
    ).pack(fill=X, expand=TRUE, side=LEFT)
Button(
    ws, 
    text="NETFLIX", 
    font=f,
    command=Netflix_b
    ).pack(fill=X, expand=TRUE, side=RIGHT)


ws.protocol('WM_DELETE_WINDOW', hide_window)
ws.mainloop()
