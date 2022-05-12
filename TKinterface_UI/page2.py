from tkinter import *
from tkinter import messagebox
from pystray import MenuItem as item
import pystray
from PIL import Image, ImageTk
import cv2 as cv # OpenCV
import os
import ctypes

ws = Tk()
ws.geometry('400x300')
ws.title('SELECT PLATFORM')
ws['bg']='#ffbf00'

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
        menu=(item('Quit', quit_window), item('Show', show_window))
        icon=pystray.Icon("name", image, "My System Tray Icon", menu)
        icon.run()
    else:
        ws.destroy()

def nextPage():
    ws.destroy()
    import page3

def prevPage():
    ws.destroy()
    import page1

Label(
    ws,
    text="This is Second page",
    padx=20,
    pady=20,
    bg='#ffbf00',
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
    ).pack(fill=X, expand=TRUE, side=LEFT)


ws.protocol('WM_DELETE_WINDOW', hide_window)
ws.mainloop()