from tkinter.ttk import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd

import Traindata
import Detect
import Detectcam

master = tk.Tk()
master.title("Road Damage Detection")
master.geometry("900x400")
master.resizable(False, False)
master.configure(background='#EFE4B0')


    
def file_opener1():
    #input11 =filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("[png]","*.png*"),("[PNG]","*.PNG*"),("[jpg]","*.jpg*"),("[JPG]","*.JPG*"),("all files","*.*")))
    input11 =filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("[mp4]","*.mp4*"),("[MP4]","*.MP4*"),("[avi]","*.avi*"),("[AVI]","*.AVI*"),("all files","*.*")))
    filepath.set(input11)

def Detect_Trafic():
    if filepath.get()!='':
        print(filepath.get())
        #Detect.Detect_Start(str(filepath.get()))
        Detectcam.Detect_Start(str(filepath.get()))
    else:
        messagebox.showinfo(title="Select Image/Video File", message="Select Subject")
    
def Train_file():
    pass

def Process_Train():
    Traindata.Train_Start()
        

label = tk.Label(master ,width=40,text = "Road Damage Detection",font=("arial italic", 30), bg="#0000FF", fg="white").grid(row=0, column=0,columnspan=2)


a1 = tk.Label(master ,width=25,text = "Start Detect",font=("arial italic", 15), bg="#EFE4B0", fg="#0000FF").grid(row=2, column=0,padx=1, pady=1)

btnd1 = tk.Button(master,text="Select Video File",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=lambda:file_opener1()).grid(row=3, column=0,padx=1, pady=20)

filepath = tk.StringVar()
aa11 = tk.Entry(master,font=("arial italic", 15), bg="white", fg="Blue",width=25,textvariable=filepath).grid(row=4, column=0,padx=1, pady=1)
filepath.set("")

btnd2 = tk.Button(master,text="Start",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=lambda:Detect_Trafic()).grid(row=5, column=0,padx=1, pady=20)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

a22 = tk.Label(master ,width=25,text = "Start Train Model",font=("arial italic", 15), bg="#EFE4B0", fg="#0000FF").grid(row=2, column=1,padx=1, pady=1)
btn2 = tk.Button(master,text="Model Train",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=lambda:Process_Train()).grid(row=3, column=1,padx=1, pady=20)

btn3 = tk.Button(master,text="Exit",font=("arial italic", 15), bg="#0000FF", fg="white",width=25,command=master.destroy).grid(row=8, column=0,padx=1, pady=20)


master.mainloop()
