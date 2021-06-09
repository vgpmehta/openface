# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:04:23 2020

@author: DiogoBranco
"""


import tkinter as tk
from tkinter import filedialog
import csv

root=tk.Tk()
root.title("Blinks")

root.resizable(0, 0)


def addFile():
    Dirname=filedialog.askopenfile(mode ='r', filetypes =[('Comma Separated Files', '*.csv')])
    #print(Dirname)
    Dirname=str(Dirname)
    Dirname=Dirname.replace("<_io.TextIOWrapper name='",'')
    Dirname=Dirname.replace("' mode='r' encoding='cp1252'>",'')
    #print(Dirname)
    
    
    
    blinks=[]
    with open(Dirname, newline='') as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            blinks.append(row[' AU45_c'])
    
    
    a=0
    numberofblinks=0
    while a!=len(blinks)-1:

        b=a+1
    
        if blinks[a]== ' 1.00' and blinks[b] == ' 0.00':
            numberofblinks=numberofblinks+1
        a=a+1
    
    #print("Number of blinks: ",numberofblinks)
            
    popupmsg("Number of blinks: "+str(numberofblinks))
    
    
    
    
    
    
def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = tk.Label(popup, text=msg)
    label.pack(anchor= "center", fill="x", pady=10)
    B1 = tk.Button(popup, text="Close", command = popup.destroy)
    B1.pack()
    popup.mainloop()



canvas=tk.Canvas(root, height=100, width=200, bg="black")
label1 =tk.Label(root,text="Select a .csv file", bg="white")    
canvas.create_window(100,50,window=label1)    
canvas.pack()
    
openFile=tk.Button(root,text ="Select File", padx=50,pady=5,fg="white",bg="#263D42",command=addFile)
openFile.pack()
root.mainloop()









