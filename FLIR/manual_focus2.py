from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
import time, traceback

top=Tk()
top.title("MANUAL FOCUS TOOL")

###focus control
focus= Scale(orient='vertical', label='focus control',from_=100, length=500,resolution=1, to=4207)
focus.pack(side=RIGHT)

url='rr+tcp://192.168.55.10:60827/?service=camera'

c1=RRN.ConnectService(url)

def update_focus():
    try:
        c1.setf_param("focus_pos", RR.VarValue(int(focus.get()),"int32"))
        top.after(10,update_focus)
    except:
        traceback.print_exc()
update_focus()
top.mainloop()
