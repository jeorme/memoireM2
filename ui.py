#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.23
#  in conjunction with Tcl version 8.6
#    May 21, 2019 12:26:14 PM CEST  platform: Windows NT

import sys
import statsmodels.api as sm
from scipy import stats

import matplotlib.pyplot as plt

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import ui_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    ui_support.set_Tk_var()
    top = Toplevel1 (root)
    ui_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    ui_support.set_Tk_var()
    top = Toplevel1 (w)
    ui_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font10 = "-family {Courier New} -size 10 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("600x450+650+150")
        top.title("New Toplevel")
        top.configure(background="#d9d9d9")

        self.generateBt = tk.Button(top,command=self.generate)
        self.generateBt.place(relx=0.617, rely=0.133, height=34, width=87)
        self.generateBt.configure(activebackground="#ececec")
        self.generateBt.configure(activeforeground="#000000")
        self.generateBt.configure(background="#d9d9d9")
        self.generateBt.configure(disabledforeground="#a3a3a3")
        self.generateBt.configure(foreground="#000000")
        self.generateBt.configure(highlightbackground="#d9d9d9")
        self.generateBt.configure(highlightcolor="black")
        self.generateBt.configure(pady="0")
        self.generateBt.configure(text='''Generate''')
        self.generateBt.configure(width=87)

        self.modelCombo = ttk.Combobox(top)
        self.modelCombo.place(relx=0.2, rely=0.133, relheight=0.047
                , relwidth=0.205)
        self.value_list = ["AR","MA","ARMA","ARIMA","ARCH","GARCH","VAR","VMA","VARMA"]
        self.modelCombo.configure(values=self.value_list)
        self.modelCombo.configure(textvariable=ui_support.combobox)
        self.modelCombo.configure(width=123)
        self.modelCombo.configure(takefocus="")
        self.modelCombo.bind("<<ComboboxSelected>>",self.callback)

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.033, rely=0.111, height=41, width=64)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''model''')
        self.Label1.configure(width=64)

        self.Label2 = tk.Label(top)
        self.Label2.place(relx=0.05, rely=0.356, height=31, width=44)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''p''')
        self.Label2.configure(width=44)

        self.pVal = tk.Entry(top)
        self.pVal.place(relx=0.133, rely=0.356,height=30, relwidth=0.14)
        self.pVal.configure(background="white")
        self.pVal.configure(disabledforeground="#a3a3a3")
        self.pVal.configure(font=font10)
        self.pVal.configure(foreground="#000000")
        self.pVal.configure(insertbackground="black")
        self.pVal.configure(width=84)

        self.Label2_3 = tk.Label(top)
        self.Label2_3.place(relx=0.333, rely=0.356, height=31, width=44)
        self.Label2_3.configure(activebackground="#f9f9f9")
        self.Label2_3.configure(activeforeground="black")
        self.Label2_3.configure(background="#d9d9d9")
        self.Label2_3.configure(disabledforeground="#a3a3a3")
        self.Label2_3.configure(foreground="#000000")
        self.Label2_3.configure(highlightbackground="#d9d9d9")
        self.Label2_3.configure(highlightcolor="black")
        self.Label2_3.configure(text='''q''')

        self.qVal = tk.Entry(top)
        self.qVal.place(relx=0.4, rely=0.356,height=30, relwidth=0.14)
        self.qVal.configure(background="white")
        self.qVal.configure(disabledforeground="#a3a3a3")
        self.qVal.configure(font=font10)
        self.qVal.configure(foreground="#000000")
        self.qVal.configure(highlightbackground="#d9d9d9")
        self.qVal.configure(highlightcolor="black")
        self.qVal.configure(insertbackground="black")
        self.qVal.configure(selectbackground="#c4c4c4")
        self.qVal.configure(selectforeground="black")

        self.Label2_5 = tk.Label(top)
        self.Label2_5.place(relx=0.6, rely=0.356, height=31, width=44)
        self.Label2_5.configure(activebackground="#f9f9f9")
        self.Label2_5.configure(activeforeground="black")
        self.Label2_5.configure(background="#d9d9d9")
        self.Label2_5.configure(disabledforeground="#a3a3a3")
        self.Label2_5.configure(foreground="#000000")
        self.Label2_5.configure(highlightbackground="#d9d9d9")
        self.Label2_5.configure(highlightcolor="black")
        self.Label2_5.configure(text='''d''')

        self.dVal = tk.Entry(top)
        self.dVal.place(relx=0.683, rely=0.356,height=30, relwidth=0.14)
        self.dVal.configure(background="white")
        self.dVal.configure(disabledforeground="#a3a3a3")
        self.dVal.configure(font=font10)
        self.dVal.configure(foreground="#000000")
        self.dVal.configure(highlightbackground="#d9d9d9")
        self.dVal.configure(highlightcolor="black")
        self.dVal.configure(insertbackground="black")
        self.dVal.configure(selectbackground="#c4c4c4")
        self.dVal.configure(selectforeground="black")

        self.Label2_7 = tk.Label(top)
        self.Label2_7.place(relx=0.05, rely=0.222, height=31, width=84)
        self.Label2_7.configure(activebackground="#f9f9f9")
        self.Label2_7.configure(activeforeground="black")
        self.Label2_7.configure(background="#d9d9d9")
        self.Label2_7.configure(disabledforeground="#a3a3a3")
        self.Label2_7.configure(foreground="#000000")
        self.Label2_7.configure(highlightbackground="#d9d9d9")
        self.Label2_7.configure(highlightcolor="black")
        self.Label2_7.configure(text='''sample size''')
        self.Label2_7.configure(width=84)

        self.sampleSize = tk.Entry(top)
        self.sampleSize.place(relx=0.2, rely=0.222,height=30, relwidth=0.14)
        self.sampleSize.configure(background="white")
        self.sampleSize.configure(disabledforeground="#a3a3a3")
        self.sampleSize.configure(font=font10)
        self.sampleSize.configure(foreground="#000000")
        self.sampleSize.configure(highlightbackground="#d9d9d9")
        self.sampleSize.configure(highlightcolor="black")
        self.sampleSize.configure(insertbackground="black")
        self.sampleSize.configure(selectbackground="#c4c4c4")
        self.sampleSize.configure(selectforeground="black")

        self.Label2_8 = tk.Label(top)
        self.Label2_8.place(relx=0.417, rely=0.222, height=31, width=64)
        self.Label2_8.configure(activebackground="#f9f9f9")
        self.Label2_8.configure(activeforeground="black")
        self.Label2_8.configure(background="#d9d9d9")
        self.Label2_8.configure(disabledforeground="#a3a3a3")
        self.Label2_8.configure(foreground="#000000")
        self.Label2_8.configure(highlightbackground="#d9d9d9")
        self.Label2_8.configure(highlightcolor="black")
        self.Label2_8.configure(text='''dimension''')
        self.Label2_8.configure(width=64)

        self.dimension = tk.Entry(top)
        self.dimension.place(relx=0.533, rely=0.222,height=30, relwidth=0.14)
        self.dimension.configure(background="white")
        self.dimension.configure(disabledforeground="#a3a3a3")
        self.dimension.configure(font=font10)
        self.dimension.configure(foreground="#000000")
        self.dimension.configure(highlightbackground="#d9d9d9")
        self.dimension.configure(highlightcolor="black")
        self.dimension.configure(insertbackground="black")
        self.dimension.configure(selectbackground="#c4c4c4")
        self.dimension.configure(selectforeground="black")

    def generate(self):
        d = int(self.dVal)
        p = int(self.pVal)
        q = int(self.qVal)
        n = int(self.sampleSize)
        if self.modelCombo.get()=="AR":
            sample = sm.tsa.arma_generate_sample(d,0,n)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(sample)

    def callback(self,event):
        if self.modelCombo.get() == "AR":
            self.Label2_3.pack_forget()







if __name__ == '__main__':
    vp_start_gui()





