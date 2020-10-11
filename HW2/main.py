# -*- coding: utf-8 -*-

from sys import argv
from os import getcwd
from pandas import read_csv
from numpy import zeros, arange, float32

from PyQt5.QtCore import (QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, QFileDialog)

from MainWindow import Ui_MainWindow

'''
Read .64 file and convert to int array[0, 32)
'''
def to_raw_pic(file):
    img = read_csv(file)
    img = img.to_numpy().flatten()
    strings = ''

    for row in img:
        strings += row

    dic = {}
    for s in strings:
        if not s == '\x1a':
            dic[f'{str(s)}'] = strings.count(str(s))

    if img[-1] == '\x1a':
        new_img = zeros(shape = (img[:-1].shape[0], 64))
    else:
        new_img = zeros(shape = (img[:].shape[0], 64))

    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            last_row = -1 if img[-1] == '\x1a' else new_img.shape[1] + 1

            if img[:last_row][row][col].isalpha():
                new_img[row][col] = ord(img[:last_row][row][col]) - 55
            else:
                new_img[row][col] = img[:last_row][row][col]

    return new_img

'''
Main ui
'''

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.cwd = getcwd()
        self.raw_img = zeros(shape = (64, 64)).astype(float32)

    
    '''
    Click to read the second picture
    '''

    def onclick(self, e):
        
        imgChoose, _ = QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.64)") 

        if imgChoose == "":
            return
        
        self.raw2_img = to_raw_pic(imgChoose)
        self.title2 = imgChoose.split(imgChoose[:imgChoose.rfind('/') + 1])[1]

        self.raw2_widget.axes.cla()
        self.raw2_widget.axes.imshow(self.raw2_img, cmap = 'gray')
        # self.raw2_widget.axes.set_title(self.title2)
        self.raw2_widget.axes.set_axis_off()
        self.raw2_widget.draw()


    # def onclick2(self, e):
    #     print(e)
        
    '''
    NOT DONE YET
    '''
    def saveImage(self):
        imageChoose, _ = QFileDialog.getSaveFileName(self,  
                                    "Save image",  
                                    self.title + '.png',
                                    "Image Files (*.png, *.jpg)")  

        if imageChoose == "":
            return

    '''
    Replotting histogram and processed image when value changes
    '''
    def rePlot(self, hist_widget, processed_widget, img):
       
        processed_widget.axes.cla()
        processed_widget.axes.imshow(img, cmap = 'gray', vmin = 0, vmax = 31)
        processed_widget.axes.set_title(self.title)
        processed_widget.axes.set_axis_off()
        processed_widget.draw()

        hist_widget.axes.cla()
        hist_widget.axes.hist(img.flatten(), bins = arange(33) - .5)
        hist_widget.axes.set_title(self.title)
        hist_widget.axes.set_xticks(ticks = list(range(0, 32, 5)), minor = False)
        hist_widget.draw()

    '''
    Open read file dialog
    '''
    def onReadFile(self):
        self.mulValue.setProperty("value", 1.0)
        self.addValue.setProperty("value", 0.0)
        imgChoose, _ = QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.64)") 

        if imgChoose == "":
            return
        
        self.raw_img = to_raw_pic(imgChoose)
        self.title = imgChoose.split(imgChoose[:imgChoose.rfind('/') + 1])[1]

        self.raw_widget.axes.cla()
        self.raw_widget.axes.imshow(self.raw_img, cmap = 'gray')
        self.raw_widget.axes.set_title(self.title)
        self.raw_widget.axes.set_axis_off()
        self.raw_widget.draw()

        self.rePlot(self.histogram_widget, self.processed_widget, self.raw_img)
        
    '''
    Adding activate when value change
    '''
    def add(self):
        self.mulValue.setProperty("value", 1.0)

        old = self.raw_img + int(self.addValue.value())
        new_img = zeros(shape = self.raw_img.shape)

        for r in range(self.raw_img.shape[0]):
            for c in range(self.raw_img.shape[1]):
                if old[r][c] > 31:
                    new_img[r][c] = 31
                elif old[r][c] < 0:
                    new_img[r][c] = 0
                else:
                    new_img[r][c] = old[r][c]

        self.rePlot(self.histogram_widget, self.processed_widget, new_img)
        
    '''
    Multiplying activate when value change
    '''
    def multiply(self):
        self.addValue.setProperty("value", 0.0)

        old = self.raw_img * float(self.mulValue.value())
        new_img = zeros(shape = self.raw_img.shape)

        for r in range(self.raw_img.shape[0]):
            for c in range(self.raw_img.shape[1]):
                if old[r][c] > 31:
                    new_img[r][c] = 31
                else:
                    new_img[r][c] = old[r][c]

        self.rePlot(self.histogram_widget, self.processed_widget, new_img)
        
    '''
    Shifting the image 
    '''
    def shift(self):
        shift_img = zeros(shape = self.raw_img.shape)
        for row in range(self.raw_img.shape[0]):
            for col in range(self.raw_img.shape[1]):

                if col > 0:
                    shift_img[row][col] = self.raw_img[row][col] -  self.raw_img[row][col - 1]
                else:
                    shift_img[row][col] = 0
        
        self.addValue.setProperty("value", 0.0)
        self.mulValue.setProperty("value", 1.0)
        self.rePlot(self.histogram_widget, self.processed_widget, shift_img)
        
    '''
    Averaging two pictures
    '''
    def average(self):
        self.mulValue.setProperty("value", 1.0)
        self.addValue.setProperty("value", 0.0)
        self.rePlot(self.histogram_widget, self.processed_widget, (self.raw_img + self.raw2_img)/2)

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()