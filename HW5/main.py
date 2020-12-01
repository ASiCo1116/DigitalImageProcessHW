# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
from time import time
from sys import argv
from os import getcwd
from cv2 import imread, split, merge, dft, idft
from numpy import zeros, arange, float32, sum, array, histogram, cumsum, int, around, divmod, floor, ceil, reshape, any, dot, random, median, amax, amin, ones, log, angle, arccos
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, laplace
from PyQt5.QtCore import (QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, QFileDialog, QTableWidget, QTableWidgetItem)

from mainwindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    '''
    Main ui
    '''
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.cwd = getcwd()
                        
        self.raw.fig.canvas.mpl_connect('button_press_event', self.readFile)
        self.apply_1.clicked.connect(self.converting_planes)

    def readFile(self, e):
        '''
        Read a picture
        '''
        imgChoose, _ = QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.png *.jpg *.jpeg *.bmp)") 

        if imgChoose == "":
            return

        self.raw_img = imread(imgChoose)
        self.b,self.g,self.r = split(self.raw_img)
        self.raw_img = merge([self.r, self.g, self.b])
        # self.gray_img = sum(self.raw_img, axis=2) / 3.0
        self.processed_img = self.raw_img.copy()
        self.raw.axes.cla()
        self.raw.axes.imshow(self.raw_img)
        self.raw.axes.set_axis_off()
        self.raw.draw()
        # self.filtered_img = zeros(shape=self.raw_img.shape)

    def saveImage(self, e):
        '''
        Save an image
        '''
        imageName, _ = QFileDialog.getSaveFileName(self,  
                                    "Save Image As",  
                                    "",
                                    "Image Files (*.png *.jpg)")  

        if imageName == "":
            return

        if imageName:
            e.canvas.figure.savefig(imageName)
    
    def reset(self):

        self.filtered_img = zeros(shape = self.raw_img.shape)
        self.filter.axes.cla()
        self.filter.axes.imshow(self.filtered_img, cmap='gray', vmin=0, vmax=255)
        self.filter.axes.set_axis_off()
        self.filter.draw()
    
    def converting_planes(self):
        btnId = self.plane_btns.checkedId()  #-2 ~ -7
        
        if btnId == -2:
            self.rgb()
        elif btnId == -3:
            self.cmy()
        elif btnId == -4:
            self.hsi()
        elif btnId == -5:
            self.xyz()
        elif btnId == -6:
            self.lab()
        elif btnId == -7:
            self.yuv()

    def rgb(self):
        self.plane1.axes.cla()
        self.plane1.axes.imshow(self.r, cmap='gray')
        self.plane1.axes.set_axis_off()
        self.plane1.draw()

        self.plane2.axes.cla()
        self.plane2.axes.imshow(self.g, cmap='gray')
        self.plane2.axes.set_axis_off()
        self.plane2.draw()

        self.plane3.axes.cla()
        self.plane3.axes.imshow(self.b, cmap='gray')
        self.plane3.axes.set_axis_off()
        self.plane3.draw()
    
    def cmy(self):
        self.plane1.axes.cla()
        self.plane1.axes.imshow(255 - self.r, cmap='gray')
        self.plane1.axes.set_axis_off()
        self.plane1.draw()

        self.plane2.axes.cla()
        self.plane2.axes.imshow(255 - self.g, cmap='gray')
        self.plane2.axes.set_axis_off()
        self.plane2.draw()

        self.plane3.axes.cla()
        self.plane3.axes.imshow(255 - self.b, cmap='gray')
        self.plane3.axes.set_axis_off()
        self.plane3.draw()
    
    def hsi(self):
        self.r, self.g, self.b = self.r.astype(np.float32), self.g.astype(np.float32), self.b.astype(np.float32)
        self.h = arccos((self.r - self.g + self.r - self.b) * .5 / ((self.r - self.g)** 2 + (self.r - self.b) * (self.g - self.b))** .5)
        
        for x in range(self.h.shape[0]):
            for y in range(self.h.shape[1]):
                if self.b[x][y] > self.g[x][y]:
                    self.h[x][y] = 360.0 - self.h[x][y]
        self.s = np.zeros(shape = self.h.shape)
        for x in range(self.r.shape[0]):
            for y in range(self.r.shape[1]):
                self.s[x][y] = 1.0 - 3 / (self.r[x][y] + self.g[x][y] + self.b[x][y]) * min(self.r[x][y], self.g[x][y], self.b[x][y])

        self.i = (self.r + self.g + self.b) / 3

        self.plane1.axes.cla()
        self.plane1.axes.imshow(self.h, cmap='gray')
        self.plane1.axes.set_axis_off()
        self.plane1.draw()

        self.plane2.axes.cla()
        self.plane2.axes.imshow(self.s, cmap='gray')
        self.plane2.axes.set_axis_off()
        self.plane2.draw()

        self.plane3.axes.cla()
        self.plane3.axes.imshow(self.i, cmap='gray')
        self.plane3.axes.set_axis_off()
        self.plane3.draw()
    
    def xyz(self):
        self.x, self.y, self.z = zeros(shape=self.r.shape), zeros(shape=self.r.shape), zeros(shape=self.r.shape)
        
        for x in range(self.r.shape[0]):
            for y in range(self.r.shape[1]):
                self.x[x][y] = 0.412453 * self.r[x][y] + 0.35758 * self.g[x][y] + 0.180423 * self.b[x][y]
                self.y[x][y] = 0.212671 * self.r[x][y] + 0.71516 * self.g[x][y] + 0.072169 * self.b[x][y]
                self.z[x][y] = 0.019334 * self.r[x][y] + 0.119193 * self.g[x][y] + 0.950227 * self.b[x][y]

        self.plane1.axes.cla()
        self.plane1.axes.imshow(self.x, cmap='gray')
        self.plane1.axes.set_axis_off()
        self.plane1.draw()

        self.plane2.axes.cla()
        self.plane2.axes.imshow(self.y, cmap='gray')
        self.plane2.axes.set_axis_off()
        self.plane2.draw()

        self.plane3.axes.cla()
        self.plane3.axes.imshow(self.z, cmap='gray')
        self.plane3.axes.set_axis_off()
        self.plane3.draw()
    
    def lab(self):
        

        self.plane1.axes.cla()
        self.plane1.axes.imshow(255 - self.r, cmap='gray')
        self.plane1.axes.set_axis_off()
        self.plane1.draw()

        self.plane2.axes.cla()
        self.plane2.axes.imshow(255 - self.g, cmap='gray')
        self.plane2.axes.set_axis_off()
        self.plane2.draw()

        self.plane3.axes.cla()
        self.plane3.axes.imshow(255 - self.b, cmap='gray')
        self.plane3.axes.set_axis_off()
        self.plane3.draw()
    
    def yuv(self):
        self.plane1.axes.cla()
        self.plane1.axes.imshow(255 - self.r, cmap='gray')
        self.plane1.axes.set_axis_off()
        self.plane1.draw()

        self.plane2.axes.cla()
        self.plane2.axes.imshow(255 - self.g, cmap='gray')
        self.plane2.axes.set_axis_off()
        self.plane2.draw()

        self.plane3.axes.cla()
        self.plane3.axes.imshow(255 - self.b, cmap='gray')
        self.plane3.axes.set_axis_off()
        self.plane3.draw()

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()