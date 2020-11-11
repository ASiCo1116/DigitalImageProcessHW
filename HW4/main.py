# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
from time import time
from sys import argv
from os import getcwd
from cv2 import imread, split, merge, dft, idft
from numpy import zeros, arange, float32, sum, array, histogram, cumsum, int, around, divmod, floor, ceil, reshape, any, dot, random, median, amax, amin, ones, log, angle
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
        self.spectrum.fig.canvas.mpl_connect('button_press_event', self.saveImage)
        self.angle.fig.canvas.mpl_connect('button_press_event', self.saveImage)
        self.ifft.fig.canvas.mpl_connect('button_press_event', self.saveImage)
        self.apply_1.clicked.connect(self.filter_radio)
        self.reset_1.clicked.connect(self.reset)

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
        b,g,r = split(self.raw_img)
        self.raw_img = merge([r, g, b])
        self.gray_img = sum(self.raw_img, axis=2) / 3.0
        self.processed_img = self.gray_img.copy()
        self.raw.axes.cla()
        self.raw.axes.imshow(self.gray_img, cmap='gray')
        self.raw.axes.set_axis_off()
        self.raw.draw()
        self.filtered_img = zeros(shape=self.raw_img.shape)


        fft, angle, ifft = self._problem1(self.processed_img)

        self.spectrum.axes.cla()
        self.spectrum.axes.imshow(fft, cmap='gray')
        self.spectrum.axes.set_axis_off()
        self.spectrum.draw()

        self.angle.axes.cla()
        self.angle.axes.imshow(angle, cmap='gray')
        self.angle.axes.set_axis_off()
        self.angle.draw()

        #some bug in ifft
        self.ifft.axes.cla()
        self.ifft.axes.imshow(ifft, cmap='gray')
        self.ifft.axes.set_axis_off()
        self.ifft.draw()


    
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
    
    def _problem1(self, img):
        angle_img = fft2(img)
        fft2_img = dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        fft2_img_shift = fftshift(fft2_img)
        mag = cv.magnitude(fft2_img_shift[:,:, 0], fft2_img_shift[:,:, 1])
        _min, _max = log(1 + abs(mag.min())), log(1 + abs(mag.max()))
        new_img = 255 * (log(1 + abs(mag)) - _min) / (_max - _min)

        idft_img = idft(ifftshift(fft2_img_shift), flags=cv.DFT_SCALE)
        new_img_ifft = cv.magnitude(idft_img[:,:, 0], idft_img[:,:, 1])
        new_img_ifft[new_img_ifft > 255] = 255
        new_img_ifft[new_img_ifft < 0] = 0

        return new_img, angle(angle_img), new_img_ifft

    def filter_radio(self):
        btnId = self.filterButton.checkedId()  #-2 ~ -10
        
        if btnId == -2:  #IHP
            self._IP('high')
        elif btnId == -3: #BWHP
            self._BW('high')
        elif btnId == -4: #GHP
            self._GP('high')
        elif btnId == -5: #ILP
            self._IP('low')
        elif btnId == -6: #BWLP
            self._BW('low')
        elif btnId == -7: #GLP
            self._GP('low')
        elif btnId == -8:
            self._HOMO()
        elif btnId == -9:
            self._MB()
        elif btnId == 10:
            self._MBN()

        self.filter.axes.cla()
        self.filter.axes.imshow(self.filtered_img, cmap='gray')
        self.filter.axes.set_axis_off()
        self.filter.draw()

    
    def _IP(self, mode):
        
        if mode == 'low':

            filter = zeros(shape=self.processed_img.shape)
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    if ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5 <= self.cutoff_box.value():
                        filter[i][j] = 1

        if mode == 'high':

            filter = ones(shape=self.processed_img.shape)
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    if ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5 <= self.cutoff_box.value():
                        filter[i][j] = 0

        _dft = fftshift(dft(self.processed_img, flags=cv.DFT_COMPLEX_OUTPUT))
        self.filtered_img = _dft.copy()
        self.filtered_img[:,:, 0] = _dft[:,:, 0] * filter
        self.filtered_img[:,:, 1] = _dft[:,:, 1] * filter
        self.filtered_img = idft(ifftshift(self.filtered_img), flags=cv.DFT_SCALE)
        self.filtered_img = cv.magnitude(self.filtered_img[:,:, 0], self.filtered_img[:,:, 1])
        self.filtered_img[self.filtered_img > 255] = 255
        self.filtered_img[self.filtered_img < 0] = 0
    
    def _BW(self, mode):
        if mode == 'high':
            filter = zeros(shape=self.processed_img.shape)
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    d = ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5
                    filter[i][j] = 1 / (1 + (self.cutoff_box.value() / d)**(2 * self.order_box.value()))
                    
        if mode == 'low':
            filter = zeros(shape=self.processed_img.shape)
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    d = ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5
                    filter[i][j] = 1 / (1 + (d / self.cutoff_box.value())**(2 * self.order_box.value()))
        
        _dft = fftshift(dft(self.processed_img, flags=cv.DFT_COMPLEX_OUTPUT))
        self.filtered_img = _dft.copy()
        self.filtered_img[:,:, 0] = _dft[:,:, 0] * filter
        self.filtered_img[:,:, 1] = _dft[:,:, 1] * filter
        self.filtered_img = idft(ifftshift(self.filtered_img), flags=cv.DFT_SCALE)
        self.filtered_img = cv.magnitude(self.filtered_img[:,:, 0], self.filtered_img[:,:, 1])
        self.filtered_img[self.filtered_img > 255] = 255
        self.filtered_img[self.filtered_img < 0] = 0


    def _GP(self, mode):
        if mode == 'high':
            filter = zeros(shape=self.processed_img.shape)
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    d = ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5
                    filter[i][j] = 1 - math.exp(-d ** 2 / 2 / self.cutoff_box.value() / self.cutoff_box.value())
        
        if mode == 'low':
            filter = zeros(shape=self.processed_img.shape)
            for i in range(filter.shape[0]):
                for j in range(filter.shape[1]):
                    d = ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5
                    filter[i][j] = math.exp(-d ** 2 / 2 / self.cutoff_box.value() / self.cutoff_box.value())
        
        _dft = fftshift(dft(self.processed_img, flags=cv.DFT_COMPLEX_OUTPUT))
        self.filtered_img = _dft.copy()
        self.filtered_img[:,:, 0] = _dft[:,:, 0] * filter
        self.filtered_img[:,:, 1] = _dft[:,:, 1] * filter
        self.filtered_img = idft(ifftshift(self.filtered_img), flags=cv.DFT_SCALE)
        self.filtered_img = cv.magnitude(self.filtered_img[:,:, 0], self.filtered_img[:,:, 1])
        self.filtered_img[self.filtered_img > 255] = 255
        self.filtered_img[self.filtered_img < 0] = 0
    
    def _HOMO(self):
        filter = zeros(shape=self.processed_img.shape)
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                d = ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5
                filter[i][j] = (self.rH_box.value() - self.rL_box.value()) * (1 - math.exp(-d ** 2 / self.cutoff_box.value() / self.cutoff_box.value())) + self.rL_box.value()
        
        _dft = fftshift(dft(self.processed_img, flags=cv.DFT_COMPLEX_OUTPUT))
        self.filtered_img = _dft.copy()
        self.filtered_img[:,:, 0] = _dft[:,:, 0] * filter
        self.filtered_img[:,:, 1] = _dft[:,:, 1] * filter
        self.filtered_img = idft(ifftshift(self.filtered_img), flags=cv.DFT_SCALE)
        self.filtered_img = cv.magnitude(self.filtered_img[:,:, 0], self.filtered_img[:,:, 1])
        self.filtered_img[self.filtered_img > 255] = 255
        self.filtered_img[self.filtered_img < 0] = 0
    
    def _MB(self):

        filter = zeros(shape=self.processed_img.shape)
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                ab = (0.1 * i + 0.1 * j)
                if i == 0 and j == 0:
                    filter[i][j] = 0
                    continue
                filter[i][j] = (1 / np.pi / ab) * np.sin(np.pi * ab) * np.exp(-1j * np.pi * ab)

        _dft = fftshift(dft(self.processed_img, flags=cv.DFT_COMPLEX_OUTPUT))
        self.filtered_img = _dft.copy()
        # self.filtered_img = _dft * filter
        self.filtered_img[:,:, 0] = _dft[:,:, 0] * filter
        self.filtered_img[:,:, 1] = _dft[:,:, 1] * filter
        self.filtered_img = idft(ifftshift(self.filtered_img), flags=cv.DFT_SCALE)
        self.filtered_img = cv.magnitude(self.filtered_img[:,:, 0], self.filtered_img[:,:, 1])
        self.filtered_img[self.filtered_img > 255] = 255
        self.filtered_img[self.filtered_img < 0] = 0

    def _MBN(self):

        pass

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()