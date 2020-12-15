# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
from time import time
from sys import argv
from os import getcwd
from cv2 import imread, split, merge, dft, idft
from numpy import zeros, arange, float32, sum, array, histogram, cumsum, int, around, divmod, floor, ceil, reshape, any, dot, random, median, amax, amin, ones, log, angle, arccos, dstack, float32, uint8
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, laplace
from PyQt5.QtGui import (QColor)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QColorDialog)

from mainwindow import Ui_MainWindow

def show_on_widget(widget, image, cmap = None, vmin = None, vmax = None):
    widget.axes.cla()
    widget.axes.imshow(image, cmap = cmap, vmin = vmin, vmax = vmax)
    widget.axes.set_axis_off()
    widget.draw()

class MainWindow(QMainWindow, Ui_MainWindow):
    '''
    Main ui
    '''
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.tab.mouseReleaseEvent = self.read_image
        self.tab_2.mouseReleaseEvent = self.read_image_2
        self.tab_3.mouseReleaseEvent = self.read_image_3
    
    def read_image(self, e):
        img, _ = QFileDialog.getOpenFileName(self,  
                                    "Open an image",  
                                    "",  
                                    "Image files (*.png *.jpg *.jpeg *.bmp)") 

        if img == "":
            return

        self.label_4.setText('')

        self.raw_img = imread(img)
        b, g, r = split(self.raw_img)
        self.raw_img = merge([r, g, b])
        show_on_widget(self.tab, self.raw_img)
    
    def read_image_2(self, e):
        img, _ = QFileDialog.getOpenFileName(self,  
                                    "Open an image",  
                                    "",  
                                    "Image files (*.png *.jpg *.jpeg *.bmp)") 

        if img == "":
            return

        self.label_5.setText('')

        self.raw_img_2 = imread(img)
        b, g, r = split(self.raw_img_2)
        self.raw_img_2 = merge([r, g, b])
        show_on_widget(self.tab_2, self.raw_img_2)

    def read_image_3(self, e):
        img, _ = QFileDialog.getOpenFileName(self,  
                                    "Open an image",  
                                    "",  
                                    "Image files (*.png *.jpg *.jpeg *.bmp)") 

        if img == "":
            return

        self.label_6.setText('')

        self.raw_img_3 = imread(img)
        b, g, r = split(self.raw_img_3)
        self.raw_img_3 = merge([r, g, b])
        show_on_widget(self.tab_3, self.raw_img_3)

def main():

    app = QApplication(argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()