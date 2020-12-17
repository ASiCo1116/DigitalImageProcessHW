# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
import pywt
from functools import reduce
from time import time
from sys import argv
from os import getcwd
from cv2 import imread, split, merge, dft, idft
from numpy import zeros, arange, float32, sum, array, histogram, cumsum, int, around, divmod, floor, ceil, reshape, any, dot, random, median, amax, amin, ones, log, angle, arccos, dstack, float32, uint8
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, laplace
from skimage.measure import perimeter
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
        self.pushButton_1.clicked.connect(self.trapezoidal)
        self.pushButton_2.clicked.connect(self.wavy)
        self.pushButton_3.clicked.connect(self.circular)
        self.pushButton_4.clicked.connect(self.reset)
        self.pushButton_5.clicked.connect(self.fusion)
        self.pushButton_6.clicked.connect(self.hough_transform)
        self.gray_img = None
        self.gray_img_2 = None
        self.gray_img_3 = None
    
    def read_image(self, e):
        img, _ = QFileDialog.getOpenFileName(self,  
                                    "Open an image",  
                                    "",  
                                    "Image files (*.png *.jpg *.jpeg *.bmp)") 

        if img == "":
            return
        self.cv_img = img
        self.label_4.setText('')

        self.raw_img = imread(img)
        b, g, r = split(self.raw_img)
        self.raw_img = merge([r, g, b])
        self.gray_img = (r * .299 + g * .587 + b * .114).astype(np.int)
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
        self.gray_img_2 = (r * .299 + g * .587 + b * .114).astype(np.int)
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
        self.gray_img_3 = (r * .299 + g * .587 + b * .114).astype(np.int)
        show_on_widget(self.tab_3, self.raw_img_3)

    def trapezoidal(self):
        new_img = zeros(self.gray_img.shape)

        for r in range(self.gray_img.shape[0]):
            for c in range(self.gray_img.shape[1]):
                new_x = int(3 * r / 4 + r * c / (self.gray_img.shape[0] * self.gray_img.shape[1]))
                new_y = int(c + r / 4 - c * r / (2 * self.gray_img.shape[1]))
                new_img[new_x][new_y] = self.gray_img[r][c]

        show_on_widget(self.widget_precessed, new_img, cmap='gray')

    def wavy(self):
        new_img = zeros(self.gray_img.shape)

        for r in range(self.gray_img.shape[0]):
            for c in range(self.gray_img.shape[1]):
                new_x = int(c - 32 * math.sin(r / 32))
                new_y = int(r - 32 * math.sin(c / 32))
                if new_x >= 1 and new_x <= self.gray_img.shape[0] and new_y >= 1 and new_y <= self.gray_img.shape[1]:
                    new_img[c][r] = self.gray_img[new_x - 1][new_y - 1]

        show_on_widget(self.widget_precessed, new_img, cmap='gray')

    def circular(self):
        new_img = zeros(self.gray_img.shape)

        for r in range(1, self.gray_img.shape[0]):
            for c in range(1, self.gray_img.shape[1]):

                d = ((self.gray_img.shape[0] / 2)** 2 - ((self.gray_img.shape[0] / 2) - r)** 2)** .5
                new_x = int( (c - self.gray_img.shape[1] / 2) * self.gray_img.shape[1] / (d * 2) + self.gray_img.shape[1] / 2)
                new_y = r
                
                if new_x >= 1 and new_x <= self.gray_img.shape[1] and new_y >= 1 and new_y <= self.gray_img.shape[1]:
                    new_img[r][c] = self.gray_img[new_y - 1][new_x - 1]

        show_on_widget(self.widget_precessed, new_img, cmap='gray')
    
    def reset(self):
        show_on_widget(self.widget_precessed, zeros(self.gray_img.shape), cmap='gray')

    def fusion(self):
        db_value = self.spinBox_db.value()

        if not isinstance(self.gray_img_3, np.ndarray):
            (ll, (lh, hl, hh)) = pywt.dwt2(self.gray_img, f'db{db_value}')
            (ll2, (lh2, hl2, hh2)) = pywt.dwt2(self.gray_img_2, f'db{db_value}')
            ll_avg = (ll + ll2) / 2
            lh_max, hl_max, hh_max = np.maximum(lh, lh2), np.maximum(hl, hl2), np.maximum(hh, hh2)
            new_img = pywt.idwt2((ll_avg, (lh_max, hl_max, hh_max)), f'db{db_value}')
        else:
            (ll, (lh, hl, hh)) = pywt.dwt2(self.gray_img, f'db{db_value}')
            (ll2, (lh2, hl2, hh2)) = pywt.dwt2(self.gray_img_2, f'db{db_value}')
            (ll3, (lh3, hl3, hh3)) = pywt.dwt2(self.gray_img_3, f'db{db_value}')
            ll_avg = (ll + ll2 + ll3) / 3
            lh_max, hl_max, hh_max = np.maximum.reduce([lh, lh2, lh3]), np.maximum.reduce([hl, hl2, hl3]), np.maximum.reduce([hh, hh2, hh3])
            new_img = pywt.idwt2((ll_avg, (lh_max, hl_max, hh_max)), f'db{db_value}')

        show_on_widget(self.widget_precessed, new_img, 'gray')

    def hough_transform(self):
        img = cv.imread(self.cv_img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        filter_img = zeros((gray.shape[0] + 4, gray.shape[1] + 4))

        for (r, c) in np.ndindex(gray.shape):
            filter_img[r + 2][c + 2] = gray[r][c]
            
        edges = cv.Canny(gray, 50, 150, apertureSize=3)
        lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # img[img >= 128] = 255
        # img[img < 128] = 0
        show_on_widget(self.widget_precessed, img)

        # print(perimeter(img, neighbourhood=4))

def main():

    app = QApplication(argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()