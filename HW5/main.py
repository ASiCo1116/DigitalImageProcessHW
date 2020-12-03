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

def dum(a,an):
    if (a/an > 0.008856):
        return (a/an)**(1/3)
    else:
        return 7.787 * (a/an) + (16/116)

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
        self.apply_2.clicked.connect(self.pseudo)
        self.apply_3.clicked.connect(self.kmeans_planes)
        self.color_widget_1.mouseReleaseEvent = self.color1
        self.color_widget_2.mouseReleaseEvent = self.color2
        self.color_1 = QColor(255, 255, 255)
        self.color_2 = QColor(0, 0, 0)

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
    
    def draw_on(self, p, img, cmap = None, vmin = None, vmax = None):
        p.axes.cla()
        p.axes.imshow(img, cmap = cmap, vmin = vmin, vmax = vmax)
        p.axes.set_axis_off()
        p.draw()

    def color1(self, e):
        self.color_1 = QColorDialog.getColor()
        r, g, b = str(self.color_1.red()), str(self.color_1.green()), str(self.color_1.blue())
        self.color_widget_1.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

    def color2(self, e):
        self.color_2 = QColorDialog.getColor()
        r, g, b = str(self.color_2.red()), str(self.color_2.green()), str(self.color_2.blue())
        self.color_widget_2.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
    
    def reset(self):

        self.filtered_img = zeros(shape = self.raw_img.shape)
        self.filter.axes.cla()
        self.filter.axes.imshow(self.filtered_img, cmap='gray', vmin=0, vmax=255)
        self.filter.axes.set_axis_off()
        self.filter.draw()
    
    def pseudo(self):
        btnId = self.pseudo_group.checkedId()  #-2 ~ -7
        
        if btnId == -2:
            im_gray = self.processed_img
            im_color = cv.applyColorMap(im_gray, cv.COLORMAP_AUTUMN)
            r, g, b = im_gray[:,:, 0], im_gray[:,:, 1], im_gray[:,:, 2]

            self.draw_on(self.pseudo1, r, 'gray')
            self.draw_on(self.pseudo2, g, 'gray')
            self.draw_on(self.pseudo3, b, 'gray')
            self.draw_on(self.mixed_pseudo, im_color)

        elif btnId == -3:
            im_gray = self.processed_img
            im_color = cv.applyColorMap(im_gray, cv.COLORMAP_JET)
            r, g, b = im_gray[:,:, 0], im_gray[:,:, 1], im_gray[:,:, 2]
            
            self.draw_on(self.pseudo1, r, 'gray')
            self.draw_on(self.pseudo2, g, 'gray')
            self.draw_on(self.pseudo3, b, 'gray')
            self.draw_on(self.mixed_pseudo, im_color)

        elif btnId == -4:
            im_gray = self.processed_img
            im_color = cv.applyColorMap(im_gray, cv.COLORMAP_RAINBOW)
            r, g, b = im_gray[:,:, 0], im_gray[:,:, 1], im_gray[:,:, 2]
            
            self.draw_on(self.pseudo1, r, 'gray')
            self.draw_on(self.pseudo2, g, 'gray')
            self.draw_on(self.pseudo3, b, 'gray')
            self.draw_on(self.mixed_pseudo, im_color)

        elif btnId == -5:
            gray = 0.299 * self.r + 0.587 * self.g + 0.114 * self.b
            level = self.c_level.value() - 1

            r1, g1, b1 = self.color_1.red(), self.color_1.green(), self.color_1.blue()
            r2, g2, b2 = self.color_2.red(), self.color_2.green(), self.color_2.blue()

            r_step = (r2 - r1) / level
            g_step = (g2 - g1) / level
            b_step = (b2 - b1) / level

            new_r = zeros(self.r.shape)
            new_g = zeros(self.r.shape)
            new_b = zeros(self.r.shape)

            for x in range(self.r.shape[0]):
                for y in range(self.r.shape[1]):

                    new_r[x][y] = around(r1 + np.round(gray[x][y] * level / 255) * r_step)
                    new_g[x][y] = around(g1 + np.round(gray[x][y] * level / 255) * g_step)
                    new_b[x][y] = around(b1 + np.round(gray[x][y] * level / 255) * b_step)
            
            new_r, new_g, new_b = new_r.astype(np.int), new_g.astype(np.int), new_b.astype(np.int)
            mix = dstack([new_r, new_g, new_b])
            mix[mix > 255] = 255
            
            self.draw_on(self.pseudo1, self.r, 'gray')
            self.draw_on(self.pseudo2, self.g, 'gray')
            self.draw_on(self.pseudo3, self.b, 'gray')
            self.draw_on(self.mixed_pseudo, mix)
        
    def converting_planes(self):
        btnId = self.plane_btns.checkedId()  #-2 ~ -7
        
        if btnId == -2:
            self.rgb(self.plane1, self.plane2, self.plane3, self.mixed_plane)
        elif btnId == -3:
            self.cmy(self.plane1, self.plane2, self.plane3, self.mixed_plane)
        elif btnId == -4:
            self.hsi(self.plane1, self.plane2, self.plane3, self.mixed_plane)
        elif btnId == -5:
            self.xyz(self.plane1, self.plane2, self.plane3, self.mixed_plane)
        elif btnId == -6:
            self.lab(self.plane1, self.plane2, self.plane3, self.mixed_plane)
        elif btnId == -7:
            self.yuv(self.plane1, self.plane2, self.plane3, self.mixed_plane)
        
    def kmeans_planes(self):
        btnId = self.btns_2.checkedId()

        if btnId == -2:
            self.kmeans(self.rgb(self.kplane1, self.kplane2, self.kplane3, self.kmeans_processed), self.k_level.value())
        elif btnId == -3:
            self.kmeans(self.cmy(self.kplane1, self.kplane2, self.kplane3, self.kmeans_processed), self.k_level.value() )
        elif btnId == -4:
            self.kmeans(self.hsi(self.kplane1, self.kplane2, self.kplane3, self.kmeans_processed), self.k_level.value())
        elif btnId == -5:
            self.kmeans(self.xyz(self.kplane1, self.kplane2, self.kplane3, self.kmeans_processed), self.k_level.value())
        elif btnId == -6:
            self.kmeans(self.lab(self.kplane1, self.kplane2, self.kplane3, self.kmeans_processed), self.k_level.value())
        elif btnId == -7:
            self.kmeans(self.yuv(self.kplane1, self.kplane2, self.kplane3, self.kmeans_processed), self.k_level.value())

    def rgb(self, p1, p2, p3, p4):
        self.draw_on(p1, self.r, 'gray', 0, 255)
        self.draw_on(p2, self.g, 'gray', 0, 255)
        self.draw_on(p3, self.b, 'gray', 0, 255)
        mix = dstack([self.r, self.g, self.b])
        self.draw_on(p4, mix, None, 0, 255)

        return mix
    
    def cmy(self, p1, p2, p3, p4):
        self.draw_on(p1, 255-self.r, 'gray', 0, 255)
        self.draw_on(p2, 255-self.g, 'gray', 0, 255)
        self.draw_on(p3, 255-self.b, 'gray', 0, 255)
        mix = dstack([255-self.r, 255-self.g, 255-self.b])
        self.draw_on(p4, mix, None, 0, 255)

        return mix
    
    def hsi(self, p1, p2, p3, p4):
        r, g, b = self.r / 255, self.g / 255, self.b / 255
        h, s, i = zeros(r.shape), zeros(r.shape), zeros(r.shape)

        for x in range(r.shape[0]):
            for y in range(r.shape[1]):
                min_ = min(r[x][y], g[x][y], b[x][y])
                
                s[x][y] = 1 - (3 / (r[x][y] + g[x][y] + b[x][y])) * min_
                i[x][y] = (r[x][y] + g[x][y] + b[x][y]) / 3

                up = ((r[x][y] - g[x][y]) + (r[x][y] - b[x][y])) * .5
                down = ((r[x][y] - g[x][y])** 2 + (r[x][y] - b[x][y]) * (g[x][y] - b[x][y]))** .5
                degree = arccos(up / down)
                
                if b[x][y] > g[x][y]:
                    h[x][y] = 360 - np.degrees(degree)
                else:
                    h[x][y] = np.degrees(degree)
        
        h = h * 255 / 360
        h[np.isnan(h)] = 0

        s = s * 255
        s[np.isnan(s)] = 0

        i = i * 255
        i[np.isnan(i)] = 0

        h, s, i = h.astype(np.int), s.astype(np.int), i.astype(np.int)

        self.draw_on(p1, h, 'gray')
        self.draw_on(p2, s, 'gray')
        self.draw_on(p3, i, 'gray')
        mix = dstack([h, s, i])
        self.draw_on(p4, mix)
        
        return mix
    
    def xyz(self, p1, p2, p3, p4):
        x_, y_, z = zeros(self.r.shape), zeros(self.r.shape), zeros(self.r.shape)
        
        for x in range(self.r.shape[0]):
            for y in range(self.r.shape[1]):
                x_[x][y] = 0.412453 * self.r[x][y] + 0.35758 * self.g[x][y] + 0.180423 * self.b[x][y]
                y_[x][y] = 0.212671 * self.r[x][y] + 0.71516 * self.g[x][y] + 0.072169 * self.b[x][y]
                z[x][y] = 0.019334 * self.r[x][y] + 0.119193 * self.g[x][y] + 0.950227 * self.b[x][y]

        x_, y_, z = x_.astype(np.int), y_.astype(np.int), z.astype(np.int)
        self.draw_on(p1, x_, 'gray')
        self.draw_on(p2, y_, 'gray')
        self.draw_on(p3, z, 'gray')
        mix = dstack([x_, y_, z])
        mix[mix > 255] = 255
        self.draw_on(p4, mix)

        return mix
    
    def lab(self, p1, p2, p3, p4):
        x_, y_, z = zeros(self.r.shape), zeros(self.r.shape), zeros(self.r.shape)
        
        for x in range(self.r.shape[0]):
            for y in range(self.r.shape[1]):
                x_[x][y] = 0.412453 * self.r[x][y] + 0.35758 * self.g[x][y] + 0.180423 * self.b[x][y]
                y_[x][y] = 0.212671 * self.r[x][y] + 0.71516 * self.g[x][y] + 0.072169 * self.b[x][y]
                z[x][y] = 0.019334 * self.r[x][y] + 0.119193 * self.g[x][y] + 0.950227 * self.b[x][y]

        x_, y_, z = x_ / 255, y_ / 255, z / 255
        
        xn = 0.950456
        yn = 1.0
        zn = 1.088754
        l, a, b = zeros(self.r.shape), zeros(self.r.shape), zeros(self.r.shape)

        for x in range(self.r.shape[0]):
            for y in range(self.r.shape[1]):
                if y_[x][y] / yn > .008856:
                    l[x][y] = 116 * (y_[x][y]/yn)**(1/3) - 16
                else:
                    903.3 * y_[x][y]/yn
                a[x][y] = 500 * (dum(x_[x][y], xn) - dum(y_[x][y], yn))
                b[x][y] = 200 * (dum(y_[x][y], yn) - dum(z[x][y], zn))
        
        l = l * 255
        a = a * 255
        b = b * 255

        l, a, b = l.astype(np.int), a.astype(np.int), b.astype(np.int)
        mix = dstack([l, a, b])
        mix[mix > 255] = 255

        self.draw_on(p1, l, 'gray')
        self.draw_on(p2, a, 'gray')
        self.draw_on(p3, b, 'gray')
        self.draw_on(p4, mix)

        return mix

    def yuv(self, p1, p2, p3, p4):
        y_, u, v = zeros(self.r.shape), zeros(self.r.shape), zeros(self.r.shape)
        
        for x in range(self.r.shape[0]):
            for y in range(self.r.shape[1]):

                y_[x][y] = 0.299 * self.r[x][y] + 0.587 * self.g[x][y] + 0.114 * self.b[x][y]
                u[x][y] = -0.169 * self.r[x][y] + -0.331 * self.g[x][y] + 0.5 * self.b[x][y] + 128
                v[x][y] = 0.5 * self.r[x][y] + -0.419 * self.g[x][y] + 0.081 * self.b[x][y] + 128
        
        y_, u, v = y_.astype(np.int), u.astype(np.int), v.astype(np.int)
        mix = dstack([y_, u, v])
        mix[mix > 255] = 255

        self.draw_on(p1, y_, 'gray')
        self.draw_on(p2, u, 'gray')
        self.draw_on(p3, v, 'gray')
        self.draw_on(p4, mix)

        return mix

    def kmeans(self, img, k):
        Z = img.reshape((-1, 3))
        Z = float32(Z)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        center = uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        
        self.mixed_kplane.axes.cla()
        self.mixed_kplane.axes.imshow(res2)
        self.mixed_kplane.axes.set_axis_off()
        self.mixed_kplane.draw()

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()