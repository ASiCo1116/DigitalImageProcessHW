# -*- coding: utf-8 -*-

from math import floor, ceil
from cv2 import imread, split, merge
from sys import argv
from os import getcwd
# from pandas import read_csv
from numpy import zeros, arange, float32, sum, array

from PyQt5.QtCore import (QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, QFileDialog)

from MainWindow import Ui_MainWindow

def dotFunction(img, x, y):
    
    return array([[x, 0], [0, y]])

def bilinearInterpolation(image, ratio):

    img_height, img_width = image.shape[:2]
    height, width = int(img_height * (ratio/100)), int(img_width * (ratio/100))

    resized = zeros(shape = (height, width))

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    for i in range(height):
        for j in range(width):

            x_l, y_l = floor(x_ratio * j), floor(y_ratio * i)
            x_h, y_h = ceil(x_ratio * j), ceil(y_ratio * i)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            pixel = a * (1 - x_weight) * (1 - y_weight) + \
                    b * x_weight * (1 - y_weight) + \
                    c * y_weight * (1 - x_weight) + \
                    d * x_weight * y_weight

            resized[i][j] = pixel

    return resized

'''
Main ui
'''

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.cwd = getcwd()
        self.raw.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        
        self.resize_slider.setProperty('value', 100)
        self.resize_slider.setMinimum(10)
        self.resize_slider.setMaximum(200)

        self.binary_slider.setProperty('value', 128)
        self.binary_slider.setMinimum(0)
        self.binary_slider.setMaximum(255)

        self.bright_slider.setProperty('value', 0)
        self.bright_slider.setMinimum(-255)
        self.bright_slider.setMaximum(255)

        self.contrast_slider.setProperty('value', 0)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)

        self.binary_slider.valueChanged.connect(self.binaryValueChanged)
        self.bright_slider.valueChanged.connect(self.brightnessValueChanged)
        self.contrast_slider.valueChanged.connect(self.contrastValueChanged)
        self.resize_slider.valueChanged.connect(self.resizeValueChanged)
    
    '''
    Read picture function
    '''

    def onclick(self, e):
        
        imgChoose, _ = QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.png *.jpg *.jpeg)") 

        if imgChoose == "":
            return

        self.raw_img = imread(imgChoose)
        b,g,r = split(self.raw_img)
        self.raw_img = merge([r,g,b])
        self.raw.axes.cla()
        self.raw.axes.imshow(self.raw_img)
        self.raw.axes.set_axis_off()
        self.raw.draw()

        self.gray1_img = sum(self.raw_img, axis=2) / 3.0
        
        self.gray1.axes.cla()
        self.gray1.axes.imshow(self.gray1_img, cmap='gray')
        self.gray1.axes.set_axis_off()
        self.gray1.draw()
        
        self.gray1_hist.axes.cla()
        self.gray1_hist.axes.hist(self.gray1_img.flatten(), bins=arange(255) - .5)
        # self.gray1_hist.axes.set_title('grayscale histogram', fontsize=12)
        self.gray1_hist.axes.set_axis_off()
        self.gray1_hist.draw()

        self.gray2_img = self.raw_img[:, :, 0] * 0.299 + self.raw_img[:, :, 1] * 0.587 + self.raw_img[:, :, 2] * 0.114
        self.gray2.axes.cla()
        self.gray2.axes.imshow(self.gray2_img, cmap='gray')
        self.gray2.axes.set_axis_off()
        self.gray2.draw()
        
        self.gray2_hist.axes.cla()
        self.gray2_hist.axes.hist(self.gray2_img.flatten(), bins=arange(255) - .5)
        # self.gray2_hist.axes.set_title('grayscale histogram', fontsize=12)
        self.gray2_hist.axes.set_axis_off()
        self.gray2_hist.draw()

        self.subtract_hist.axes.cla()
        self.subtract_hist.axes.hist((self.gray2_img - self.gray1_img).flatten(), bins = arange(255) - .5)
        self.subtract_hist.axes.set_axis_off()
        self.subtract_hist.draw()

        self.binary_img = zeros(shape = self.gray1_img.shape)
        self.binary_img[self.gray1_img >= 128] = 255
        self.binary_img[self.gray1_img < 128] = 0

        self.binary.axes.cla()
        self.binary.axes.imshow(self.binary_img, cmap='gray', vmin=0, vmax=255)
        self.binary.axes.set_axis_off()
        self.binary.draw()

        self.bright_contrast_img = self.gray1_img.copy()
        self.bright_contrast.axes.cla()
        self.bright_contrast.axes.imshow(self.bright_contrast_img, cmap='gray')
        self.bright_contrast.axes.set_axis_off()
        self.bright_contrast.draw()

        self.resize_gray_img = self.gray1_img.copy()
        self.resize_gray.axes.cla()
        self.resize_gray.axes.imshow(self.resize_gray_img, cmap='gray')
        self.resize_gray.axes.set_axis_off()
        self.resize_gray.draw()

    def binaryValueChanged(self, v):
        self.binary_img[self.gray1_img >= v] = 255
        self.binary_img[self.gray1_img <= v] = 0

        self.binary.axes.cla()
        self.binary.axes.imshow(self.binary_img, cmap='gray', vmin=0, vmax=255)
        self.binary.axes.set_axis_off()
        self.binary.draw()

    def brightnessValueChanged(self, v):
        self.bright_contrast.axes.cla()
        self.bright_contrast.axes.imshow(self.brightness_contrast(v, self.contrast_slider.value()), cmap='gray', vmin=0, vmax=255)
        self.bright_contrast.axes.set_axis_off()
        self.bright_contrast.draw()
    
    def contrastValueChanged(self, v):  #v from -254 to 258
        self.bright_contrast.axes.cla()
        self.bright_contrast.axes.imshow(self.brightness_contrast(self.bright_slider.value(), v), cmap='gray', vmin=0, vmax=255)
        self.bright_contrast.axes.set_axis_off()
        self.bright_contrast.draw()
    
    def brightness_contrast(self, b, c):
        b_img = (self.bright_contrast_img + b).copy()
        b_img[b_img > 255] = 255
        b_img[b_img < 0] = 0

        factor = 259 * (c + 255) / (255 * (259 - c))
        new_image = (factor * (b_img - 128) + 128).copy()

        new_image[new_image > 255] = 255
        new_image[new_image < 0] = 0
        return new_image

    def resizeValueChanged(self, v):
        self.resize_gray.axes.cla()
        self.resize_gray.axes.imshow(bilinearInterpolation(self.resize_gray_img, v), cmap='gray', vmin=0, vmax=255)
        self.resize_gray.axes.set_axis_off()
        self.resize_gray.draw()

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