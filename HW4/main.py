# -*- coding: utf-8 -*-

from time import time
from sys import argv
from os import getcwd
from cv2 import imread, split, merge
from numpy import zeros, arange, float32, sum, array, histogram, cumsum, int, around, divmod, floor, ceil, reshape, any, dot, random, median, amax, amin, ones, log, angle
from numpy.fft import fft2, ifft2, fftshift
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
        # self.ifft.axes.cla()
        # self.ifft.axes.imshow(ifft, cmap='gray')
        # self.ifft.axes.set_axis_off()
        # self.ifft.draw()


    
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
        fft2_img = fft2(img).copy()
        _min, _max = log(1 + abs(fft2_img.min())), log(1 + abs(fft2_img.max()))
        new_img = 255 * (log(1 + abs(fft2_img)) - _min) / (_max - _min)
        return new_img, angle(fft2_img), ifft2(new_img)

    def filter_radio(self):
        btnId = self.filterButton.checkedId()  #-2 ~ -10
        
        if btnId == -5:
            self._ILP()

        self.filter.axes.cla()
        self.filter.axes.imshow(self.filtered_img, cmap='gray')
        self.filter.axes.set_axis_off()
        self.filter.draw()

    
    def _ILP(self):
        
        filter = zeros(shape=self.processed_img.shape)
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                if ((i - filter.shape[0] / 2)** 2 + (j - filter.shape[1] / 2)** 2)** .5 <= self.cutoff_box.value():
                    filter[i][j] = 1

        self.filtered_img[:, 0] = fft2(self.processed_img)[:, 0] * filter
        self.filtered_img[:, 1] = fft2(self.processed_img)[:, 1] * filter


    # def process(self):
    #     start_time = time()
        
    #     btnId = self.buttonGroup.checkedId()

    #     if btnId == -7:
    #         print('conv')
    #         self._conv()
    #     elif btnId == -2:
    #         print('median')
    #         self._median()
    #     elif btnId == -3:
    #         print('max')
    #         self._max()
    #     elif btnId == -4:
    #         print('min')
    #         self._min()
    #     elif btnId == -5:
    #         print('gaussian')
    #         self._gaussian()
    #     elif btnId == -6:
    #         print('sobel')
    #         self._sobel()
    #     elif btnId == -8:
    #         print('LoG')
    #         self._LoG()
        
    
app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()