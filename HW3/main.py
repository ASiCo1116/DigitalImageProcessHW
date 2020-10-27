# -*- coding: utf-8 -*-

from time import time
from sys import argv
from os import getcwd
from cv2 import imread, split, merge
from numpy import zeros, arange, float32, sum, array, histogram, cumsum, int, around, divmod, floor, ceil, reshape, any, dot, random, median, amax, amin, ones
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
        self.current_size = 3
        self.tableWidget.setColumnCount(self.current_size)
        self.tableWidget.setRowCount(self.current_size)
        self.raw_img = zeros(shape=(500, 500))
        self.processed_img = zeros(shape=(500, 500))
        # QTableWidget.resizeColumnsToContents(self.tableWidget)
        # QTableWidget.resizeRowsToContents(self.tableWidget)

        for r in range(self.current_size):
            for c in range(self.current_size):
                self.tableWidget.setItem(r, c, QTableWidgetItem('1'))
                
        self.raw.fig.canvas.mpl_connect('button_press_event', self.readFile)
        self.spinBox.valueChanged.connect(self.spinBoxValueChanged)
        self.startBtn.clicked.connect(self.process)
        self.randomBtn.clicked.connect(self.random)
        self.resetBtn.clicked.connect(self.reset)
        self.processed.fig.canvas.mpl_connect('button_press_event', self.saveImage)

    def readFile(self, e):
        '''
        Read a picture
        '''
        imgChoose, _ = QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.png *.jpg *.jpeg)") 

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

        self.processed.axes.cla()
        self.processed.axes.imshow(self.processed_img, cmap='gray', vmin=0, vmax=255)
        self.processed.axes.set_axis_off()
        self.processed.draw()
    
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

    def spinBoxValueChanged(self, v):
        self.current_size = v
        self.tableWidget.setColumnCount(v)
        self.tableWidget.setRowCount(v)

        for r in range(v):
            for c in range(v):
                self.tableWidget.setItem(r, c, QTableWidgetItem('1'))

        # QTableWidget.resizeColumnsToContents(self.tableWidget)
        # QTableWidget.resizeRowsToContents(self.tableWidget)

    def reset(self):
        self.processed_img = self.gray_img.copy()
        self.processed.axes.cla()
        self.processed.axes.imshow(self.processed_img, cmap='gray', vmin=0, vmax=255)
        self.processed.axes.set_axis_off()
        self.processed.draw()

    def random(self):
        rd = ((random.rand(self.current_size, self.current_size) - .5) * 20).astype(int)

        for r in range(self.current_size):
            for c in range(self.current_size):
                self.tableWidget.setItem(r, c, QTableWidgetItem(str(rd[r][c])))

    def process(self):
        start_time = time()
        
        btnId = self.buttonGroup.checkedId()

        if btnId == -7:
            print('conv')
            self._conv()
        elif btnId == -2:
            print('median')
            self._median()
        elif btnId == -3:
            print('max')
            self._max()
        elif btnId == -4:
            print('min')
            self._min()
        elif btnId == -5:
            print('gaussian')
            self._gaussian()
        elif btnId == -6:
            print('sobel')
            self._sobel()
        elif btnId == -8:
            print('LoG')
            self._LoG()
        
        self.processed.axes.cla()
        self.processed.axes.imshow(self.processed_img, cmap='gray', vmin=0, vmax=255)
        self.processed.axes.set_axis_off()
        self.processed.draw()

        print(f'process done.')
        print(f'total time: {time()-start_time:.2f}')
    
    def _conv(self):
        self.kernel = \
            zeros(shape=(self.tableWidget.rowCount(), self.tableWidget.columnCount()))
        
        for r in range(self.tableWidget.rowCount()):
            for c in range(self.tableWidget.columnCount()):
                self.kernel[r][c] = float(self.tableWidget.item(r, c).text())
        
        self.processing_img = \
            zeros(shape=(self.processed_img.shape[0] - self.current_size + 1, self.processed_img.shape[1] - self.current_size + 1))
        
        for r in range(self.processed_img.shape[0] - self.current_size):
            for c in range(self.processed_img.shape[1] - self.current_size):
                self.processing_img[r][c] = dot(self.kernel, self.processed_img[r:r+self.current_size, c:c+self.current_size]).sum()
        
        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()

    def _median(self):
        
        self.processing_img = \
            zeros(shape=(self.processed_img.shape[0] - self.maxBox.value() + 1, self.processed_img.shape[1] - self.medianBox.value() + 1))

        for r in range(self.processed_img.shape[0] - self.medianBox.value()):
            for c in range(self.processed_img.shape[1] - self.medianBox.value()):
                self.processing_img[r][c] = median(self.processed_img[r:r+self.medianBox.value(), c:c+self.medianBox.value()])
        
        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()
    
    def _max(self):

        self.processing_img = \
            zeros(shape=(self.processed_img.shape[0] - self.maxBox.value() + 1, self.processed_img.shape[1] - self.maxBox.value() + 1))

        for r in range(self.processed_img.shape[0] - self.maxBox.value()):
            for c in range(self.processed_img.shape[1] - self.maxBox.value()):
                self.processing_img[r][c] = amax(self.processed_img[r:r+self.maxBox.value(), c:c+self.maxBox.value()])
        
        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()
    
    def _min(self):

        self.processing_img = \
            zeros(shape=(self.processed_img.shape[0] - self.minBox.value() + 1, self.processed_img.shape[1] - self.minBox.value() + 1))

        for r in range(self.processed_img.shape[0] - self.minBox.value()):
            for c in range(self.processed_img.shape[1] - self.minBox.value()):
                self.processing_img[r][c] = amin(self.processed_img[r:r+self.minBox.value(), c:c+self.minBox.value()])
        
        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()
    
    def _gaussian(self):
        self.processing_img = gaussian_filter(self.processed_img, sigma = self.gauBox.value())

        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()
    
    def _sobel(self):
        self.processing_img = sobel(self.processed_img)

        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()

    def _LoG(self):

        self.processing_img = gaussian_laplace(self.processed_img, sigma = self.LoGBox.value())

        # self.processing_img = laplace(gaussian_filter(self.processed_img, sigma = self.gauBox.value()))

        self.processing_img[self.processing_img >= 255] = 255
        self.processing_img[self.processing_img <= 0] = 0
        self.processed_img = self.processing_img.copy()

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()