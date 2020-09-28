from sys import argv
from os import getcwd
from numpy import arange

from PyQt5.QtCore import (QSize)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, QFileDialog)

from MainWindow_3 import Ui_MainWindow
from functions import myComputing

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.cwd = getcwd()

        self.actionreadImage.triggered.connect(self.onReadFile)
        self.addBtn.clicked.connect(self.add(self.addValue))

    def onReadFile(self):
        imgChoose, _ = QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.64)") 

        if imgChoose == "":
            return
        
        f = myComputing(imgChoose)
        f.to_raw_and_hist()
        self.raw_img = f.raw_img
        
        self.histogram_widget.axes.cla()
        self.histogram_widget.axes.bar(f.key, f.height)
        self.histogram_widget.axes.set_title(f.title)
        self.histogram_widget.axes.set_xticks(ticks = list(range(0, 32, 1)), minor = True)
        self.histogram_widget.draw()

        self.raw_widget.axes.cla()
        self.raw_widget.axes.imshow(self.raw_img, cmap = 'gray')
        self.raw_widget.axes.set_title(f.title)
        self.raw_widget.draw()

        self.processed_widget.axes.cla()
        self.processed_widget.axes.imshow(self.raw_img, cmap = 'gray')
        self.processed_widget.axes.set_title(f.title)
        self.processed_widget.draw()
    
    def add(self, value):
        self.processed_widget.axes.cla()
        self.processed_widget.axes.imshow(self.raw_img + value, cmap = 'gray')
        # self.processed_widget.axes.set_title(f.title)
        self.processed_widget.draw()

        
        
    

app = QApplication(argv)

window = MainWindow()
window.show()
app.exec()