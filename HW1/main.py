import matplotlib
matplotlib.use("Qt5Agg")

from os import getcwd
from sys import argv
from numpy import arange
from pandas import read_csv
from matplotlib.pyplot import bar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, uic, QtCore, QtGui

from MainWindow import Ui_MainWindow

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=1, height=1, dpi=75):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.cwd = getcwd()

        self.readButton.clicked.connect(self.onReadFile)

        

        # self.pixmap1 = QtGui.QPixmap('./test.jpg')
        # self.label = QtWidgets.QLabel(self)
        # self.label.setPixmap(self.pixmap1) #將 image 加入 label
        # self.label.setGeometry(60,60,300,300) # 大小
    
    def onReadFile(self):
        imgChoose, filetype = \
            QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Open image",  
                                    self.cwd,  
                                    "Image Files (*.64)") 

        if imgChoose == "":
            return

        print(imgChoose)
        self.computeHistogram(imgChoose)
        print("file type",filetype)
    
    def computeHistogram(self, file):
        title = file.split(file[:file.rfind('/') + 1])[1]
        img = read_csv(file)
        img = img.to_numpy().flatten()
        strings = ''

        for row in img:
            strings += row

        dic = {}
        for s in strings:
            if not s == '\x1a':
                dic[f'{str(s)}'] = strings.count(str(s))
        
        value = list(dic.values())
        key = list(dic.keys())
        sort = sorted(dic.items(), key = lambda x: x[0])
        
        sc = MplCanvas(self, dpi=100)
        # sc.axes.clf(keep_observers=True)
        sc.axes.bar(arange(len(key)), [x[1] for x in sort], tick_label = list(range(0, 32, 1)))
        sc.axes.set_title(title)

        self.stackedWidget.addWidget(sc)
        self.stackedWidget.setCurrentIndex(2)

        # layout = QtWidgets.QVBoxLayout(self)
        # layout.addWidget(sc)
        # self.histwidget.setLayout(layout)
        
app = QtWidgets.QApplication(argv)

window = MainWindow()
window.show()
app.exec()