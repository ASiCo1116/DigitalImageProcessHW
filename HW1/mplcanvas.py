import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.pyplot import bar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore
import matplotlib.pyplot as plt

plt.axis('off')

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=75):
        self.fig = Figure(figsize=(1, 1), dpi=150)
        # self.fig.set_facecolor('#222222')
        # self.fig = self.fig.add_subplot(111)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_off()
        super().__init__(self.fig)

    # def event(self, e):
    #     if e.button() == QtCore.Qt.LeftButton:
    #         e.
    
    # def mousePressEvent(self, event):
    #     super(MplCanvas, self).mousePressEvent(event)
    #     if event.button() == QtCore.Qt.LeftButton:
    #         self.emit(QtCore.SIGNAL("mousePressed()"))
