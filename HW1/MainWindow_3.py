# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\mainwindow_3.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1120, 861)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(250, 30))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.raw_widget = MplCanvas(self.centralwidget)
        self.raw_widget.setMaximumSize(QtCore.QSize(550, 16777215))
        self.raw_widget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.raw_widget.setObjectName("raw_widget")
        self.horizontalLayout.addWidget(self.raw_widget)
        self.processed_widget = MplCanvas(self.centralwidget)
        self.processed_widget.setMaximumSize(QtCore.QSize(550, 16777215))
        self.processed_widget.setObjectName("processed_widget")
        self.horizontalLayout.addWidget(self.processed_widget)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.histogram_widget = MplCanvas(self.centralwidget)
        self.histogram_widget.setMaximumSize(QtCore.QSize(550, 16777215))
        self.histogram_widget.setObjectName("histogram_widget")
        self.horizontalLayout_2.addWidget(self.histogram_widget)
        self.widget_4 = QtWidgets.QWidget(self.centralwidget)
        self.widget_4.setMaximumSize(QtCore.QSize(560, 16777215))
        self.widget_4.setObjectName("widget_4")
        self.addValue = QtWidgets.QDoubleSpinBox(self.widget_4)
        self.addValue.setGeometry(QtCore.QRect(90, 10, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.addValue.setFont(font)
        self.addValue.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.addValue.setMinimum(-31.0)
        self.addValue.setMaximum(31.0)
        self.addValue.setSingleStep(1.0)
        self.addValue.setObjectName("addValue")
        self.avgBtn = QtWidgets.QPushButton(self.widget_4)
        self.avgBtn.setGeometry(QtCore.QRect(90, 110, 231, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.avgBtn.setFont(font)
        self.avgBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.avgBtn.setObjectName("avgBtn")
        self.shiftBtn = QtWidgets.QPushButton(self.widget_4)
        self.shiftBtn.setGeometry(QtCore.QRect(90, 160, 231, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.shiftBtn.setFont(font)
        self.shiftBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shiftBtn.setObjectName("shiftBtn")
        self.mulValue = QtWidgets.QDoubleSpinBox(self.widget_4)
        self.mulValue.setGeometry(QtCore.QRect(90, 60, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.mulValue.setFont(font)
        self.mulValue.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.mulValue.setMaximum(10.0)
        self.mulValue.setSingleStep(0.5)
        self.mulValue.setProperty("value", 1.0)
        self.mulValue.setObjectName("mulValue")
        self.label_5 = QtWidgets.QLabel(self.widget_4)
        self.label_5.setGeometry(QtCore.QRect(210, 10, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.widget_4)
        self.label_6.setGeometry(QtCore.QRect(210, 60, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.widget_4)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(330, 50, 171, 131))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.raw2_widget = MplCanvas(self.horizontalLayoutWidget_5)
        self.raw2_widget.setMaximumSize(QtCore.QSize(300, 300))
        self.raw2_widget.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.raw2_widget.setObjectName("raw2_widget")
        self.horizontalLayout_7.addWidget(self.raw2_widget)
        self.label_7 = QtWidgets.QLabel(self.widget_4)
        self.label_7.setGeometry(QtCore.QRect(340, 10, 151, 41))
        self.label_7.setMaximumSize(QtCore.QSize(200, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.widget_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionreadImage = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\C1HW01-2020/500_F_219110502_6EtOxEEa9KQCbholMIqObqKCgArSGfvC.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionreadImage.setIcon(icon)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        self.actionreadImage.setFont(font)
        self.actionreadImage.setVisible(True)
        self.actionreadImage.setObjectName("actionreadImage")
        self.toolBar.addAction(self.actionreadImage)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p>Raw image</p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p>Processed image</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "Histogram"))
        self.label_3.setText(_translate("MainWindow", "Action"))
        self.avgBtn.setText(_translate("MainWindow", "Average"))
        self.shiftBtn.setText(_translate("MainWindow", "Shift"))
        self.label_5.setText(_translate("MainWindow", "Add"))
        self.label_6.setText(_translate("MainWindow", "Multiply"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p>Raw image 2</p></body></html>"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionreadImage.setText(_translate("MainWindow", "readImage"))
        self.actionreadImage.setToolTip(_translate("MainWindow", "read image"))
        self.actionreadImage.setStatusTip(_translate("MainWindow", "Read a image"))
        self.actionreadImage.setShortcut(_translate("MainWindow", "Ctrl+O"))
from mplcanvas import MplCanvas
