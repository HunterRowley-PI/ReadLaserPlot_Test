# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 14:54:16 2014

@author: mabsh
"""
import time

import datetime
import scipy.fftpack
from scipy.signal import butter, lfilter, freqz
from scipy import stats
from scipy.ndimage import gaussian_filter

import numpy as np

from readlaserplot_ui_5 import Ui_MainWindow
# from hp5508a_PowerPMAC_ui import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
# from matplotlib.widgets import Cursor

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from myFunctions import getIndex, tiltCorrection, raw2TwoD

import os
import io

engUnit = u" (µm)\r\n"
frmt = '{:.6f}'.format
i = 0

title = 'Read Laser Plot 5.3 | April 2, 2020'

class MyMain(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMain, self).__init__(parent)
        self.setupUi(self)
        self.naviToolBar = NavigationToolbar(self.laserPlot, self)
        self.verticalLayout_3.addWidget(self.naviToolBar)
        self.setWindowTitle(title)
        # self.setWindowIcon(QtGui.QIcon('C:\\Users\\mabsh\\Desktop\\trad.png'))
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        self.initLaserPlot()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.constantUpdate)
        self.timeOut = 20
        self.timer.start(self.timeOut)
        self.count = 0
        self.fcounter = 0.0
        self.fcounter1 = 0.0

        self.savePlot.clicked.connect(self.onSavePlot)

        self.readPlotBtn.clicked.connect(self.onReadPlot)

        self.readMultiPlotsBtn.clicked.connect(self.onReadMultiPlots)
        self.lines = []

        self.tickIncrement.valueChanged.connect(self.updatePlot)

        self.newTitle.textChanged.connect(self.updatePlot)
        self.newXLabel.textChanged.connect(self.updatePlot)
        self.newYLabel.textChanged.connect(self.updatePlot)
        self.yScaleMinimum.valueChanged.connect(self.onyScaleMinimum)
        self.yScaleMaximum.valueChanged.connect(self.onyScaleMaximum)
        self.xScaleMinimum.valueChanged.connect(self.onxScaleMinimum)
        self.xScaleMaximum.valueChanged.connect(self.onxScaleMaximum)

        self.yOffset.stateChanged.connect(self.updatePlot)
        self.linearFit.stateChanged.connect(self.updatePlot)
        self.addTilt.valueChanged.connect(self.updatePlot)
        self.addOffset.valueChanged.connect(self.updatePlot)
        self.addXOffset.valueChanged.connect(self.updatePlot)
        self.runningAverageN2.valueChanged.connect(self.updatePlot)

        self.filterType.activated.connect(self.onFilterSelection)
        self.cutoffFrequency.valueChanged.connect(self.updatePlot)
        self.sampleRate.valueChanged.connect(self.updatePlot)
        self.order.valueChanged.connect(self.updatePlot)
        self.thermalCorrection.clicked.connect(self.updatePlot)
        self.thermalCorrectionAmount.valueChanged.connect(self.updatePlot)

        self.histogram.clicked.connect(self.updatePlot)
        self.cpk.clicked.connect(self.updatePlot)
        self.usl.valueChanged.connect(self.onUsl)
        self.usl.setValue(2.0)
        self.lsl.valueChanged.connect(self.updatePlot)
        self.histogram.setVisible(True)

        self.autoScaleX.clicked.connect(self.updatePlot)  # onDataManipulation)
        self.autoScaleY.clicked.connect(self.updatePlot)  # onDataManipulation)
        self.leftTextOff.clicked.connect(self.updatePlot)  # onDataManipulation)
        self.rightTextOff.clicked.connect(self.updatePlot)  # onDataManipulation)
        self.radians.clicked.connect(self.updatePlot)  # onDataManipulation)

        self.xScaleFactor.valueChanged.connect(self.onScaleFactor)
        self.yScaleFactor.valueChanged.connect(self.onScaleFactor)

        self.fftBtn.clicked.connect(self.onFFTButton)
        self.toVelocityBtn.clicked.connect(self.onToVelocity)

        self.overTravel = 0.5
        self.cmdPosition = ""
        self.positionData = ""
        self.positionSampleData = ""
        self.iteration = 0

        self.color = QtGui.QColor(0, 0, 255)
        self.setColor()
        self.pmacCommand = ''
        self.pmacAnswer = ''
        self.stripChartData = []
        self.thisy = np.array([])
        self.averageY = np.array([])
        self.averageY1 = np.array([])
        self.tempPath = ''

        name = ['No Filter', 'Running Average', 'Lowpass Filter', 'Highpass Filter', 'Bandpass Filter',
                'Gaussian Filter']
        self.filterType.addItems(name)
        self.onFilterSelection()

        name = ['New X Label', 'Position (mm-derived from Time)', 'Position (mm)', 'Position (deg)',
                'Position (radians)', 'Position (arcsec)', 'Frequency (Hz)', 'Current (A)', 'Time (sec)', 'Time (min)',
                'Time (hour)', 'Diagonal Distance (mm)']
        self.xLabels.addItems(name)
        self.xLabels.currentIndexChanged.connect(self.onxLabels)

        name = ['New Y Label', 'Error (µm)', 'Error (arcsec)', 'Error (mrad)', 'Velocity (mm/sec)',
                'Velocity (deg/sec)', 'Displacement (µm)', 'Displacement (arcsec)', 'Position (deg)',
                'Position (arcsec)', 'Amplitude (dB)', 'Force (N)', 'Jitter (nm)', 'Current (Amps)']
        self.yLabels.addItems(name)
        self.yLabels.currentIndexChanged.connect(self.onyLabels)

        self.polarPlot.clicked.connect(self.onPolarPlot)
        name = ['0.0001', '0.001', '0.01', '0.1', '1.0', '10.0', '100.0']
        self.singleStepSize_XOffset.addItems(name)
        self.singleStepSize_Tilt.addItems(name)
        self.singleStepSize_YOffset.addItems(name)

        self.singleStepSize_XOffset.currentIndexChanged.connect(self.onsingleStepSize_XOffset)
        self.singleStepSize_Tilt.currentIndexChanged.connect(self.onsingleStepSize_Tilt)
        self.singleStepSize_YOffset.currentIndexChanged.connect(self.onsingleStepSize_YOffset)
        self.path = self.readFilePath()

        self.plot0Xroll.valueChanged.connect(self.onYarray2D)
        self.plot1Xroll.valueChanged.connect(self.onYarray2D)
        self.plot2Xroll.valueChanged.connect(self.onYarray2D)
        self.plot3Xroll.valueChanged.connect(self.onYarray2D)
        self.plot0Yoffset.valueChanged.connect(self.onYarray2D)
        self.plot1Yoffset.valueChanged.connect(self.onYarray2D)
        self.plot2Yoffset.valueChanged.connect(self.onYarray2D)
        self.plot3Yoffset.valueChanged.connect(self.onYarray2D)
        self.yarray2D = []
        self.saveCompTable.clicked.connect(self.onSaveCompTable)
        self.compTableRoll.valueChanged.connect(self.onYarray2D)
        self.compYoffset.valueChanged.connect(self.onYarray2D)
        self.compName.setText('MON 1')
        self.compScaleFactor.setValue(-51.28)
        self.reverse.setChecked(True)
        self.reverse.clicked.connect(self.onYarray2D)
        self.compScaleFactor.valueChanged.connect(self.onYarray2D)
        self.gaussian_filter_sigma.valueChanged.connect(self.onYarray2D)
        self.displayCompPlot.clicked.connect(self.onYarray2D)
        self.resetAll.clicked.connect(self.onResetAll)

        self.selectDirectory.clicked.connect(self.onSelectDirectory)
        self.originalData = 0.0

    def onSelectDirectory(self):
        self.path, _ = QFileDialog.getOpenFileName(self, 'Open file', self.path)

        if self.path:
            print(os.path())

    def onResetAll(self):
        self.resetAll.setChecked(False)
        self.plot0Xroll.setValue(0)
        self.plot1Xroll.setValue(0)
        self.plot2Xroll.setValue(0)
        self.plot3Xroll.setValue(0)
        self.plot0Yoffset.setValue(0)
        self.plot1Yoffset.setValue(0)
        self.plot2Yoffset.setValue(0)
        self.plot3Yoffset.setValue(0)

    def onUsl(self, value):
        self.lsl.setValue(-value)

    def saveFilePath(self):
        fin = open('C:\\Python_Projects\\HP5508A Laser System\\ReadLaserFilePath.txt', 'w')
        fin.write("File Path: " + self.path)
        fin.close()

    def readFilePath(self):
        try:
            fin = open('C:\\Python_Projects\\HP5508A Laser System\\ReadLaserFilePath.txt', 'r')
            for i in fin:
                i = i.strip();
                j = i[i.find(":") + 1:].strip()
                if i.startswith('File Path: '):
                    self.path = j
            fin.close()
            return j
        except:
            print('no filepath.txt found')

    def onsingleStepSize_XOffset(self):
        self.addXOffset.setSingleStep(float(self.singleStepSize_XOffset.currentText()))

    def onsingleStepSize_Tilt(self):
        self.addTilt.setSingleStep(float(self.singleStepSize_Tilt.currentText()))

    def onsingleStepSize_YOffset(self):
        self.addOffset.setSingleStep(float(self.singleStepSize_YOffset.currentText()))

    def onyScaleMinimum(self, value):
        self.autoScaleY.setChecked(False)
        self.yScaleMaximum.setValue(-value)
        self.updatePlot()

    def onyScaleMaximum(self):
        self.autoScaleY.setChecked(False)
        self.updatePlot()

    def onxScaleMinimum(self):
        self.autoScaleX.setChecked(False)
        self.updatePlot()

    def onxScaleMaximum(self):
        self.autoScaleX.setChecked(False)
        self.updatePlot()

    def onPolarPlot(self):
        twoDdata = np.reshape(self.yarray, (20, 61))
        avg = []
        for i in np.transpose(twoDdata):
            avg.append(np.mean(i))
        linspace = np.linspace(int(avg[0]), int(avg[-1]), 61)
        print(np.size(linspace))
        self.initLaserPlot()

        for i in (twoDdata - linspace):
            self.axes.plot(linspace, i, 'o-')  # , color=(1, 0, 1))

        self.axes.set_title('PEG Repeatability')
        self.axes.set_xlabel('Target Position (µm)')
        # self.newXLabel.setText('Position (µm)')
        self.axes.set_ylabel('Error (µm)')
        # self.newYLabel.setText('Error (µm)')
        self.laserPlot.adjustSize()
        self.laserPlot.draw()

    def onxLabels(self, value):
        self.newXLabel.setText(self.xLabels.currentText())

    def onyLabels(self, value):
        self.newYLabel.setText(self.yLabels.currentText())

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))

    def onselect(self):
        # set useblit True on gtkagg for enhanced performance
        indmin, indmax = np.searchsorted(self.xarray, (self.indexStart.value(), self.indexStop.value()))
        indmax = min(len(self.xarray) - 1, indmax)
        self.thisx = list(self.xarray[indmin:indmax])
        self.thisy = list(self.yarray[indmin:indmax])
        if self.runningAverageN.value() > 0:
            self.thisy = np.convolve(self.thisy,
                                     np.ones((self.runningAverageN.value(),)) / self.runningAverageN.value())[
                         (self.runningAverageN.value() - 1):]
            if self.runningAverageN.value() > 0:
                self.thisx = self.thisx[:-(self.runningAverageN.value())]
                self.thisy = self.thisy[:-(self.runningAverageN.value())]

        if self.endPointsTiltFit.checkState() == 2:
            j = 0
            k = float(len(self.thisy) - 1)
            tempDataArray = ''
            for i in self.thisy:
                tempDataArray = tempDataArray + str(float(i) - float(self.thisy[-1]) / k * float(j)) + '\n'
                j = j + 1
            self.thisy = map(float, tempDataArray.strip().split('\n'))
        self.review3axis.cla()
        self.review3axis.plot(self.thisx, self.thisy, '-')
        self.review3.draw()

    def plotDataPoints(self):
        self.createCompTable.setEnabled(False)
        self.compTableNo.setEnabled(False)

        xstr = (self.timeData)
        ystr = (self.positionData)
        x = map(float, xstr.split())
        y = map(float, ystr.split())
        if self.laserSetupMode == 1:
            engUnit = u" (µm)\r\n"
            frmt = '{:.4f}'.format
            self.yarray = np.array(y) * 1000  # convert to µm
        else:
            engUnit = u" (arcsec)\r\n"
            frmt = '{:.2f}'.format
            self.yarray = np.array(y) * 1  # no scale change
        self.xarray = np.array(x)  # convert List to xArray
        self.xarray = self.xarray - min(self.xarray)

        if self.zmi1000.isChecked() == True:
            if self.positionMode.text() == 'Time Mode':
                self.yarray = self.yarray * self.zmiResolution  # convert List to yArray
                self.yarray = self.yarray - self.yarray[0]
                q = self.xarray[1] - self.xarray[0]
                for i in range(0, np.size(x)):
                    self.xarray[i] = float(i) * q
                self.xarray = self.xarray * 1.5625e-8

                self.createCompTable.setEnabled(True)
                self.compTableNo.setEnabled(True)

        self.statisticAnalysis.setText("Minimum: " + str(frmt(min(self.yarray))) + engUnit +
                                       "Maximum: " + str(frmt(max(self.yarray))) + engUnit +
                                       "Peak to Peak: " + str(frmt(max(self.yarray) - min(self.yarray))) + engUnit +
                                       "Average: " + str(frmt(np.mean(self.yarray))) + engUnit +
                                       "RMS: " + str(frmt(np.sqrt(np.mean(np.square(self.yarray))))) + engUnit +
                                       "No. of Samples: " + str(len(self.yarray)))
        self.axes.cla()
        self.axes.grid(visible=True)
        self.axes.set_title(self.comments.toPlainText().strip())

        if self.positionMode.text() == "Position Mode":
            self.axes.set_xlabel('Position (mm)')
            if self.laserDisplacement.title() == "Laser Displacement (arcsec)":
                self.axes.set_ylabel(u'Error (arcsec)')
            else:
                self.axes.set_ylabel(u'Error (µm)')

            # TODO:
            self.axes.plot(self.xarray, self.yarray, 'h:')

        else:
            self.axes.set_xlabel('Time (sec)')
            self.axes.set_ylabel('Error ' + engUnit.strip())
            self.axes.plot(self.xarray, self.yarray, '-')
        self.laserPlot.adjustSize()
        self.laserPlot.draw()
        self.tabControl.setCurrentIndex(1)
        if self.positionMode.text() == "Time Mode":
            self.setScaleValue()
            self.indexStart.setValue(self.xScaleMinimum.value())
            self.indexStop.setValue(self.xScaleMaximum.value())

    def initLaserPlot(self):
        dpi = 100
        self.laserPlot.figure = Figure((7.0, 4.5), dpi=dpi, facecolor=(1, 1, 1), edgecolor=(1, 1, 1))
        self.axes = self.laserPlot.figure.add_subplot(111, facecolor='#FFFFFF')

        self.axes.grid(visible=True, which='both', color='gray', linestyle=':')  # color='k'
        self.laserPlot.canvas = Canvas(self.laserPlot.figure)
        self.laserPlot.mpl_connect('button_press_event', self.onButtonPress)
        self.laserPlot.mpl_connect('button_release_event', self.onButtonRelease)

        if self.xticks.isChecked() == True:
            self.majorLocator = MultipleLocator(self.tickIncrement.value())
            self.axes.xaxis.set_major_locator(self.majorLocator)

    def onButtonRelease(self, event):
        if event.button == event.button.RIGHT:
            x, y = event.xdata, event.ydata
            t = "{0:.8e}".format(self.originalData)
            r = "{0:.8e}".format(y)
            self.statisticAnalysis.setText(self.statisticAnalysis.toPlainText().replace(t, r))
            self.updatePlot()

    def onButtonPress(self, event):
        if event.button == event.button.RIGHT:
            try:
                x, y = event.xdata, event.ydata
                twoD, _ = raw2TwoD(self.statisticAnalysis.toPlainText())
                row = []
                for j in twoD[0]:
                    row.append(abs(j - x))
                rowIndex = np.argmin(np.array(row))
                col = []
                for k in np.transpose(twoD)[rowIndex][1:7]:
                    col.append(abs(k - y))
                colIndex = np.argmin(np.array(col))
                self.originalData = np.transpose(twoD)[rowIndex][1:7][colIndex]
                # if event.button == 2:
                #     self.xOffset.setValue(self.xOffset.value() - x)
                # else:
                #     # print(event)
                #     k = 0
                #     if np.size(self.xarray) > 0:
                #         for i in self.xarray:
                #             if i >= x:
                #                 break
                #             k += 1
                #         if event.button == 1 and event.dblclick:
                #             self.marker1location.setValue(k)
                #             self.addMarkerToLeft.setChecked(True)
                #         if event.button == 3 and event.dblclick:
                #             self.marker2location.setValue(k)
                #             self.addMarkerToLeft.setChecked(True)
            except:
                None


    def onScaleFactor(self):
        self.autoScaleX.setChecked(True)
        self.autoScaleY.setChecked(True)
        self.updatePlot()

    def onReadPlot(self):
        self.xLabels.currentIndexChanged.disconnect(self.onxLabels)
        self.xLabels.setCurrentIndex(0)
        self.xLabels.currentIndexChanged.connect(self.onxLabels)
        file_choices = "*.txt | *.dat | *.rta"  # | *.rtl"
        self.path, _ = QFileDialog.getOpenFileName(self, 'Open file', self.path, file_choices)

        if self.path:
            self.statusBar.showMessage('Open %s' % self.path)
            fin = open(self.path)
            self.statisticAnalysis.setText(fin.read().replace('Ã‚', ''))
            fin.close()
            self.updatePlot()
            self.newTitle.setText('')
            self.saveFilePath()
        self.xScaleFactor.setValue(1.0)
        self.yScaleFactor.setValue(1.0)

    def updatePlot(self):
        rad = self.radians.isChecked()
        self.initLaserPlot()
        data = self.statisticAnalysis.toPlainText().replace(u'Â', '')
        if data.find('Renishaw') > 0:  # Data from Renishaw Laser System
            if self.newTitle.toPlainText() == '':
                self.ttl = '[Theta Axis Accuracy/Repeatability]'
            else:
                self.ttl = self.newTitle.toPlainText()

            x = data[data.find('Targets :') + len('Targets :'): data.find('Flags')].strip().replace('  ', '\n')
            self.woNsn = '(' + data[data.find('Machine: ') + len('Machine: '): data.find('Serial No:')].strip().replace(
                '_', ', ') + '_' + data[data.find('Serial No:') + len('Serial No:'): data.find('Date:')].strip() + ')'
            self.xarray = np.array([*map(float, x.split('\n'))])
            y = data[data.find('Run Target Data:') + len('Run Target Data:'): data.find('EQUIPMENT::')].strip()
            yy = []
            scalefactor = 0.206264806247
            if rad:
                scalefactor = 180.0 / np.pi * 3600.0 / 1000.0 * 4.84813681109536
            else:
                scalefactor = 180.0 / np.pi * 3600.0 / 1000.0  # * 4.84813681109536
            for i in y.split('\n'):
                yy.append(float(i.split('    ')[-1]) * scalefactor)
            YY = np.array(yy)
            self.yarray = YY[:int(len(yy) / 2)]
            # shift in Y scale to display +/-
            shift = ((np.max(self.yarray) + np.min(self.yarray)) / 2)
            yarray0 = tiltCorrection(self.yarray - shift) + self.addOffset.value()

            # end of shift in Y
            start = self.yarray[0]
            self.axes.plot(self.xarray, yarray0, '-o')  # , color=(1, 0, 1))
            self.yarray = YY[int(len(yy) / 2):][::-1]
            # thermal drift correction
            end = self.yarray[0]
            if start != end:
                drift = np.arange(start, end, (end - start) / np.size(self.yarray))
                self.yarray = self.yarray - drift[::-1]
            # end of thermal drift correction
            # shift in Y scale to display +/-
            shift = ((np.max(self.yarray) + np.min(self.yarray)) / 2)
            yarray1 = tiltCorrection(self.yarray - shift) + self.addOffset.value()
            # end of shift in Y
            self.axes.plot(self.xarray, yarray1, '-o')  # , color=(1, 0, 1))
            self.axes.set_title(self.ttl + '\n' + self.woNsn)
            self.axes.set_xlabel('Target (degrees)')
            if rad:
                self.axes.set_ylabel('Error (µrad)')
                unit = 'µrad'
            else:
                self.axes.set_ylabel('Error (arc-sec)')
                unit = 'arcsec'
            self.majorLocator = MultipleLocator(60)

            y2DArray = np.vstack((yarray0, yarray1))
            righttext = 'Statistical Analysis:\n' + self.getStatistics(unit, y2DArray)
            self.laserPlot.figure.text(0.66, 0.14, righttext, color='#000080', size='small', multialignment='right')

            self.axes.xaxis.set_major_locator(self.majorLocator)
            self.axes.set_xlim(0, 360)
            self.axes.set_ylim(self.yScaleMinimum.value(), self.yScaleMaximum.value())
            self.laserPlot.adjustSize()
            self.laserPlot.draw()
            # desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'temp.png')
            # self.laserPlot.figure.savefig(desktop, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            #                               papertype=None, format=None, transparent=False, bbox_inches=None,
            #                               pad_inches=0.1, metadata=None)

            buf = io.BytesIO()
            self.laserPlot.figure.savefig(buf)
            QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
            buf.close()

        else:
            self.ttl = data[data.find('Comments: ') + len('Comments: '):data.find(']') + 1]
            lt = data[data.find('Environmental Data'):].strip().replace(u'Â', '').split('\n')
            lefttext = ''
            for i in lt:
                # if i.startswith('Hum') or i.startswith('VOL') or i.startswith('MT3'):
                # if i.startswith('MT2') or i.startswith('VOL') or i.startswith('MT3'):
                # if i.startswith('VOL') or i.startswith('MT3') or i.find('99999') > 0 or i.find('20.0') > 0 or i.startswith('ZMI2000'):
                if i.startswith('MT2') or i.startswith('VOL') or i.startswith('MT3') or i.startswith('ZMI2000'):
                    # if i.startswith('VOL') or i.startswith('MT3') or i.find('99999') > 0 or i.find('20.0') > 0:
                    lefttext = lefttext
                else:
                    try:
                        if i.startswith('Expansion Coefficient'):
                            i = i[:i.find(':') + 2] + (str('{:.2f}'.format(float(i[i.find(':') + 1:].strip()))))
                        else:
                            i = i[:i.find(':') + 2] + (str('{:.2f}'.format(float(i[i.find(':') + 1:].strip()))))
                    except:
                        None
                    lefttext = lefttext + i + '\n'
            lefttext = lefttext.strip()
            righttext = data[data.find('Statistical Analysis:'):data.find('Raw Data:')].strip().replace(u'Â', '')

            # self.woNsn =('(WO:'+ data[data.find('Work Order Number: ')+len('Work Order Number: ') :data.find('\n',data.find('Work Order Number: ') )]  +
            #            ', SN:'+ data[data.find('Serial Number: ')+len('Serial Number: ') :data.find('\n', data.find('Serial Number: '))] + ')')

            self.woNsn = ('(' + data[data.find('Work Order Number: ') + len('Work Order Number: '):data.find('\n',
                                                                                                             data.find(
                                                                                                                 'Work Order Number: '))] +
                          ', ' + data[data.find('Serial Number: ') + len('Serial Number: '):data.find('\n', data.find(
                        'Serial Number: '))] + ')')
            # self.ttl = self.ttl.replace('X Axis', data[data.find('Serial Number: ')+len('Serial Number: ')+10 :data.find('\n', data.find('Serial Number: '))])
            if data.find('(arcsec)') > 0:
                yl = 'arcsec'
            else:
                if data.find('mrad') > 0:
                    yl = u'mrad'
                else:
                    yl = u'µm'

            if data.find('Time	Position') > 0:  # Time Data
                data = data[data.find('Time	Position') + len('Time Position') + 1:data.find('Environmental Data')]
                dataarray = np.fromstring(data, dtype=float, sep='\n')
                self.xarray = (np.reshape(dataarray, (-1, 2)))[:, 0] * self.xScaleFactor.value()
                self.xarray = self.xarray - self.xarray[0]
                self.yarray = (np.reshape(dataarray, (-1, 2)))[:, 1] * self.yScaleFactor.value()
                xlabel = u'Time (sec)'
                if self.ttl.find('Minimum Incremental Move') > 0:
                    ylabel = u'Displacement (' + yl + ')'
                else:
                    ylabel = u'Error (' + yl + ')'
                if self.filterType.currentText() == 'Running Average':
                    if self.runningAverageN2.value() > 0:
                        self.yarray = np.convolve(self.yarray, np.ones(
                            (self.runningAverageN2.value(),)) / self.runningAverageN2.value())[
                                      (self.runningAverageN2.value() - 1):]
                        if self.runningAverageN2.value() > 0:
                            self.xarray = self.xarray[:-(self.runningAverageN2.value())]
                            self.yarray = self.yarray[:-(self.runningAverageN2.value())]

                if self.yOffset.checkState() == 'Lowpass Filter':
                    self.yarray = self.acCoupling(self.yarray)
                if self.linearFit.checkState() == 2:
                    self.yarray = self.yarray - np.linspace(self.yarray[0], self.yarray[-1], np.size(self.yarray))
                self.yarray = self.yarray + np.linspace(0, self.addTilt.value(),
                                                        np.size(self.yarray)) + self.addOffset.value()

                if self.filterType.currentText() == 'Highpass Filter':
                    nyq = 0.5 * self.sampleRate.value()
                    normal_cutoff = self.cutoffFrequency.value() / nyq
                    b, a = butter(self.order.value(), normal_cutoff, btype='high', analog=False)
                    self.yarray = lfilter(b, a, self.yarray)
                if self.filterType.currentText() == 'Bandpass Filter':
                    nyq = 0.5 * self.sampleRate.value()
                    normal_cutoff = self.cutoffFrequency.value() / nyq
                    b, a = butter(self.order.value(), normal_cutoff, btype='high', analog=False)
                    self.yarray = lfilter(b, a, self.yarray)

                if self.filterType.currentText() == 'Gaussian Filter':
                    self.yarray = gaussian_filter(self.yarray, sigma=self.cutoffFrequency.value(), order=0)

                self.axes.plot(np.array(self.xarray) + self.addXOffset.value(), self.yarray, '-')
            else:
                pdata = data[data.find('Target Position'): data.find('Environmental Data:')].strip().split('\n')
                a = []
                rawdata = []
                for i in pdata:
                    b = []
                    for j in i.split('\t'):
                        b.append(j)
                    a.append(b)
                if data.find('Bi-directional Repeatability') > 0 or data.find('Uni-directional Repeatability') > 0:
                    NoOfRow, NoOfColumn = np.shape(np.array(a))
                    xlabel = u'Number of Sample'
                    ylabel = u'Error (' + yl + ')'
                    if NoOfRow == 4:
                        repData = [*map(float, (a[2][1:-5]))]
                    else:
                        repData = [*map(float, (a[1][1:-5]))]
                    repData = repData - np.mean(repData) + self.addOffset.value()
                    self.axes.plot(repData[int(self.addXOffset.value()):], 'H')
                    if righttext.find('arcsec') > 0:
                        engUnit = u" (arcsec)\n"
                    else:
                        if righttext.find('mrad') > 0:
                            engUnit = u" (mrad)\n"
                        else:
                            engUnit = u" (µm)\n"

                    frmt = '{:.4f}'.format
                    righttext = ('Statistical Analysis:\n' +
                                 "Minimum: " + str(frmt(np.min(repData))) + engUnit +
                                 "Maximum: " + str(frmt(np.max(repData))) + engUnit +
                                 u"Repeatability: ±" + str(frmt((np.max(repData) - np.min(repData)) / 2)) + engUnit +
                                 "Sigma: " + str(frmt(np.std(repData))) + engUnit)

                else:
                    if data.find('arcsec') > 0:
                        xlabel = u'Position (deg)'
                    else:
                        xlabel = u'Position (mm)'
                    ylabel = u'Error (' + yl + ')'
                    b = np.transpose(a)
                    k = 0
                    lbl = []
                    # if self.thermalCorrection.checkState() == 2:
                    b, righttext = self.rearrangText(b, yl, self.ttl)
                    for i in b:
                        name = i[0]
                        d = []
                        for j in (i[1:]):
                            if k == 0:
                                d.append(float(j))  # * self.xScaleFactor.value())
                            else:
                                d.append(float(j))  # * self.yScaleFactor.value())

                        if k == 0:
                            self.xarray = np.array(d)  # + self.addXOffset.value()
                        else:
                            #                            if self.yOffset.checkState() == 2:
                            #                                d = d + (np.max(d)-np.min(d))/2
                            #                            if self.linearFit.checkState() == 2:
                            #                                d = d - np.linspace(d[0],d[-1],np.size(d))
                            self.yarray = d  # + np.linspace(0, self.addTilt.value(), np.size(d)) + self.addOffset.value()
                            if name.startswith('RUN'):
                                if self.histogram.isChecked() == False or self.histogram.isChecked() == True:
                                    if k % 2 == 1:
                                        self.axes.plot(self.xarray, self.yarray, ':+', linewidth=0.5, markerSize=3)
                                    else:
                                        self.axes.plot(self.xarray, self.yarray, ':x', linewidth=0.5, markerSize=3)
                                rawdata.append(self.yarray)
                                lbl.append(str(name))

                            if name.startswith('Average'):
                                self.axes.plot(self.xarray, self.yarray, '-o', color='#000000', markerSize=3)
                                lbl.append(str(name))
                        k += 1
            #            if  self.histogram.isChecked() == True:
            #                self.initLaserPlot()
            #                rd = np.array(rawdata)
            #                x, y = np.shape(rd)
            #                ave = np.average(rd)
            #                sigma = np.std(rd)
            #                bins = np.arange(self.lsl.value(), self.usl.value(), (self.usl.value()-self.lsl.value())/100.0)
            #                histogram = np.histogram(np.reshape(rd, x*y), bins = bins, normed = True)[0]
            #                bins = 0.5 * (bins[1:]+ bins[:-1])
            #                print (np.size(bins))
            #
            #                b = stats.norm.pdf(bins)
            #                self.axes.plot(bins, histogram, '-')
            #                self.axes.plot(bins, b)
            #                USL = self.usl.value()
            #                LSL = self.lsl.value()
            #                CpU = (USL-ave)/(3*sigma)
            #                CpL = (ave - LSL)/(3*sigma)
            #                Cpk = np.min([CpU, CpL])#Process capability index
            #                print ("CpU = %.3f" %CpU, "CpL = %.3f" %CpL, "Cpk = %.3f" %Cpk,)
            #                xlabel = ylabel
            #                ylabel = '(Histogram)'

            if self.leftTextOff.checkState() != 2:
                self.laserPlot.figure.text(0.15, 0.14, lefttext, color='#000080', size='small')
            if self.rightTextOff.checkState() != 2:
                self.laserPlot.figure.text(0.66, 0.14, righttext, color='#000080', size='small', multialignment='right')

            self.axes.set_title(self.ttl.replace('um Step', u'µm Step').replace(' deg', '°').replace('A Axis',
                                                                                                     u'Focus Axis') + '\n' + self.woNsn)
            self.axes.set_xlabel(xlabel)

            self.axes.set_ylabel(ylabel)

            # self.axes.set_ylabel('Displacement (arcsec)')
            # TODO:

            if self.newTitle.toPlainText().strip() != '':
                self.axes.set_title(
                    self.newTitle.toPlainText().strip().replace('um Step', u'µm Step').replace(' deg', '°'))
            else:
                self.newTitle.setText(self.ttl.replace(' deg', '°') + '\n' + self.woNsn)

            if self.newXLabel.text().strip() != '':
                self.axes.set_xlabel(self.newXLabel.text().strip())
            if self.newYLabel.text().strip() != '':
                self.axes.set_ylabel(self.newYLabel.text().strip())

            if self.autoScaleX.isChecked() == False:
                self.axes.set_xlim(self.xScaleMinimum.value(), self.xScaleMaximum.value())
            if self.autoScaleY.isChecked() == False:
                self.axes.set_ylim(self.yScaleMinimum.value(), self.yScaleMaximum.value())

            self.laserPlot.adjustSize()
            self.laserPlot.draw()
            self.setScaleValue()
            # self.newXLabel.setText(xlabel)
            # self.newYLabel.setText(ylabel)
            if data.find('Time	Position') > 0:  # Time Data
                self.indexStart.setValue(self.xScaleMinimum.value())
                self.indexStop.setValue(self.xScaleMaximum.value())

            # desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'temp.png')
            # self.laserPlot.figure.savefig(desktop, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            #                               papertype=None, format=None, transparent=False, bbox_inches=None,
            #                               pad_inches=0.1, metadata=None)

            buf = io.BytesIO()
            self.laserPlot.figure.savefig(buf)
            QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
            buf.close()

    def getStatistics(self, name, array):
        y2dnew = np.transpose(array)
        meanAtPoint = []
        maxAtPoint = []
        minAtPoint = []
        stdAtPoint = []
        ptpAtPoint = []
        for i in y2dnew:
            meanAtPoint = np.append(meanAtPoint, np.mean(i))
            maxAtPoint = np.append(maxAtPoint, np.max(i))
            minAtPoint = np.append(minAtPoint, np.min(i))
            stdAtPoint = np.append(stdAtPoint, np.std(i))
            ptpAtPoint = np.append(ptpAtPoint, np.max(i) - np.min(i))
        maximumError = np.max(maxAtPoint)
        minimumError = np.min(minAtPoint)
        averageError = np.mean(meanAtPoint)
        repeatability = np.max(ptpAtPoint)
        oneSigma = np.max(stdAtPoint)
        if name == "µrad":
            engUnit = u" (µrad)\n"
            frmt = '{:.1f}'.format
        else:
            engUnit = u" (arcsec)\n"
            frmt = '{:.2f}'.format
        t = ("Minimum: " + str(frmt(minimumError)) + engUnit +
             "Maximum: " + str(frmt(maximumError)) + engUnit +
             u"Accuracy: ±" + str(frmt((maximumError - minimumError) / 2)) + engUnit +
             u"Repeatability: ±" + str(frmt(repeatability / 2)) + engUnit +
             "Average: " + str(frmt(averageError)) + engUnit +
             "Sigma: " + str(frmt(oneSigma)) + engUnit)
        return t

    def rearrangText(self, twoDdata, yAxisLabel, plotTitle):  # 2D data,  y-axis label
        b = twoDdata
        yl = yAxisLabel
        c, r = (np.shape(b))
        header = (b[:, 0])
        rawdata = (b[1:c - 5, 1:])
        targetP = (b[0, 1:])
        targetPf = np.array([*map(float, targetP)]) * self.xScaleFactor.value() + self.addXOffset.value()
        data = []
        j = 0
        for i in rawdata:
            temp = np.array([*map(float, i)]) * self.yScaleFactor.value()
            if self.thermalCorrection.checkState() == 2:
                if j % 2 == 0:
                    temp = temp + np.linspace(0, self.thermalCorrectionAmount.value(), np.size(temp))
                else:
                    temp = temp - np.linspace(0, self.thermalCorrectionAmount.value(), np.size(temp))
            if self.yOffset.checkState() == 2:
                temp = self.acCoupling(temp)
            if self.linearFit.checkState() == 2:
                temp = temp - np.linspace(temp[0], temp[-1], np.size(temp))

            temp = temp + np.linspace(0, self.addTilt.value(), np.size(temp)) + self.addOffset.value()
            data.append(temp)
            j += 1
        nparrayData = np.array(data)

        if self.histogram.isChecked() == True:
            meanRemoved = []
            for i in np.transpose(nparrayData):
                meanRemoved.append(i - np.mean(i))
            nparrayData = (np.transpose(meanRemoved))
        Max = []
        Min = []
        Average = []
        PTP = []
        STDEV = []
        rep = []
        for i in np.transpose(nparrayData):
            Max.append(str(np.max(i)))
            Min.append(str(np.min(i)))
            Average.append(str(np.mean(i)))
            PTP.append(str(np.max(i) - np.min(i)))
            STDEV.append(str(np.std(i)))
            rep.append(np.max(i) - np.min(i))
        b = np.vstack((header, np.transpose(np.vstack((targetPf, (nparrayData), Max, Min, PTP, Average, STDEV)))))

        engUnit = " (" + yl + ")\n"
        if yl == 'arcsec' and plotTitle.find('Accuracy') == -1:
            frmt = '{:.2f}'.format
            acc = str(frmt((np.max(nparrayData) - np.min(nparrayData)) / 1.0))
        else:
            frmt = '{:.4f}'.format
            if plotTitle.find('Straightness') > 0 or plotTitle.find('Flatness') > 0:
                acc = str(frmt((np.max(nparrayData) - np.min(nparrayData)) / 1.0))
            else:
                acc = "±" + str(frmt((np.max(nparrayData) - np.min(nparrayData)) / 2.0))
        USL = self.usl.value()
        LSL = self.lsl.value()
        sigma = np.std(nparrayData)
        average = np.mean(nparrayData)
        if self.cpk.isChecked():
            righttext = ('Statistical Analysis:\n' +
                         u"Minimum: " + str(frmt(np.min(nparrayData))) + engUnit +
                         u"Maximum: " + str(frmt(np.max(nparrayData))) + engUnit +
                         u"Accuracy: " + acc + engUnit +
                         u"Repeatability: ±" + str(frmt((np.max(rep) / 2.0))) + engUnit +
                         u"Average: " + str(frmt(np.mean([*map(float, Average)]))) + engUnit +
                         #               u"Sigma: " + str(frmt(np.max([*map(float,STDEV)]))) + engUnit +
                         u"Sigma: " + str(frmt(sigma)) + engUnit +
                         u"Cpk: " + str('{:.2f}'.format(np.min(
                        [(USL - average) / (3 * sigma), (np.mean(average - LSL) / (3 * sigma))]))) + " @ ± " + str(
                        self.usl.value()) + " (" + yl + ")")
        else:
            righttext = ('Statistical Analysis:\n' +
                         u"Minimum: " + str(frmt(np.min(nparrayData))) + engUnit +
                         u"Maximum: " + str(frmt(np.max(nparrayData))) + engUnit +
                         u"Accuracy: " + acc + engUnit +
                         u"Repeatability: ±" + str(frmt((np.max(rep) / 2.0))) + engUnit +
                         u"Average: " + str(frmt(np.mean([*map(float, Average)]))) + engUnit +
                         #                u"Sigma: " + str(frmt(np.max([*map(float,STDEV)]))) + engUnit)
                         u"Sigma: " + str(frmt(sigma)) + engUnit)

        return np.transpose(b), righttext.strip()

    def analyzeData(self):
        # Get Position Information from xarray
        if self.comments.toPlainText().find('Uni-directional') == -1:
            self.xarray = self.xarray[0:len(self.xarray) / self.noOfCycles.value() / 2]
            y2d = np.reshape(self.yarray, (
            self.noOfCycles.value() * 2, len(self.yarray) / (self.noOfCycles.value() * 2)))  # convert to 2D
            y2dnew = []
            for i in range(0, int(self.noOfCycles.value())):
                # print i
                trans = y2d[i * 2 + 1]
                trans = trans[::-1]  # reverse array
                # print np.vstack((y2d[i*2],trans))
                y2dnew = np.append(y2dnew, np.vstack((y2d[i * 2], trans)))

            y2dnew = np.reshape(y2dnew, (self.noOfCycles.value() * 2, len(self.yarray) / (self.noOfCycles.value() * 2)))
        else:
            self.xarray = self.xarray[0:len(self.xarray) / self.noOfCycles.value()]
            y2d = np.reshape(self.yarray, (
            self.noOfCycles.value() * 1, len(self.yarray) / (self.noOfCycles.value() * 1)))  # convert to 2D
            y2dnew = []
            for i in range(0, int(self.noOfCycles.value())):
                trans = y2d[i]
                y2dnew = np.append(y2dnew, trans)
            y2dnew = np.reshape(y2dnew, (self.noOfCycles.value() * 1, len(self.yarray) / (self.noOfCycles.value() * 1)))

        clmlbl = ""
        j = 0
        k = 1
        for i in y2dnew:
            l = j % 2
            if l == 1:
                lbl = "RUN" + str(k) + "-"
                k = k + 1
            else:
                lbl = "RUN" + str(k) + "+"
            clmlbl = clmlbl + "\t" + lbl
            j = j + 1

        rawData = np.vstack((self.xarray, y2dnew))

        y2dnew = np.transpose(y2dnew)
        meanAtPoint = [];
        maxAtPoint = [];
        minAtPoint = [];
        stdAtPoint = [];
        ptpAtPoint = []
        for i in y2dnew:
            meanAtPoint = np.append(meanAtPoint, np.mean(i))
            maxAtPoint = np.append(maxAtPoint, np.max(i))
            minAtPoint = np.append(minAtPoint, np.min(i))
            stdAtPoint = np.append(stdAtPoint, np.std(i))
            ptpAtPoint = np.append(ptpAtPoint, np.max(i) - np.min(i))

        # print meanAtPoint, maxAtPoint,  minAtPoint,  stdAtPoint, ptpAtPoint
        maximumError = np.max(maxAtPoint)
        minimumError = np.min(minAtPoint)
        averageError = np.mean(meanAtPoint)
        repeatability = np.max(ptpAtPoint)
        oneSigma = np.max(stdAtPoint)

        rawData = np.vstack((rawData, maxAtPoint, minAtPoint, ptpAtPoint, meanAtPoint, stdAtPoint))

        rawData = np.transpose(rawData)
        e = "Raw Data:\r\nTarget Position" + clmlbl + "\tMax\tMin\tPTP\tAverage\tSTDEV"
        for i in rawData:
            d = str(i).replace("[", "").strip()
            d = d.replace("]", "").strip()
            d = d.replace(" ", "\t")
            d = d.replace("\n", "\t")
            while d.find("\t\t") > 0:
                d = d.replace("\t\t", "\t")
            e = e + "\r\n" + d
        rawData = e
        if self.laserSetupMode == 1:
            engUnit = u" (µm)\r\n"
            frmt = '{:.4f}'.format
        else:
            engUnit = u" (arcsec111)\r\n"
            frmt = '{:.2f}'.format
        if self.laserDisplacement.title() == "Laser Displacement (arcsec)":
            a = u"Accuracy: " + str(frmt((maximumError - minimumError) / 1)) + engUnit
        else:
            a = u"Accuracy: ±" + str(frmt((maximumError - minimumError) / 2)) + engUnit
        self.statisticAnalysis.setText("Minimum: " + str(frmt(minimumError)) + engUnit +
                                       "Maximum: " + str(frmt(maximumError)) + engUnit +
                                       # u"Accuracy: ±" + str(frmt((maximumError - minimumError)/2)) + engUnit +
                                       # u"Accuracy: " + str(frmt((maximumError - minimumError)/1)) + engUnit +
                                       a +
                                       u"Repeatability: ±" + str(frmt(repeatability / 2)) + engUnit +
                                       "Average: " + str(frmt(averageError)) + engUnit +
                                       "Sigma: " + str(frmt(oneSigma)) + engUnit)
        self.statisticAnalysis.append("\r\n" + rawData)
        self.statisticAnalysis.append("\r\n" + "Environmental Data:" + "\r\n"
                                      + self.environmentalData.toPlainText() + "\n"
                                      + 'Expansion Coefficient (ppm): ' + self.expansionCoefficient.text())

        if self.thermalComp.isChecked() == True:
            self.statisticAnalysis.append('Thermal Compensation: On')
        else:
            self.statisticAnalysis.append('Thermal Compensation: Off')
        if self.volComp.isChecked() == True:
            self.statisticAnalysis.append('Velocity of Light Compensation: On')
        else:
            self.statisticAnalysis.append('Velocity of Light Compensation: Off')

        self.createCompTable.setEnabled(True)
        self.compTableNo.setEnabled(True)
        self.setScaleValue()
        self.indexStart.setValue(self.xScaleMinimum.value())
        self.indexStop.setValue(self.xScaleMaximum.value())

    def onReadMultiPlots(self):

        if self.resetPlotBtn.checkState() == 2:
            self.resetPlotBtn.setCheckState(0)
            self.initLaserPlot()
            self.thisy = np.array([])
            self.lines = []
            self.yarray2D = []
        file_choices = "*.txt | *.dat"
        # self.path = unicode(QFileDialog.getOpenFileName(self,
        #                    'Open file', self.path,
        #                    file_choices))
        self.path, _ = QFileDialog.getOpenFileName(self, 'Open file', self.path, file_choices)

        if self.path:
            self.statusBar.showMessage('Open %s' % self.path)
            fin = open(self.path)
            self.statisticAnalysis.setText(fin.read())
            fin.close()
            fileName = self.path.split('/')[-1]
            data = self.statisticAnalysis.toPlainText().replace(u'Â', '')
            self.ttl = data[data.find('Comments: ') + len('Comments: '):data.find(']') + 1]

            if data.find('(arcsec)') > 0:
                yl = 'arcsec'
            else:
                yl = u'µm'
            if data.find('Time	Position') > 0:  # Time Data
                data = data[data.find('Time	Position') + len('Time Position') + 1:data.find('Environmental Data')]
                self.xarray = []
                self.yarray = []
                datalist = data.strip().split('\n')
                for i in datalist:
                    j = i.split('\t')
                    self.xarray.append(float(j[0]) * self.xScaleFactor.value())
                    self.yarray.append(float(j[1]) * self.yScaleFactor.value())
                xlabel = u'Time (sec)'
                ylabel = u'Error (' + yl + ')'
                ya = list(np.array(self.yarray))
                if self.yOffset.checkState() == 2:
                    ya = self.acCoupling(ya)
                if self.linearFit.checkState() == 2:
                    ya = ya - np.linspace(ya[0], ya[-1], np.size(ya))
                ya = ya + np.linspace(0, self.addTilt.value(), np.size(ya)) + self.addOffset.value()
                # self.getAverage(ya)   # average

                p1, = self.axes.plot(np.array(self.xarray) + self.addXOffset.value(), ya, '-', label=fileName)
                self.lines.append(p1)

                self.axes.legend(self.lines, [l.get_label() for l in self.lines], fontsize='x-small', frameon=False)
                # print self.addXOffset.value()
                self.axes.set_title(self.ttl)
                self.axes.set_xlabel(xlabel)
                self.axes.set_ylabel(ylabel)

            elif data.find('Target Position') > 0:  # position based data
                data = self.statisticAnalysis.toPlainText().replace(u'Â', '')
                pdata = data[data.find('Target Position'): data.find('Environmental Data:')].strip().split('\n')
                a = []
                for i in pdata:
                    b = []
                    for j in i.split('\t'):
                        b.append(j)
                    a.append(b)

                b = np.transpose(a)
                k = 0
                # if self.thermalCorrection.checkState() == 2:
                b, righttext = self.rearrangText(b, yl, self.ttl)
                for i in b:
                    name = i[0]
                    d = []
                    for j in (i[1:]):
                        if k == 0:
                            d.append(float(j))  # * self.xScaleFactor.value())
                        else:
                            d.append(float(j))  # * self.yScaleFactor.value())

                    if k == 0:
                        self.xarray = np.array(d)  # + self.addXOffset.value()
                    else:
                        self.yarray = d  # + np.linspace(0, self.addTilt.value(), np.size(d)) + self.addOffset.value()

                        if name.startswith('Average'):
                            p1, = self.axes.plot(self.xarray, self.yarray, '-o', label=fileName, markerSize=5)
                            # lbl.append(str(name))
                            self.lines.append(p1)
                            self.yarray2D.append(d)
                            index = getIndex(self.yarray, np.max(self.yarray))
                            # print("peak at ", np.size(self.yarray)-index)
                            # print(type(d[0]))#-np.linspace(0,((float(d[-1])-fd[0])/np.size(d))))
                    k += 1

                self.axes.legend(self.lines, [l.get_label() for l in self.lines], fontsize='x-small', frameon=False)
                self.axes.set_xlabel('Position (mm)')
                self.axes.set_ylabel('Error (µm)')
                self.axes.set_title(fileName)
                # self.axes.set_ylim(-1,  1)
                # self.axes.set_title('[Y Encoder Cyclical Error_X at -1.00 mm]')
            else:
                pdata = self.statisticAnalysis.toPlainText().replace(u'Â', '').strip()  # Read MON table
                x = []
                y = []
                for i in pdata.split('\n'):
                    z = i.split(' ')
                    x.append(float(z[2]))  # * self.xScaleFactor.value())
                    y.append(float(z[3]))  # * self.yScaleFactor.value())
                self.xarray = np.array(x)  # + self.addXOffset.value()
                self.yarray = np.roll(np.array(y), int(
                    self.addXOffset.value()))  # + np.linspace(0, self.addTilt.value(), np.size(d)) + self.addOffset.value()
                p1, = self.axes.plot(self.xarray, self.yarray, '-o', label=fileName, markerSize=5)
                # lbl.append(str(name))
                self.lines.append(p1)

                self.axes.legend(self.lines, [l.get_label() for l in self.lines], fontsize='x-small', frameon=False)
                self.axes.set_title(fileName)
                self.axes.set_xlabel('Point')
                self.axes.set_ylabel('[LUT]')
                self.majorLocator = MultipleLocator(12.8)
                self.axes.xaxis.set_major_locator(self.majorLocator)
                self.axes.set_ylim(-102.56, 102.56)
                self.axes.set_ylim(-80, 80)

            self.laserPlot.adjustSize()
            self.laserPlot.draw()

    def onYarray2D(self):
        self.initLaserPlot()
        j = 0
        yarray2DNew = []
        for i in self.yarray2D:
            if j == 0:
                yarray2DNew.append(np.roll(i, self.plot0Xroll.value()) - self.plot0Yoffset.value())
            if j == 1:
                yarray2DNew.append(np.roll(i, self.plot1Xroll.value()) - self.plot1Yoffset.value())
            if j == 2:
                yarray2DNew.append(np.roll(i, self.plot2Xroll.value()) - self.plot2Yoffset.value())
            if j == 3:
                yarray2DNew.append(np.roll(i, self.plot3Xroll.value()) - self.plot3Yoffset.value())
            j += 1

        if self.compPrediction.isChecked():
            i = 0
            for yarray, fileName in zip(yarray2DNew, self.lines):
                if i > 1:
                    p1, = self.axes.plot(self.xarray, yarray, '-o', label=fileName.get_label(), markerSize=5)
                else:
                    p1, = self.axes.plot(self.xarray, yarray, '-', label=fileName.get_label(), markerSize=5)
                i += 1
            p1, = self.axes.plot(self.xarray, yarray2DNew[0] - yarray2DNew[1], '-o', label='compPrediction',
                                 markerSize=5)
        else:
            for yarray, fileName in zip(yarray2DNew, self.lines):
                p1, = self.axes.plot(self.xarray, yarray, '-o', label=fileName.get_label(), markerSize=5)

        if self.displayCompPlot.isChecked():
            self.axis2 = self.axes.twinx()
            self.axis2.cla()
            j = 1
            comptable = ''
            averageArray = []
            cmp = []
            for i in np.transpose(np.array(yarray2DNew)):
                averageArray.append(np.mean(i))

            if self.reverse.isChecked():
                averageArray = averageArray[::-1]
            print(averageArray)
            # -------------------
            # gaussian filter
            averageArray = gaussian_filter(averageArray, sigma=self.gaussian_filter_sigma.value(), order=0)
            averageArray = averageArray - np.linspace(averageArray[0], averageArray[-1], len(averageArray))

            for i in np.roll(averageArray, self.compTableRoll.value()):
                a = float(i) * self.compScaleFactor.value() + self.compYoffset.value()
                cmp.append(a)
                compdata = str('{:.0f}'.format(a))
                comptable = comptable + self.compName.text() + ' ' + str(j) + ' ' + compdata + '\n'
                j = j + 1

            self.axis2.plot(self.xarray, np.array(cmp), color='red')
            self.compTableDisplay.setText(comptable.strip())

        self.axes.legend(self.lines, [l.get_label() for l in self.lines], fontsize='x-small', frameon=False)
        self.axes.set_xlabel('Position (mm)')
        self.axes.set_ylabel('Error (µm)')
        self.axes.set_title(fileName)

        self.laserPlot.adjustSize()
        self.laserPlot.draw()

    def onSaveCompTable(self):
        file_choices = "*.txt | *.dat"
        self.path, _ = QFileDialog.getSaveFileName(self, 'Save file', self.path, file_choices)
        if self.path:
            self.statusBar.showMessage('Saved to %s' % self.path)
            f = open(self.path, 'w')
            f.write(self.compTableDisplay.toPlainText())  # .encode('utf-8'))
            f.close()

    def acCoupling(self, yarray):
        midpoint = (np.max(yarray) + np.min(yarray)) / 2
        return yarray - midpoint

    def getAverage(self, ya):
        if len(self.thisy) == len(ya):
            j = 0
            for i in ya:
                self.thisy[j] = (self.thisy[j] + i) / 2
                j += 1
        else:
            self.thisy = ya
        self.review3axis.cla()
        self.review3axis.grid(True)
        self.review3axis.plot(self.thisy, '-')
        self.review3.adjustSize()
        self.review3.draw()
        self.yarray = self.thisy
        # self.xarray = np.linspace(0, 1, len(self.thisy))
        # self.indexStart.setValue(np.min(self.xarray))
        # self.indexStop.setValue(np.max(self.xarray))

    def onPlotFromScript(self):
        self.initLaserPlot()
        text = self.plotScript.toPlainText()
        if text.find('=') > 0:
            text = text.split('\n')
            data = ''
            for i in text:
                data = data + i[i.find('= ') + 1:].replace(';', ',')
            ya = data.split(',')
            yaa = []
            for j in ya:
                if len(j.strip()) > 0:
                    yaa.append(float(j))
            self.axes.plot(yaa, '-')
            self.plotScript.setText(unicode(yaa))
        else:
            text = text.split('\n')
            for i in text:
                ya = i.split(',')
                yaa = []
                for j in ya:
                    if len(j.strip()) > 0:
                        yaa.append(float(j))
                self.axes.plot(yaa, '-')
        self.axes.set_title('From Plot Script')
        self.axes.set_xlabel('')
        self.axes.set_ylabel('')
        self.laserPlot.adjustSize()
        self.laserPlot.draw()

    def setScaleValue(self):
        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()
        # self.xScaleMinimum.setValue(xmin)
        # self.xScaleMaximum.setValue(xmax)
        # self.yScaleMinimum.setValue(ymin)
        # self.yScaleMaximum.setValue(ymax)
        try:
            self.xarraySubset = self.xarray[:]
            self.yarraySubset = self.yarray[:]
        except:
            None

    def onReadData(self):
        file_choices = "*.txt | *.dat"
        self.path, _ = QFileDialog.getOpenFileName(self, 'Open file', self.path, file_choices)
        if self.path:
            if self.tempPath[:-5] != self.path[:-5]:
                self.averageY = np.array([])
                self.review1axis.cla()
            self.tempPath = self.path
            self.saveFilePath()
            self.statusBar.showMessage('Open %s' % self.path)
            fin = open(self.path)
            self.statisticAnalysis.setText(fin.read())
            fin.close()
            self.createCompTable.setEnabled(True)
            self.compTableNo.setEnabled(True)

            data = self.statisticAnalysis.toPlainText()
            if data.find('Time	Position') > 0:  # Time Data
                data = data[data.find('Time	Position') + len('Time Position') + 1:data.find('Environmental Data')]
                self.timeData2 = []
                self.positionData2 = []
                datalist = data.strip().split('\n')
                for i in datalist:
                    j = i.split('\t')
                    self.timeData2.append(float(j[0]))
                    self.positionData2.append(float(j[1]))

                if len(self.averageY) == len(self.positionData2):
                    j = 0
                    for i in self.positionData2:
                        self.averageY[j] = (self.averageY[j] + i) / 2
                        j += 1
                else:
                    self.averageY = np.array(self.positionData2)

                # self.review1axis.cla()
                self.review1axis.grid(visible=True)
                self.review1axis.plot(self.timeData2, self.averageY, '-')
                # self.review1axis.plot(self.timeData2, self.positionData2, ':')
                self.review1.adjustSize()
                self.review1.draw()

    def onReadData2(self):
        file_choices = "*.txt | *.dat"
        self.path, _ = QFileDialog.getOpenFileName(self, 'Open file', self.path, file_choices)
        if self.path:
            if self.tempPath[:-5] != self.path[:-5]:
                self.averageY1 = np.array([])
                self.review2axis.cla()
            self.tempPath = self.path

            self.statusBar.showMessage('Open %s' % self.path)
            fin = open(self.path)
            data = fin.read()
            fin.close()
            if data.find('Time	Position') > 0:  # Time Data
                data = data[data.find('Time	Position') + len('Time Position') + 1:data.find('Environmental Data')]
                self.timeData3 = []
                self.positionData3 = []
                datalist = data.strip().split('\n')
                for i in datalist:
                    j = i.split('\t')
                    self.timeData3.append(float(j[0]))
                    self.positionData3.append(float(j[1]))

                if len(self.averageY1) == len(self.positionData3):
                    j = 0
                    for i in self.positionData3:
                        self.averageY1[j] = (self.averageY1[j] + i) / 2
                        j += 1
                else:
                    self.averageY1 = np.array(self.positionData3)

            # self.review2axis.cla()
            self.review2axis.grid(visible=True)
            self.review2axis.plot(self.timeData3, self.averageY1, '-')
            # self.review2axis.plot(self.timeData3, self.positionData3, ':')
            self.review2.adjustSize()
            self.review2.draw()

    def onGetBidirectional(self):
        self.review1axis.cla()
        if self.midPoint2.value() == 0 and self.trim2.value() == 0:
            thisy21 = list(self.averageY[self.indexStart2.value():self.indexStop2.value()])
            self.review1axis.plot(thisy21, '-')
        else:
            thisy21 = list(self.averageY[self.indexStart2.value():self.midPoint2.value() - self.trim2.value()])
            thisy22 = list(self.averageY[self.midPoint2.value() + self.trim2.value():self.indexStop2.value()][::-1])
            self.review1axis.plot(thisy21, '-')
            self.review1axis.plot(thisy22, '-')
        self.review1.adjustSize()
        self.review1.draw()
        if self.midPoint2.value() == 0 and self.trim2.value() == 0:
            self.average1 = thisy21
        else:
            self.average1 = []
            j = 0
            if len(thisy21) < len(thisy22):
                for i in thisy21:
                    self.average1.append((i + thisy22[j]) / 2)
                    j += 1
            else:
                for i in thisy22:
                    self.average1.append((i + thisy21[j]) / 2)
                    j += 1
        self.yarray = list(self.average1)
        self.xarray = np.linspace(0, 1, len(self.yarray))
        try:
            self.review2axis.cla()
            if self.midPoint3.value() == 0 and self.trim3.value() == 0:
                thisy31 = list(self.averageY1[self.indexStart3.value():self.indexStop3.value()])
                self.review2axis.plot(thisy31, '-')
            else:
                thisy31 = list(self.averageY1[self.indexStart3.value():self.midPoint3.value() - self.trim3.value()])
                thisy32 = list(
                    self.averageY1[self.midPoint3.value() + self.trim3.value():self.indexStop3.value()][::-1])
                self.review2axis.plot(thisy31, '-')
                self.review2axis.plot(thisy32, '-')
            self.review2.adjustSize()
            self.review2.draw()
            if self.midPoint3.value() == 0 and self.trim3.value() == 0:
                self.average2 = thisy31
            else:
                self.average2 = []
                j = 0
                if len(thisy31) < len(thisy32):
                    for i in thisy31:
                        self.average2.append((i + thisy32[j]) / 2)
                        j += 1
                else:
                    for i in thisy32:
                        self.average2.append((i + thisy31[j]) / 2)
                        j += 1
            self.yarray = list(self.average2)
            self.xarray = np.linspace(0, 1, len(self.yarray))
        except:
            pass  # "no data yet"

        try:
            thisy2 = list(self.average1)
            thisy3 = list(self.average2)
            l = thisy2[-1]
            for i in thisy3:
                thisy2.append(i - thisy3[0] + l)

            self.review3axis.cla()
            self.review3axis.plot(thisy2, '-')
            self.review3.adjustSize()
            self.review3.draw()
            self.yarray = thisy2
            self.xarray = np.linspace(0, 1, len(thisy2))
            self.indexStart.setValue(np.min(self.xarray))
            self.indexStop.setValue(np.max(self.xarray))
        except:
            pass  # "no data"

        self.average2 = thisy22

    def onFFTButton(self):
        # TODO:
        if self.fftBtn.text() == 'FFT':
            self.statusBar.showMessage('Show FFT Plot')
            self.fftBtn.setText('TIME')
            N = np.size(self.xarray)
            # sample spacing
            T = self.xarray[1] - self.xarray[0]
            # print T
            yf = scipy.fftpack.fft(self.yarray)
            xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

            self.axes.cla()
            self.axes.grid(visible=True)
            self.axes.set_title(u'FFT')
            self.axes.set_xlabel(u'Frequency (Hz)')
            self.axes.set_ylabel(u'Amplitude (dB)')
            self.axes.plot(np.delete(xf, 0), np.delete(2.0 / N * np.abs(yf[0:N / 2]), 0), '-')
            self.laserPlot.adjustSize()
            self.laserPlot.draw()
        else:
            self.statusBar.showMessage('Show Time Base Plot')
            self.fftBtn.setText('FFT')
            self.axes.cla()
            self.axes.grid(visible=True)
            self.axes.set_xlabel(u'Time (sec)')
            """
            if self.laserSetupMode == 1:
                self.axes.set_ylabel(u'Position (µm)')
            else:
                self.axes.set_ylabel(u'Error (arcsec)')
            """
            self.axes.plot(self.xarray, self.yarray, '-')
            self.laserPlot.adjustSize()
            self.laserPlot.draw()

    def onToVelocity(self):
        if self.toVelocityBtn.text() == 'Velocity':
            self.statusBar.showMessage('Show Velocity Plot')
            self.toVelocityBtn.setText('Position')
            v = []
            for i in range(0, len(self.xarray) - 1):
                v.append((self.yarray[i + 1] - self.yarray[i]) / (self.xarray[i + 1] - self.xarray[i]))

            self.axes.cla()
            self.axes.grid(visible=True)
            self.axes.set_title(self.newTitle.toPlainText().strip())
            self.axes.set_xlabel(u'Time (sec)')
            self.axes.set_ylabel(self.newYLabel.text())
            self.axes.plot(self.xarray[:-1], v, '-')

            if not self.autoScaleX.isChecked():
                self.axes.set_xlim(self.xScaleMinimum.value(), self.xScaleMaximum.value())
            if not self.autoScaleY.isChecked():
                self.axes.set_ylim(self.yScaleMinimum.value(), self.yScaleMaximum.value())

            self.laserPlot.adjustSize()
            self.laserPlot.draw()
        else:
            self.statusBar.showMessage('Show Position Plot')
            self.toVelocityBtn.setText('Velocity')
            self.axes.cla()
            self.axes.grid(visible=True)
            self.axes.set_xlabel(u'Time (sec)')
            self.axes.set_ylabel(u'Position (µm)')
            self.axes.set_title(self.ttl + '\n' + self.woNsn)
            self.axes.plot(self.xarray, self.yarray, '-')
            self.laserPlot.adjustSize()
            self.laserPlot.draw()

    def onFilterSelection(self):
        if self.filterType.currentText() == 'No Filter':
            self.runningAvg2Label.setVisible(False)
            self.runningAverageN2.setVisible(False)
            self.cutoffLabel.setVisible(False)
            self.cutoffFrequency.setVisible(False)
            self.highCutoffLabel.setVisible(False)
            self.highCutoffFrequency.setVisible(False)
            self.fsLabel.setVisible(False)
            self.sampleRate.setVisible(False)
            self.orderLabel.setVisible(False)
            self.order.setVisible(False)
        if self.filterType.currentText() == 'Running Average':
            self.runningAvg2Label.setVisible(True)
            self.runningAverageN2.setVisible(True)
            self.cutoffLabel.setVisible(False)
            self.cutoffFrequency.setVisible(False)
            self.highCutoffLabel.setVisible(False)
            self.highCutoffFrequency.setVisible(False)
            self.fsLabel.setVisible(False)
            self.sampleRate.setVisible(False)
            self.orderLabel.setVisible(False)
            self.order.setVisible(False)
        if self.filterType.currentText() == 'Lowpass Filter' or self.filterType.currentText() == 'Highpass Filter':
            self.runningAvg2Label.setVisible(False)
            self.runningAverageN2.setVisible(False)
            self.cutoffLabel.setVisible(True)
            self.cutoffFrequency.setVisible(True)
            self.highCutoffLabel.setVisible(False)
            self.highCutoffFrequency.setVisible(False)
            self.fsLabel.setVisible(True)
            self.sampleRate.setVisible(True)
            self.orderLabel.setVisible(True)
            self.order.setVisible(True)
        if self.filterType.currentText() == 'Bandpass Filter' or self.filterType.currentText() == 'Gaussian Filter':
            self.runningAvg2Label.setVisible(False)
            self.runningAverageN2.setVisible(False)
            self.cutoffLabel.setVisible(True)
            self.cutoffFrequency.setVisible(True)
            self.highCutoffLabel.setVisible(True)
            self.highCutoffFrequency.setVisible(True)
            self.fsLabel.setVisible(True)
            self.sampleRate.setVisible(True)
            self.orderLabel.setVisible(True)
            self.order.setVisible(True)
        try:
            self.updatePlot()
        except:
            pass

    def onDataManipulation(self):
        xa = self.xarray * self.xScaleFactor.value()
        ya = self.yarray * self.yScaleFactor.value()
        if self.filterType.currentText() == 'Running Average':  # running average filter
            if self.runningAverageN2.value() > 0:
                ya = np.convolve(ya, np.ones((self.runningAverageN2.value(),)) / self.runningAverageN2.value())[
                     (self.runningAverageN2.value() - 1):]
                if self.runningAverageN2.value() > 0:
                    xa = xa[:-(self.runningAverageN2.value())]
                    ya = ya[:-(self.runningAverageN2.value())]

                    comptable = ya - ya[0]
                    comptable2 = comptable - np.arange(comptable[0], comptable[-1], comptable[-1] / np.size(comptable))
                    for i in comptable2:
                        print(i)

        if self.yOffset.checkState() == 2:
            ya = self.acCoupling(ya)
        if self.linearFit.checkState() == 2:
            ya = ya - np.linspace(ya[0], ya[-1], np.size(ya))

        ya = ya + np.linspace(0, self.addTilt.value(), np.size(ya)) + self.addOffset.value()

        if self.filterType.currentText() == 'Lowpass Filter':
            nyq = 0.5 * self.sampleRate.value()
            normal_cutoff = self.cutoffFrequency.value() / nyq
            b, a = butter(self.order.value(), normal_cutoff, btype='low', analog=False)
            y = lfilter(b, a, ya)
            ya = y
        if self.filterType.currentText() == 'Highpass Filter':
            nyq = 0.5 * self.sampleRate.value()
            normal_cutoff = self.cutoffFrequency.value() / nyq
            b, a = butter(self.order.value(), normal_cutoff, btype='high', analog=False)
            y = lfilter(b, a, ya)
            ya = y
        if self.filterType.currentText() == 'Bandpass Filter':
            nyq = 0.5 * self.sampleRate.value()
            normal_cutoff = self.cutoffFrequency.value() / nyq
            high = self.highCutoffFrequency.value() / nyq
            b, a = butter(self.order.value(), [normal_cutoff, high], btype='band', analog=False)
            y = lfilter(b, a, ya)
            ya = y
        if self.filterType.currentText() == 'Gaussian Filter':
            ya = gaussian_filter(ya, sigma=self.cutoffFrequency.value(), order=0)

        self.axes.cla()
        self.axes.plot(xa + self.addXOffset.value(), ya, '-')
        self.axes.grid(visible=True)
        if self.newTitle.toPlainText().strip() != '':
            self.axes.set_title(self.newTitle.toPlainText().strip())
        if self.newXLabel.text().strip() != '':
            self.axes.set_xlabel(self.newXLabel.text().strip())
        if self.newYLabel.text().strip() != '':
            self.axes.set_ylabel(self.newYLabel.text().strip())

        self.axes.set_xlim(self.xScaleMinimum.value(), self.xScaleMaximum.value())
        self.axes.set_ylim(self.yScaleMinimum.value(), self.yScaleMaximum.value())
        ###
        self.laserPlot.adjustSize()
        self.laserPlot.draw()
        self.rawX = xa
        self.rawY = ya

    def onRedrawPlot(self):
        if self.newTitle.toPlainText().strip() != '':
            self.axes.set_title(self.newTitle.toPlainText().strip())
        if self.newXLabel.text().strip() != '':
            self.axes.set_xlabel(self.newXLabel.text().strip())
        if self.newYLabel.text().strip() != '':
            self.axes.set_ylabel(self.newYLabel.text().strip())

        self.axes.set_xlim(self.xScaleMinimum.value(), self.xScaleMaximum.value())
        self.axes.set_ylim(self.yScaleMinimum.value(), self.yScaleMaximum.value())
        self.laserPlot.adjustSize()
        self.laserPlot.draw()

    def setColor(self):
        alpha = 140
        values = "{r}, {g}, {b}, {a}".format(r=self.color.red(),
                                             g=self.color.green(),
                                             b=self.color.blue(),
                                             a=alpha
                                             )
        self.readPlotBtn.setStyleSheet('QPushButton { color: rgba(' + values + '); }')

    def onSavePlot(self):
        self.onTestInfo()
        fin = open('C:\\Python_Projects\\HP5508A Laser System\\testinfo.txt')
        testinfo = fin.read()
        fin.close()

        file_choices = "*.txt | *.dat"
        if len(self.path) > 10:
            p = self.path[:self.path.rfind('/') + 1] + self.comments.toPlainText()[1:-1].replace('/', '') + '.txt'
            self.path = p

        self.path, _ = QFileDialog.getSaveFileName(self, 'Save file', self.path, file_choices)
        if self.path:
            self.statusBar.showMessage('Open %s' % self.path)
            f = open(self.path, 'w')
            if self.positionMode.text() == "Position Mode":
                datastring = 'Tested Date: ' + str(datetime.datetime.now()).split('.')[
                    0] + '\n' + testinfo + "\n\nStatistical Analysis:\n" + self.statisticAnalysis.toPlainText()
                f.write(datastring)  # .encode('utf-8'))
                f.close()

            if self.positionMode.text() == "Time Mode":
                rawDataArray = np.vstack((self.xarray, self.yarray))
                rawDataArray = np.transpose(rawDataArray)
                e = "Raw Data:\r\nTime\tPosition"
                for i in rawDataArray:
                    d = str(i).replace("[", "").strip()
                    d = d.replace("]", "").strip()
                    d = d.replace(" ", "\t")
                    d = d.replace("\n", "\t")
                    while d.find("\t\t") > 0:
                        d = d.replace("\t\t", "\t")
                    e = e + "\n" + d

                rawDataStr = e

                datastring = ('Tested Date: ' + str(datetime.datetime.now()).split('.')[0] + '\n'
                              + testinfo + "\n\nStatistical Analysis:\n"
                              + self.statisticAnalysis.toPlainText() + "\n\n"
                              + rawDataStr + "\n\nEnvironmental Data\n"
                              + self.environmentalData.toPlainText() + '\n'
                              + 'Expansion Coefficient (ppm): ' + self.expansionCoefficient.text() + '\n'
                              + 'Thermal Compensation: ' + str(self.thermalComp.isChecked()) + '\n'
                              + 'Velocity of Light Compensation: ' + str(self.volComp.isChecked()))

                f.write(datastring)  # .encode('utf-8'))
                f.close()
            self.tabControl.setCurrentIndex(0)

    def savePlt(self):
        self.onTestInfo()
        fin = open('C:\\Python_Projects\\HP5508A Laser System\\testinfo.txt')
        testinfo = fin.read()
        fin.close()

        if self.path:
            self.statusBar.showMessage('Open %s' % self.path)
            f = open(self.path, 'w')
            if self.positionMode.text() == "Position Mode":
                datastring = 'Tested Date: ' + str(datetime.datetime.now()).split('.')[
                    0] + '\n' + testinfo + "\n\nStatistical Analysis:\n" + self.statisticAnalysis.toPlainText()
                f.write(datastring.encode('utf-8'))
                f.close()

            if self.positionMode.text() == "Time Mode":
                rawDataArray = np.vstack((self.xarray, self.yarray))
                rawDataArray = np.transpose(rawDataArray)
                e = "Raw Data:\r\nTime\tPosition"
                for i in rawDataArray:
                    d = str(i).replace("[", "").strip()
                    d = d.replace("]", "").strip()
                    d = d.replace(" ", "\t")
                    d = d.replace("\n", "\t")
                    while d.find("\t\t") > 0:
                        d = d.replace("\t\t", "\t")
                    e = e + "\n" + d

                rawDataStr = e

                datastring = ('Tested Date: ' + str(datetime.datetime.now()).split('.')[0] + '\n'
                              + testinfo + "\n\nStatistical Analysis:\n"
                              + self.statisticAnalysis.toPlainText() + "\n\n"
                              + rawDataStr + "\n\nEnvironmental Data\n"
                              + self.environmentalData.toPlainText() + '\n'
                              + 'Expansion Coefficient (ppm): ' + self.expansionCoefficient.text() + '\n'
                              + 'Thermal Compensation: ' + str(self.thermalComp.isChecked()) + '\n'
                              + 'Velocity of Light Compensation: ' + str(self.volComp.isChecked()))
                f.write(datastring.encode('utf-8'))
                f.close()
            self.tabControl.setCurrentIndex(0)

    def constantUpdate(self):
        global i
        i += 1
        color = str(255 - i % 255) + "," + str(255 - i % 255) + "," + str(255)
        self.fftBtn.setStyleSheet("background-color: rgb(" + color + ")")

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = MyMain()
    main_window.show()
    app.exec_()
    print("exited successfully")


if __name__ == '__main__':
    main()
