import numpy as np
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

import struct
import pyaudio
from scipy.fftpack import fft

import sys
import time


class AudioStream(object):
    def __init__(self):

        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 115, 1910, 1070)

        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])

        wf_ylabels = [(0, '0'), (127, '128'), (512, '512')]
        wf_yaxis = pg.AxisItem(orientation='left')
        wf_yaxis.setTicks([wf_ylabels])

        sp_xlabels = [
            (np.log10(10), '10'), (np.log10(100), '100'),
            (np.log10(1000), '1000'), (np.log10(22050), '22050')
        ]
        sp_xaxis = pg.AxisItem(orientation='bottom')
        sp_xaxis.setTicks([sp_xlabels])

        self.waveform = self.win.addPlot(
            title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        )
        self.spectrum = self.win.addPlot(
            title='SPECTRUM', row=2, col=1, axisItems={'bottom': sp_xaxis},
        )

        # pyaudio stuff
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 1024 * 2

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        # waveform and spectrum x points
        self.x = np.arange(0, 2 * self.CHUNK, 2)
        self.f = np.linspace(0, self.RATE // 2, self.CHUNK // 2)
        self.win.show()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec()

    def close(self):
        print("Closing stream...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
                self.waveform.setYRange(0, 512, padding=0)
                self.waveform.setXRange(0, 2 * self.CHUNK, padding=0.005)
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='m', width=3)
                self.spectrum.setLogMode(x=True, y=True)
                self.spectrum.setYRange(-4, 0, padding=0)
                self.spectrum.setXRange(
                    np.log10(20), np.log10(self.RATE / 2), padding=0.005)

    def update(self):
        stime = time.time()
        try:
            wf_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            wf_data = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
            wf_data = np.array(wf_data, dtype=np.int16)[::2] + 128
            self.set_plotdata(name='waveform', data_x=self.x, data_y=wf_data, )

            sp_data = fft(np.array(wf_data, dtype=np.int16) - 128)
            sp_data = np.abs(sp_data[0:int(self.CHUNK / 2)]
                             ) * 2 / (128 * self.CHUNK)
            self.set_plotdata(name='spectrum', data_x=self.f, data_y=sp_data)
        except OSError as e:
            # This can happen if the stream is closed while an update is pending
            print(f"Error reading from stream: {e}")
            return
        print('{:.0f} FPS'.format(1 / (time.time() - stime)))

    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)
        self.start()


if __name__ == '__main__':
    audio_app = AudioStream()
    try:
        audio_app.animation()
    except KeyboardInterrupt:
        pass
    finally:
        audio_app.close()