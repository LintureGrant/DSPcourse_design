# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal

import FilterDesigner#布局文件
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
import sys
import wave
import time
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from matplotlib.patches import Circle
##信号处理函数类，里面的是一些信号处理的函数
class ProcessFunction(object):  ##这里负责写一些数字信号处理的方法
    def Audio_TimeDomain(self,feature):  ##时域
        f = wave.open(feature.path,"rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])
        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)

        feature.textBrowser_2.append("AUDIO INFO:   Number of channel: " + str(nchannels))
        feature.textBrowser_2.append("AUDIO INFO:   Sampling Frequency: " + str(framerate)+" Hz")
        feature.textBrowser_2.append("AUDIO INFO:   Sampling number: " + str(nframes))
        feature.textBrowser_2.append("AUDIO INFO:   Sampling duration: " + str(nframes/framerate)+" seconds")

        ax = feature.fig5.add_subplot(111)
        ###进度条显示******
        feature.progressBar.setValue(10)
        #***************
        #调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        #ax.clear()
        ax.plot(time, wave_data[:, 0])
        ax.set_title('Normalized Magnitude')
        ax.set_xlabel('Time [sec]')

        feature.fig5.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas5.draw()  # TODO:这里开始绘制

        feature.progressBar.setValue(20)

    def Audio_FrequencyDomain(self,feature):
        #*********************STFT图像绘制*****************************
        sampling_freq, audio = wavfile.read(feature.path)
        T = 20  # 短时傅里叶变换的时长 单位 ms
        fs = sampling_freq  # 采样频率是这么多，那么做出来的频谱宽度就是这么多
        N = len(audio)  # 采样点的个数
        audio = audio * 1.0 / (max(abs(audio)))

        # 计算并绘制STFT的大小
        f, t, Zxx = signal.stft(audio, fs, nperseg=T * fs / 1000)

        feature.progressBar.setValue(30)

        ax = feature.fig7.add_subplot(111)
        feature.fig7.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # ax=plt.figure()
        ax.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.1)
        ax.set_title('STFT Magnitude')
        #feature.fig6.colorbar(ax=ax)
        #feature.fig6.colorbar(feature.fig6)
        ####还存在的问题是colorbar显示不了的问题####
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
        feature.canvas7.draw()  # TODO:这里开始绘制
        feature.progressBar.setValue(40)
        #**************************************************************

        # *******************FFT图像绘制*********************************
        fft_signal = np.fft.fft(audio)
        fft_signal = abs(fft_signal)
        # 建立频率轴

        # 建立频率轴

        fft_signal = np.fft.fftshift(fft_signal)

        fft_signal = fft_signal[int(fft_signal.shape[0] / 2):]

        freqInteral = (sampling_freq / len(fft_signal))  ###频率轴的间隔

        Freq = np.arange(0, sampling_freq / 2, sampling_freq / (2*len(fft_signal)))

        feature.progressBar.setValue(50)
        #
        highFreq=  (np.argmax(fft_signal[int(len(fft_signal) / 2):len(fft_signal)]) )*freqInteral
        feature.textBrowser_2.append("FFT INFO:   Highest frequency: "+str(highFreq))


        ax = feature.fig6.add_subplot(111)
        # 调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        ax.plot(Freq, fft_signal, color='red')
        ax.set_title('FFT Figure')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Am')
        feature.fig6.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas6.draw()  # TODO:这里开始绘制

        feature.progressBar.setValue(60)

    def Audio_YuPuDomain(self, feature):
        #*******************语谱图绘制***********************************
        f = wave.open(feature.path, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        waveData = np.reshape(waveData, [nframes, nchannels]).T
        f.close()
        feature.progressBar.setValue(70)

        ax = feature.fig8.add_subplot(111)
        feature.fig8.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # ax=plt.figure()
        ax.specgram(waveData[0],Fs = framerate, scale_by_freq = True, sides = 'default')
        ax.set_title('Spectrogram')
        # feature.fig6.colorbar(ax=ax)
        # feature.fig6.colorbar(feature.fig6)
        ####还存在的问题是colorbar显示不了的问题####
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
        feature.canvas8.draw()  # TODO:这里开始绘制
        feature.progressBar.setValue(80)

    def IIR_Designer(self,feature):
        if str(feature.iirType)=='Butterworth':##巴特沃斯 双线性变换法间接设计模拟滤波器
            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType)=="Bandpass" or str(feature.filterType)=="bandstop":
                ##如果是带通带阻需要输入四组频率数据
                ##########频率预畸##################
                wp=str(feature.An_wp).split()#切分开之后再转换为array
                wp0=float(wp[0]) * (2 * np.pi / fs)
                wp1=float(wp[1]) * (2 * np.pi / fs)

                #************************************
                wp[0]=(2 * fs) * np.tan(wp0 / 2)#双线性变换
                wp[1]=(2 * fs) * np.tan(wp1 / 2)

                omiga_p=[float(wp[0]),float(wp[1])]
                wst=str(feature.An_wst).split()#切分开之后再转换为array
                wst0=float(wst[0]) * (2 * np.pi / fs)
                wst1=float(wst[1]) * (2 * np.pi / fs)
                #************************************

                wst[0]=(2 * fs) * np.tan(wst0 / 2)#双线性变换
                wst[1]=(2 * fs) * np.tan(wst1 / 2)

                omiga_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                ##########频率预畸##################
                omiga_p = (2 * fs) * np.tan(wp / 2)
                omiga_st = (2 * fs) * np.tan(wst / 2)
            feature.Rp=float(feature.Rp)
            feature.As=float(feature.As)
            N, Wn = signal.buttord(omiga_p, omiga_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.butter(N, Wn, btype=str(feature.filterType),
                                              analog=True))
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z, feature.p = signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig1.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            #ax.clear()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                     label=r'$|H_z(e^{j \omega})|$')
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Butterworth')
            feature.fig1.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas1.draw()  # TODO:这里开始绘制

            ###绘制零极点图########
            ax = feature.fig3.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            ##参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x,y,color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit quantization" % N)
            feature.fig3.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas3.draw()  # TODO:这里开始绘制

        if str(feature.iirType) == 'Chebyshev I':  ##切比雪夫一型

            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType)=="Bandpass" or str(feature.filterType)=="bandstop":
                ##如果是带通带阻需要输入四组频率数据
                ##########频率预畸##################
                wp=str(feature.An_wp).split()#切分开之后再转换为array
                wp0=float(wp[0]) * (2 * np.pi / fs)
                wp1=float(wp[1]) * (2 * np.pi / fs)

                #************************************
                wp[0]=(2 * fs) * np.tan(wp0 / 2)#双线性变换
                wp[1]=(2 * fs) * np.tan(wp1 / 2)

                omiga_p=[float(wp[0]),float(wp[1])]
                wst=str(feature.An_wst).split()#切分开之后再转换为array
                wst0=float(wst[0]) * (2 * np.pi / fs)
                wst1=float(wst[1]) * (2 * np.pi / fs)
                #************************************

                wst[0]=(2 * fs) * np.tan(wst0 / 2)#双线性变换
                wst[1]=(2 * fs) * np.tan(wst1 / 2)

                omiga_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                ##########频率预畸##################
                omiga_p = (2 * fs) * np.tan(wp / 2)
                omiga_st = (2 * fs) * np.tan(wst / 2)

            if len(str(feature.Rp).split())>1: #纹波参数
                Rpinput=str(feature.Rp).split()
                feature.Rp = float(Rpinput[0])
                feature.As = float(feature.As)
                rp_in=float(Rpinput[1])
            else:
                feature.Rp=float(feature.Rp)
                feature.As=float(feature.As)
                rp_in = 0.1*feature.Rp

            #N, Wn = signal.cheb1ord(wp, wst, feature.Rp, feature.As, True)
            N, Wn = signal.cheb1ord(omiga_p, omiga_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.cheby1(N, rp_in,Wn, btype=str(feature.filterType),
                                              analog=True))##切比雪夫是还有一个纹波参数
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z,feature.p=signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig1.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            #ax.clear()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                     label=r'$|H_z(e^{j \omega})|$')
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Chebyshev I')
            feature.fig1.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas1.draw()  # TODO:这里开始绘制

            ###绘制零极点图########
            ax = feature.fig3.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            ##参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x,y,color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit quantization" % N)
            feature.fig3.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas3.draw()  # TODO:这里开始绘制

        if str(feature.iirType) == 'Chebyshev II':  ##切比雪夫二型

            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType)=="Bandpass" or str(feature.filterType)=="bandstop":
                ##如果是带通带阻需要输入四组频率数据
                ##########频率预畸##################
                wp=str(feature.An_wp).split()#切分开之后再转换为array
                wp0=float(wp[0]) * (2 * np.pi / fs)
                wp1=float(wp[1]) * (2 * np.pi / fs)

                #************************************
                wp[0]=(2 * fs) * np.tan(wp0 / 2)#双线性变换
                wp[1]=(2 * fs) * np.tan(wp1 / 2)

                omiga_p=[float(wp[0]),float(wp[1])]
                wst=str(feature.An_wst).split()#切分开之后再转换为array
                wst0=float(wst[0]) * (2 * np.pi / fs)
                wst1=float(wst[1]) * (2 * np.pi / fs)
                #************************************

                wst[0]=(2 * fs) * np.tan(wst0 / 2)#双线性变换
                wst[1]=(2 * fs) * np.tan(wst1 / 2)

                omiga_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                ##########频率预畸##################
                omiga_p = (2 * fs) * np.tan(wp / 2)
                omiga_st = (2 * fs) * np.tan(wst / 2)
            if len(str(feature.As).split())>1: #纹波参数
                Asinput=str(feature.As).split()
                feature.As = float(Asinput[0])
                feature.Rp = float(feature.Rp)
                rs_in=float(Asinput[1])
            else:
                feature.Rp=float(feature.Rp)
                feature.As=float(feature.As)
                rs_in = 0.1*feature.As
            N, Wn = signal.cheb2ord(omiga_p, omiga_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.cheby2(N, rs_in,Wn, btype=str(feature.filterType),
                                              analog=True))##切比雪夫是还有一个纹波参数
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z,feature.p=signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig1.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            #ax.clear()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                     label=r'$|H_z(e^{j \omega})|$')
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Chebyshev I')
            feature.fig1.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas1.draw()  # TODO:这里开始绘制

            ###绘制零极点图########
            ax = feature.fig3.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            ##参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x,y,color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit quantization" % N)
            feature.fig3.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas3.draw()  # TODO:这里开始绘制

        if str(feature.iirType) == 'Cauer/elliptic':
            fs = float(feature.fs)
            feature.textBrowser_3.append(str(feature.filterType))
            feature.textBrowser_3.append(str(feature.iirType))
            if str(feature.filterType)=="Bandpass" or str(feature.filterType)=="bandstop":
                ##如果是带通带阻需要输入四组频率数据
                ##########频率预畸##################
                wp=str(feature.An_wp).split()#切分开之后再转换为array
                wp0=float(wp[0]) * (2 * np.pi / fs)
                wp1=float(wp[1]) * (2 * np.pi / fs)

                #************************************
                wp[0]=(2 * fs) * np.tan(wp0 / 2)#双线性变换
                wp[1]=(2 * fs) * np.tan(wp1 / 2)

                omiga_p=[float(wp[0]),float(wp[1])]
                wst=str(feature.An_wst).split()#切分开之后再转换为array
                wst0=float(wst[0]) * (2 * np.pi / fs)
                wst1=float(wst[1]) * (2 * np.pi / fs)
                #************************************

                wst[0]=(2 * fs) * np.tan(wst0 / 2)#双线性变换
                wst[1]=(2 * fs) * np.tan(wst1 / 2)

                omiga_st = [wst[0], wst[1]]
            else:
                wp = float(feature.An_wp) * (2 * np.pi / fs)
                wst = float(feature.An_wst) * (2 * np.pi / fs)
                ##########频率预畸##################
                omiga_p = (2 * fs) * np.tan(wp / 2)
                omiga_st = (2 * fs) * np.tan(wst / 2)

            Asinput = str(feature.As).split()
            Rpinput = str(feature.Rp).split()
            feature.As = float(Asinput[0])
            feature.Rp = float(Rpinput[0])
            rs_in = float(Asinput[1])
            rp_in = float(Rpinput[1])

            feature.Rp=float(feature.Rp)
            feature.As=float(feature.As)
            N, Wn = signal.ellipord(omiga_p, omiga_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.ellip(N, rp_in,rs_in,Wn, btype=str(feature.filterType),
                                              analog=True))##切比雪夫是还有一个纹波参数
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            feature.z,feature.p=signal.bilinear(feature.filts.num, feature.filts.den, fs)

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig1.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            #ax.clear()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                     label=r'$|H_z(e^{j \omega})|$')
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Cauer/elliptic')
            feature.fig1.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas1.draw()  # TODO:这里开始绘制

            ###绘制零极点图########
            ax = feature.fig3.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            z1, p1, k1 = signal.tf2zpk(feature.z, feature.p)  # zero, pole and gain
            c = np.vstack((feature.p, feature.z))
            Max = (abs(c)).max()  # find the largest value
            a = feature.p / Max  # normalization
            b = feature.z / Max
            Ra = (a * (2 ** ((N - 1) - 1))).astype(int)  # quantizan and truncate
            Rb = (b * (2 ** ((N - 1) - 1))).astype(int)
            z2, p2, k2 = signal.tf2zpk(Rb, Ra)
            ##参数方程画圆
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x,y,color='black')
            for i in p1:
                ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
            for i in z1:
                ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
            for i in p2:
                ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
            for i in z2:
                ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.2, 1.2)
            ax.grid()
            ax.set_title("%d bit quantization" % N)
            feature.fig3.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas3.draw()  # TODO:这里开始绘制

        feature.textBrowser.setText("PARAMETER OF THIS FILTER")
        feature.textBrowser.append("*********" )
        feature.textBrowser.append("FILTER TPYE=" +str(feature.filterType))
        feature.textBrowser.append("IIR TPYE=" + str(feature.iirType))
        feature.textBrowser.append("ORDER=" + str(N))
        feature.textBrowser.append("b="+str(feature.z))
        feature.textBrowser.append("a="+str(feature.p))
        feature.textBrowser.append()
    def apply_IIR(self,feature):
        f = wave.open(feature.path,"rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        maximum=max(abs(wave_data))
        wave_data = wave_data * 1.0 / (maximum)
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])

        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)
        #print(time)
        t = np.linspace(0, nframes/ framerate, nframes, endpoint=False)
        print("p maxium")
        print(feature.p)
        print(feature.z)
        feature.yout=signal.filtfilt(feature.z,feature.p,wave_data[:, 0],method='gust')
        print(max(feature.yout))
        ax = feature.fig2.add_subplot(111)
        #调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        #ax.clear()
        ax.plot(t, feature.yout)
        ax.set_title('Passed Filter')
        ax.set_xlabel('Time [sec]')

        feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas2.draw()  # TODO:这里开始绘制

        ##绘制出时域的图像之后，再到频率分析
        #FFT变换#
        fft_signal = np.fft.fft(feature.yout)
        fft_signal = np.fft.fftshift(abs(fft_signal))
        fft_signal = fft_signal[int(fft_signal.shape[0]/2):]
        # 建立频率轴
        Freq = np.arange(0, framerate / 2, framerate / (2*len(fft_signal)))

        ####绘图######
        ax = feature.fig4.add_subplot(111)
        # 调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        ax.plot(Freq, fft_signal, color='red')
        ax.set_title('FFT Figure')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Am')
        feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas4.draw()  # TODO:这里开始绘制


        #feature.precessed_Audio=feature.filtz.output(wave_data,time,X0=None)#求系统的零状态响应
        # feature.precessed_Audio =feature.precessed_Audio.tostring()
        # feature.process_flag=1#标志位为1，代表处理好了，否则的话就代表没有
        ##写音频文件###

        feature.yout = feature.yout * maximum  # 去归一化
        feature.yout = feature.yout.astype(np.short)
        f = wave.open(feature.saveDatepath_IIR, "wb")  ##
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.setnframes(nframes)
        f.writeframes(feature.yout.tostring())
        f.close()

        feature.process_flag=1#代表本次处理完毕

    def FIR_Designer(self,feature):
        kaiser_para=0.85
        if feature.filterType_FIR == 'Lowpass':
            numtaps=int(feature.filter_length)
            fcut=feature.f2*2/feature.fs_FIR
            if str(feature.firType)=='kaiser':
                width=kaiser_para
            else:
                width=None
            feature.FIR_b = signal.firwin(numtaps, fcut,width=width, window=str(feature.firType))  #
        if feature.filterType_FIR == 'Highpass':
            numtaps=int(feature.filter_length)
            fcut=feature.f2*2/feature.fs_FIR
            if str(feature.firType)=='kaiser':
                width=kaiser_para
            else:
                width=None
            feature.FIR_b = signal.firwin(numtaps, fcut,width=width,window=str(feature.firType),pass_zero=False)  #
        if feature.filterType_FIR == 'Bandpass':
            numtaps = int(feature.filter_length)
            fcut = [feature.f1*2/feature.fs_FIR,feature.f2*2/feature.fs_FIR]
            if str(feature.firType)=='kaiser':
                width=kaiser_para
            else:
                width=None
            feature.FIR_b = signal.firwin(numtaps, fcut,width=width,window=str(feature.firType), pass_zero=False)  #
        if feature.filterType_FIR == 'Band-stop pass':
            numtaps = int(feature.filter_length)
            fcut = [feature.f1*2/feature.fs_FIR,feature.f2*2/feature.fs_FIR]
            if str(feature.firType)=='kaiser':
                width=kaiser_para
            else:
                width=None
            feature.FIR_b = signal.firwin(numtaps, fcut,width=width,window=str(feature.firType))  #


        feature.textBrowser_10.append("FilterType_FIR:"+feature.filterType_FIR)
        feature.textBrowser_10.append("FirType"+str(feature.firType))
        feature.textBrowser_10.append("INFO:     *****Succeed!*****    ")
        # 绘制频率响应：
        wz, hz = signal.freqz(feature.FIR_b)

        ax = feature.fig25.add_subplot(111)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # ax.clear()
        ax.semilogx(wz * feature.fs_FIR / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                    label=r'$|H_z(e^{j \omega})|$')
        ax.set_xlabel('Hz')
        ax.set_ylabel('dB')
        ax.set_title(str(feature.firType))
        feature.fig25.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas25.draw()  # TODO:这里开始绘制


        #####绘制零极点图###############
        ##只有零点没有极点###
        ax = feature.fig26.add_subplot(111)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        fir_a=np.zeros(numtaps)
        fir_a[numtaps-1]=1
        z1, p1, k1 = signal.tf2zpk(feature.FIR_b, fir_a)  # zero, pole and gain
        c = np.vstack((fir_a, feature.FIR_b))
        Max = (abs(c)).max()  # find the largest value
        a = fir_a / Max  # normalization
        b = feature.FIR_b / Max
        Ra = (a * (2 ** ((numtaps - 1) - 1))).astype(int)  # quantizan and truncate
        Rb = (b * (2 ** ((numtaps - 1) - 1))).astype(int)
        z2, p2, k2 = signal.tf2zpk(Rb, Ra)
        ##参数方程画圆
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, color='black')
        for i in p1:
            ax.plot(np.real(i), np.imag(i), 'bx')  # pole before quantization
        for i in z1:
            ax.plot(np.real(i), np.imag(i), 'bo')  # zero before quantization
        for i in p2:
            ax.plot(np.real(i), np.imag(i), 'rx')  # pole after quantization
        for i in z2:
            ax.plot(np.real(i), np.imag(i), 'ro')  # zero after quantization
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.2, 1.2)
        ax.grid()
        ax.set_title("%d bit quantization" % numtaps)
        feature.fig26.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas26.draw()  # TODO:这里开始绘制
        feature.textBrowser_11.setText("PARAMETER OF THIS FILTER")
        feature.textBrowser_11.append("*********")
        feature.textBrowser_11.append("FILTER TPYE=" + str(feature.filterType_FIR))
        feature.textBrowser_11.append("FIR TPYE=" + str(feature.firType))
        feature.textBrowser_11.append("ORDER(length of filter)=" + str(numtaps))
        feature.textBrowser_11.append("b=" + str(feature.FIR_b))
        feature.textBrowser_11.append("a=" + str(fir_a))
    def apply_FIR(self,feature):
        f = wave.open(feature.path,"rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        maximum=max(abs(wave_data))
        wave_data = wave_data * 1.0 / (maximum)
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])

        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)
        #print(time)
        t = np.linspace(0, nframes/ framerate, nframes, endpoint=False)

        feature.yout=signal.filtfilt(feature.FIR_b,1,wave_data[:, 0])

        ax = feature.fig27.add_subplot(111)
        #调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        #ax.clear()
        ax.plot(t, feature.yout)
        ax.set_title('Passed Filter')
        ax.set_xlabel('Time [sec]')

        feature.fig27.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas27.draw()  # TODO:这里开始绘制

        ##绘制出时域的图像之后，再到频率分析
        #FFT变换#
        fft_signal = np.fft.fft(feature.yout)
        fft_signal = np.fft.fftshift(abs(fft_signal))[int(fft_signal.shape[0]/2):]
        # 建立频率轴
        Freq = np.arange(0, framerate / 2, framerate / (2*len(fft_signal)))

        ####绘图######
        ax = feature.fig28.add_subplot(111)
        # 调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        ax.plot(Freq, fft_signal, color='red')
        ax.set_title('FFT Figure')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Am')
        feature.fig28.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas28.draw()  # TODO:这里开始绘制


        #feature.precessed_Audio=feature.filtz.output(wave_data,time,X0=None)#求系统的零状态响应
        # feature.precessed_Audio =feature.precessed_Audio.tostring()
        # feature.process_flag=1#标志位为1，代表处理好了，否则的话就代表没有
        ##写音频文件###
        feature.yout = feature.yout * maximum  # 去归一化
        feature.yout = feature.yout.astype(np.short)
        f = wave.open(feature.saveDatepath_FIR, "wb")  ##
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.setnframes(nframes)
        f.writeframes(feature.yout.tostring())
        f.close()
        feature.process_flag=1#代表本次处理完毕

#MyMainForm类里面负责写窗体的一些逻辑控制以及方法调用
#
#
class MyMainForm(QMainWindow, FilterDesigner.Ui_FilterDesigner):#因为界面py文件和逻辑控制py文件分开的，所以在引用的时候要加上文件名再点出对象
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)#从父类哪里继承下来
        self.Process=ProcessFunction()#process对象包含了所有的信号处理函数及其画图
        self.saveDatepath_IIR = os.getcwd().replace('\\','/')+"/ProcessedSignal/sweep.wav"##处理过后的数据保存位置
        self.saveDatepath_FIR = os.getcwd().replace('\\','/')+"/ProcessedSignal/sweepfir.wav"  ##处理过后的数据保存位置
        #self.saveDatepath_IIR = os.getcwd()+"/ProcessedSignal/sweep.wav"##处理过后的数据保存位置
        #self.saveDatepath_FIR = os.getcwd()+"/sweepfir.wav"  ##处理过后的数据保存位置
        #print(self.saveDatepath_IIR)

        self.setupUi(self)#setupUi是Ui_FilterDesigner类里面的一个方法，这里的self是两个父类的子类的一个实例
        self.progressBar.setValue(0)#进度条初始化为0
        #***************标志位的初始化*******************
        self.process_flag=0#处理完毕标志位
        self.isPlay = 0#播放器播放标志位
        self.isPlay_IIR = 0  # 播放器播放标志位
        self.isPlay_FIR = 0

        #**************播放器的设定**********************

        self.player = QMediaPlayer(self) #这个播放器是播放原声的
        self.player_IIR = QMediaPlayer(self)#定义两个对象出来，这个负责播放处理过后的
        self.player_FIR = QMediaPlayer(self)

        self.horizontalSlider_2.sliderMoved[int].connect(lambda: self.player.setPosition(self.horizontalSlider_2.value()))
        self.horizontalSlider_2.setStyle(QStyleFactory.create('Fusion'))

        self.horizontalSlider_3.sliderMoved[int].connect(lambda: self.player.setPosition(self.horizontalSlider_3.value()))
        self.horizontalSlider_3.setStyle(QStyleFactory.create('Fusion'))

        self.horizontalSlider_11.sliderMoved[int].connect(lambda: self.player.setPosition(self.horizontalSlider_11.value()))
        self.horizontalSlider_11.setStyle(QStyleFactory.create('Fusion'))
        ##IIR
        self.horizontalSlider_4.sliderMoved[int].connect(lambda: self.player_IIR.setPosition(self.horizontalSlider_4.value()))
        self.horizontalSlider_4.setStyle(QStyleFactory.create('Fusion'))
        ##FIR
        self.horizontalSlider_12.sliderMoved[int].connect(lambda: self.player_FIR.setPosition(self.horizontalSlider_12.value()))
        self.horizontalSlider_12.setStyle(QStyleFactory.create('Fusion'))



        self.timer = QTimer(self)
        self.timer.start(1000)##定时器设定为1s，超时过后链接到playRefresh刷新页面
        self.timer.timeout.connect(self.playRefresh)##



        #**************菜单栏的事件绑定*******************
        self.action_2.triggered.connect(self.onFileOpen)##菜单栏的action打开文件
        self.actionExit.triggered.connect(self.close)#菜单栏的退出action
        self.Timelayout_()##时间域的四个图窗布局
        self.Iirlayout_()##IIR设计界面的四个图窗布局
        self.Firlayout_()##FIR设计界面的四个图窗布局

        #**************第一个界面的事件绑定配置*************
        self.dial.setValue(20)#默认音量大小为20
        self.dial.valueChanged.connect(self.changeVoice0)##音量圆盘控制事件绑定,如果值被改变就调起事件
        self.pushButton_analyse.clicked.connect(self.Analyse_btn_start)  # 给pushButton_3添加一个点击事件
        self.pushButton_3.clicked.connect(self.palyMusic)

        #**************第二个界面的事件绑定配置*************
        self.dial_2.setValue(20)  # 默认音量大小为20
        self.dial_2.valueChanged.connect(self.changeVoice1)  ##音量圆盘控制事件绑定,如果值被改变就调起事件
        self.dial_3.setValue(20)  # 默认音量大小为20
        self.dial_3.valueChanged.connect(self.changeVoice)
        self.pushButton.clicked.connect(self.desigenIIR)#点击开始设计IIR滤波器按钮之后，调用函数
        self.pushButton_2.clicked.connect(self.applyIIR)#点击应用滤波器
        self.pushButton_4.clicked.connect(self.palyMusic)
        self.pushButton_5.clicked.connect(self.playIIRaudio)

        #**************第二个界面的事件绑定配置*************
        self.dial_10.setValue(20)  # 默认音量大小为20
        self.dial_10.valueChanged.connect(self.changeVoice2)  ##音量圆盘控制事件绑定,如果值被改变就调起事件
        self.dial_11.setValue(20)  # 默认音量大小为20
        self.dial_11.valueChanged.connect(self.changeVoice)
        self.pushButton_18.clicked.connect(self.designFIR)#点击开始设计IIR滤波器按钮之后，调用函数
        self.pushButton_19.clicked.connect(self.applyFIR)#点击应用滤波器
        self.pushButton_16.clicked.connect(self.palyMusic)
        self.pushButton_17.clicked.connect(self.playFIRaudio)

    def Timelayout_(self):
        self.fig5 = plt.figure()
        self.canvas5 = FigureCanvas(self.fig5)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas5)
        self.graphicsView_5.setLayout(layout)  # 设置好布局之后调用函数

        self.fig6 = plt.figure()
        self.canvas6 = FigureCanvas(self.fig6)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas6)
        self.graphicsView_6.setLayout(layout)  # 设置好布局之后调用函数

        self.fig7 = plt.Figure()
        self.canvas7 = FigureCanvas(self.fig7)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas7)
        self.graphicsView_7.setLayout(layout)  # 设置好布局之后调用函数

        self.fig8 = plt.Figure()
        self.canvas8 = FigureCanvas(self.fig8)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas8)
        self.graphicsView_8.setLayout(layout)  # 设置好布局之后调用函数

    def Iirlayout_(self):
        self.fig1 = plt.figure()
        self.canvas1 = FigureCanvas(self.fig1)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas1)
        self.graphicsView.setLayout(layout)  # 设置好布局之后调用函数

        self.fig2 = plt.figure()
        self.canvas2 = FigureCanvas(self.fig2)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas2)
        self.graphicsView_2.setLayout(layout)  # 设置好布局之后调用函数

        self.fig3 = plt.Figure()
        self.canvas3 = FigureCanvas(self.fig3)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas3)
        self.graphicsView_3.setLayout(layout)  # 设置好布局之后调用函数

        self.fig4 = plt.Figure()
        self.canvas4 = FigureCanvas(self.fig4)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas4)
        self.graphicsView_4.setLayout(layout)  # 设置好布局之后调用函数

    def Firlayout_(self):
        self.fig25 = plt.figure()
        self.canvas25 = FigureCanvas(self.fig25)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas25)
        self.graphicsView_25.setLayout(layout)  # 设置好布局之后调用函数

        self.fig26 = plt.figure()
        self.canvas26 = FigureCanvas(self.fig26)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas26)
        self.graphicsView_26.setLayout(layout)  # 设置好布局之后调用函数

        self.fig27 = plt.Figure()
        self.canvas27 = FigureCanvas(self.fig27)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas27)
        self.graphicsView_27.setLayout(layout)  # 设置好布局之后调用函数

        self.fig28 = plt.Figure()
        self.canvas28 = FigureCanvas(self.fig28)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas28)
        self.graphicsView_28.setLayout(layout)  # 设置好布局之后调用函数
    ###############对应的一些触发方法######################
    def onFileOpen(self): ##打开文件
        self.path, _ = QFileDialog.getOpenFileName(self, '打开文件', '', '音乐文件 (*.wav)')

        if self.path:##选中文件之后就选中了需要播放的音乐，并同时显示出来
            self.isPlay=0#每次打开文件的时候就需要暂停播放，无论是否在播放与否
            self.isPlay_IIR=0

            self.player.pause()
            self.player_IIR.pause()

            self.player.setMedia(QMediaContent(QUrl(self.path)))  ##选中需要播放的音乐
            self.horizontalSlider_2.setMinimum(0)
            self.horizontalSlider_2.setMaximum(self.player.duration())
            self.horizontalSlider_2.setValue(self.horizontalSlider_2.value() + 1000)
            self.horizontalSlider_2.setSliderPosition(0)

            self.horizontalSlider_3.setMinimum(0)
            self.horizontalSlider_3.setMaximum(self.player.duration())
            self.horizontalSlider_3.setValue(self.horizontalSlider_2.value() + 1000)
            self.horizontalSlider_3.setSliderPosition(0)

            self.label_17.setText("Current File:  "+os.path.basename(self.path))
            self.label_18.setText("Current File:  " + os.path.basename(self.path))

    def Analyse_btn_start(self):##这里对应的是打开文件，并点击按钮
        try:
            if self.path:##要必须在打开文件之后才允许进行处理
                self.textBrowser_2.append("*********This file :"+str(os.path.basename(self.path))+"*********")
                self.progressBar.setValue(0)  ##每次允许处理时进度条归0
                self.Process.Audio_TimeDomain(self)  ##把实例传入进去
                self.Process.Audio_FrequencyDomain(self)
                self.Process.Audio_YuPuDomain(self)
                self.progressBar.setValue(100)
                self.textBrowser_2.append("Analyse Succeed!")
                self.textBrowser_2.append("---------  "+str(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))+"  ---------")

        except Exception as e:
            print(e)
            self.textBrowser_2.setText("There are some errors occuring when programme trying to open file")

    def palyMusic(self):
        try:
            if self.path:#这个path是当前的路径，如果path变了，那么就意味着更换了文件
                if not self.isPlay:##如果isPaly=0，那就说明播放器并没有打开，且此时按下了播放按钮，就开始播放
                    self.player.play()
                    self.isPlay=1##播放之后同时置为1，代表播放器目前正在播放
                else:
                    self.player.pause()
                    self.isPlay = 0  ##暂停之后同时置为0，代表播放器目前没有播放
        except Exception as e:
            print(e)
            self.textBrowser_2.setText("There are some errors occuring when playing audio")

    def playRefresh(self):
        if self.isPlay:
            #print(self.player.duration())
            self.horizontalSlider_2.setMinimum(0)
            self.horizontalSlider_2.setMaximum(self.player.duration())
            self.horizontalSlider_2.setValue(self.horizontalSlider_2.value() + 1000)

            self.horizontalSlider_3.setMinimum(0)
            self.horizontalSlider_3.setMaximum(self.player.duration())
            self.horizontalSlider_3.setValue(self.horizontalSlider_3.value() + 1000)
        elif self.isPlay_IIR:
            self.horizontalSlider_4.setMinimum(0)
            self.horizontalSlider_4.setMaximum(self.player_IIR.duration())
            self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)
        elif self.isPlay_FIR:
            self.horizontalSlider_12.setMinimum(0)
            self.horizontalSlider_12.setMaximum(self.player_FIR.duration())
            self.horizontalSlider_12.setValue(self.horizontalSlider_12.value() + 1000)
        #ORIGINAL AUDIO
        self.label_14.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.label_15.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        self.label_19.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.label_20.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        self.label_68.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.label_69.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

        #IIR
        self.label_22.setText(time.strftime('%M:%S', time.localtime(self.player_IIR.position() / 1000)))
        self.label_23.setText(time.strftime('%M:%S', time.localtime(self.player_IIR.duration() / 1000)))
        #FIR
        self.label_71.setText(time.strftime('%M:%S', time.localtime(self.player_FIR.position() / 1000)))
        self.label_72.setText(time.strftime('%M:%S', time.localtime(self.player_FIR.duration() / 1000)))

    def changeVoice(self):
        #print(self.dial.value())
        self.player_IIR.setVolume(self.dial_3.value())
        self.player_FIR.setVolume(self.dial_11.value())

    def changeVoice0(self):
        self.player.setVolume(self.dial.value())
        self.dial_2.setValue(self.dial.value())
        self.dial_10.setValue(self.dial.value())
    def changeVoice1(self):
        self.player.setVolume(self.dial_2.value())
        self.dial.setValue(self.dial_2.value())
        self.dial_10.setValue(self.dial_2.value())
    def changeVoice2(self):
        self.player.setVolume(self.dial_10.value())
        self.dial.setValue(self.dial_10.value())
        self.dial_2.setValue(self.dial_10.value())


    def desigenIIR(self):
        ###获取到输入参数：滤波器四个指标
        self.progressBar.setValue(0)
        try:
            self.An_wp=self.lineEdit_3.text()
            self.An_wst=self.lineEdit_2.text()
            self.Rp=self.lineEdit.text()
            self.As=self.lineEdit_4.text()

            self.fs=self.lineEdit_5.text()

            self.filterType=self.comboBox_2.currentText()
            self.iirType=self.comboBox_3.currentText()
            self.progressBar.setValue(10)
            self.Process.IIR_Designer(self)
        except Exception as e:
            print(e)
        self.progressBar.setValue(100)
    def applyIIR(self):
        self.progressBar.setValue(0)
        try:
            self.process_flag=0
            self.player_IIR.pause()
            self.player_IIR.setMedia(QMediaContent(QUrl(self.path)))  # 先把绑定改过去，不然文件占用
            self.progressBar.setValue(20)
            self.Process.apply_IIR(self)
            if self.process_flag:  # 如果处理好了
                try:
                    self.isPlay = 0
                    self.isPlay_IIR = 0
                    self.isPlay_FIR = 0
                    self.player.pause()  # 暂停另外的播放器
                    self.player_FIR.pause()
                    self.horizontalSlider_4.setMinimum(0)
                    self.horizontalSlider_4.setMaximum(self.player_IIR.duration())
                    self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)
                    self.horizontalSlider_4.setSliderPosition(0)
                    self.label_21.setText("Processed Audio: " + os.path.basename(self.path))

                    self.player_IIR.setMedia(QMediaContent(QUrl(self.saveDatepath_IIR)))  ##选中需要播放的音乐
                    #self.player_IIR.setMedia(QMediaContent(), buf)  # 从缓存里面读出来的

                #self.player_IIR.setMedia(QMediaContent(QUrl(self.path)))
                except Exception as e:
                    print(e)
            else:#没有处理好，也就是没有进行滤波操作
                self.textBrowser_3.setText("please choose a audio to design filter and apply before previewing")
        except Exception as e:
            print(e)
        self.progressBar.setValue(100)
    def playIIRaudio(self):
        try:
            self.isPlay=0#点按任意一个播放器的播放暂停按钮都会停止
            self.player.pause()
            if self.process_flag:  # 如果处理好了
                if not self.isPlay_IIR:
                    self.horizontalSlider_4.setValue(self.player_IIR.position())
                    self.player_IIR.play()
                    self.isPlay_IIR=1
                    self.textBrowser_3.append("play")
                else:##如果发现播放器正在播放
                    self.player_IIR.pause()
                    self.isPlay_IIR=0
                    self.textBrowser_3.append("pause")
            else:#没有处理好，也就是没有进行滤波操作
                self.textBrowser_3.setText("please choose a audio to design filter and apply before previewing")
        except Exception as e:
            print(e)

    def designFIR(self):
        try:
            self.progressBar.setValue(0)
            self.f1=float(self.lineEdit_17.text())

            self.f2=float(self.lineEdit_6.text())

            self.filter_length=float(self.lineEdit_16.text())

            self.fs_FIR = float(self.lineEdit_20.text())

            self.filterType_FIR=self.comboBox_8.currentText()

            self.firType=self.comboBox_9.currentText()

            self.progressBar.setValue(20)
            self.Process.FIR_Designer(self)
            self.progressBar.setValue(100)
        except Exception as e:
            print(e)

    def applyFIR(self):
        self.progressBar.setValue(0)
        try:
            #处理之前全部关掉播放器对音乐的链接，不然会导致文件写不进去
            self.isPlay = 0
            self.isPlay_IIR = 0
            self.isPlay_FIR = 0
            self.player.pause()  # 暂停另外的播放器
            self.player_IIR.pause()
            self.player_FIR.pause()
            self.player_FIR.setMedia(QMediaContent(QUrl(self.path)))  # 先绑定到其他地方去，不然文件占用会导致写不进去文件
            self.process_flag=0

            self.progressBar.setValue(10)

            self.Process.apply_FIR(self)

            print(self.process_flag)
            if self.process_flag:  # 如果处理好了
                self.horizontalSlider_12.setMinimum(0)
                self.horizontalSlider_12.setMaximum(self.player_FIR.duration())
                self.horizontalSlider_12.setValue(self.horizontalSlider_12.value() + 1000)
                self.horizontalSlider_12.setSliderPosition(0)
                self.label_21.setText("Processed Audio: " + os.path.basename(self.path))

                self.player_FIR.setMedia(QMediaContent(QUrl(self.saveDatepath_FIR)))  ##选中需要播放的音乐
                #self.player_IIR.setMedia(QMediaContent(QUrl(self.path)))

            else:#没有处理好，也就是没有进行滤波操作
                self.textBrowser_3.setText("please choose a audio to design filter and apply before previewing")
        except Exception as e:
            print(e)
        self.progressBar.setValue(100)
    def playFIRaudio(self):
        try:
            self.isPlay=0#点按任意一个播放器的播放暂停按钮都会停止
            self.isPlay_IIR = 0
            self.player.stop()
            self.player_IIR.stop()
            print(self.process_flag)
            if self.process_flag:  # 如果处理好了
                if not self.isPlay_FIR:
                    self.horizontalSlider_12.setValue(self.player_FIR.position())#从停止位置继续播放
                    self.player_FIR.play()
                    self.isPlay_FIR=1
                    self.textBrowser_10.append("play")
                else:##如果发现播放器正在播放
                    self.player_FIR.pause()
                    self.isPlay_FIR=0
                    self.textBrowser_10.append("pause")
            else:#没有处理好，也就是没有进行滤波操作
                self.textBrowser_10.setText("please choose a audio to design filter and apply before previewing")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())
