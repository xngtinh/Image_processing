# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import math


class MyForm(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyForm, self).__init__()
        self.fileName = None
        self.size = None
        self.ui = loadUi('imageshow.ui', self)
        self.ui.btnOpen.clicked.connect(self.displayMethod)
        self.ui.btnExit.clicked.connect(self.exitMethod)
        self.ui.btnLight.clicked.connect(self.lightImageMethod)
        self.ui.btnDark.clicked.connect(self.darkImageMethod)
        self.ui.btnMultiple.clicked.connect(self.multipleImageMethod)
        self.ui.btnLogarit.clicked.connect(self.logaritImageMethod)
        self.ui.btnGamma.clicked.connect(self.gammaImageMethod)
        self.ui.btnHistogram.clicked.connect(self.histogramMethod)
        self.ui.btnEqualHist.clicked.connect(self.equalhistogramMethod)
        self.ui.btnTuyentinh.clicked.connect(self.tuyentinhMethod)
        self.ui.btnGaussian.clicked.connect(self.gaussianMethod)
        self.ui.btnBlur.clicked.connect(self.blurMethod)
        self.ui.btnMedian.clicked.connect(self.medianMethod)
        self.ui.btnSharpening.clicked.connect(self.sharpeningMethod)
        self.ui.btnGx.clicked.connect(self.gxMethod)
        self.ui.btnGy.clicked.connect(self.gyMethod)
        self.ui.btnGxplusGy.clicked.connect(self.gx_plus_gyMethod)
        self.ui.btnWireFame.clicked.connect(self.Gaussian_filter_3D_wirefame)
        self.ui.btnSurface.clicked.connect(self.Gaussian_filter_3D_surface)
        self.ui.btnIdeallp.clicked.connect(self.ideallp)
        self.ui.btnButterworthlp.clicked.connect(self.butterworthlp)
        self.ui.btnGaussianlp.clicked.connect(self.gaussianlp)
        self.ui.btnIdealhp.clicked.connect(self.idealhp)
        self.ui.btnButterworthhp.clicked.connect(self.butterworthhp)
        self.ui.btnGaussianhp.clicked.connect(self.gaussianhp)
        self.ui.btnApplyIdeal_lp.clicked.connect(self.ideallp_applyimg)
        self.ui.btnApplyBut_lp.clicked.connect(self.butterworthlp_applyimg)
        self.ui.btnApplyGau_lp.clicked.connect(self.gaussianlp_applyimg)
        self.ui.btnApplyIdeal_hp.clicked.connect(self.idealhp_applyimg)
        self.ui.btnApplyBut_hp.clicked.connect(self.butterworthhp_applyimg)
        self.ui.btnApplyGau_hp.clicked.connect(self.gaussianhp_applyimg)
        self.ui.btnRotate90.clicked.connect(self.rotate90imageMethod)
        self.ui.btnRotate180.clicked.connect(self.rotate180imageMethod)
        self.ui.btnRotate270.clicked.connect(self.rotate270imageMethod)
        self.ui.btnSobel_x.clicked.connect(self.sobel_x)
        self.ui.btnSobel_y.clicked.connect(self.sobel_y)
        self.ui.btnSobel.clicked.connect(self.sobel)
        self.ui.btnPrewitt_x.clicked.connect(self.prewitt_x)
        self.ui.btnPrewitt_y.clicked.connect(self.prewitt_y)
        self.ui.btnPrewitt.clicked.connect(self.prewitt)
        self.ui.btnThresholdBinary.clicked.connect(self.adaptiveThreshold_binary)
        self.ui.btnThresholdTozero.clicked.connect(self.adaptiveThreshold_tozero)
        self.ui.btnThresholdTrunc.clicked.connect(self.adaptiveThreshold_trunc)
        self.ui.btnThresholOtsu.clicked.connect(self.adaptiveThreshold_otsu)
        self.ui.btnThresholdGaussianC.clicked.connect(self.adaptiveThreshold_gaussianC)
        self.ui.btnThresholdMeanC.clicked.connect(self.adaptiveThreshold_meanC)
        self.ui.btnThresholdGaussianC_2.clicked.connect(self.adaptiveThreshold_gaussianC_2)
        self.ui.btnThresholdMeanC_2.clicked.connect(self.adaptiveThreshold_meanC_2)
        self.ui.btnKmeans.clicked.connect(self.KmeansMethod)
        self.show()

    def actionClicked(self, action):
        print('Action: ', action.text())

    def displayMethod(self):
        self.fileName = QFileDialog.getOpenFileName()[0]
        self.ui.Old_Picture.setScaledContents(True)
        pixmap = QPixmap(self.fileName)
        self.ui.Old_Picture.setPixmap(pixmap)
        self.ui.Old_Picture.repaint()
        QApplication.processEvents()

    def lightImageMethod(self):
        img_origin = cv2.imread(self.fileName, 0)
        img_light = img_origin + 20
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/img_light.png", img_light)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_light.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def darkImageMethod(self):
        img_origin = cv2.imread(self.fileName, 0)
        img_dark = img_origin - 5
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/img_dark.png", img_dark)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_dark.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def multipleImageMethod(self):
        img_origin = cv2.imread(self.fileName, 0)
        img_mul = img_origin * 20
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/img_mul.png", img_mul)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_mul.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def logaritImageMethod(self):
        img_origin = cv2.imread(self.fileName)
        img_log = np.uint8(np.log1p(img_origin))
        thresh = 1
        img_log_trans = cv2.threshold(img_log, thresh, 255, cv2.THRESH_BINARY)[1]
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/img_log.png", img_log_trans)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_log.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gammaImageMethod(self):
        img_noise = cv2.imread(self.fileName, 0)
        gamma = 2
        img_gamma = np.power(img_noise, gamma)
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/img_gamma.png", img_gamma)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_gamma.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def histogramMethod(self):
        img = cv2.imread(self.fileName, 0)
        # vẽ histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title('Histogram for gray scale picture')
        plt.savefig('./images/histogram.png')
        plt.close()
        # hiển thị hình ảnh kết quả
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/histogram.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def equalhistogramMethod(self):
        img = cv2.imread(self.fileName, 0)
        # can bang anh
        img_equal = cv2.equalizeHist(img)
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/img_equal.png", img_equal)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_equal.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def tuyentinhMethod(self):
        # bien doi tuyen tinh
        img = cv2.imread(self.fileName, 0)
        imgMean = np.mean(img)
        imgStd = np.std(img)
        outMean = 100
        outStd = 20
        scale = outStd / imgStd
        shift = outMean - scale * imgMean
        imgLinear = shift + scale * img
        # hiển thị hình ảnh kết quả
        cv2.imwrite("./images/linear.png", imgLinear)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/linear.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gaussianMethod(self):
        img_noise = cv2.imread(self.fileName, 0)
        # Gaussian
        self.size = int(self.ui.txtFilter.text())
        img_gauss = cv2.GaussianBlur(img_noise, (self.size, self.size), 0)
        # Show pictures
        cv2.imwrite("./images/Gaussian_img.png", img_gauss)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/Gaussian_img.png", img_gauss)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def blurMethod(self):
        img_noise = cv2.imread(self.fileName, 0)

        # Blur
        self.size = int(self.ui.txtFilter.text())
        img_blur = cv2.blur(img_noise, (self.size, self.size), 0)

        # Show Pictures
        cv2.imwrite("./images/Blur_img.png", img_blur)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/Blur_img.png", img_blur)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def medianMethod(self):
        img_noise = cv2.imread(self.fileName, 0)

        # Median
        self.size = int(self.ui.txtFilter.text())
        img_median = cv2.medianBlur(img_noise, self.size)

        # Show picture
        cv2.imwrite("./images/Median_img.png", img_median)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/Median_img.png", img_median)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def sharpeningMethod(self):
        img_origin = cv2.imread(self.fileName)

        dst = np.empty_like(img_origin)  # create empty array the size of the image
        noise = cv2.randn(dst, (0, 0, 0), (20, 20, 20))  # add random img noise
        # Pass img through noise filter to add noise
        img_noise = cv2.addWeighted(img_origin, 0.5, noise, 0.5, 50)

        # Blur
        self.size = int(self.ui.txtFilter.text())
        img_gauss = cv2.GaussianBlur(img_noise, (self.size, self.size), 0)

        # Sharpening
        img_sharpening = img_origin + (img_origin - img_gauss)

        # Show pictures
        cv2.imwrite("./images/Sharpening_img.png", img_sharpening)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/Sharpening_img.png", img_sharpening)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gxMethod(self):
        img_origin = cv2.imread(self.fileName)
        dst = np.empty_like(img_origin)  # create empty array the size of the image
        noise = cv2.randn(dst, (0, 0, 0), (20, 20, 20))  # add random img noise
        # Pass img through noise filter to add noise
        img_noise = cv2.addWeighted(img_origin, 0.5, noise, 0.5, 50)

        arrx = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
        img_filtedx = cv2.filter2D(img_origin, -1, arrx)

        img_gx = img_origin - img_filtedx

        # Show pictures
        cv2.imwrite("./images/img_gx.png", img_gx)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_gx.png", img_gx)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gyMethod(self):
        img_origin = cv2.imread(self.fileName)

        arry = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
        img_filtedy = cv2.filter2D(img_origin, -1, arry)
        img_gy = img_origin - img_filtedy

        # Show pictures

        cv2.imwrite("./images/img_gy.png", img_gy)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_gy.png", img_gy)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gx_plus_gyMethod(self):
        img_origin = cv2.imread(self.fileName)

        arrx = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
        img_filtedx = cv2.filter2D(img_origin, -1, arrx)
        arry = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
        img_filtedy = cv2.filter2D(img_origin, -1, arry)

        img_gx = img_origin + (img_origin - img_filtedx)
        img_gy = img_origin + (img_origin - img_filtedy)
        img_gx_plus_gy = img_gx + img_gy

        # Show pictures
        cv2.imwrite("./images/img_gx_plus_gy.png", img_gx_plus_gy)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_gx_plus_gy.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def Gaussian_filter_3D_wirefame(self):

        x = np.arange(-5, 5, 0.5)
        y = np.arange(-5, 5, 0.5)
        x, y = np.meshgrid(x, y)
        array = np.sqrt(x ** 2 + y ** 2)
        z = np.sin(array)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(x, y, z, color='black')
        plt.savefig('./images/wirefame.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/wirefame.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def Gaussian_filter_3D_surface(self):

        x = np.arange(-5, 5, 0.5)
        y = np.arange(-5, 5, 0.5)
        x, y = np.meshgrid(x, y)
        array = np.sqrt(x ** 2 + y ** 2)
        z = np.sin(array)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.savefig('./images/surface.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/surface.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    # Lowpass
    def ideallp(self):
        img = cv2.imread(self.fileName)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                if d[i, j] <= 100:  # d0 = 100
                    H[i, j] = 1
                else:
                    H[i, j] = 0

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, H, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.savefig('./images/Ideal_low_pass.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/Ideal_low_pass.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def ideallp_applyimg(self):
        img = cv2.imread(self.fileName, 0)
        # print(img)
        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                if d[i, j] <= 40:  # d0 = 40
                    H[i, j] = 1
                else:
                    H[i, j] = 0

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_ideal_lp = np.uint8(abs(np.fft.ifft2(np.fft.ifftshift(img_apply))))

        # Show pictures
        cv2.imwrite("./images/img_ideal_lp.png", img_ideal_lp)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_ideal_lp.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def butterworthlp(self, do=50, n=2):
        img = cv2.imread(self.fileName)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                H[i, j] = 1 / (1 + (d[i, j] / 50) ** (2 * n))  # cho Do = 50, n =2

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, H, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.savefig('./images/butterworth_low_pass.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/butterworth_low_pass.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def butterworthlp_applyimg(self, do=50, n=2):
        img = cv2.imread(self.fileName, 0)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                H[i, j] = 1 / (1 + (d[i, j] / 50) ** (2 * n))  # cho Do = 50, n =2

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_butterworth_lp = abs(np.fft.ifft2(np.fft.ifftshift(img_apply)))
        img_butterworth_lp = np.uint8(img_butterworth_lp)

        # Show pictures
        cv2.imwrite("./images/img_butterworth_lp.png", img_butterworth_lp)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_butterworth_lp.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gaussianlp(self):
        img = cv2.imread(self.fileName)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        H = pow(math.e, (-d ** 2 / (2 * (10 ** 2))))  # cho sigma=50

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, H, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.savefig('./images/gaussian_low_pass.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/gaussian_low_pass.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gaussianlp_applyimg(self, do=50, n=2):
        img = cv2.imread(self.fileName, 0)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        H = pow(math.e, (-d ** 2 / (2 * (10 ** 2))))  # cho sigma=10

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_gaussian_lp = abs(np.fft.ifft2(np.fft.ifftshift(img_apply)))
        img_gaussian_lp = np.uint8(img_gaussian_lp)

        # Show pictures
        cv2.imwrite("./images/img_gaussian_lp.png", img_gaussian_lp)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_gaussian_lp.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    # Highpass
    def idealhp(self):
        img = cv2.imread(self.fileName)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                if d[i, j] <= 100:  # d0 = 100
                    H[i, j] = 0
                else:
                    H[i, j] = 1

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, H, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.savefig('./images/ideal_high_pass.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/ideal_high_pass.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def idealhp_applyimg(self):
        img = cv2.imread(self.fileName, 0)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                if d[i, j] <= 40:  # d0 = 100
                    H[i, j] = 0
                else:
                    H[i, j] = 1

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_ideal_hp = np.uint8(abs(np.fft.ifft2(np.fft.ifftshift(img_apply))))

        # Show pictures
        cv2.imwrite("./images/img_ideal_hp.png", img_ideal_hp)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_ideal_hp.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def butterworthhp(self, do=50, n=2):
        img = cv2.imread(self.fileName)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                H[i, j] = 1 / (1 + (50 / d[i, j]) ** (2 * n))  # cho Do = 5, n =2

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, H, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.savefig('./images/butterworth_high_pass.png')
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/butterworth_high_pass.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def butterworthhp_applyimg(self, do=50, n=2):
        img = cv2.imread(self.fileName, 0)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        for i in range(0, d.shape[0]):
            for j in range(0, d.shape[1]):
                H[i, j] = 1 / (1 + (50 / d[i, j]) ** (2 * n))  # cho Do = 5, n =2

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_butterworth_hp = abs(np.fft.ifft2(np.fft.ifftshift(img_apply)))
        img_butterworth_hp = np.uint8(img_butterworth_hp)

        # Show pictures
        cv2.imwrite("./images/img_butterworth_hp.png", img_butterworth_hp)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_butterworth_hp.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gaussianhp(self):
        img = cv2.imread(self.fileName)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        H = d.copy()
        H = 1 - pow(math.e, (-d ** 2 / (2 * (50 ** 2))))  # cho sigma=50

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x, y, H, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        fig.savefig("./images/gaussian_high_pass.png")
        plt.close()

        # Show pictures
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap('./images/gaussian_high_pass.png')
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def gaussianhp_applyimg(self, do=50, n=2):
        img = cv2.imread(self.fileName, 0)

        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        # H = d.copy()
        H = 1 - pow(math.e, (-d ** 2 / (2 * (50 ** 2))))  # cho sigma=50

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_gaussian_hp = abs(np.fft.ifft2(np.fft.ifftshift(img_apply)))
        img_gaussian_hp = np.uint8(img_gaussian_hp)

        # Show pictures
        cv2.imwrite("./images/img_gaussian_hp.png", img_gaussian_hp)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_gaussian_hp.png")
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_binary(self):
        img = cv2.imread(self.fileName, 0)
        threshold_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        # Show pictures
        cv2.imwrite("./images/threshold_binary.png", threshold_binary)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary.png", threshold_binary)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_tozero(self):
        img = cv2.imread(self.fileName, 0)
        thresholdTozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)[1]
        # Show pictures
        cv2.imwrite("./images/threshold_binary.png", thresholdTozero)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary.png", thresholdTozero)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_trunc(self):
        img = cv2.imread(self.fileName, 0)
        threshold_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)[1]

        # Show pictures
        cv2.imwrite("./images/threshold_binary.png", threshold_trunc)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary.png", threshold_trunc)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_otsu(self):
        img = cv2.imread(self.fileName, 0)
        threshold_otsu = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)[1]
        # Show pictures
        cv2.imwrite("./images/threshold_otsu.png", threshold_otsu)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_otsu.png", threshold_otsu)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_gaussianC(self):
        img = cv2.imread(self.fileName, 0)
        threshold_gaussianC = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        # Show pictures
        cv2.imwrite("./images/threshold_binary.png", threshold_gaussianC)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary.png", threshold_gaussianC)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_meanC(self):
        img = cv2.imread(self.fileName, 0)
        threshold_meanC = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
        # Show pictures
        cv2.imwrite("./images/threshold_binary.png", threshold_meanC)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary.png", threshold_meanC)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_gaussianC_2(self):
        img = cv2.imread(self.fileName, 0)
        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        # H = d.copy()
        H = pow(math.e, (-d ** 2 / (2 * (10 ** 2))))  # cho sigma=50

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_gaussian_lp = abs(np.fft.ifft2(np.fft.ifftshift(img_apply)))
        img_gaussian_lp = np.uint8(img_gaussian_lp)
        threshold_gaussianC_radiograph = cv2.adaptiveThreshold(img_gaussian_lp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 305, 0)
        # Show pictures
        cv2.imwrite("./images/threshold_binary_radiograph.png", threshold_gaussianC_radiograph)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary_radiograph.png", threshold_gaussianC_radiograph)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def adaptiveThreshold_meanC_2(self):
        img = cv2.imread(self.fileName, 0)
        sx, sy = img.shape[:2]
        y = np.arange(-sx / 2, sx / 2)  # tâm là (0,0)
        x = np.arange(-sy / 2, sy / 2)

        x, y = np.meshgrid(x, y)
        d = np.sqrt(x ** 2 + y ** 2)

        # H = d.copy()
        H = pow(math.e, (-d ** 2 / (2 * (10 ** 2))))  # cho sigma=50

        g = np.fft.fftshift(np.fft.fft2(img))  # fft and shift to center
        img_apply = g * H  # apply filter
        img_gaussian_lp = abs(np.fft.ifft2(np.fft.ifftshift(img_apply)))
        img_gaussian_lp = np.uint8(img_gaussian_lp)
        threshold_meanC_radiograph = cv2.adaptiveThreshold(img_gaussian_lp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 305, 0)
        # Show pictures
        cv2.imwrite("./images/threshold_binary.png", threshold_meanC_radiograph)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/threshold_binary.png", threshold_meanC_radiograph)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def rotate90imageMethod(self):
        img_rotate = cv2.imread(self.fileName, 0)
        # xoay 90 do theo chieu kim dong ho
        img_rotate_90 = cv2.rotate(img_rotate, cv2.ROTATE_90_CLOCKWISE)
        # Show pictures
        cv2.imwrite("./images/img_rotate_90.png", img_rotate_90)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_rotate_90.png", img_rotate_90)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def rotate180imageMethod(self):
        img_rotate = cv2.imread(self.fileName, 0)
        # xoay 180 do theo chieu kim dong ho
        img_rotate_180 = cv2.rotate(img_rotate, cv2.ROTATE_180)
        # Show pictures
        cv2.imwrite("./images/img_rotate_180.png", img_rotate_180)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_rotate_180.png", img_rotate_180)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def rotate270imageMethod(self):
        img_rotate = cv2.imread(self.fileName, 0)
        # xoay 270do theo chieu kim dong ho
        img_rotate_270 = cv2.rotate(img_rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Show pictures
        cv2.imwrite("./images/img_rotate_270.png", img_rotate_270)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/img_rotate_270.png", img_rotate_270)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def sobel_x(self):
        img = cv2.imread(self.fileName, 0)
        sobel_x = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
        # Show pictures
        cv2.imwrite("./images/sobel_x.png", sobel_x)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/sobel_x.png", sobel_x)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def sobel_y(self):
        img = cv2.imread(self.fileName, 0)
        sobel_y = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
        # Show pictures
        cv2.imwrite("./images/sobel_y.png", sobel_y)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/sobel_y.png", sobel_y)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def sobel(self):
        img = cv2.imread(self.fileName, 0)
        img_sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
        img_sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)
        img_sobel = img_sobelx + img_sobely
        # Show pictures
        cv2.imwrite("./images/sobel.png", img_sobel)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/sobel.png", img_sobel)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def prewitt_x(self):
        img = cv2.imread(self.fileName, 0)
        kernelx = np.array([[-5, -5, -5],
                       [0, 0, 0],
                       [5, 5, 5]])
        img_prewittx = cv2.filter2D(img, -1, kernelx)
        # Show pictures
        cv2.imwrite("./images/prewitt_x.png", img_prewittx)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/prewitt_x.png", img_prewittx)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def prewitt_y(self):
        img = cv2.imread(self.fileName, 0)
        kernely = np.array([[-5, 0, 5],
                        [-5, 0, 5],
                        [-5, 0, 5]])
        img_prewitty = cv2.filter2D(img, -1, kernely)

        # Show pictures
        cv2.imwrite("./images/prewitt_y.png", img_prewitty)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/prewitt_y.png", img_prewitty)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def prewitt(self):
        img = cv2.imread(self.fileName, 0)
        kernelx = np.array([[-5, -5, -5],
                            [0, 0, 0],
                            [5, 5, 5]])
        kernely = np.array([[-5, 0, 5],
                            [-5, 0, 5],
                            [-5, 0, 5]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img, -1, kernelx)
        img_prewitty = cv2.filter2D(img, -1, kernely)
        img_prewitt = img_prewittx+img_prewitty

        # Show pictures
        cv2.imwrite("./images/prewitt.png", img_prewitt)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/prewitt.png", img_prewitt)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def KmeansMethod(self):
        img = cv2.imread(self.fileName, 0)

        self.size = int(self.ui.txtKmeans.text())
        z = img.reshape((-1, 3))
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(z, self.size, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        img_kmeans = res.reshape(img.shape)

        # Show Pictures
        cv2.imwrite("./images/Kmeans.png", img_kmeans)
        self.ui.New_Picture.setScaledContents(True)
        pixmap = QPixmap("./images/Kmeans.png", img_kmeans)
        self.ui.New_Picture.setPixmap(pixmap)
        self.ui.New_Picture.repaint()
        QApplication.processEvents()

    def exitMethod(self):
        QApplication.instance().quit()


app = QtWidgets.QApplication(sys.argv)
window = MyForm()
app.exec_()
