import os
import cv2
import numpy as np
from ftplib import FTP
import shutil
import random
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import sys 
import time 
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import math
import sympy as sp

MIRROR_FILE_PATH = r'mirror_command/mirror_change.txt'
DISPERSION_FILE_PATH = r'dazzler_command/dispersion.txt'

with open(MIRROR_FILE_PATH, 'r') as file:
    content = file.read()
mirror_values = list(map(int, content.split()))

with open(DISPERSION_FILE_PATH, 'r') as file:
    content = file.readlines()

dispersion_values = {
    0: int(content[0].split('=')[1].strip()),  # 0 is the key for 'order2'
}

class BetatronApplication(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super(BetatronApplication, self).__init__(*args, **kwargs)

        self.new_focus = 0  
        self.new_second_dispersion = 0  

        self.count_grad = 0
        self.dir_run_count = 0
        self.run_count = 0
        self.count_history = []
        self.focus_learning_rate = 4
        self.second_dispersion_learning_rate = 4
        self.momentum = 0.999

    # ------------ Plotting ------------ #

        self.second_dispersion_der_history = []
        self.focus_der_history = []
        self.total_gradient_history = []

        self.iteration_data = []
        self.der_iteration_data = []
        self.count_data = []
        
        self.count_plot_widget = pg.PlotWidget()
        self.count_plot_widget.setWindowTitle('count optimization')
        self.count_plot_widget.setLabel('left', 'count')
        self.count_plot_widget.setLabel('bottom', 'iteration')
        self.count_plot_widget.showGrid(x=True, y=True)
        self.count_plot_widget.show()

        self.main_plot_window = pg.GraphicsLayoutWidget()
        self.main_plot_window.show()

        layout = self.main_plot_window.addLayout(row=0, col=0)

        self.count_plot_widget = layout.addPlot(title='count vs iteration')
        self.focus_plot = layout.addPlot(title='count_focus_derivative')
        self.second_dispersion_plot = layout.addPlot(title='count_second_dispersion_derivative')
        self.total_gradient_plot = layout.addPlot(title='total_gradient')

        subplots = [self.count_plot_widget, self.focus_plot, self.second_dispersion_plot, self.total_gradient_plot]
        for subplot in subplots:
            subplot.showGrid(x=True, y=True)

        self.plot_curve = self.count_plot_widget.plot(pen='r')
        self.focus_curve = self.focus_plot.plot(pen='r', name='focus derivative')
        self.second_dispersion_curve = self.second_dispersion_plot.plot(pen='g', name='second dispersion derivative')
        self.total_gradient_curve = self.total_gradient_plot.plot(pen='y', name='total gradient')

        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.focus_curve.setData(self.der_iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.der_iteration_data, self.second_dispersion_der_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

    # ------------ Deformable mirror ------------ #

        # init -150
        self.MIRROR_HOST = "192.168.200.3"
        self.MIRROR_USER = "Utilisateur"
        self.MIRROR_PASSWORD = "alls"    

        self.initial_focus = -240
        self.focus_history = []    
        # self.FOCUS_LOWER_BOUND = max(self.initial_focus - 20, -200)
        # self.FOCUS_UPPER_BOUND = min(self.initial_focus + 20, 200)

        self.FOCUS_LOWER_BOUND = -99999
        self.FOCUS_UPPER_BOUND = +99999

        self.tolerance = 1
        
    # ------------ Dazzler ------------ #

        self.DAZZLER_HOST = "192.168.58.7"
        self.DAZZLER_USER = "fastlite"
        self.DAZZLER_PASSWORD = "fastlite"

        # 36100 initial 
        self.initial_second_dispersion = -240
        self.second_dispersion_history = []
        # self.SECOND_DISPERSION_LOWER_BOUND = max(self.initial_second_dispersion - 500, 30000)
        # self.SECOND_DISPERSION_UPPER_BOUND = min(self.initial_second_dispersion + 500, 40000)

        self.SECOND_DISPERSION_LOWER_BOUND = -99999
        self.SECOND_DISPERSION_UPPER_BOUND = +99999

        self.random_direction = [random.choice([-1, 1]) for _ in range(4)]

    def upload_files(self):
        mirror_ftp = FTP()
        dazzler_ftp = FTP()

        mirror_ftp.connect(host=self.MIRROR_HOST)
        mirror_ftp.login(user=self.MIRROR_USER, passwd=self.MIRROR_PASSWORD)

        dazzler_ftp.connect(host=self.DAZZLER_HOST)
        dazzler_ftp.login(user=self.DAZZLER_USER, passwd=self.DAZZLER_PASSWORD)

        mirror_files = [os.path.basename(MIRROR_FILE_PATH)]
        dazzler_files = [os.path.basename(DISPERSION_FILE_PATH)]

        for mirror_file_name in mirror_files:
            for dazzler_file_name in dazzler_files:
                focus_file_path = MIRROR_FILE_PATH
                dispersion_file_path = DISPERSION_FILE_PATH

                if os.path.isfile(focus_file_path) and os.path.isfile(dispersion_file_path):
                    copy_mirror_IMG_PATH = os.path.join('mirror_command', f'copy_{mirror_file_name}')
                    copy_dazzler_IMG_PATH = os.path.join('dazzler_command', f'copy_{dazzler_file_name}')

                    try:
                        os.makedirs(os.path.dirname(copy_mirror_IMG_PATH))
                        os.makedirs(os.path.dirname(copy_dazzler_IMG_PATH))
                    except OSError:
                        pass

                    shutil.copy(focus_file_path, copy_mirror_IMG_PATH)
                    shutil.copy(dispersion_file_path, copy_dazzler_IMG_PATH)

                    with open(copy_mirror_IMG_PATH, 'rb') as local_file:
                        mirror_ftp.storbinary(f'STOR {mirror_file_name}', local_file)
                        print(f"Uploaded to mirror FTP: {mirror_file_name}")

                    with open(copy_dazzler_IMG_PATH, 'rb') as local_file:
                        dazzler_ftp.storbinary(f'STOR {dazzler_file_name}', local_file)
                        print(f"Uploaded to dazzler FTP: {dazzler_file_name}")

                    os.remove(copy_mirror_IMG_PATH)
                    os.remove(copy_dazzler_IMG_PATH)

    def write_values(self):

        self.new_focus = round(np.clip(self.focus_history[-1], self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND))
        self.new_second_dispersion = round(np.clip(self.second_dispersion_history[-1], self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND))

        mirror_values[0] = self.new_focus
        dispersion_values[0] = self.new_second_dispersion

        with open(MIRROR_FILE_PATH, 'w') as file:
            file.write(' '.join(map(str, mirror_values)))

        with open(DISPERSION_FILE_PATH, 'w') as file:
            file.write(f'order2 = {dispersion_values[0]}\n')

        # self.upload_files() # send files to second computer

        QtCore.QCoreApplication.processEvents()

    def count_function(self):

        x = self.focus_history[-1]
        y = self.second_dispersion_history[-1]

        count_func = (((0.1 * (x + y)))** 2 * np.sin(0.01 * (x + y)))

        self.count_history.append(count_func) # this is the count for the value
            
    def calc_derivatives(self):
        x = self.focus_history[-1]
        y = self.second_dispersion_history[-1]

        self.count_focus_der = 0.2*(0.1*(x+y))*np.sin(0.01*(x+y))+0.01*(np.cos(0.01*(x+y)))*(0.1*(x+y))**2
        self.count_second_dispersion_der = 0.2*(0.1*(x+y))*np.sin(0.01*(x+y))+0.01*(np.cos(0.01*(x+y)))*(0.1*(x+y))**2

        self.focus_der_history.append(self.count_focus_der)
        self.second_dispersion_der_history.append(self.count_second_dispersion_der)

        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1])

        self.total_gradient_history.append(self.total_gradient)
        
        return {"focus":self.count_focus_der,"second_dispersion":self.count_second_dispersion_der}

    def optimize_count(self):

        if np.abs(self.focus_learning_rate * self.focus_der_history[-1]) > 1:
            self.new_focus = (self.focus_history[-1] + (self.momentum*(self.focus_history[-1]-self.focus_history[-2])) - self.focus_learning_rate*self.focus_der_history[-1])

            self.new_focus = np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)
            self.new_focus = round(self.new_focus)

            self.focus_history.append(self.new_focus)
            mirror_values[0] = self.focus_history[-1]

        if np.abs(self.second_dispersion_learning_rate * self.second_dispersion_der_history[-1]) > 1:

            self.new_second_dispersion = (self.second_dispersion_history[-1] + (self.momentum*(self.second_dispersion_history[-1]-self.second_dispersion_history[-2])) - self.second_dispersion_learning_rate*self.second_dispersion_der_history[-1])
                                                       
            self.new_second_dispersion = np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)
            self.new_second_dispersion = round(self.new_second_dispersion)

            self.second_dispersion_history.append(self.new_second_dispersion)
            dispersion_values[0] = self.second_dispersion_history[-1]

    def process_images(self):
        self.run_count += 1
        self.iteration_data.append(self.run_count)
        
        if self.run_count == 1 or self.run_count == 2:


            if self.run_count == 1:
                print('-------------')      
                self.focus_history.append(self.initial_focus)                       
                self.second_dispersion_history.append(self.initial_second_dispersion)

            if self.run_count == 2:
                self.new_focus = self.focus_history[-1] +1
                self.new_second_dispersion =  self.second_dispersion_history[-1] +1

                self.focus_history.append(self.new_focus)
                self.second_dispersion_history.append(self.new_second_dispersion)

            self.count_function()   
            self.calc_derivatives()
            print(f"count {self.count_history[-1]}, focus = {self.focus_history[-1]}, disp2 = {self.second_dispersion_history[-1]}")

        if self.run_count > 2:
            self.count_function()   
            self.calc_derivatives()            
            self.optimize_count()

            print(f"count {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}")

        self.write_values()

        print(f"{self.focus_der_history[-1], self.second_dispersion_der_history[-1]}")
        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.focus_curve.setData(self.iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.iteration_data, self.second_dispersion_der_history)
        self.total_gradient_curve.setData(self.iteration_data, self.total_gradient_history)

        print('-------------')

if __name__ == "__main__":
    app = BetatronApplication([])

    for _ in range(100):
        app.process_images()

    win = QtWidgets.QMainWindow()
    sys.exit(app.exec_())