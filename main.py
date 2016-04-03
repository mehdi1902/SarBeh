# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:15:29 2016

@author: mehdi
"""

from PyQt4 import QtGui # Import the PyQt4 module we'll need
import sys # We need sys so that we can pass argv to QApplication
import os
import GUI # This file holds our MainWindow and all design related things
              # it also keeps events etc that we defined in Qt Designer
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import segmentation
import learning
from skimage.io import imread, imshow, imsave


class ExampleApp(QtGui.QMainWindow, GUI.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.btn_browse_input.clicked.connect(self.browse_folder)
        self.btn_browse_output.clicked.connect(self.browse_folder)
        self.btn_browse_target.clicked.connect(self.browse_folder)
        self.btn_browse_cores.clicked.connect(self.browse_folder)
        
        
        self.btn_segmentation.clicked.connect(self.segmentation)
        self.btn_learn.clicked.connect(self.learning)
        
    def learning(self):
        cores_path = str(self.text_cores.text())
        target_path = str(self.text_target.text())
        
        learning.learn(cores_path, target_path)

    def browse_folder(self):
        sending_btn = self.sender()
        directory = QtGui.QFileDialog.getExistingDirectory(self,u'انتخاب پوشه')
        if sending_btn.objectName()=='btn_browse_input':
            self.text_input.setText(directory+'/')
        elif sending_btn.objectName()=='btn_browse_output':
            self.text_output.setText(directory+'/')
        elif sending_btn.objectName()=='btn_browse_cores':
            self.text_cores.setText(directory+'/')
        elif sending_btn.objectName()=='btn_browse_target':
            self.text_target.setText(directory+'/')
        
    def segmentation(self):
        input_dir = str(self.text_input.text())
        output_dir = str(self.text_output.text())


        if not os.path.isdir(input_dir):
            QMessageBox.information(self, u'مسیر اشتباه', u'لطفا مسیر درستی برای پوشه ورودی انتخاب کنید.')
        elif not os.path.isdir(output_dir):
            QMessageBox.information(self, u'مسیر اشتباه', u'لطفا مسیر درستی برای پوشه خروجی انتخاب کنید.')
        else:
            
            samples = []
#            counter = 0
            cores = []
            '''
            
            change here
            
            '''
            
            for image in os.listdir(input_dir):
                if (image.split('.')[-1] in ['jpg', 'JPG', 'png', 'PNG']):
                    samples.append(input_dir+image)
            for sample in samples:
                cores.extend(segmentation.segmentation(sample))
#                for i in range(counter, counter+len(cores)):
            for i in range(len(cores)):
                image_name = '%s/%i.png'%(output_dir, i)
                imsave(image_name, cores[i])
#                counter += len(cores)
            QMessageBox.information(self, u'پایان', u'تقسیم‌ بندی با موفقیت انجام شد!')

        

def main():
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()
    app.deleteLater()
    sys.exit()
#    sys.exit(app.quit())
#    quit()
#    sys.exit()
#    exit()
    


if __name__ == '__main__':              # if we're running file directly and not importing it
    main()                              # run the main function