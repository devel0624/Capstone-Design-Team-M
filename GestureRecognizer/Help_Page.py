import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap

import Image_Resource_rc

form_class = uic.loadUiType("Help_Page.ui")[0]

class Help_Page(QDialog,QWidget, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.show()


#       self.Button_Home.clicked.connect(self.Home)
        

#    def Home(self):
#        self.close()

