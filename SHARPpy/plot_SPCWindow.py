import sys
import os
import numpy as np
from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtCore import Qt
import sharppy
import sharppy.sharptab as tab
from sharppy.sharptab.constants import *
from sharppy.io.spc_decoder import SPCDecoder
from sharppy.viz.SPCWindow import SPCWindow
from sutils.config import Config

def main():
    """
    Create a new SPCWindow and display it.
    """
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # Create a configuration object
    cfg = Config(os.path.join(os.path.expanduser("~"), ".sharppy", "sharppy.ini"))
    
    # Create the main window
    window = MainWindow(cfg)
    
    # Start the application
    sys.exit(app.exec_())

class MainWindow(QtWidgets.QMainWindow):
    config_changed = QtCore.Signal()
    
    def __init__(self, cfg):
        super(MainWindow, self).__init__()
        self.cfg = cfg
        
        # Create the SPCWindow
        self.spc_window = SPCWindow(parent=self, cfg=self.cfg)
        
        # Load a sounding profile
        self.loadSounding()
        
        # Show the window
        self.spc_window.show()
    
    def loadSounding(self):
        """
        Load a sounding profile from a file and add it to the SPCWindow.
        """
        # Load the sounding data from a file
        decoder = SPCDecoder('examples/data/14061619.OAX')
        prof_collection = decoder.getProfiles()
        
        # Set required metadata
        from datetime import datetime
        prof_collection.setMeta('model', 'Observed')
        prof_collection.setMeta('run', datetime.now())
        prof_collection.setMeta('loc', 'OAX')
        prof_collection.setMeta('observed', True)
        
        # Add the profile collection to the SPCWindow
        self.spc_window.addProfileCollection(prof_collection)
    
    def preferencesbox(self):
        """
        Dummy method to handle preferences menu item.
        """
        pass

if __name__ == '__main__':
    main()
