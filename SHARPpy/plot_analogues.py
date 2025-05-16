"""
Plot a sounding analogues display using SHARPpy's native Qt visualization
"""
import os
import sys
import numpy as np
from datetime import datetime

# Set Qt API before importing Qt modules
os.environ['QT_API'] = 'pyside6'
from qtpy.QtWidgets import QApplication, QMainWindow
from qtpy.QtCore import Qt

from sharppy.sharptab import profile, params, interp, thermo, winds
from sharppy.io.spc_decoder import SPCDecoder
from sharppy.viz.analogues import plotAnalogues

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SHARPpy Sounding Analogues')
        self.resize(800, 600)

        # Load sounding data
        # Use absolute path to the data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'examples', 'data', '14061619.OAX')
        decoder = SPCDecoder(data_path)
        self.prof_collection = decoder.getProfiles()
        
        # Create the analogues display
        self.analogues = plotAnalogues()
        
        # Set the profile
        if self.prof_collection.hasCurrentProf():
            prof = self.prof_collection.getHighlightedProf()
            self.analogues.setProf(prof)

        # Set the widget as the central widget
        self.setCentralWidget(self.analogues)

def main():
    app = QApplication(sys.argv)
    # Handle high DPI displays
    try:
        app.setAttribute(Qt.AA_EnableHighDpiScaling)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    except AttributeError:
        # These attributes might not be available in some Qt versions
        pass
    
    window = MainWindow()
    window.show()
    
    # Handle different Qt versions
    if hasattr(app, 'exec'):
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()
