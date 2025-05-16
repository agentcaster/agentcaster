"""
Plot EF-scale Probabilities based on V-rot using SHARPpy's native Qt visualization
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
from sharppy.viz.vrot import plotVROT

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SHARPpy EF-scale Probabilities based on V-rot')
        self.resize(800, 600)
        
        # Create the VROT display
        self.vrot = plotVROT()
        
        # Set the widget as the central widget
        self.setCentralWidget(self.vrot)
        
        # Display instructions
        print("Double-click on the plot to enter a V-rot value.")

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
