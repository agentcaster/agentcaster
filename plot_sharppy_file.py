import sys
import os
import numpy as np
import time
import argparse

sys.path.insert(0, os.path.abspath('SHARPpy'))

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtCore import Qt
import sharppy
import sharppy.sharptab as tab
from sharppy.sharptab.constants import *
from sharppy.io.spc_decoder import SPCDecoder
from sharppy.viz.SPCWindow import SPCWindow
from sutils.config import Config

from convert_bufkit_to_sharppy import convert_bufkit_to_sharppy

def main():
    parser = argparse.ArgumentParser(description='Convert BUFKIT file to SHARPpy format and plot the sounding.')
    parser.add_argument('bufkit_file', help='Path to the input BUFKIT file.')
    parser.add_argument('fcst_hour', type=int, help='Forecast hour to extract and plot.')
    
    args = parser.parse_args()
    bufkit_file = args.bufkit_file
    fcst_hour = args.fcst_hour

    try:
        bufkit_dir = os.path.dirname(bufkit_file)
        if not bufkit_dir or not os.path.basename(bufkit_dir) == 'bufkit':
            raise ValueError("Input BUFKIT file must be in a directory named 'bufkit'.")
        date_dir = os.path.dirname(bufkit_dir)
        sharppy_dir = os.path.join(date_dir, "sharppy_files")
        base_name = os.path.splitext(os.path.basename(bufkit_file))[0]
        sharppy_filename = f"{base_name}_f{fcst_hour:02d}.sharppy"
        sharppy_file_path = os.path.join(sharppy_dir, sharppy_filename)
    except Exception as e:
        print(f"Error determining intermediate SHARPpy file path: {e}")
        sys.exit(1)

    try:
        convert_bufkit_to_sharppy(bufkit_file, fcst_hour, sharppy_file_path)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"Error during BUFKIT to SHARPpy conversion: {str(e)}")
        sys.exit(1)

    if not os.path.exists(sharppy_file_path):
        print(f"Conversion failed: SHARPpy file not found at {sharppy_file_path}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    cfg = Config(os.path.join(os.path.expanduser("~"), ".sharppy", "sharppy.ini"))
    
    window = MainWindow(cfg, sharppy_file_path)
    
    app.exec_()

class MainWindow(QtWidgets.QMainWindow):
    config_changed = QtCore.Signal()
    
    def __init__(self, cfg, file_path):
        super(MainWindow, self).__init__()
        self.cfg = cfg
        self.file_path = file_path
        
        self.spc_window = SPCWindow(parent=self, cfg=self.cfg)
        
        self.loadSounding()
        
        self.spc_window.show()
        
        QtCore.QTimer.singleShot(100, self.saveImageAndClose)
    
    def loadSounding(self):
        decoder = SPCDecoder(self.file_path)
        prof_collection = decoder.getProfiles()
        
        import os
        import re
        from datetime import datetime, timedelta
        
        file_name = os.path.basename(self.file_path)
        fcst_hour = 0
        match = re.search(r'_(\d+)z\.txt$', file_name)
        if match:
            fcst_hour = int(match.group(1))
        
        date_str = None
        match = re.search(r'_(\d{6})_', file_name)
        if match:
            date_str = match.group(1)
        
        highlighted_prof = prof_collection.getHighlightedProf()
        profile_date = highlighted_prof.date
        
        model_run_hour = None
        lat = None
        lon = None
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[:5]:
                model_info = re.search(r'\((.*?) BUFKIT F(\d+)\)', line)
                if model_info:
                    model_run_hour = model_info.group(1)
                    fcst_hour = int(model_info.group(2))
                
                coord_info = re.search(r'(\d+\.\d+),(-\d+\.\d+)', line)
                if coord_info:
                    lat = float(coord_info.group(1))
                    lon = float(coord_info.group(2))
                    break
        
        run_time = profile_date - timedelta(hours=fcst_hour)
        
        prof_collection.setMeta('model', 'BUFKIT')
        prof_collection.setMeta('run', run_time)
        prof_collection.setMeta('loc', highlighted_prof.location)
        prof_collection.setMeta('observed', False)
        
        prof_collection.setMeta('base_time', run_time)
        
        prof_collection.setMeta('fhour', fcst_hour)
        prof_collection.setMeta('forecast_hour', fcst_hour)
        
        if lat is not None and lon is not None:
            prof_collection.setMeta('lat', lat)
            prof_collection.setMeta('lon', lon)
            
            highlighted_prof.latitude = lat
            highlighted_prof.longitude = lon
        
        highlighted_prof.fhour = fcst_hour
        
        self.spc_window.addProfileCollection(prof_collection)
    
    def saveImageAndClose(self):
        input_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        output_filename = os.path.join(input_dir, base_name + ".png")

        pixmap = self.spc_window.grab()

        pixmap.save(output_filename, "PNG")
        print(f"Image saved as {output_filename}")

        QtWidgets.QApplication.quit()
    
    def preferencesbox(self):
        pass

if __name__ == '__main__':
    main()
