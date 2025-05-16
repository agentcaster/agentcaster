import os
import sys
from datetime import datetime, timedelta
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath('SHARPpy'))

from sharppy.io.buf_decoder import BufDecoder

def extract_coordinates_from_bufkit(bufkit_file, profile=None):
    lat = None if profile is None else profile.latitude
    lon = None
    
    try:
        with open(bufkit_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "STID" in line and i+1 < len(lines):
                    latlon_line = lines[i+1]
                    parts = latlon_line.split()
                    
                    for j in range(len(parts)):
                        if parts[j] == 'SLAT' and j+2 < len(parts):
                            file_lat = float(parts[j+2])
                            if lat is None:
                                lat = file_lat
                        if parts[j] == 'SLON' and j+2 < len(parts):
                            lon = float(parts[j+2])
                    break
    except Exception as e:
        print(f"Warning: Could not extract lat/lon from file: {str(e)}")
        return None, None
    
    if lat is None or lon is None:
        return None, None
    
    return lat, lon

def convert_bufkit_to_sharppy(bufkit_file, fcst_hour, output_file, filter_missing=True, max_pressure=100.0):
    print(f"Reading BUFKIT file: {bufkit_file}")
    print(f"Extracting forecast hour: {fcst_hour}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        decoder = BufDecoder(bufkit_file)
        prof_collection = decoder.getProfiles()
    except Exception as e:
        print(f"Error reading BUFKIT file: {str(e)}")
        sys.exit(1)
    
    base_time = prof_collection.getMeta('base_time')
    
    valid_time = base_time + timedelta(hours=fcst_hour)
    
    prof_collection.setCurrentDate(valid_time)
    
    closest_profile = prof_collection.getHighlightedProf()
    
    if closest_profile is None:
        print(f"No profile found for forecast hour {fcst_hour}")
        sys.exit(1)
    
    site_id = closest_profile.location
    profile_time = closest_profile.date
    profile_time_str = profile_time.strftime("%y%m%d/%H%M")
    
    base_time_hour = base_time.strftime("%Hz")
    
    lat, lon = extract_coordinates_from_bufkit(bufkit_file, closest_profile)
    
    if lat is None or lon is None:
        print(f"Could not extract coordinates from {bufkit_file}. Skipping.")
        sys.exit(1)
    
    p = closest_profile.pres
    h = closest_profile.hght
    t = closest_profile.tmpc
    td = closest_profile.dwpc
    wdir = closest_profile.wdir
    wspd = closest_profile.wspd
    
    if filter_missing:
        valid_mask = ~np.ma.getmaskarray(td)
        
        p = p[valid_mask]
        h = h[valid_mask]
        t = t[valid_mask]
        td = td[valid_mask]
        wdir = wdir[valid_mask]
        wspd = wspd[valid_mask]
        
        troposphere_mask = p >= max_pressure
        
        p = p[troposphere_mask]
        h = h[troposphere_mask]
        t = t[troposphere_mask]
        td = td[troposphere_mask]
        wdir = wdir[troposphere_mask]
        wspd = wspd[troposphere_mask]
        
        print(f"Filtering applied: Removed levels with missing data.")
        print(f"Troposphere filtering: kept only levels at or below {max_pressure} mb.")
    
    content = []
    content.append("%TITLE%")
    content.append(f" {site_id}   {profile_time_str} {lat:.2f},{lon:.2f} ({base_time_hour} BUFKIT F{fcst_hour:03d})")
    content.append("")
    content.append("   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD")
    content.append("-------------------------------------------------------------------")
    content.append("%RAW%")
    
    for i in range(len(p)):
        content.append(f" {p[i]:.2f},    {h[i]:.2f},     {t[i]:.2f},     {td[i]:.2f},    {wdir[i]:.2f},     {wspd[i]:.2f}")
    
    content.append("%END%")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(content))
    
    print(f"SHARPpy-readable file created: {output_file}")
    print(f"Profile has {len(p)} levels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert BUFKIT file to SHARPpy format for a specific forecast hour.')
    parser.add_argument('bufkit_file', help='Path to the input BUFKIT file.')
    parser.add_argument('fcst_hour', type=int, help='Forecast hour to extract.')
    parser.add_argument('output_file', help='Path to save the output SHARPpy-readable file.')
    parser.add_argument('--no-filter', action='store_true', help='Disable filtering of missing data (not recommended)')
    parser.add_argument('--max-pressure', type=float, default=100.0, help='Maximum pressure level to include (mb), default=100.0')
    
    args = parser.parse_args()

    try:
        convert_bufkit_to_sharppy(
            args.bufkit_file, 
            args.fcst_hour, 
            args.output_file,
            filter_missing=not args.no_filter,
            max_pressure=args.max_pressure
        )
    except Exception as e:
        print(f"Error processing BUFKIT file: {str(e)}")
        sys.exit(1)
