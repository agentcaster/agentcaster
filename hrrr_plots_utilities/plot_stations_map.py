import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import ssl
import warnings
import sys

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore', category=UserWarning)
def extract_coordinates_from_bufkit_header(bufkit_file):
    lat, lon = None, None
    try:
        with open(bufkit_file, 'r', encoding='latin-1') as f:
            lines = [f.readline() for _ in range(6)]
            for i, line in enumerate(lines):
                if "STID" in line:

                    if i + 1 < len(lines):
                        latlon_line = lines[i+1]
                        lat_match = re.search(r'SLAT\s*=\s*(-?\d+\.?\d*)', latlon_line)
                        lon_match = re.search(r'SLON\s*=\s*(-?\d+\.?\d*)', latlon_line)
                        if lat_match:
                            lat = float(lat_match.group(1))
                        if lon_match:
                            lon = float(lon_match.group(1))
                    break
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"Warning: Could not parse header for {bufkit_file}: {str(e)}", file=sys.stderr)
        return None, None
    
    if lat is not None and lon is not None:
        return lat, lon
    else:
        return None, None
bufkit_dir = "dataset/hrrr_20250301_00z/bufkit/"

station_lats = []
station_lons = []

if not os.path.isdir(bufkit_dir):
    print(f"Error: Representative BUFKIT directory not found: {bufkit_dir}", file=sys.stderr)
    print("Please ensure data for 20250301 exists or modify the 'bufkit_dir' variable.", file=sys.stderr)
    sys.exit(1)

print(f"Reading station locations from: {bufkit_dir}")
bufkit_files = glob.glob(os.path.join(bufkit_dir, "hrrr_*.buf"))

if not bufkit_files:
    print(f"Error: No BUFKIT files found in {bufkit_dir}", file=sys.stderr)
    sys.exit(1)

for bufkit_file in bufkit_files:
    lat, lon = extract_coordinates_from_bufkit_header(bufkit_file)
    if lat is not None and lon is not None:
        station_lats.append(lat)
        station_lons.append(lon)

print(f"Found {len(station_lats)} stations with valid coordinates.")
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-95, central_latitude=35))


ax.set_extent([-125, -66, 25, 50], crs=ccrs.PlateCarree())


ax.coastlines(resolution='110m')
ax.add_feature(cfeature.BORDERS.with_scale('110m'), linestyle=':')
ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray')


gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax.scatter(station_lons, station_lats,
           transform=ccrs.PlateCarree(), 
           s=10, color='red', alpha=0.7, 
           label=f'Stations ({len(station_lats)} total)')

plt.title('Available Bufkit Station Locations')
legend = ax.legend(loc='lower right')


plt.savefig('station_map.png', dpi=300, bbox_inches='tight')