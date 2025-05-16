import pandas as pd
import numpy as np
import pyproj
import xarray as xr
import os
import warnings
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects
import glob
import re

GRID_80km_PROJ_PARAMS = {
    "proj": "lcc",
    "lat_1": 25.0,
    "lat_2": 25.0,
    "lat_0": 25.0,
    "lon_0": -95.0,
    "a": 6371200,
    "b": 6371200,
    "units": "m"
}
GRID_80km_NX = 93
GRID_80km_NY = 65
GRID_80km_DX = 81270.5
GRID_80km_DY = 81270.5

GRID_5km_PROJ_PARAMS = GRID_80km_PROJ_PARAMS
GRID_5km_DX = 40635.0 / 8.0
GRID_5km_DY = 40635.0 / 8.0
GRID_5km_NX = 185 * 8
GRID_5km_NY = 129 * 8

CONUS_LON_MIN, CONUS_LON_MAX = -121.0, -65.0
CONUS_LAT_MIN, CONUS_LAT_MAX = 22.0, 50.0

proj_grid = pyproj.Proj(GRID_80km_PROJ_PARAMS)
crs_grid_5km = pyproj.CRS(GRID_5km_PROJ_PARAMS)
proj_latlon = pyproj.Proj("epsg:4326")
transformer_ll_to_grid = pyproj.Transformer.from_proj(proj_latlon, proj_grid, always_xy=True)
transformer_grid_to_ll = pyproj.Transformer.from_proj(proj_grid, proj_latlon, always_xy=True)

SPC_RISK_COLORS = {
    '0.02': '#80c480',
    '0.05': '#c6a393',
    '0.10': '#ffeb80',
    '0.15': '#ff8080',
    '0.30': '#ff80ff',
    '0.45': '#c896f7',
    '0.60': '#0f4e8b',
}

CONTOUR_COLORS = {
    '0.02': '#008200',
    '0.05': '#8b4825',
    '0.10': '#ff9601',
    '0.15': '#ff0000',
    '0.30': '#ff00ff',
    '0.45': '#912dee',
}

SPC_RISK_LABELS = {
    '0.02': '2% Tornado Risk',
    '0.05': '5% Tornado Risk',
    '0.10': '10% Tornado Risk',
    '0.15': '15% Tornado Risk',
    '0.30': '30% Tornado Risk',
    '0.45': '45% Tornado Risk',
    '0.60': '60% Tornado Risk',
}

def plot_spc_outlook(spc_shp_path, output_png_path, date_str):
    print(f"Plotting SPC tornado outlook to PNG: {output_png_path}")
    
    try:
        spc_gdf = gpd.read_file(spc_shp_path)
        if spc_gdf.empty:
            print("SPC outlook shapefile is empty. Generating blank map.")
        else:
            print(f"Successfully read SPC outlook with {len(spc_gdf)} risk areas")
            
        map_proj = ccrs.LambertConformal(
            central_longitude=GRID_5km_PROJ_PARAMS['lon_0'],
            central_latitude=GRID_5km_PROJ_PARAMS['lat_0'],
            standard_parallels=(GRID_5km_PROJ_PARAMS['lat_1'], GRID_5km_PROJ_PARAMS['lat_2']),
            globe=ccrs.Globe(ellipse=None, semimajor_axis=GRID_5km_PROJ_PARAMS['a'], semiminor_axis=GRID_5km_PROJ_PARAMS['b'])
        )
        
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=map_proj)
        
        min_x, min_y = transformer_ll_to_grid.transform(CONUS_LON_MIN, CONUS_LAT_MIN)
        max_x, max_y = transformer_ll_to_grid.transform(CONUS_LON_MAX, CONUS_LAT_MAX)
        ax.set_extent([min_x, max_x, min_y, max_y], crs=map_proj)
        
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#d3e4f5')
        ax.add_feature(cfeature.LAND, zorder=0, facecolor='#ffffff')
        ax.add_feature(cfeature.LAKES, zorder=1, alpha=0.5, facecolor='#d3e4f5')
        
        legend_handles = []
        legend_labels = []
        risk_levels_present = set()
        
        if not spc_gdf.empty:
            spc_gdf = spc_gdf.sort_values('DN')
            
            if spc_gdf.crs is None:
                spc_gdf.set_crs("EPSG:4326", inplace=True)
            
            for _, row in spc_gdf.iterrows():
                if 'LABEL' in spc_gdf.columns:
                    risk_level = row['LABEL']
                elif 'DN' in spc_gdf.columns:
                    dn_value = row['DN']
                    if dn_value > 0:
                        risk_level = f"{dn_value/100:.2f}"
                    else:
                        continue
                else:
                    continue

                if risk_level in SPC_RISK_COLORS:
                    color = SPC_RISK_COLORS[risk_level]
                    
                    edge_color = CONTOUR_COLORS.get(risk_level, '#000000')
                    
                    ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(),
                                      facecolor=color, edgecolor=edge_color,
                                      linewidth=0.5, alpha=1.0, zorder=1)
                    
                    if risk_level not in risk_levels_present:
                        risk_levels_present.add(risk_level)
                        patch = Rectangle((0, 0), 1, 1, facecolor=color)
                        legend_handles.append(patch)
                        legend_labels.append(f"{float(risk_level)*100:.0f}%")
        
        ax.add_feature(cfeature.COASTLINE, zorder=2, edgecolor='#949494', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, zorder=2, linestyle='-', edgecolor='#949494', linewidth=0.5)
        ax.add_feature(cfeature.STATES, zorder=2, linestyle='-', edgecolor='#949494', linewidth=0.3)
        
        if legend_handles:
            all_categories = ['2%', '5%', '10%', '15%', '30%', '45%', '60%']
            
            all_colors = []
            for cat in all_categories:
                percentage = int(cat.strip('%'))
                if percentage < 10:
                    key = f"0.0{percentage}"
                else:
                    key = f"0.{percentage}"
                all_colors.append(SPC_RISK_COLORS[key])
            
            consistent_legend_handles = []
            for color in all_colors:
                rect = Rectangle((0, 0), 1, 1, facecolor=color)
                consistent_legend_handles.append(rect)
            
            legend = ax.legend(consistent_legend_handles, all_categories,
                            loc='lower right',
                            frameon=True,
                            ncol=7,
                            handlelength=1.5,
                            handleheight=1.0,
                            fontsize=9,
                            borderpad=0.4,
                            labelspacing=0.2,
                            columnspacing=0.5,
                            handletextpad=0.4)
            
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('#949494')
            legend.get_frame().set_linewidth(0.5)
        
        forecast_hour = "06Z"
        plt.title(f"SPC Day 1 Tornado Outlook - {date_str} {forecast_hour}")
        
        output_dir = os.path.dirname(output_png_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
        print(f"Successfully saved SPC tornado outlook plot to {output_png_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting SPC tornado outlook: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    base_data_dir = "dataset"
    hrrr_run = "00z"
    
    date_dirs = glob.glob(os.path.join(base_data_dir, f"hrrr_*_{hrrr_run}"))
    
    if not date_dirs:
        print(f"No date directories found in {base_data_dir}")
        exit(1)
    
    date_pattern = re.compile(r'hrrr_(\d{8})_' + hrrr_run)
    
    for date_dir in sorted(date_dirs):
        match = date_pattern.search(date_dir)
        if not match:
            print(f"Could not extract date from directory: {date_dir}, skipping...")
            continue
            
        target_date_str = match.group(1)
        print(f"\n{'='*80}\nProcessing date: {target_date_str}\n{'='*80}")
        
        spc_outlook_dir = os.path.join(date_dir, "spc_outlooks")
        spc_outlook_path = os.path.join(spc_outlook_dir, f"day1otlk_{target_date_str}_1200_torn.shp")
        
        output_dir = os.path.join("spc_output", target_date_str)
        os.makedirs(output_dir, exist_ok=True)
        output_png_path = os.path.join(output_dir, f"SPC Day 1 Tornado Outlook - {target_date_str}.png")
        
        if not os.path.exists(spc_outlook_path):
            print(f"Warning: SPC outlook file not found: {spc_outlook_path}")
            alternate_patterns = [f"day1otlk_{target_date_str}_0[0-9]00_torn.shp", 
                                 f"day1otlk_{target_date_str}_1[0-9]00_torn.shp"]
            
            found_alternative = False
            for pattern in alternate_patterns:
                alt_files = glob.glob(os.path.join(spc_outlook_dir, pattern))
                if alt_files:
                    spc_outlook_path = alt_files[0]
                    print(f"Using alternative SPC outlook file: {spc_outlook_path}")
                    found_alternative = True
                    break
            
            if not found_alternative:
                print(f"No SPC outlook files found for date {target_date_str}")
                continue
        
        plot_spc_outlook(spc_outlook_path, output_png_path, target_date_str)
    
    print("\nProcessed all available dates.")