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
import json

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
    '<2%': '#FFFFFF',
    '2%': '#80c480',
    '5%': '#c6a393',
    '10%': '#ffeb80',
    '15%': '#ff8080',
    '30%': '#ff80ff',
    '45%': '#c896f7',
    '60%': '#0f4e8b',
}

CONTOUR_COLORS = {
    '2%': '#008200',
    '5%': '#8b4825',
    '10%': '#ff9601',
    '15%': '#ff0000',
    '30%': '#ff00ff',
    '45%': '#912dee',
    '60%': '#000000'
}

PLOT_ORDER = ['<2%', '2%', '5%', '10%', '15%', '30%', '45%', '60%']

def risk_level_sort_key(risk_level):
    if isinstance(risk_level, (int, float)):
        risk_level = str(int(risk_level)) + '%'
    if risk_level == '<2%':
        return 0.01
    try:
        return float(risk_level.strip('%')) / 100.0
    except ValueError:
        return -1

def plot_llm_prediction(geojson_path, output_png_path, date_str, model_name):
    print(f"Plotting LLM prediction [{model_name}] from {os.path.basename(geojson_path)} to PNG: {output_png_path}")
    
    try:
        try:
            llm_gdf = gpd.read_file(geojson_path)
        except Exception as read_err:
            print(f"  Error reading GeoJSON file with geopandas: {read_err}")
            try:
                with open(geojson_path, 'r') as f:
                    geojson_data = json.load(f)
                if geojson_data.get('type') == 'FeatureCollection' and 'features' in geojson_data:
                     features_list = []
                     for feature in geojson_data['features']:
                         geom_dict = feature.get('geometry')
                         if geom_dict:
                             geom = shape(geom_dict)
                             props = feature.get('properties', {})
                             features_list.append({'geometry': geom, **props})
                         else:
                             print(f"Warning: Feature missing geometry in {geojson_path}")
                     llm_gdf = gpd.GeoDataFrame(features_list, crs="EPSG:4326")
                     print("  Successfully read GeoJSON via json fallback.")
                else:
                    raise ValueError("GeoJSON is not a valid FeatureCollection")
            except Exception as fallback_err:
                 print(f"  Fallback json reading also failed: {fallback_err}")
                 print("Could not read GeoJSON file. Skipping plot.")
                 return
                 
        if llm_gdf.empty:
            print("LLM prediction GeoJSON is empty or unreadable. Generating blank map.")
        else:
            if 'risk_level' not in llm_gdf.columns:
                 print("Error: GeoJSON file is missing the 'risk_level' property in features. Skipping plot.")
                 return
            print(f"Successfully read LLM prediction with {len(llm_gdf)} risk areas")
            
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
        
        if not llm_gdf.empty:
            llm_gdf['sort_key'] = llm_gdf['risk_level'].apply(risk_level_sort_key)
            llm_gdf = llm_gdf.sort_values('sort_key')
            
            if llm_gdf.crs is None:
                print("Warning: LLM GeoJSON missing CRS information. Assuming WGS84 (EPSG:4326).")
                llm_gdf.set_crs("EPSG:4326", inplace=True)
            
            for _, row in llm_gdf.iterrows():
                risk_level = row['risk_level']
                geometry = row['geometry']

                if risk_level not in SPC_RISK_COLORS or risk_level == '<2%' or geometry is None or not geometry.is_valid:
                    if geometry is None or not geometry.is_valid:
                        print(f"Skipping invalid geometry for risk level: {risk_level}")
                    else:
                        print(f"Skipping plot fill for risk level: {risk_level}")
                    continue

                color = SPC_RISK_COLORS[risk_level]
                
                edge_color = CONTOUR_COLORS.get(risk_level, '#000000')
                
                try:
                    ax.add_geometries([geometry], crs=ccrs.PlateCarree(), 
                                      facecolor=color, edgecolor=edge_color,
                                      linewidth=0.5, alpha=1.0, zorder=1)
                except Exception as plot_geom_err:
                     print(f"Error adding geometry for risk level {risk_level}: {plot_geom_err}")
                     continue
                    
                if risk_level not in risk_levels_present:
                    risk_levels_present.add(risk_level)
                    patch = Rectangle((0, 0), 1, 1, facecolor=color)
                    legend_handles.append(patch)
                    legend_labels.append(risk_level)
        
        ax.add_feature(cfeature.COASTLINE, zorder=2, edgecolor='#949494', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, zorder=2, linestyle='-', edgecolor='#949494', linewidth=0.5)
        ax.add_feature(cfeature.STATES, zorder=2, linestyle='-', edgecolor='#949494', linewidth=0.3)
        
        if legend_handles:
            all_categories = PLOT_ORDER[1:]
            all_colors = [SPC_RISK_COLORS[cat] for cat in all_categories]
            
            consistent_legend_handles = [Rectangle((0, 0), 1, 1, facecolor=color) for color in all_colors]
            
            legend = ax.legend(consistent_legend_handles, all_categories,
                            loc='lower right',
                            frameon=True,
                            ncol=len(all_categories),
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
        
        plt.title(f"LLM Prediction [{model_name}] - Tornado Risk - {date_str}")
        
        output_dir = os.path.dirname(output_png_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
        print(f"Successfully saved LLM prediction plot to {output_png_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting LLM prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    base_prediction_dir = "llm_predictions"
    
    date_dirs = []
    try:
        for entry in os.scandir(base_prediction_dir):
            if entry.is_dir():
                if re.fullmatch(r'\d{8}', entry.name):
                    date_dirs.append(entry.path)
                else:
                    print(f"Skipping directory, name does not match YYYYMMDD format: {entry.name}")
    except FileNotFoundError:
        print(f"Error: Base prediction directory not found: {base_prediction_dir}")
        exit(1)
        
    if not date_dirs:
        print(f"No date directories found in {base_prediction_dir}")
        exit(1)
    
    for date_dir in sorted(date_dirs):
        target_date_str = os.path.basename(date_dir)
        print(f"\n{'='*80}\nProcessing date: {target_date_str}\n{'='*80}")
        
        geojson_files = glob.glob(os.path.join(date_dir, 'prediction_*.geojson'))
        
        if not geojson_files:
            print(f"No prediction GeoJSON file found in {date_dir}")
            continue
            
        for geojson_path in geojson_files:
            geojson_filename = os.path.basename(geojson_path)
            
            model_name_for_plot = "Unknown Model"
            try:
                base_name = geojson_filename.replace("prediction_", "").replace(".geojson", "")
                parts = base_name.split(f'_{target_date_str}')
                if len(parts) > 1 and parts[0]:
                    model_name_extracted = parts[0]
                    model_name_for_plot = model_name_extracted
                else:
                    print(f"Warning: Could not reliably extract model name from {geojson_filename} using date {target_date_str}")
                    
            except Exception as e:
                 print(f"Warning: Error parsing model name from {geojson_filename}: {e}")
        
            output_png_path = geojson_path.replace(".geojson", ".png")
            
            plot_llm_prediction(geojson_path, output_png_path, target_date_str, model_name_for_plot)
    
    print("\nProcessed all available prediction dates.")