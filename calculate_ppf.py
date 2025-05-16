import pandas as pd
import numpy as np
import pyproj
import xarray as xr
import os
import warnings
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects
import glob
import re
from affine import Affine
import scipy.signal as signal


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
CELL_AREA_M2 = GRID_5km_DX * GRID_5km_DY
print(f"Calculated 5km grid cell area: {CELL_AREA_M2:.1f} m^2")

CONVOLUTION_RADIUS_M = 40000.0
CONVOLUTION_RADIUS_PIXELS = int(np.ceil(CONVOLUTION_RADIUS_M / GRID_5km_DX))
print(f"Convolution kernel radius: {CONVOLUTION_RADIUS_M / 1000.0:.1f} km ({CONVOLUTION_RADIUS_PIXELS} pixels)")

def create_disk_kernel(radius_pixels):
    diameter = 2 * radius_pixels + 1
    y, x = np.ogrid[-radius_pixels:radius_pixels+1, -radius_pixels:radius_pixels+1]
    mask = x**2 + y**2 <= radius_pixels**2
    kernel = np.zeros((diameter, diameter), dtype=np.float32)
    kernel[mask] = 1.0
    print(f"Created convolution kernel of size {diameter}x{diameter} with {kernel.sum()} non-zero elements.")
    return kernel

CONVOLUTION_KERNEL = create_disk_kernel(CONVOLUTION_RADIUS_PIXELS)

ALPHA_SCALING_FACTOR = 0.25
print(f"Using Alpha scaling factor for lambda: {ALPHA_SCALING_FACTOR:.4f}")
CONUS_LON_MIN, CONUS_LON_MAX = -121.0, -65.0
CONUS_LAT_MIN, CONUS_LAT_MAX = 22.0, 50.0

proj_grid = pyproj.Proj(GRID_80km_PROJ_PARAMS)
crs_grid_5km = pyproj.CRS(GRID_5km_PROJ_PARAMS)
proj_latlon = pyproj.Proj("epsg:4326")
transformer_ll_to_grid = pyproj.Transformer.from_proj(proj_latlon, proj_grid, always_xy=True)
transformer_grid_to_ll = pyproj.Transformer.from_proj(proj_grid, proj_latlon, always_xy=True)
center_x_80, center_y_80 = transformer_ll_to_grid.transform(GRID_80km_PROJ_PARAMS['lon_0'], GRID_80km_PROJ_PARAMS['lat_0'])
grid_80km_ll_x = center_x_80 - (GRID_80km_NX / 2) * GRID_80km_DX
grid_80km_ll_y = center_y_80 - (GRID_80km_NY / 2) * GRID_80km_DY

print(f"Using NCEP Grid 211 (~80km) for calculation, interpolating to ~5km grid for output.")
print(f"Calculated Grid 211 SW corner (x, y): ({grid_80km_ll_x:.1f}, {grid_80km_ll_y:.1f})")

x_coords_80km = grid_80km_ll_x + np.arange(GRID_80km_NX) * GRID_80km_DX
y_coords_80km = grid_80km_ll_y + np.arange(GRID_80km_NY) * GRID_80km_DY

x_grid_2d_80km, y_grid_2d_80km = np.meshgrid(x_coords_80km, y_coords_80km)

transform = from_origin(x_coords_80km[0] - GRID_80km_DX / 2.0, y_coords_80km[-1] + GRID_80km_DY / 2.0, GRID_80km_DX, GRID_80km_DY)
center_x_5, center_y_5 = transformer_ll_to_grid.transform(GRID_5km_PROJ_PARAMS['lon_0'], GRID_5km_PROJ_PARAMS['lat_0'])
grid_5km_ll_x = center_x_5 - (GRID_5km_NX / 2) * GRID_5km_DX
grid_5km_ll_y = center_y_5 - (GRID_5km_NY / 2) * GRID_5km_DY
x_coords_5km = grid_5km_ll_x + np.arange(GRID_5km_NX) * GRID_5km_DX
y_coords_5km = grid_5km_ll_y + np.arange(GRID_5km_NY) * GRID_5km_DY

origin_x = x_coords_5km[0] - GRID_5km_DX / 2.0
origin_y = y_coords_5km[0] - GRID_5km_DY / 2.0

transform_5km = (
    Affine.translation(origin_x, origin_y) *
    Affine.scale(GRID_5km_DX, GRID_5km_DY)
)

def calculate_ppf_interp(report_df, sigma_meters=121906.0):

    if report_df.empty or 'Lat' not in report_df.columns or 'Lon' not in report_df.columns:
        print("No reports found. Returning zero probability field on 5km grid.")
        ppf_prob_field_5km = np.zeros((GRID_5km_NY, GRID_5km_NX))
        original_max_80km_density = 0.0
        interpolated_max_5km_density = 0.0
        max_lambda = 0.0
        max_prob = 0.0
    else:
        print("Calculating Normalized Gaussian KDE PPF (Density) on 80km grid...")
        report_lons = report_df['Lon'].values
        report_lats = report_df['Lat'].values
        report_x, report_y = transformer_ll_to_grid.transform(report_lons, report_lats)

        grid_points_flat_80km = np.stack([x_grid_2d_80km.ravel(), y_grid_2d_80km.ravel()], axis=-1)
        report_points = np.stack([report_x, report_y], axis=-1)

        diff = grid_points_flat_80km[:, np.newaxis, :] - report_points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        sigma_sq = sigma_meters**2

        norm_factor = 1.0 / (2.0 * np.pi * sigma_sq)

        gaussian_values_unnormalized = np.exp(-0.5 * dist_sq / sigma_sq)
        gaussian_values_normalized = gaussian_values_unnormalized * norm_factor

        ppf_flat_80km_density = np.sum(gaussian_values_normalized, axis=1)
        ppf_field_80km_density = ppf_flat_80km_density.reshape((GRID_80km_NY, GRID_80km_NX))
        original_max_80km_density = ppf_field_80km_density.max()
        print(f"Calculated 80km Normalized Density field max: {original_max_80km_density:.4e} m^-2")

        ppf_da_80km_density = xr.DataArray(
            ppf_field_80km_density,
            coords=[('y', y_coords_80km), ('x', x_coords_80km)],
            name='ppf_normalized_80km_density'
        )
        print("Interpolating Normalized Density from 80km to 5km grid...")
        ppf_da_5km_density_interp = ppf_da_80km_density.interp(
            y=y_coords_5km, x=x_coords_5km, method='linear',
            kwargs={"fill_value": 0.0}
        )
        ppf_field_5km_density = ppf_da_5km_density_interp.values
        interpolated_max_5km_density = ppf_field_5km_density.max()
        print(f"Interpolated 5km Normalized Density field max: {interpolated_max_5km_density:.4e} m^-2")

        ppf_field_5km_density = np.maximum(ppf_field_5km_density, 0.0)

        print(f"Convolving 5km density field with {CONVOLUTION_RADIUS_M / 1000.0:.1f} km disk kernel...")
        convolved_density_sum = signal.fftconvolve(ppf_field_5km_density, CONVOLUTION_KERNEL, mode='same')

        expected_count_lambda = convolved_density_sum * CELL_AREA_M2
        max_lambda = expected_count_lambda.max()
        print(f"Maximum expected count (lambda) after convolution: {max_lambda:.4f}")

        with np.errstate(over='ignore'):
             ppf_prob_field_5km = 1.0 - np.exp(-ALPHA_SCALING_FACTOR * expected_count_lambda)
        ppf_prob_field_5km = np.clip(ppf_prob_field_5km, 0.0, 1.0)
        max_prob = ppf_prob_field_5km.max()
        print(f"Maximum point probability after 1-exp(-alpha*lambda): {max_prob:.4f}")

    ppf_prob_da_5km = xr.DataArray(
        ppf_prob_field_5km,
        coords=[('y', y_coords_5km), ('x', x_coords_5km)],
        name='ppf_interp5km_scaled_convolved_prob',
        attrs={
            'long_name': f'Probability (1-exp(-alpha*lambda)) from {CONVOLUTION_RADIUS_M / 1000.0:.0f}km Convolved Density',
            'units': 'probability',
            'grid': '~5km (Interpolated from NCEP 211)',
            'projection_proj4': proj_grid.srs,
            'sigma_meters_80km_kde_calc': sigma_meters,
            'alpha_scaling_factor': ALPHA_SCALING_FACTOR,
            'convolution_radius_meters': CONVOLUTION_RADIUS_M,
            'transform_5km': transform_5km,
            'original_80km_density_max': float(original_max_80km_density),
            'interpolated_5km_density_max': float(interpolated_max_5km_density),
            'max_lambda_after_convolution': float(max_lambda),
            'max_probability': float(max_prob)
        }
    )
    return ppf_prob_da_5km
PPF_CATEGORIES = ['0%', '2%', '5%', '10%', '15%', '30%', '45%', '60%']
PPF_THRESHOLDS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, np.inf]
PPF_COLORS = {
    '0%': '#FFFFFF',
    '2%': '#80c480',
    '5%': '#c6a393',
    '10%': '#ffeb80',
    '15%': '#ff8080',
    '30%': '#ff80ff',
    '45%': '#c896f7',
    '60%': '#0f4e8b'
}

def categorize_ppf(ppf_prob_da):
    print("Categorizing probability field using standard SPC thresholds...")
    indices = np.digitize(ppf_prob_da.values, PPF_THRESHOLDS[1:], right=False)
    category_field = np.array(PPF_CATEGORIES)[indices]

    ppf_cat_da = xr.DataArray(
        category_field,
        coords=ppf_prob_da.coords,
        name='ppf_category_from_scaled_convolved_prob',
        attrs={
            'long_name': 'Categorical PPF (from Scaled Convolved Probability)',
            'grid': GRID_5km_PROJ_PARAMS['proj'],
            'projection_proj4': GRID_5km_PROJ_PARAMS['proj'],
            'thresholds_applied': str(PPF_THRESHOLDS),
            'applied_to': f'probability (from alpha={ALPHA_SCALING_FACTOR:.3f} scaled convolved density)',
            'labels': str(PPF_CATEGORIES),
            'transform_5km': transform_5km
        }
    )
    return ppf_cat_da

def save_ppf_categories_to_geojson(ppf_cat_da, output_geojson_path):
    print(f"Converting PPF categories (from scaled convolved prob) to GeoJSON (WGS84): {output_geojson_path}")
    all_features = []
    raster_transform = ppf_cat_da.attrs.get('transform_5km')
    grid_crs = crs_grid_5km
    if raster_transform is None:
        print("Error: 5km affine transform not found in DataArray attributes.")
        return

    data = ppf_cat_da.values.astype(str)

    for category in PPF_CATEGORIES:
        if category == '0%':
            continue
        mask = (data == category).astype(np.uint8)
        try:
            results = features.shapes(mask, mask=mask, transform=raster_transform)

            for poly_info, value in results:
                if value == 1:
                    geom = shape(poly_info)
                    if geom.is_valid:
                         all_features.append({'geometry': geom, 'risk_level': category})
                    else:
                         buffered_geom = geom.buffer(0)
                         if buffered_geom.is_valid:
                              all_features.append({'geometry': buffered_geom, 'risk_level': category})
                         else:
                             print(f"Warning: Skipping invalid geometry for category {category} at {output_geojson_path}")
        except Exception as e:
             print(f"Warning: Error during rasterio.features.shapes for category {category}: {e}")
             continue

    if not all_features:
        print("No non-zero PPF category areas found to save in GeoJSON.")
        gdf = gpd.GeoDataFrame([], columns=['geometry', 'risk_level'], crs=grid_crs)
    else:
        gdf = gpd.GeoDataFrame(all_features, crs=grid_crs)

    print(f"Reprojecting {len(gdf)} features to EPSG:4326 before saving...")
    try:
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
    except Exception as e:
        print(f"Error reprojecting to EPSG:4326: {e}")
        print(f"Attempting to save in original CRS ({grid_crs.to_string()}) instead.")
        gdf_wgs84 = gdf

    output_dir = os.path.dirname(output_geojson_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        gdf_wgs84.to_file(output_geojson_path, driver='GeoJSON')
        print(f"Successfully saved PPF categories GeoJSON ({gdf_wgs84.crs.to_string()}) derived from scaled convolved prob to {output_geojson_path}")
    except Exception as e:
        print(f"Error saving PPF GeoJSON derived from scaled convolved prob: {e}")

def plot_ppf_categories(ppf_cat_da, output_png_path, date_str, report_df=None):
    print(f"Plotting PPF categories (derived from scaled convolved prob) to PNG: {output_png_path}")

    map_proj = ccrs.LambertConformal(
        central_longitude=GRID_5km_PROJ_PARAMS['lon_0'],
        central_latitude=GRID_5km_PROJ_PARAMS['lat_0'],
        standard_parallels=(GRID_5km_PROJ_PARAMS['lat_1'], GRID_5km_PROJ_PARAMS['lat_2']),
        globe=ccrs.Globe(ellipse=None, semimajor_axis=GRID_5km_PROJ_PARAMS['a'], semiminor_axis=GRID_5km_PROJ_PARAMS['b'])
    )

    cmap_dict = {i: PPF_COLORS[cat] for i, cat in enumerate(PPF_CATEGORIES)}
    initial_cmap = mcolors.ListedColormap([cmap_dict[i] for i in range(len(PPF_CATEGORIES))])
    norm = mcolors.BoundaryNorm(np.arange(len(PPF_CATEGORIES) + 1) - 0.5, initial_cmap.N)

    cmap_colors = initial_cmap.colors
    cmap_colors[0] = (*mcolors.to_rgb(cmap_colors[0]), 0.0)
    cmap = mcolors.ListedColormap(cmap_colors)

    cat_to_index = {cat: i for i, cat in enumerate(PPF_CATEGORIES)}
    plot_data = np.vectorize(cat_to_index.get)(ppf_cat_da.values)

    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=map_proj)

    min_x, min_y = transformer_ll_to_grid.transform(CONUS_LON_MIN, CONUS_LAT_MIN)
    max_x, max_y = transformer_ll_to_grid.transform(CONUS_LON_MAX, CONUS_LAT_MAX)
    ax.set_extent([min_x, max_x, min_y, max_y], crs=map_proj)

    x_edges = np.append(ppf_cat_da['x'] - GRID_5km_DX/2, ppf_cat_da['x'][-1] + GRID_5km_DX/2)
    y_edges = np.append(ppf_cat_da['y'] - GRID_5km_DY/2, ppf_cat_da['y'][-1] + GRID_5km_DY/2)
    mesh = ax.pcolormesh(x_edges, y_edges, plot_data, cmap=cmap, norm=norm, transform=map_proj, shading='auto', zorder=1)

    contour_colors = ['#008200', '#8b4825', '#ff9601', '#ff0000', '#ff00ff', '#912dee']
    contour_levels = np.array([0, 1, 2, 3, 4, 5])

    ax.contour(ppf_cat_da['x'], ppf_cat_da['y'], plot_data,
               levels=contour_levels,
               colors=contour_colors,
               linewidths=0.5, transform=map_proj, zorder=2)

    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#d3e4f5')
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='#ffffff')
    ax.add_feature(cfeature.COASTLINE, zorder=2, edgecolor='#949494', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, zorder=2, linestyle='-', edgecolor='#949494', linewidth=0.5)
    ax.add_feature(cfeature.STATES, zorder=2, linestyle='-', edgecolor='#949494', linewidth=0.3)
    ax.add_feature(cfeature.LAKES, zorder=1, alpha=0.5, facecolor='#d3e4f5')

    if report_df is not None and not report_df.empty and 'Lat' in report_df.columns and 'Lon' in report_df.columns:
        valid_reports = report_df.dropna(subset=['Lat', 'Lon'])

        if not valid_reports.empty:
            print(f"Plotting {len(valid_reports)} tornado reports as black dots")
            report_x, report_y = transformer_ll_to_grid.transform(valid_reports['Lon'].values, valid_reports['Lat'].values)
            ax.scatter(report_x, report_y, s=10, color='black', marker='.',
                      transform=map_proj, zorder=5, alpha=0.8)

    categories_for_legend = ['2%', '5%', '10%', '15%', '30%', '45%', '60%']
    legend_colors = [PPF_COLORS[cat] for cat in categories_for_legend]

    legend_handles = []
    for i, cat in enumerate(categories_for_legend):
        rect = Rectangle((0, 0), 1, 1, facecolor=legend_colors[i])
        legend_handles.append(rect)

    legend = ax.legend(legend_handles,
                     categories_for_legend,
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

    plt.title(f"Ground Truth Tornado Risk - {date_str}")

    output_dir = os.path.dirname(output_png_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
    print(f"Successfully saved PPF plot (scaled convolved prob) to {output_png_path}")
    plt.close()


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

        ground_truth_dir = os.path.join(date_dir, "ground_truth")
        yy = target_date_str[2:4]
        mm = target_date_str[4:6]
        dd = target_date_str[6:8]
        report_filename = f"{yy}{mm}{dd}_rpts_torn.csv"
        report_path = os.path.join(ground_truth_dir, report_filename)

        output_dir = os.path.join("ppf_output", target_date_str)
        os.makedirs(output_dir, exist_ok=True)
        output_ppf_cat_geojson = os.path.join(output_dir, f"ground_truth_{target_date_str}.geojson")
        output_ppf_cat_png = os.path.join(output_dir, f"Ground Truth Tornado Risk - {target_date_str}.png")

        if not os.path.exists(report_path):
            print(f"Error: Ground truth file not found: {report_path}")
        else:
            print(f"Reading ground truth reports from: {report_path}")
            try:
                with open(report_path, 'r') as f:
                    lines = f.readlines()
                header_line_index = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith('Time,F_Scale'):
                        header_line_index = i
                        break
                if header_line_index == -1:
                     print("Warning: Header 'Time,F_Scale' not found, attempting read_csv without skiprows adjustment.")
                     report_data = pd.read_csv(report_path)
                     if 'Lat' not in report_data.columns or 'Lon' not in report_data.columns:
                         raise ValueError("Could not find required 'Lat'/'Lon' columns after basic read.")
                else:
                    report_data = pd.read_csv(report_path, skiprows=header_line_index)
                print(f"Read {len(report_data)} reports.")

                report_data['Lat'] = pd.to_numeric(report_data['Lat'], errors='coerce')
                report_data['Lon'] = pd.to_numeric(report_data['Lon'], errors='coerce')
                report_data.dropna(subset=['Lat', 'Lon'], inplace=True)
                print(f"Processing {len(report_data)} reports after cleaning.")

                print(f"Calculating KDE Density (sigma=120km on 80km), interpolating, convolving (40km), scaling lambda (alpha={ALPHA_SCALING_FACTOR:.3f}), converting to probability...")
                ppf_final_5km_prob = calculate_ppf_interp(report_data)

                if ppf_final_5km_prob is not None:
                    print("Categorizing final 5km Scaled Convolved Probability PPF field...")
                    ppf_categorical = categorize_ppf(ppf_final_5km_prob)

                    save_ppf_categories_to_geojson(ppf_categorical, output_ppf_cat_geojson)
                    plot_ppf_categories(ppf_categorical, output_ppf_cat_png, target_date_str, report_data)

                    print(f"PPF calculation complete for {target_date_str}.")

            except pd.errors.EmptyDataError:
                print(f"Ground truth file is empty: {report_path}")
                print("Calculating zero PPF field...")
                ppf_final_5km_prob = calculate_ppf_interp(pd.DataFrame())
                print("Categorizing zero PPF field...")
                ppf_categorical = categorize_ppf(ppf_final_5km_prob)
                save_ppf_categories_to_geojson(ppf_categorical, output_ppf_cat_geojson)
                plot_ppf_categories(ppf_categorical, output_ppf_cat_png, target_date_str, pd.DataFrame())
                print(f"PPF calculation complete for {target_date_str} (zero field/empty outputs).")
            except Exception as e:
                print(f"An error occurred during PPF calculation for {target_date_str}: {e}")
                import traceback
                traceback.print_exc()

    print("\nProcessed all available dates.") 