#!/usr/bin/env python

import os
import sys
import re
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

def sanitize_filename(name):
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', name)
    return name

def plot_single_message(grb, msg_num, file_basename, base_output_dir):
    try:
        if grb.name == "unknown":
            return f"Skipped unknown variable in {file_basename}"

        data, lats, lons = grb.data()

        base_desc = f"{grb.name}_at_{grb.level}_{grb.typeOfLevel}"
        layer_suffix = ""
        try:
            type_of_level_str = str(grb.typeOfLevel).lower()
            if 'layer' in type_of_level_str or 'between' in type_of_level_str:
                bottom_level = grb.bottomLevel
                top_level = grb.level
                if bottom_level != top_level:
                    layer_units = ''
                    if 'height' in type_of_level_str and 'ground' in type_of_level_str:
                        layer_units = 'm'
                    elif 'isobaric' in type_of_level_str:
                        layer_units = 'hPa'
                    elif 'sigma' in type_of_level_str:
                        layer_units = 'sigma'
                    elif 'depth' in type_of_level_str and 'below' in type_of_level_str:
                        layer_units = 'm'
                    elif 'pressure' in type_of_level_str and 'ground' in type_of_level_str:
                        layer_units = 'Pa'

                    layer_suffix = f"_Layer{bottom_level}{layer_units}"
        except (KeyError, AttributeError, Exception):
            pass

        variable_desc_full = base_desc + layer_suffix

        try:
            if grb.name == 'Layer Thickness' and grb.level == 261:
                variable_desc_full = "Layer_Thickness_261K-256K_Layer"
        except AttributeError:
             pass

        variable_folder_name = sanitize_filename(variable_desc_full)

        output_subdir = base_output_dir / variable_folder_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        var_name_lower = grb.name.lower()
        title_suffix = f"at {grb.level} {grb.typeOfLevel}"
        cmap = plt.cm.viridis

        try:
            type_of_level_str = str(grb.typeOfLevel).lower()
            if 'layer' in type_of_level_str or 'between' in type_of_level_str:
                bottom_level = grb.bottomLevel
                top_level = grb.level
                if bottom_level != top_level:
                    layer_units = ''
                    if 'height' in type_of_level_str and 'ground' in type_of_level_str:
                        layer_units = 'm'
                    elif 'isobaric' in type_of_level_str:
                        layer_units = 'hPa'
                    elif 'sigma' in type_of_level_str:
                        layer_units = 'sigma'
                    elif 'depth' in type_of_level_str and 'below' in type_of_level_str:
                        layer_units = 'm'
                    elif 'pressure' in type_of_level_str and 'ground' in type_of_level_str:
                        layer_units = 'Pa'
                    title_suffix = f"at {top_level}-{bottom_level} {grb.typeOfLevel} ({layer_units})"
        except (KeyError, AttributeError, Exception):
            pass

        try:
            if grb.name == 'Layer Thickness' and grb.level == 261:
                title_suffix = "261K-256K Layer"
        except AttributeError:
            pass

        if "Temperature" in grb.name:
            cmap = plt.cm.RdBu_r
        elif "humidity" in var_name_lower:
            cmap = plt.cm.Blues
        elif "wind" in var_name_lower:
            cmap = plt.cm.viridis
        elif "CAPE" in grb.name:
            colors = ["#FFFFFF", "#92D050", "#FFFF00", "#FFC000", "#FF0000", "#C00000", "#7030A0"]
            cmap = LinearSegmentedColormap.from_list("cape_cmap", colors)
        elif "precipitation" in var_name_lower or "rain" in var_name_lower:
            cmap = plt.cm.Blues
        elif "pressure" in var_name_lower:
            cmap = plt.cm.viridis
        elif "cloud" in var_name_lower:
            cmap = plt.cm.Greys

        plot_title = f"{grb.name} {title_suffix}".strip()

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5))

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS)
        if isinstance(data, np.ma.MaskedArray):
             data = data.filled(np.nan)

        plot_data = data
        units = grb.units

        vmin = np.nanmin(plot_data)
        vmax = np.nanmax(plot_data)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        mesh = ax.pcolormesh(lons, lats, plot_data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)

        fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label=units)

        forecast_hour = file_basename.split('f')[-1]
        try:
            valid_time_str = grb.validDate.strftime('%Y-%m-%d %H:%M UTC')
        except AttributeError:
            valid_time_str = "N/A"
        try:
            name = grb.name
            short_name = grb.shortName
            level = grb.level
            type_of_level = grb.typeOfLevel

            if name == "Derived radar reflectivity" and level == 263 and type_of_level == "isothermal":
                try:
                    step_type = grb.stepType
                    if step_type == 'max':
                        variable_identifier = "refd_1hr_max"
                    else:
                        variable_identifier = "refd_instant"
                except (KeyError, AttributeError, Exception):
                    variable_identifier = short_name
            elif name == 'Layer Thickness' and level == 261:
                 variable_identifier = "layth_261K-256K"
            else:
                variable_identifier = short_name

        except (KeyError, AttributeError, Exception):
            variable_identifier = f"Msg {msg_num}"

        ax.set_title(f"HRRR {plot_title} ({variable_identifier})\nForecast Hour: {forecast_hour}, Valid: {valid_time_str}")

        ax.set_extent([-125, -66, 23, 50], crs=ccrs.PlateCarree())

        filename_parts = [file_basename, f"Msg{msg_num}"]
        filename_parts.append(variable_folder_name)

        output_filename_stem = "_".join(filename_parts)
        output_filename_stem_safe = sanitize_filename(output_filename_stem)
        output_file = output_subdir / f"{output_filename_stem_safe}.png"

        plt.savefig(output_file, dpi=150, bbox_inches='tight')

        plt.close(fig)

        return f"✓ Saved plot: {output_file.relative_to(base_output_dir.parent)}"

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return f"✗ Error plotting '{grb.name}' from {file_basename}: {e} in {fname} at line {exc_tb.tb_lineno}"


def process_grib2_file(grib2_file_path, base_output_dir):
    results = []
    file_basename = grib2_file_path.name.rsplit('.', 1)[0]

    try:
        grbs = pygrib.open(str(grib2_file_path))
        num_messages = grbs.messages

        for i, grb in enumerate(tqdm(grbs, total=num_messages, desc=f"Vars in {file_basename}", leave=False, unit="var")):
            msg_num = i + 1
            plot_result = plot_single_message(grb, msg_num, file_basename, base_output_dir)
            if plot_result:
                results.append(plot_result)

        grbs.close()

    except Exception as e:
        results.append(f"✗ Error opening or reading {grib2_file_path.name}: {e}")

    return results


def process_date_folder(date_folder_path, max_workers=5):
    if not date_folder_path.is_dir():
        print(f"Error: Date folder {date_folder_path} does not exist or is not a directory.")
        return False

    maps_dir = date_folder_path / "hrrr_maps"
    if not maps_dir.is_dir():
        print(f"Error: hrrr_maps subfolder not found in {date_folder_path}")
        return False

    grib2_files = sorted(list(maps_dir.glob("*.grib2")))
    if not grib2_files:
        print(f"No GRIB2 files found in {maps_dir}")
        return False

    plots_dir = date_folder_path / "hrrr_plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {len(grib2_files)} GRIB2 files in {date_folder_path.name} using {max_workers} workers...")
    print(f"Output base directory: {plots_dir}")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_grib2_file, grib2_file, plots_dir): grib2_file
            for grib2_file in grib2_files
        }

        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Plotting {date_folder_path.name}", unit="file"):
            grib2_file = future_to_file[future]
            try:
                file_results = future.result()
                all_results.extend(file_results)
            except Exception as e:
                 tqdm.write(f"✗✗ Unexpected error processing {grib2_file.name}: {e}")
                 all_results.append(f"✗✗ Unexpected error processing {grib2_file.name}: {e}")

    print(f"\nFinished processing for {date_folder_path.name}.")
    errors = [r for r in all_results if r.startswith("✗")]
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for err in errors[:10]:
             print(f"  {err}")
        if len(errors) > 10:
             print(f"  ... and {len(errors) - 10} more.")
    else:
        print("✓ Processing completed without errors.")

    return True

def main():
    parser = argparse.ArgumentParser(description='Process and plot HRRR GRIB2 files from dataset folders.')
    parser.add_argument('--dataset-dir', default='dataset',
                        help='Path to the base dataset directory (default: dataset)')
    parser.add_argument('--date', default='20250301',
                        help='Specific date to process (format: YYYYMMDD). Default: 20250301. Ignored if --all-dates is used.')
    parser.add_argument('--all-dates', action='store_true',
                        help='Process all hrrr_YYYYMMDD_00z folders found in the dataset directory.')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum number of parallel workers for plotting GRIB files (default: 5)')

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Error: Dataset directory '{dataset_dir}' not found or not a directory.")
        sys.exit(1)

    target_folders = []
    if args.all_dates:
        print(f"Processing all date folders in {dataset_dir}...")
        date_folders = sorted(dataset_dir.glob("hrrr_*_00z"))
        if not date_folders:
            print(f"No date folders matching 'hrrr_*_00z' found in {dataset_dir}")
            sys.exit(0)
        target_folders.extend(date_folders)
    else:
        print(f"Processing single date: {args.date}")
        date_folder = dataset_dir / f"hrrr_{args.date}_00z"
        if not date_folder.is_dir():
            print(f"Error: Date folder for {args.date} ('{date_folder}') not found or not a directory.")
            sys.exit(1)
        target_folders.append(date_folder)

    print(f"Found {len(target_folders)} date folder(s) to process.")

    for date_folder in target_folders:
        process_date_folder(date_folder, args.max_workers)

    print("\nAll requested processing complete.")

if __name__ == "__main__":
    main()