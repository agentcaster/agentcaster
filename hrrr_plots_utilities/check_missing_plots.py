#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

import pygrib
import argparse
from pathlib import Path
import re
import sys
from collections import defaultdict

def plot_grib_message_for_check(grb, message_number, output_filename_base):
    try:
        data, lats, lons = grb.data()
        plot_title = f"Msg {message_number}: {grb.name} at {grb.level} {grb.typeOfLevel}"
        output_filename = f"{output_filename_base}_Msg{message_number}.png"

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5))
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS)

        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        plot_data = data
        units = grb.units
        if units == 'K' and "Temperature" in grb.name:
            plot_data = data - 273.15
            units = '°C'

        vmin = np.nanmin(plot_data)
        vmax = np.nanmax(plot_data)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
             plt.close(fig)
             print(f"  -> Plotting skipped for Msg {message_number}: Data is all NaN")
             return False
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        cmap = plt.cm.viridis
        mesh = ax.pcolormesh(lons, lats, plot_data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label=units)
        ax.set_title(plot_title + f"\nValid: {grb.validDate.strftime('%Y-%m-%d %H:%M UTC')}")
        ax.set_extent([-125, -66, 23, 50], crs=ccrs.PlateCarree())

        plt.savefig(output_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved comparison plot: {output_filename}")
        return True

    except Exception as e:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        print(f"  -> Error plotting Msg {message_number} ({grb.name}): {e}")
        return False

def sanitize_filename(name):
    name = re.sub(r'[\/:*?"<>|\s]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

def check_plots(grib_file_path, plots_dir_path):
    if not grib_file_path.is_file():
        print(f"Error: GRIB file not found at {grib_file_path}")
        sys.exit(1)

    if not plots_dir_path or not plots_dir_path.is_dir():
        print(f"Error: Plots directory not found or not specified ({plots_dir_path}).")
        sys.exit(1)
    print(f"Checking GRIB file: {grib_file_path.name}")
    print(f"Against plot folders in: {plots_dir_path}")

    expected_folders_details = defaultdict(list)
    plottable_count = 0
    unknown_count = 0

    try:
        grbs = pygrib.open(str(grib_file_path))
        print(f"Opened GRIB file. Found {grbs.messages} total messages.")

        for i, grb in enumerate(grbs):
            msg_num = i + 1
            if grb.name == "unknown":
                unknown_count += 1
                continue

            plottable_count += 1
            original_str_grb = str(grb)

            variable_desc = f"{grb.name}_at_{grb.level}_{grb.typeOfLevel}"
            sanitized_name = sanitize_filename(variable_desc)

            expected_folders_details[sanitized_name].append((msg_num, original_str_grb))

        grbs.seek(0)

        print(f" -> Found {plottable_count} plottable messages.")
        print(f" -> Generated {len(expected_folders_details)} unique expected folder names using process_and_plot logic.")
        print(f" -> Found {unknown_count} 'unknown' messages (skipped).")

        duplicates_found_details = {name: details for name, details in expected_folders_details.items() if len(details) > 1}
        if duplicates_found_details:
            colliding_msg_count = sum(len(details) for details in duplicates_found_details.values())
            num_duplicate_names = len(duplicates_found_details)
            print("\n--- START Duplicate Name Report ---")
            print(f"Total Duplicate Folder Names: {num_duplicate_names}")
            print(f"Total Colliding Messages: {colliding_msg_count}")
            for name, details_list in duplicates_found_details.items():
                print(f"\nFolder: \"{name}\"")
                message_numbers = [d[0] for d in details_list]
                print(f"  Messages: {', '.join(map(str, sorted(message_numbers)))}")
            print("--- END Duplicate Name Report ---")
        else:
             print("\n--- No Duplicate Names Found --- ")

        if duplicates_found_details:
            print("\n--- Plotting First Two Colliding Messages for Each Duplicate Name ---")
            for duplicate_name, colliding_messages_details in duplicates_found_details.items():
                print(f"\nProcessing duplicate: '{duplicate_name}'")
                if len(colliding_messages_details) >= 2:
                    msg_details_to_plot = colliding_messages_details[:2]
                    output_base = f"duplicate_plot_{duplicate_name}"

                    grbs_to_plot_found = 0
                    grb1, grb2 = None, None
                    msg_num1 = msg_details_to_plot[0][0]
                    msg_num2 = msg_details_to_plot[1][0]

                    try:
                        grb1 = grbs.message(msg_num1)
                        grb2 = grbs.message(msg_num2)
                        if grb1 and grb2:
                            grbs_to_plot_found = 2
                        else:
                             print(f"Error: grbs.message() returned None for {msg_num1} or {msg_num2}.")
                    except ValueError as e:
                        print(f"Error retrieving messages by number: {e}")
                    except Exception as e:
                         print(f"Error accessing messages via grbs.message(): {e}")

                    if grbs_to_plot_found == 2:
                        print(f"\n--- Detailed Metadata for Msg {msg_num1} ({duplicate_name}) ---")
                        for key in grb1.keys():
                            try:
                                print(f"  {key}: {grb1[key]}")
                            except Exception:
                                print(f"  {key}: Error reading key")
                        try:
                            data1, _, _ = grb1.data()
                            if isinstance(data1, np.ma.MaskedArray):
                                data1 = data1.filled(np.nan)
                            if np.all(np.isnan(data1)):
                                print("  Data Stats: All NaN")
                            else:
                                print(f"  Data Stats: Min={np.nanmin(data1):.2f}, Max={np.nanmax(data1):.2f}, Mean={np.nanmean(data1):.2f}")
                        except Exception as e:
                            print(f"  Data Stats: Error getting data stats - {e}")

                        print(f"\n--- Detailed Metadata for Msg {msg_num2} ({duplicate_name}) ---")
                        for key in grb2.keys():
                            try:
                                print(f"  {key}: {grb2[key]}")
                            except Exception:
                                print(f"  {key}: Error reading key")
                        try:
                            data2, _, _ = grb2.data()
                            if isinstance(data2, np.ma.MaskedArray):
                                data2 = data2.filled(np.nan)
                            if np.all(np.isnan(data2)):
                                print("  Data Stats: All NaN")
                            else:
                                print(f"  Data Stats: Min={np.nanmin(data2):.2f}, Max={np.nanmax(data2):.2f}, Mean={np.nanmean(data2):.2f}")
                        except Exception as e:
                            print(f"  Data Stats: Error getting data stats - {e}")

                        print(f"\n  Plotting Message {msg_num1}... ({duplicate_name})")
                        plot_grib_message_for_check(grb1, msg_num1, output_base)
                        print(f"  Plotting Message {msg_num2}... ({duplicate_name})")
                        plot_grib_message_for_check(grb2, msg_num2, output_base)
                    else:
                        print(f"  -> Could not find both messages ({msg_num1}, {msg_num2}) to plot.")
                else:
                     print(f"  -> Warning: Found duplicate name '{duplicate_name}' but with less than 2 messages listed.")
            print("\nFinished plotting detected duplicates.")

        grbs.close()

    except Exception as e:
        print(f"Error reading GRIB file {grib_file_path}: {e}")
        sys.exit(1)

    actual_folders = set()
    try:
        for item in plots_dir_path.iterdir():
            if item.is_dir():
                actual_folders.add(item.name)
        print(f"\nFound {len(actual_folders)} actual plot folders in the directory.")
    except Exception as e:
        print(f"Error listing plots directory {plots_dir_path}: {e}")
        sys.exit(1)

    print("--- Debugging Comparison ---")
    print(f"Size of expected_folders set: {len(expected_folders_details)}")
    print(f"Size of actual_folders set: {len(actual_folders)}")

    missing_folders = set(expected_folders_details.keys()) - actual_folders
    extra_folders = actual_folders - set(expected_folders_details.keys())

    print(f"Size of missing_folders set (expected - actual): {len(missing_folders)}")
    print(f"Size of extra_folders set (actual - expected): {len(extra_folders)}")
    print("--- End Debugging ---")

    print("\n--- Comparison Results ---")

    if not missing_folders:
        print("✓ All expected variable folders were found.")
    else:
        print(f"✗ Found {len(missing_folders)} MISSING variable folders:")
        for folder in sorted(list(missing_folders)):
            print(f"  - {folder}")

    if not extra_folders:
        print("✓ No unexpected extra folders were found.")
    else:
        print(f"⚠ Found {len(extra_folders)} UNEXPECTED extra folders:")
        for folder in sorted(list(extra_folders)):
            print(f"  - {folder}")

def main():
    parser = argparse.ArgumentParser(description='Compare GRIB variables to plot folders and optionally plot duplicates.')
    parser.add_argument('--grib-file', required=True,
                        help='Path to a single representative GRIB2 file (e.g., dataset/YYYYMMDD/hrrr_maps/hrrr...f12.grib2)')
    parser.add_argument('--plots-dir', required=True,
                        help='Path to the corresponding hrrr_plots directory (e.g., dataset/YYYYMMDD/hrrr_plots)')

    args = parser.parse_args()

    grib_path = Path(args.grib_file)
    plots_path = Path(args.plots_dir)

    check_plots(grib_path, plots_path)

if __name__ == "__main__":
    main()