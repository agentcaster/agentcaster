import os
import sys
import glob
import math
import subprocess
import argparse
import re
import csv

BROKEN_STATIONS_FILE = "broken_sounding_stations.csv"

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r

def extract_coordinates_from_bufkit_header(bufkit_file):
    stid, lat, lon = None, None, None
    try:
        with open(bufkit_file, 'r', encoding='latin-1') as f:
            lines = [f.readline() for _ in range(6)]
            for i, line in enumerate(lines):
                if "STID" in line:
                    stid_match = re.search(r'STID\s*=\s*(\w+)', line)
                    if stid_match:
                        stid = stid_match.group(1)
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
        print(f"Warning: BUFKIT file not found: {bufkit_file}", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"Warning: Could not parse header for {bufkit_file}: {str(e)}", file=sys.stderr)
        return None, None, None
    if stid and lat is not None and lon is not None:
        return stid, lat, lon
    else:
        return None, None, None

def load_broken_stations(filename=BROKEN_STATIONS_FILE):
    broken_set = set()
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            if header[0].strip() != 'Broken Station ID':
                 print(f"Warning: Unexpected header in {filename}: {header}", file=sys.stderr)
            for row in reader:
                if row:
                    broken_set.add(row[0].strip())
        print(f"Loaded {len(broken_set)} broken stations from {filename}.")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. No stations will be excluded as broken.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error reading {filename}: {e}. No stations will be excluded.", file=sys.stderr)
    return broken_set

def find_nearest_station(date_str, target_lat, target_lon, base_data_dir="dataset"):
    broken_stations = load_broken_stations()
    bufkit_dir = os.path.join(base_data_dir, f"hrrr_{date_str}_00z", "bufkit")
    if not os.path.isdir(bufkit_dir):
        print(f"Error: BUFKIT directory not found: {bufkit_dir}", file=sys.stderr)
        return None, None, float('inf')
    bufkit_files = glob.glob(os.path.join(bufkit_dir, "hrrr_*.buf"))
    if not bufkit_files:
        print(f"Error: No BUFKIT files found in {bufkit_dir}", file=sys.stderr)
        return None, None, float('inf')
    nearest_station_id = None
    nearest_bufkit_file_path = None
    min_distance = float('inf')
    print(f"Searching {len(bufkit_files)} stations in {bufkit_dir} (excluding broken ones)...")
    skipped_broken_count = 0
    for bufkit_file in bufkit_files:
        stid, slat, slon = extract_coordinates_from_bufkit_header(bufkit_file)
        if stid and slat is not None and slon is not None:
            if stid in broken_stations:
                skipped_broken_count += 1
                continue
            distance = haversine(target_lat, target_lon, slat, slon)
            if distance < min_distance:
                min_distance = distance
                nearest_station_id = stid
                nearest_bufkit_file_path = bufkit_file
    if skipped_broken_count > 0:
        print(f"Skipped {skipped_broken_count} stations identified as broken.")
    if nearest_station_id:
        print(f"Nearest station: {nearest_station_id} ({min_distance:.2f} km away)")
        return nearest_station_id, nearest_bufkit_file_path, min_distance
    else:
        print("Error: Could not find any valid stations or determine nearest station.", file=sys.stderr)
        return None, None, float('inf')

def plot_sounding(bufkit_file_path, fcst_hour):
    plot_script = "plot_sharppy_file.py"
    if not os.path.exists(plot_script):
        print(f"Error: Plotting script not found: {plot_script}", file=sys.stderr)
        return None
    try:
        bufkit_dir = os.path.dirname(bufkit_file_path)
        date_dir = os.path.dirname(bufkit_dir)
        sharppy_dir = os.path.join(date_dir, "sharppy_files")
        base_name = os.path.splitext(os.path.basename(bufkit_file_path))[0]
        output_png_filename = f"{base_name}_f{fcst_hour:02d}.png"
        expected_png_path = os.path.join(sharppy_dir, output_png_filename)
    except Exception as e:
         print(f"Error determining expected PNG path: {e}", file=sys.stderr)
         return None
    print(f"Calling {plot_script} for {bufkit_file_path} (F{fcst_hour})...")
    try:
        abs_bufkit_path = os.path.abspath(bufkit_file_path)
        abs_plot_script_path = os.path.abspath(plot_script)
        command = [sys.executable, abs_plot_script_path, abs_bufkit_path, str(fcst_hour)]
        print(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=os.path.dirname(abs_plot_script_path))
        print("--- plot_sharppy_file.py stdout ---")
        print(result.stdout)
        print("--- plot_sharppy_file.py stderr ---")
        print(result.stderr)
        print("--- End plot_sharppy_file.py output ---")
        if result.returncode != 0:
            print(f"Error: {plot_script} failed with return code {result.returncode}.", file=sys.stderr)
            if "No profile found for forecast hour" in result.stderr or "No profile found for forecast hour" in result.stdout:
                 print(f"Specific Error: Forecast hour {fcst_hour} not found in {os.path.basename(bufkit_file_path)}.", file=sys.stderr)
            elif "Error during BUFKIT to SHARPpy conversion" in result.stderr or "Error during BUFKIT to SHARPpy conversion" in result.stdout:
                 print(f"Specific Error: BUFKIT to SHARPpy conversion failed for {os.path.basename(bufkit_file_path)}.", file=sys.stderr)
            return None
        if os.path.exists(expected_png_path):
            print(f"Successfully generated sounding plot: {expected_png_path}")
            return expected_png_path
        else:
            print(f"Error: Plotting script completed but output PNG not found at {expected_png_path}", file=sys.stderr)
            found_path_match = re.search(r"Image saved as (.*?\.png)", result.stdout)
            if found_path_match:
                 actual_path = found_path_match.group(1)
                 print(f"Note: PNG might have been saved at: {actual_path}", file=sys.stderr)
                 return actual_path if os.path.exists(actual_path) else None
            return None
    except FileNotFoundError:
        print(f"Error: Could not execute Python interpreter '{sys.executable}' or script '{plot_script}'", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing {plot_script}: {e}", file=sys.stderr)
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Find the nearest BUFKIT station and plot its sounding for a given date, location, and forecast hour.")
    parser.add_argument("date", help="Date string in YYYYMMDD format (e.g., 20250301)")
    parser.add_argument("latitude", type=float, help="Target latitude (decimal degrees)")
    parser.add_argument("longitude", type=float, help="Target longitude (decimal degrees)")
    parser.add_argument("fcst_hour", type=int, help="Forecast hour (e.g., 12)")
    parser.add_argument("--data_dir", default="dataset", help="Base directory for HRRR data (default: dataset)")
    args = parser.parse_args()
    if not (0 <= args.fcst_hour <= 48):
         print(f"Error: Forecast hour must be between 0 and 48.", file=sys.stderr)
         sys.exit(1)
    print(f"Finding nearest station for {args.date} at ({args.latitude}, {args.longitude}), F{args.fcst_hour}...")
    nearest_station_id, nearest_bufkit_file, dist = find_nearest_station(
        args.date, args.latitude, args.longitude, args.data_dir
    )
    if nearest_station_id and nearest_bufkit_file:
        png_path = plot_sounding(nearest_bufkit_file, args.fcst_hour)
        if png_path:
            print(f"\nSuccess! Sounding plot generated for station {nearest_station_id}.")
            print(f"PNG Path: {png_path}")
        else:
            print(f"\nError: Failed to generate sounding plot for station {nearest_station_id}.", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nError: Could not find a suitable nearest station.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()