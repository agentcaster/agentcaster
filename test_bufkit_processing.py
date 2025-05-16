#!/usr/bin/env python

import os
import sys
import glob
import re
import subprocess
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

TARGET_ERROR_MSG = "BUFKIT to SHARPpy conversion failed"

def extract_info_from_bufkit(bufkit_file):
    stid, lat, lon = None, None, None
    try:
        with open(bufkit_file, 'r', encoding='latin-1') as f:
            lines = [f.readline() for _ in range(6)]
            for i, line in enumerate(lines):
                if "STID" in line:
                    stid_match = re.search(r'STID\s*=\s*(\S+)', line)
                    if stid_match:
                        stid = stid_match.group(1).strip()

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
        print(f"\nWarning: BUFKIT file not found during info extraction: {bufkit_file}", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"\nWarning: Could not parse header for {bufkit_file}: {str(e)}", file=sys.stderr)
        return None, None, None

    if stid and lat is not None and lon is not None:
        if stid.startswith("STNM="):
             stid_stnm_match = re.search(r'STNM\s*=\s*(\S+)', line)
             if stid_stnm_match:
                  stid = stid_stnm_match.group(1).strip()
             else:
                  stid = None

        return stid, lat, lon
    else:
        return None, None, None


def test_date(date_str, fcst_hour, base_data_dir="dataset"):
    print(f"\n--- Testing Date: {date_str}, Forecast Hour: {fcst_hour} ---")
    bufkit_dir = Path(base_data_dir) / f"hrrr_{date_str}_00z" / "bufkit"
    broken_stations = set()

    if not bufkit_dir.is_dir():
        print(f"Error: BUFKIT directory not found: {bufkit_dir}", file=sys.stderr)
        return broken_stations

    bufkit_files = sorted(list(bufkit_dir.glob("hrrr_*.buf")))

    if not bufkit_files:
        print(f"Error: No BUFKIT files found in {bufkit_dir}", file=sys.stderr)
        return broken_stations

    print(f"Found {len(bufkit_files)} stations to test.")

    for bufkit_file in tqdm(bufkit_files, desc=f"Processing {date_str} F{fcst_hour}", unit="station"):
        stid, lat, lon = extract_info_from_bufkit(bufkit_file)

        if stid is None:
            tqdm.write(f"Skipping {bufkit_file.name}: Could not extract info.")
            continue

        sounding_script = Path("find_and_plot_nearest_sounding.py")
        command = [
            sys.executable,
            str(sounding_script.resolve()),
            date_str,
            str(lat),
            str(lon),
            str(fcst_hour),
            "--data_dir",
            str(Path(base_data_dir).resolve())
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)

            if TARGET_ERROR_MSG in result.stdout or TARGET_ERROR_MSG in result.stderr:
                broken_stations.add(stid)

        except subprocess.TimeoutExpired:
             tqdm.write(f"BROKEN: {stid} (Timeout)")
             broken_stations.add(stid + " (Timeout)")
        except Exception as e:
            tqdm.write(f"BROKEN: {stid} (Execution Error: {e})")
            broken_stations.add(stid + " (Exec Error)")

    print(f"--- Finished Testing {date_str}: Found {len(broken_stations)} broken stations (target error) ---")
    return broken_stations


def write_csv(output_file, broken_dict, date1, date2):
    broken_only_date1 = broken_dict[date1].difference(broken_dict[date2])
    broken_only_date2 = broken_dict[date2].difference(broken_dict[date1])
    broken_on_both = broken_dict[date1].intersection(broken_dict[date2])

    print(f"\n--- Summary ---")
    print(f"Broken only on {date1}: {len(broken_only_date1)}")
    print(f"Broken only on {date2}: {len(broken_only_date2)}")
    print(f"Broken on BOTH dates: {len(broken_on_both)}")

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Broken Station ID', 'Broken Date(s)'])

            for stid in sorted(list(broken_only_date1)):
                writer.writerow([stid, date1])
            for stid in sorted(list(broken_only_date2)):
                writer.writerow([stid, date2])
            for stid in sorted(list(broken_on_both)):
                writer.writerow([stid, f"{date1}_and_{date2}"])
        print(f"\nResults saved to: {output_file}")
    except IOError as e:
        print(f"\nError writing CSV file {output_file}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Test BUFKIT sounding processing for specific dates and identify broken stations.')
    parser.add_argument('dates', nargs=2, help='Two dates to test (format: YYYYMMDD YYYYMMDD). e.g., 20250301 20250320')
    parser.add_argument('--fcst_hour', type=int, default=12, help='Forecast hour to test (default: 12)')
    parser.add_argument('--data_dir', default='dataset', help='Base directory for HRRR data (default: dataset)')
    parser.add_argument('--output_csv', default='broken_sounding_stations.csv', help='Output CSV file name (default: broken_sounding_stations.csv)')

    args = parser.parse_args()

    date1, date2 = args.dates[0], args.dates[1]
    broken_results = {}

    broken_results[date1] = test_date(date1, args.fcst_hour, args.data_dir)

    broken_results[date2] = test_date(date2, args.fcst_hour, args.data_dir)

    write_csv(args.output_csv, broken_results, date1, date2)


if __name__ == "__main__":
    main() 