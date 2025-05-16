import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

START_DATE = datetime(2025, 4, 1)
END_DATE = datetime(2025, 4, 1)

MAX_WORKERS = 10

def check_existing_folders():
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    tasks = {}
    all_components = ['hrrr_maps', 'bufkit', 'spc_outlooks', 'ground_truth']
    
    for date in [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)]:
        date_str = date.strftime('%Y%m%d')
        folder_name = f"hrrr_{date_str}_00z"
        folder_path = dataset_dir / folder_name
        
        if not folder_path.exists():
            tasks[date_str] = list(all_components)
            continue
        
        missing_components = []
        
        maps_dir = folder_path / "hrrr_maps"
        if not maps_dir.exists() or not any(maps_dir.glob("hrrr.t00z.wrfsfcf*.grib2")):
            missing_components.append('hrrr_maps')
        
        bufkit_dir = folder_path / "bufkit"
        if not bufkit_dir.exists() or not any(bufkit_dir.glob("*.buf")):
            missing_components.append('bufkit')
        
        outlooks_dir = folder_path / "spc_outlooks"
        if not outlooks_dir.exists() or not any(outlooks_dir.glob("*.shp")):
            missing_components.append('spc_outlooks')
        
        truth_dir = folder_path / "ground_truth"
        short_date = date_str[2:]
        truth_file = truth_dir / f"{short_date}_rpts_torn.csv"
        if not truth_dir.exists() or not truth_file.exists():
            missing_components.append('ground_truth')
        
        if missing_components:
            tasks[date_str] = missing_components
            
    existing_dates = [d.strftime('%Y%m%d') for d in [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)] if d.strftime('%Y%m%d') not in tasks]
    incomplete_dates = {d: c for d, c in tasks.items() if c != all_components}
    missing_dates = {d: c for d, c in tasks.items() if c == all_components}

    print(f"Found existing complete folders for: {', '.join(existing_dates) if existing_dates else 'None'}")
    print(f"Found incomplete folders needing components: { {d: c for d, c in incomplete_dates.items()} if incomplete_dates else 'None'}")
    print(f"Need to download data for missing folders: {', '.join(missing_dates.keys()) if missing_dates else 'None'}")
    
    return tasks

def create_date_folder(date_str):
    dataset_dir = Path("dataset")
    folder_name = f"hrrr_{date_str}_00z"
    folder_path = dataset_dir / folder_name
    folder_path.mkdir(exist_ok=True)
    
    (folder_path / "hrrr_maps").mkdir(exist_ok=True)
    (folder_path / "bufkit").mkdir(exist_ok=True)
    (folder_path / "spc_outlooks").mkdir(exist_ok=True)
    (folder_path / "ground_truth").mkdir(exist_ok=True)
    
    return folder_path

def download_file_with_progress(url, output_file):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                desc=f"Downloading {output_file.name}", 
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
        else:
            print(f"    ✗ Failed with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

def download_single_hrrr_file(args):
    date_str, forecast_hour, maps_dir = args
    url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/hrrr.t00z.wrfsfcf{forecast_hour:02d}.grib2"
    output_file = maps_dir / f"hrrr.t00z.wrfsfcf{forecast_hour:02d}.grib2"
    
    if output_file.exists():
        return f"File already exists: {output_file.name}"
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            return f"Successfully downloaded: {output_file.name} ({total_size/1024/1024:.1f} MB)"
        else:
            return f"Failed to download {output_file.name}: Status code {response.status_code}"
    except Exception as e:
        return f"Error downloading {output_file.name}: {e}"

def download_hrrr_maps(date_str, folder_path):
    maps_dir = folder_path / "hrrr_maps"
    date_formatted = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
    
    print(f"Downloading HRRR maps for {date_formatted}...")
    
    forecast_hours = list(range(12, 37))
    
    download_args = [(date_str, hour, maps_dir) for hour in forecast_hours]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_hour = {executor.submit(download_single_hrrr_file, args): args[1] for args in download_args}
        
        with tqdm(total=len(future_to_hour), desc="HRRR forecast hours") as pbar:
            for future in as_completed(future_to_hour):
                hour = future_to_hour[future]
                result = future.result()
                tqdm.write(f"Hour {hour:02d}: {result}")
                pbar.update(1)

def download_bufkit_files(date_str, folder_path):
    bufkit_dir = folder_path / "bufkit"
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    print(f"Downloading BUFKIT files for {year}-{month}-{day}...")
    
    base_url = f"https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/bufkit/00/hrrr/"
    
    try:
        response = requests.get(base_url)
        if response.status_code != 200:
            print(f"  ✗ Failed to fetch directory listing: Status code {response.status_code}")
            return
        
        html_content = response.text
        
        buf_files = re.findall(r'href="([^"]*\.buf)"', html_content)
        if not buf_files:
            buf_files = re.findall(r'<a href="([^"]*)">[^<]*\.buf</a>', html_content)
        
        if not buf_files:
            buf_files = re.findall(r'(hrrr%5F[^"]+\.buf)', html_content)
        
        cleaned_files = []
        for file in buf_files:
            file = file.replace('%5F', '_').replace('%23', '#')
            file = os.path.basename(file)
            cleaned_files.append(file)
        
        if not cleaned_files:
            print(f"  ✗ No BUFKIT files found in directory listing")
            return
        
        print(f"  ℹ Found {len(cleaned_files)} BUFKIT files")
        
        def download_single_bufkit(filename):
            output_file = bufkit_dir / filename
            if output_file.exists():
                return f"File already exists: {filename}"
            
            encoded_filename = filename.replace('#', '%23')
            url = f"{base_url}/{encoded_filename}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    return f"Successfully downloaded: {filename} ({len(response.content)/1024:.1f} KB)"
                else:
                    return f"Failed to download {filename}: Status code {response.status_code}"
            except Exception as e:
                return f"Error downloading {filename}: {e}"
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(download_single_bufkit, file): file for file in cleaned_files}
            
            with tqdm(total=len(future_to_file), desc="BUFKIT files") as pbar:
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    result = future.result()
                    tqdm.write(f"{filename}: {result}")
                    pbar.update(1)
        
        print(f"  ✓ BUFKIT files download complete")
    
    except Exception as e:
        print(f"  ✗ Error downloading BUFKIT files: {e}")

def download_spc_outlooks(date_str, folder_path):
    outlooks_dir = folder_path / "spc_outlooks"
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    print(f"Downloading SPC outlooks for {year}-{month}-{day}...")
    
    url = f"https://www.spc.noaa.gov/products/outlook/archive/{year}/day1otlk_{date_str}_1200-shp.zip"
    output_file = outlooks_dir / f"day1otlk_{date_str}_1200-shp.zip"
    
    if output_file.exists():
        print(f"  - File already exists: {output_file.name}")
        return
    
    try:
        print(f"  - Downloading {url}")
        success = download_file_with_progress(url, output_file)
        
        if success:
            print(f"  - Extracting ZIP file...")
            with zipfile.ZipFile(output_file) as zip_ref:
                zip_files = zip_ref.namelist()
                for file in tqdm(zip_files, desc="Extracting files"):
                    zip_ref.extract(file, outlooks_dir)
            print(f"    ✓ Success (downloaded and extracted)")
    except Exception as e:
        print(f"    ✗ Error: {e}")

def download_ground_truth(date_str, folder_path):
    truth_dir = folder_path / "ground_truth"
    short_date = date_str[2:] 
    
    print(f"Downloading ground truth tornado reports for {date_str}...")
    
    url = f"https://www.spc.noaa.gov/climo/reports/{short_date}_rpts_torn.csv"
    output_file = truth_dir / f"{short_date}_rpts_torn.csv"
    
    if output_file.exists():
        print(f"  - File already exists: {output_file.name}")
        return
    
    try:
        print(f"  - Downloading {url}")
        success = download_file_with_progress(url, output_file)
        if success:
            print(f"    ✓ Success")
    except Exception as e:
        print(f"    ✗ Error: {e}")

def download_specific_components(date_str, folder_path, components_to_download):
    component_map = {
        'hrrr_maps': download_hrrr_maps,
        'bufkit': download_bufkit_files,
        'spc_outlooks': download_spc_outlooks,
        'ground_truth': download_ground_truth
    }
    
    tasks_to_run = {name: func for name, func in component_map.items() if name in components_to_download}
    
    if not tasks_to_run:
        print(f"  ✓ No components specified for download for {date_str}.")
        return

    print(f"Downloading specific components for {date_str}: {', '.join(tasks_to_run.keys())}")
    
    with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
        futures = [executor.submit(func, date_str, folder_path) for func in tasks_to_run.values()]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  ✗ Error during component download for {date_str}: {e}")

    print(f"Completed specific component downloads for {date_str}\n")

def main():
    print("AgentCaster Data Downloader")
    print("==========================")
    print(f"Date Range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print("==========================\n")
    
    download_tasks = check_existing_folders()
    
    if not download_tasks:
        print("\nAll data folders exist and are complete. No downloads needed.")
        return
    
    print("\nStarting downloads...")
    sorted_dates = sorted(download_tasks.keys()) 
    
    for date_str in tqdm(sorted_dates, desc="Processing dates"):
        components = download_tasks[date_str]
        tqdm.write(f"\nProcessing date: {date_str}")
        
        folder_path = create_date_folder(date_str)
        
        download_specific_components(date_str, folder_path, components)
    
    print("\nData download process complete!")

if __name__ == "__main__":
    main()
