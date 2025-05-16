import pandas as pd
import geopandas as gpd
import glob
import os
import re
from collections import defaultdict, Counter

PPF_OUTPUT_DIR = "ppf_output"
BASE_DATA_DIR = "dataset"
PPF_CATEGORIES = ['2%', '5%', '10%', '15%', '30%', '45%', '60%']
SUMMARY_RISK_LEVELS = ['0%', '2%', '5%', '10%', '15%', '30%']

def count_tornado_reports(date_str):
    hrrr_run = "00z"
    date_dir = os.path.join(BASE_DATA_DIR, f"hrrr_{date_str}_{hrrr_run}")
    ground_truth_dir = os.path.join(date_dir, "ground_truth")

    yy = date_str[2:4]
    mm = date_str[4:6]
    dd = date_str[6:8]
    report_filename = f"{yy}{mm}{dd}_rpts_torn.csv"
    report_path = os.path.join(ground_truth_dir, report_filename)

    if not os.path.exists(report_path):
        return 0, []

    try:
        with open(report_path, 'r') as f:
            lines = f.readlines()

        header_line_index = -1
        required_cols = ['Time', 'F_Scale', 'Lat', 'Lon', 'State']
        header_found = False
        header_line = ''

        for i, line in enumerate(lines):
            if all(col in line for col in ['Time', 'F_Scale', 'Lat', 'Lon']):
                header_line_index = i
                header_found = True
                header_line = line.strip()
                break

        if not header_found:
            try:
                report_data = pd.read_csv(report_path, low_memory=False)
            except Exception as read_err:
                print(f"Error during basic read of {report_path}: {read_err}")
                return 0, []
        else:
             try:
                 report_data = pd.read_csv(report_path, skiprows=header_line_index)
             except Exception as read_err:
                 print(f"Error reading {report_path} after finding header: {read_err}")
                 return 0, []

        if not all(col in report_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in report_data.columns]
            if 'Lat' in report_data.columns and 'Lon' in report_data.columns:
                report_data['Lat'] = pd.to_numeric(report_data['Lat'], errors='coerce')
                report_data['Lon'] = pd.to_numeric(report_data['Lon'], errors='coerce')
                report_data.dropna(subset=['Lat', 'Lon'], inplace=True)
                return len(report_data), []
            else:
                return 0, []

        report_data['Lat'] = pd.to_numeric(report_data['Lat'], errors='coerce')
        report_data['Lon'] = pd.to_numeric(report_data['Lon'], errors='coerce')
        report_data['State'] = report_data['State'].astype(str).str.strip().str.upper()
        report_data.dropna(subset=['Lat', 'Lon', 'State'], inplace=True)
        report_data = report_data[report_data['State'] != '']

        states_list = report_data['State'].tolist()
        return len(states_list), states_list

    except pd.errors.EmptyDataError:
        return 0, []
    except Exception as e:
        print(f"Error reading tornado report file {report_path}: {e}")
        return 0, []

def analyze_geojson_files(output_dir):
    geojson_files = glob.glob(os.path.join(output_dir, "*", "ground_truth_*.geojson"))
    date_pattern = re.compile(r'(\d{8})')

    daily_data = []
    total_days = 0
    total_tornado_reports = 0

    print(f"Found {len(geojson_files)} GeoJSON files to analyze in {output_dir}")

    for filepath in sorted(geojson_files):
        match = date_pattern.search(os.path.basename(filepath))
        if not match:
            print(f"Warning: Could not extract date from filename: {filepath}. Skipping.")
            continue
        date_str = match.group(1)
        total_days += 1

        tornado_reports, states_today = count_tornado_reports(date_str)
        total_tornado_reports += tornado_reports

        day_info = {
            'date': date_str,
            'tornado_reports': tornado_reports,
            'states': states_today,
        }
        day_info.update({level: 0 for level in PPF_CATEGORIES})

        try:
            gdf = gpd.read_file(filepath)
            print(f"Processing {date_str}: Read {len(gdf)} features, {tornado_reports} tornado reports.")

            if not gdf.empty and 'risk_level' in gdf.columns:
                risk_counts = gdf['risk_level'].value_counts().to_dict()
                for level, count in risk_counts.items():
                    if level in day_info:
                        day_info[level] = count
                    else:
                        print(f"Warning: Unexpected risk level '{level}' in {filepath}")


        except Exception as e:
            print(f"\nError processing GeoJSON file {filepath}: {e}")

        daily_data.append(day_info)

    if not daily_data:
        print("No data processed. Exiting.")
        return pd.DataFrame(), pd.DataFrame(), 0, 0

    daily_df = pd.DataFrame(daily_data)
    daily_df.set_index('date', inplace=True)
    def get_max_risk(row):
        for level in reversed(PPF_CATEGORIES):
            if row[level] > 0:
                return level
        return '0%'
    daily_df['max_risk_level'] = daily_df.apply(get_max_risk, axis=1)

    table1_data = []
    for level in SUMMARY_RISK_LEVELS:
        days_at_max = daily_df[daily_df['max_risk_level'] == level]
        days_present = len(days_at_max)
        if days_present > 0:
            total_reports_for_level = days_at_max['tornado_reports'].sum()
        else:
            total_reports_for_level = 0
        table1_data.append({
            'Max Risk Level Reached': level,
            'Days Present': days_present,
            'Total Tornado Reports': total_reports_for_level
        })
    table1_df = pd.DataFrame(table1_data)
    table1_df.set_index('Max Risk Level Reached', inplace=True)
    table1_df.index.name = 'Max Risk Day'

    table2_data = []
    for date, row in daily_df.sort_index().iterrows(): 
        state_counts_day = Counter(row['states'])
        top_3_states = state_counts_day.most_common(3)
        top_3_str = ", ".join([f"{state} ({count})" for state, count in top_3_states])
        if not top_3_str:
             top_3_str = "N/A"
        table2_data.append({
            'Date': date,
            'Max Risk': row['max_risk_level'],
            'Total Reports': row['tornado_reports'],
            'Top 3 States': top_3_str
        })
    table2_df_all_days = pd.DataFrame(table2_data) 

    return table1_df, table2_df_all_days, total_days, total_tornado_reports


if __name__ == "__main__":
    print("Starting PPF output analysis...")
    table1_summary, table2_detail_all_days, total_days, total_tornado_reports = analyze_geojson_files(PPF_OUTPUT_DIR)

    print("\n--- Table 1: Summary by Max Risk Level Reached ---")
    if not table1_summary.empty:
        print(table1_summary.to_string())
    else:
        print("No summary data generated.")

    print("\n--- Table 2: Details for All Analyzed Days --- ")
    if not table2_detail_all_days.empty:
        try:
            table2_detail_all_days['Date'] = pd.to_datetime(table2_detail_all_days['Date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Warning: Could not format date column - {e}")
        print(table2_detail_all_days.to_string(index=False))
    else:
        print("No detailed daily data generated.")

    if total_days > 0:
        print(f"\n--- Overall Summary ---")
        print(f"Total days analyzed: {total_days}")
        print(f"Total tornado reports across all days: {total_tornado_reports}")
        print(f"Average tornado reports per day: {total_tornado_reports/total_days:.2f}")
        if '0%' in table1_summary.index:
            no_risk_days = table1_summary.loc['0%', 'Days Present']
            print(f"Days with no risk polygons (0% max): {no_risk_days} ({no_risk_days/total_days*100:.1f}% of total)")
            days_with_risk = total_days - no_risk_days
            print(f"Days with at least one risk polygon: {days_with_risk} ({days_with_risk/total_days*100:.1f}% of total)")

    print("\nAnalysis complete.") 