import os
import re
from collections import defaultdict

"""
terminal command to generically check file counts:
cd dataset/ && for d in hrrr_*_00z; do if [ -d "$d" ]; then echo "--- $d ---"; for sub in hrrr_maps bufkit spc_outlooks ground_truth; do if [ -d "$d/$sub" ]; then count=$(find "$d/$sub" -maxdepth 1 -type f -not -name ".DS_Store" | wc -l | xargs); echo "  $sub: $count"; else echo "  $sub: Not found"; fi; done; hrrr_plots_dir="$d/hrrr_plots"; if [ -d "$hrrr_plots_dir" ]; then plots_count=$(find "$hrrr_plots_dir" -type f -not -name ".DS_Store" | wc -l | xargs); echo "  hrrr_plots (recursive): $plots_count"; else echo "  hrrr_plots: Not found"; fi; fi; done

below is the hrrr hourly counter for hrrr_plots:
"""

def verify_hrrr_file_counts(main_dir, start_hour=12, end_hour=36):
    if not os.path.isdir(main_dir):
        print(f"Error: Main directory not found: {main_dir}")
        return

    excluded_files = [".DS_Store"]
    print(f"Excluding files: {', '.join(excluded_files)}")

    file_pattern = re.compile(r"hrrr\.t00z\.wrfsfcf(\d{2})")

    hour_counts = defaultdict(int)
    expected_hours = set(range(start_hour, end_hour + 1))
    found_hours = set()

    print(f"Scanning subdirectories in: {main_dir}")
    print("-" * 30)

    for date_folder_name in os.listdir(main_dir):
        date_folder_path = os.path.join(main_dir, date_folder_name)

        if os.path.isdir(date_folder_path):
            plots_dir = os.path.join(date_folder_path, "hrrr_plots")

            if os.path.isdir(plots_dir):
                print(f"Checking date folder: {date_folder_name}/hrrr_plots")
                try:
                    for variable_dir_name in os.listdir(plots_dir):
                        variable_dir_path = os.path.join(plots_dir, variable_dir_name)

                        if os.path.isdir(variable_dir_path):
                            try:
                                for filename in os.listdir(variable_dir_path):
                                    if filename in excluded_files:
                                        continue
                                        
                                    match = file_pattern.match(filename)
                                    if match:
                                        hour_str = match.group(1)
                                        try:
                                            hour = int(hour_str)
                                            if start_hour <= hour <= end_hour:
                                                hour_counts[hour] += 1
                                                found_hours.add(hour)
                                        except ValueError:
                                            print(f"    Warning: Could not parse hour from {filename} in {date_folder_name}/hrrr_plots/{variable_dir_name}")
                            except OSError as e:
                                 print(f"    Warning: Could not read directory {variable_dir_path}: {e}")
                except OSError as e:
                     print(f"  Warning: Could not read directory {plots_dir}: {e}")

    print("-" * 30)
    print("Total counts per forecast hour ({} to {}):".format(start_hour, end_hour))
    print("-" * 30)

    is_consistent = True
    first_count = -1
    missing_hours = expected_hours - found_hours

    for hour in range(start_hour, end_hour + 1):
        count = hour_counts[hour]
        print(f"Hour {hour:02d}: {count} files")
        if hour not in missing_hours:
            if first_count == -1:
                first_count = count
            elif count != first_count:
                is_consistent = False

    print("-" * 30)

    if missing_hours:
         print(f"Warning: No files found for hours: {sorted(list(missing_hours))}")
         is_consistent = False

    if is_consistent and first_count != -1:
        print(f"\nSuccess: All found forecast hours ({start_hour}-{end_hour}) have a consistent count of {first_count} files.")
    elif first_count == -1 and missing_hours == expected_hours:
        print("\nError: No files matching the pattern and hour range were found in any subdirectory.")
    else:
        print("\nInconsistent Counts: The number of files per forecast hour varies or some hours are missing.")
        print("Please review the counts above.")

    print("-" * 30)

if __name__ == "__main__":
    target_directory = "dataset"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    workspace_root = os.getcwd()
    absolute_target_dir = os.path.join(workspace_root, target_directory)

    if not os.path.isdir(absolute_target_dir):
         absolute_target_dir = os.path.join(script_dir, target_directory)
         if not os.path.isdir(absolute_target_dir):
              parent_dir = os.path.dirname(script_dir)
              absolute_target_dir = os.path.join(parent_dir, target_directory)

    if os.path.isdir(absolute_target_dir):
        verify_hrrr_file_counts(absolute_target_dir)
    else:
         print(f"Warning: Could not reliably determine absolute path for {target_directory}.")
         print(f"Attempting to use relative path '{target_directory}'. Ensure you run this script from the workspace root directory.")
         verify_hrrr_file_counts(target_directory)