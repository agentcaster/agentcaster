import os
import re
import argparse
import pandas as pd
from collections import defaultdict

def extract_model_name(filename):
    identifier = filename.replace("agent_interaction_", "").replace(".log", "")
    
    match_date = re.search(r"_(\d{8})$", identifier)
    if match_date:
        identifier = identifier[:match_date.start(0)]
            
    if not identifier: 
        return "unknown_model"
    return identifier

def analyze_log_file(filepath):
    assistant_turns = 0
    tool_calls = 0
    sounding_requests = 0
    filename = os.path.basename(filepath)
    model_name = extract_model_name(filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if "INFO - Assistant:" in line: 
                    assistant_turns += 1
                
                if "INFO - Executing Tool Call:" in line:
                    tool_calls += 1
                    if "Name=request_sounding" in line:
                        sounding_requests += 1
    except Exception as e:
        print(f"Error reading or parsing {filepath}: {e}")
        return None

    return {
        "filename": filename,
        "model_name": model_name,
        "assistant_turns": assistant_turns,
        "tool_calls": tool_calls,
        "sounding_requests": sounding_requests,
    }

def get_prediction_dates(base_predictions_dir="llm_predictions"):
    dates = []
    if not os.path.exists(base_predictions_dir):
        print(f"Error: Predictions directory '{base_predictions_dir}' not found. Cannot count prediction days.")
        return dates
    for item in os.listdir(base_predictions_dir):
        if os.path.isdir(os.path.join(base_predictions_dir, item)):
            if re.match(r"^\d{8}$", item):
                dates.append(item)
    return sorted(dates)

def count_successful_prediction_days(models, dates, base_predictions_dir="llm_predictions"):
    model_prediction_counts = defaultdict(int)

    if not dates:
        print("No prediction dates found in directories, so prediction day counts will be 0.")
        return model_prediction_counts

    for date_str in dates:
        date_dir_path = os.path.join(base_predictions_dir, date_str)
        if not os.path.exists(date_dir_path) or not os.path.isdir(date_dir_path):
            continue

        try:
            prediction_files = os.listdir(date_dir_path)
        except OSError:
            print(f"Warning: Could not list files in {date_dir_path}")
            continue
            
        for model_name in models:
            safe_model_name_for_regex = re.escape(model_name)
            pattern_png = re.compile(rf"prediction_{safe_model_name_for_regex}_{date_str}\.png")
            pattern_geojson = re.compile(rf"prediction_{safe_model_name_for_regex}_{date_str}\.geojson")
            
            found_for_date = False
            for fname in prediction_files:
                if pattern_png.match(fname) or pattern_geojson.match(fname):
                    model_prediction_counts[model_name] += 1
                    found_for_date = True
                    break 
    return model_prediction_counts

def main():
    parser = argparse.ArgumentParser(description="Analyze agent interaction log files and count prediction days.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory containing the log files.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="llm_predictions",
        help="Directory containing LLM predictions (for counting prediction days)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="combined_analysis_summary.csv",
        help="Filename for the output CSV summary.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' not found.")
        return

    all_results = []
    print(f"Analyzing logs in directory: {args.log_dir}\n")

    for filename in os.listdir(args.log_dir):
        if filename.endswith(".log"):
            filepath = os.path.join(args.log_dir, filename)
            result = analyze_log_file(filepath)
            if result:
                all_results.append(result)

    if not all_results:
        print("No log files found or processed in log directory.")
        return

    df_results = pd.DataFrame(all_results)

    logged_model_names = sorted(df_results['model_name'].unique())
    
    print(f"\n--- Counting Prediction Days (for models found in logs) ---")
    prediction_dates = get_prediction_dates(args.predictions_dir)
    if prediction_dates:
        print(f"Found {len(prediction_dates)} prediction dates in '{args.predictions_dir}'.")
    
    model_prediction_counts = count_successful_prediction_days(logged_model_names, prediction_dates, args.predictions_dir)
    
    df_results['prediction_days'] = df_results['model_name'].apply(lambda model_name: model_prediction_counts.get(model_name, 0))

    print("\n--- Overall Log Analysis Summary ---")
    print(f"Total conversations analyzed: {len(df_results)}")
    print(f"Average Assistant Turns per conversation: {df_results['assistant_turns'].mean():.2f}")
    print(f"Average Tool Calls per conversation: {df_results['tool_calls'].mean():.2f}")
    print(f"Average Sounding Requests per conversation: {df_results['sounding_requests'].mean():.2f}")
    print(f"Total Sounding Requests across all conversations: {df_results['sounding_requests'].sum()}")
    print(f"Maximum Sounding Requests in a single conversation: {df_results['sounding_requests'].max()}")

    print("\n--- Per-Model Average Summary ---")
    if 'model_name' in df_results.columns:
        for model_name_to_process in logged_model_names:
            model_df = df_results[df_results['model_name'] == model_name_to_process]
            if model_df.empty:
                continue

            pred_days_count = model_df['prediction_days'].iloc[0] if not model_df['prediction_days'].empty else model_prediction_counts.get(model_name_to_process, 0)

            print(f"\nModel: {model_name_to_process}")
            print(f"  Prediction Days: {pred_days_count}")
            print(f"  Avg Assistant Turns: {model_df['assistant_turns'].mean():.2f}")
            print(f"  Avg Tool Calls: {model_df['tool_calls'].mean():.2f}")
            print(f"  Avg Sounding Requests: {model_df['sounding_requests'].mean():.2f}")
            print(f"  Max Sounding Requests: {model_df['sounding_requests'].max()}")
    else:
        print("Could not generate per-model summary (model_name column not found in log results).")

    try:
        column_order = ["filename", "model_name", "assistant_turns", "tool_calls", "sounding_requests", "prediction_days"]
        existing_columns = [col for col in column_order if col in df_results.columns]
        df_results = df_results[existing_columns]
        
        df_results.to_csv(args.output_csv, index=False)
        print(f"\nDetailed combined results saved to {args.output_csv}")
    except Exception as e:
        print(f"Error saving CSV to {args.output_csv}: {e}")

if __name__ == "__main__":
    main()