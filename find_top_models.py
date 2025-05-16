import csv
from collections import defaultdict
import sys

def find_top_models(csv_filepath, target_dates):
    data_by_date = defaultdict(list)
    expected_header = [
        'date', 'model', 'per_category_iou', 'daily_mean_iou', 'daily_tornadobench',
        'daily_weight', 'is_false_alarm', 'is_hard_hallucination', 'max_risk_status',
        'daily_weighted_hallucination_loss', 'gt_overall_centroid_x', 'gt_overall_centroid_y',
        'pred_overall_centroid_x', 'pred_overall_centroid_y', 'overall_centroid_distance',
        'gt_hr_centroid_x', 'gt_hr_centroid_y', 'pred_hr_centroid_x', 'pred_hr_centroid_y',
        'hr_centroid_distance', 'gt_had_actual_risk'
    ]
    tornadobench_col_index = -1
    date_col_index = -1
    model_col_index = -1

    try:
        with open(csv_filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)

            header = next(reader)
            try:
                tornadobench_col_index = header.index('daily_tornadobench')
                date_col_index = header.index('date')
                model_col_index = header.index('model')
            except ValueError as e:
                print(f"Error: Required column not found in header - {e}", file=sys.stderr)
                return {}

            for row in reader:
                if not row:
                    continue
                try:
                    current_date = row[date_col_index]
                    if current_date in target_dates:
                        model_name = row[model_col_index]
                        score_str = row[tornadobench_col_index]
                        try:
                            score = float(score_str) if score_str else 0.0
                        except ValueError:
                            print(f"Warning: Could not convert score '{score_str}' to float for model '{model_name}' on date {current_date}. Treating as 0.0.", file=sys.stderr)
                            score = 0.0
                        data_by_date[current_date].append((model_name, score))
                except IndexError:
                    print(f"Warning: Skipping row with insufficient columns: {row}", file=sys.stderr)
                    continue

    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return {}

    top_models = {}
    for date, models_scores in data_by_date.items():
        sorted_models = sorted(models_scores, key=lambda item: item[1], reverse=True)
        top_models[date] = sorted_models[:2]

    return top_models

if __name__ == "__main__":
    csv_file = 'iou_results/daily_model_scores.csv'
    dates_to_check = ['20250314', '20250402']

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        dates_to_check = sys.argv[2:]

    results = find_top_models(csv_file, dates_to_check)

    if results:
        print(f"Top 2 models by daily_tornadobench score for dates {', '.join(dates_to_check)}:")
        for date in dates_to_check:
            if date in results:
                print(f"\nDate: {date}")
                if results[date]:
                    for i, (model, score) in enumerate(results[date]):
                        print(f"  {i+1}. Model: {model}, Score: {score:.4f}")
                else:
                    print("  No data found for this date.")
            else:
                 print(f"\nDate: {date}")
                 print("  No data found for this date.")
    else:
        print("Could not process the CSV file.") 