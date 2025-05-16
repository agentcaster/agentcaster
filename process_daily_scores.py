import pandas as pd

def process_scores(input_csv_path, output_csv_path):
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_csv_path} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_csv_path}: {e}")
        return

    df['daily_tornadobench'] = pd.to_numeric(df['daily_tornadobench'], errors='coerce')

    df['daily_tornadobench'] = (df['daily_tornadobench'] * 100).round(2)

    df.dropna(subset=['daily_tornadobench'], inplace=True)
    
    df_selected = df[['date', 'model', 'daily_tornadobench']]

    try:
        df_selected.to_csv(output_csv_path, index=False)
        print(f"Successfully created {output_csv_path}")
    except Exception as e:
        print(f"An error occurred while writing to {output_csv_path}: {e}")

if __name__ == "__main__":
    input_file = "iou_results/daily_model_scores.csv"
    output_file = "daily_model_scores_processed.csv"
    process_scores(input_file, output_file) 