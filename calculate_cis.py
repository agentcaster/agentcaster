import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

DAILY_SCORES_FILE = 'iou_results/daily_model_scores.csv'
OUTPUT_FILE = 'iou_results/model_summary_with_cis.csv'
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.9545

RNG = np.random.default_rng()

def bootstrap_weighted_mean(data, n_bootstrap, weights_col, values_col, model_denominator, confidence):
    bootstrap_means = []
    n_samples = len(data)

    if n_samples == 0:
        print(f"Warning: No samples found for bootstrapping weighted mean (values: {values_col}, weights: {weights_col}). Returning NaN CI.")
        return np.nan, np.nan
        
    weights = data[weights_col].values
    values = data[values_col].values

    if model_denominator <= 0:
        print(f"Warning: Model denominator is {model_denominator}. Weighted mean bootstrap will likely result in 0 or NaN.")
        return np.nan, np.nan

    for _ in range(n_bootstrap):
        indices = RNG.choice(np.arange(n_samples), size=n_samples, replace=True)
        bootstrap_weighted_sum = np.sum(values[indices] * weights[indices])
        bootstrap_mean = bootstrap_weighted_sum / model_denominator if model_denominator > 0 else 0.0
        bootstrap_means.append(bootstrap_mean)
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    return lower_bound, upper_bound

def bootstrap_mean(data, n_bootstrap, values_col, confidence):
    bootstrap_means = []
    n_samples = len(data)

    if n_samples == 0:
        print(f"Warning: No samples found for bootstrapping column {values_col}. Returning NaN CI.")
        return np.nan, np.nan
        
    values = data[values_col].values

    for _ in range(n_bootstrap):
        indices = RNG.choice(np.arange(n_samples), size=n_samples, replace=True)
        bootstrap_mean = np.mean(values[indices])
        bootstrap_means.append(bootstrap_mean)
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    return lower_bound, upper_bound

def main():
    parser = argparse.ArgumentParser(description="Calculate Confidence Intervals for AgentCaster Metrics using Bootstrapping.")
    parser.add_argument("--input", default=DAILY_SCORES_FILE, help=f"Path to the daily scores CSV file (default: {DAILY_SCORES_FILE})")
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"Path to the output CSV file with CIs (default: {OUTPUT_FILE})")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP, help=f"Number of bootstrap iterations (default: {N_BOOTSTRAP})")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_LEVEL, help=f"Confidence level for intervals (default: {CONFIDENCE_LEVEL:.4f}, approx 2 sigma)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator for reproducible results.")
    args = parser.parse_args()

    global RNG
    if args.seed is not None:
        RNG = np.random.default_rng(args.seed)
        print(f"Using random seed: {args.seed}")
    else:
        RNG = np.random.default_rng()
        print("Using random seed: None (results may vary on re-run)")
    try:
        df_daily = pd.read_csv(args.input)
        print(f"Successfully read daily scores from {args.input}")
    except FileNotFoundError:
        print(f"Error: Daily scores file not found at {args.input}")
        print("Please run calculate_iou.py first to generate this file.")
        return
    except Exception as e:
        print(f"Error reading daily scores file: {e}")
        return

    df_daily['daily_tornadobench'] = pd.to_numeric(df_daily['daily_tornadobench'], errors='coerce')
    df_daily['daily_weight'] = pd.to_numeric(df_daily['daily_weight'], errors='coerce')
    df_daily['is_false_alarm'] = pd.to_numeric(df_daily['is_false_alarm'], errors='coerce')
    df_daily['daily_weighted_hallucination_loss'] = pd.to_numeric(df_daily['daily_weighted_hallucination_loss'], errors='coerce')
    initial_rows = len(df_daily)
    df_daily.dropna(subset=['daily_tornadobench', 'daily_weight', 'is_false_alarm', 'daily_weighted_hallucination_loss'], inplace=True)
    dropped_rows = initial_rows - len(df_daily)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with NaN values in required columns.")

    ci_results = []
    for model, group_data in tqdm(df_daily.groupby('model'), desc="Calculating CIs per model", leave=False):
        print(f"Processing model: {model} ({len(group_data)} days)")
        
        if group_data.empty:
            print(f"  Skipping model {model} due to no valid data after cleaning.")
            ci_results.append({
                'model': model,
                'TornadoBench_CI_Lower': np.nan, 'TornadoBench_CI_Upper': np.nan,
                'TornadoHallucinationSimple_CI_Lower': np.nan, 'TornadoHallucinationSimple_CI_Upper': np.nan,
                'TornadoHallucinationHard_CI_Lower': np.nan, 'TornadoHallucinationHard_CI_Upper': np.nan,
            })
            continue
        model_unique_day_weights = group_data[['date', 'daily_weight']].drop_duplicates()
        model_denominator = model_unique_day_weights['daily_weight'].sum()
        print(f"  Model-specific denominator for TornadoBench: {model_denominator}")

        tb_lower, tb_upper = bootstrap_weighted_mean(
            group_data,
            args.n_bootstrap,
            'daily_weight',
            'daily_tornadobench',
            model_denominator,
            args.confidence
        )
        print(f"  TornadoBench CI ({args.confidence*100:.2f}%): [{tb_lower*100:.2f}%, {tb_upper*100:.2f}%]")

        ths_lower, ths_upper = bootstrap_mean(
            group_data,
            args.n_bootstrap,
            'is_false_alarm',
            args.confidence
        )
        print(f"  TornadoHallucinationSimple CI ({args.confidence*100:.2f}%): [{ths_lower:.2f}, {ths_upper:.2f}]")

        thh_lower, thh_upper = bootstrap_mean(
            group_data,
            args.n_bootstrap,
            'daily_weighted_hallucination_loss',
            args.confidence
        )
        print(f"  TornadoHallucinationHard CI ({args.confidence*100:.2f}%): [{thh_lower:.2f}, {thh_upper:.2f}]")

        ci_results.append({
            'model': model,
            'TornadoBench_CI_Lower': tb_lower,
            'TornadoBench_CI_Upper': tb_upper,
            'TornadoHallucinationSimple_CI_Lower': ths_lower,
            'TornadoHallucinationSimple_CI_Upper': ths_upper,
            'TornadoHallucinationHard_CI_Lower': thh_lower,
            'TornadoHallucinationHard_CI_Upper': thh_upper,
        })

    df_cis = pd.DataFrame(ci_results)

    try:
        df_summary = pd.read_csv('iou_results/model_summary.csv')
        df_final = pd.merge(df_summary, df_cis, on='model', how='left')
        print(f"\nMerged CIs with existing summary from iou_results/model_summary.csv")
    except FileNotFoundError:
        print("\nWarning: iou_results/model_summary.csv not found. Saving CIs only.")
        df_final = df_cis
    except Exception as e:
        print(f"\nError merging with model_summary.csv: {e}. Saving CIs only.")
        df_final = df_cis

    try:
        df_final.to_csv(args.output, index=False, float_format='%.4f')
        print(f"\nConfidence interval results saved to {args.output}")
    except Exception as e:
        print(f"\nError saving results to {args.output}: {e}")

if __name__ == "__main__":
    main() 