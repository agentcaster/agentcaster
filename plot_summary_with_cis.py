import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
import argparse

INPUT_CSV = 'iou_results/model_summary_with_cis.csv'
PLOT_DIR = 'iou_results'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_bars_with_cis(df_summary, metric_col, ci_lower_col, ci_upper_col, title, ylabel, output_filename, lower_is_better=False, color='#e74c3c'):
    if not all(col in df_summary.columns for col in [metric_col, ci_lower_col, ci_upper_col]):
        print(f"Error: Missing required columns ({metric_col}, {ci_lower_col}, {ci_upper_col}) for plotting '{title}'. Skipping plot.")
        return

    df_sorted = df_summary.sort_values(by=metric_col, ascending=lower_is_better).copy()
    
    lower_error = df_sorted[metric_col] - df_sorted[ci_lower_col]
    upper_error = df_sorted[ci_upper_col] - df_sorted[metric_col]
    yerr = np.array([lower_error.fillna(0), upper_error.fillna(0)])

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.axisbelow': True,
        'axes.edgecolor': '#444444',
        'axes.linewidth': 0.8
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    model_names = df_sorted['model'].tolist()
    display_names = [name.split('_', 1)[1] if '_' in name else name for name in model_names]
    
    x_pos = np.arange(len(model_names))
    
    bars = ax.bar(x_pos, df_sorted[metric_col], width=0.7, color=color, 
                edgecolor='none', alpha=0.85, yerr=yerr, capsize=5, 
                error_kw={'ecolor': '#444444', 'alpha': 0.7, 'capthick': 1})
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.add_patch(plt.Rectangle((x_pos[i]-0.35+0.02, 0), 0.7, height, fill=True, 
                                  color='black', alpha=0.05, zorder=0))
    
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title(title, pad=15)
    
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.3, color='#888888')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        lower_bound_val = df_sorted[ci_lower_col].iloc[i]
        upper_bound_val = df_sorted[ci_upper_col].iloc[i]

        label_y_pos = height + upper_error.iloc[i]

        if 'Bench' in ylabel:
            main_value_str = f'{height*100:.2f}%'
            interval_str = f'[{lower_bound_val*100:.2f}%, {upper_bound_val*100:.2f}%]'
            value_text = f'{main_value_str}\n{interval_str}'
        else:
            main_value_str = f'{height:.2f}'
            interval_str = f'[{lower_bound_val:.2f}, {upper_bound_val:.2f}]'
            value_text = f'{main_value_str}\n{interval_str}'

        ax.annotate(value_text,
                   xy=(x_pos[i], label_y_pos),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9,
                   fontweight='bold',
                   color='#444444')
    
    if 'Bench' in ylabel:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}'))

    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.95])
    
    try:
        filepath = os.path.join(PLOT_DIR, output_filename)
        ensure_dir(PLOT_DIR)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot model summary metrics with Confidence Intervals.")
    parser.add_argument("--input", default=INPUT_CSV, help=f"Path to the summary CSV with CIs (default: {INPUT_CSV})")
    args = parser.parse_args()

    try:
        df_summary_cis = pd.read_csv(args.input)
        print(f"Successfully read summary data with CIs from {args.input}")
    except FileNotFoundError:
        print(f"Error: Summary file with CIs not found at {args.input}")
        print("Please run calculate_cis.py first to generate this file.")
        return
    except Exception as e:
        print(f"Error reading summary file with CIs: {e}")
        return

    models_to_exclude = ['']
    original_count = len(df_summary_cis)
    df_summary_cis = df_summary_cis[~df_summary_cis['model'].isin(models_to_exclude)]
    filtered_count = len(df_summary_cis)
    if original_count > filtered_count:
        excluded_names = ", ".join(models_to_exclude)
        print(f"Filtered out models: {excluded_names}. Plotting {filtered_count}/{original_count} models.")
    
    color_tb = '#e74c3c'
    color_ths = '#2ecc71'
    color_thh = '#9b59b6'

    plot_bars_with_cis(
        df_summary_cis,
        metric_col='TornadoBench',
        ci_lower_col='TornadoBench_CI_Lower',
        ci_upper_col='TornadoBench_CI_Upper',
        title='TornadoBench',
        ylabel='TornadoBench Score (%)',
        output_filename='model_summary_TornadoBench_CI.png',
        lower_is_better=False,
        color=color_tb
    )

    plot_bars_with_cis(
        df_summary_cis,
        metric_col='TornadoHallucinationSimple',
        ci_lower_col='TornadoHallucinationSimple_CI_Lower',
        ci_upper_col='TornadoHallucinationSimple_CI_Upper',
        title='TornadoHallucinationSimple (Lower is Better)',
        ylabel='TornadoHallucinationSimple',
        output_filename='model_summary_TornadoHallucinationSimple_CI.png',
        lower_is_better=True,
        color=color_ths
    )

    plot_bars_with_cis(
        df_summary_cis,
        metric_col='TornadoHallucinationHard',
        ci_lower_col='TornadoHallucinationHard_CI_Lower',
        ci_upper_col='TornadoHallucinationHard_CI_Upper',
        title='TornadoHallucinationHard (Lower is Better)',
        ylabel='TornadoHallucinationHard',
        output_filename='model_summary_TornadoHallucinationHard_CI.png',
        lower_is_better=True,
        color=color_thh
    )

if __name__ == "__main__":
    main() 