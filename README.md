# AgentCaster: Reasoning-Guided Tornado Forecasting

Dataset: https://huggingface.co/datasets/agentcaster/agentcaster

Github: https://github.com/agentcaster/agentcaster

There is a growing need to evaluate Large Language Models (LLMs) on complex, high-impact, real-world tasks to assess their true readiness as reasoning agents. To address this gap, we introduce AgentCaster, a contamination-free framework employing multimodal LLMs end-to-end for the challenging, long-horizon task of tornado forecasting. Within AgentCaster, models interpret heterogeneous spatiotemporal data from a high-resolution convection-allowing forecast archive. We assess model performance over a 40-day period featuring diverse historical data, spanning several major tornado outbreaks and including over 500 tornado reports. Each day, models query interactively from a pool of 3,625 forecast maps and 40,125 forecast soundings for a forecast horizon of 12-36 hours. Probabilistic tornado-risk polygon predictions are verified against ground truths derived from geometric comparisons across disjoint risk bands in projected coordinate space. To quantify accuracy, we propose domain-specific TornadoBench and TornadoHallucination metrics, with TornadoBench highly challenging for both LLMs and domain expert human forecasters. Notably, human experts significantly outperform state‑of‑the‑art models, which demonstrate a strong tendency to hallucinate and overpredict risk intensity, struggle with precise geographic placement, and exhibit poor spatiotemporal reasoning in complex, dynamically evolving systems. AgentCaster aims to advance research on improving LLM agents for challenging reasoning tasks in critical domains.

## AgentCaster Quick Start

### 1) Config

Edit `config.json` and set:
 - `models`: array with your model ID(s), e.g. `["x-ai/grok-4"]`.
 - `start_date` / `end_date`: processing window in `YYYYMMDD`.
 - Create and put your key in `openrouter_api_key.txt`.

### 2) Prepare Data + Ground Truth

For the paper dataset release: https://huggingface.co/datasets/agentcaster/agentcaster

Otherwise:
- Download raw data (HRRR maps, BUFKIT, SPC outlooks, ground truth):
  - `python data_downloader.py`
  - Adjust date range (`START_DATE`, `END_DATE`) if needed.
- Generate HRRR plot PNGs:
  - `python process_and_plot_hrrr.py --all-dates`
- Generate Ground Truth polygons for scoring:
  - `python calculate_ppf.py`
  - This creates `ppf_output/{YYYYMMDD}/ground_truth_{YYYYMMDD}.geojson` per date.

### 3) Run The Agent

- `python agent_interaction.py`

Per-day prediction outputs are saved under `llm_predictions/YYYYMMDD/`. Logs are written to `logs/`.

### 4) Score And Visualize

Evaluate predictions vs. ground truth:
- `python calculate_iou.py --start YYYYMMDD --end YYYYMMDD`

Outputs in `iou_results/`.

### 5) Optional Utilities

- Confidence intervals for scores:
  - `python calculate_cis.py`
  - `python plot_summary_with_cis.py`
- Render PNGs from LLM GeoJSON predictions only (no comparisons):
  - `python calculate_llm.py`
 - SPC Day 1 outlook plots (official baseline):
   - `python calculate_spc.py`
 - Analyze ground-truth (PPF) outputs and tornado report counts:
   - `python analyze_ppf_output.py`
 - HRRR plot utilities:
   - `python hrrr_plots_utilities/plot_stations_map.py`
