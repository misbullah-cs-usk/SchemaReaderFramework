from pathlib import Path
from plot_utils import load_csv, boxplot_metric, barplot_mean_std
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "benchmark_results/benchmark_query_sensor_data_more_features_10k_valid_all_runs.csv"
    out_dir = Path("plots_output/query")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)

    # ---------- TOTAL QUERY TIME ----------
    boxplot_metric(
        df,
        metric="total_query_ms",
        title="Total Query Time Distribution (ms)",
        ylabel="Time (ms)",
        save_path=out_dir / "total_query_time_boxplot.png"
    )

    barplot_mean_std(
        df,
        metric="total_query_ms",
        title="Total Query Time Mean ± Std (ms)",
        ylabel="Time (ms)",
        save_path=out_dir / "total_query_time_bar.png"
    )

    # ---------- QUERY COMPONENTS ----------
    for metric, label in [
        ("projection_ms", "Projection Time (ms)"),
        ("filter_ms", "Filter Time (ms)"),
        ("aggregation_ms", "Aggregation Time (ms)")
    ]:
        barplot_mean_std(
            df,
            metric=metric,
            title=f"{label} Mean ± Std",
            ylabel="Time (ms)",
            save_path=out_dir / f"{metric}_bar.png"
        )

    # ---------- CPU USAGE ----------
    boxplot_metric(
        df,
        metric="cpu_avg_proc_percent",
        title="CPU Usage During Query (%)",
        ylabel="CPU (%)",
        save_path=out_dir / "cpu_query_boxplot.png"
    )

    # ---------- PEAK MEMORY DELTA ----------
    barplot_mean_std(
        df,
        metric="mem_peak_delta_mb",
        title="Peak Memory Delta During Query (MB)",
        ylabel="Peak Memory Delta (MB)",
        save_path=out_dir / "mem_peak_delta_bar.png"
    )

    print("\n[OK] Query benchmark plots generated.")


if __name__ == "__main__":
    main()

