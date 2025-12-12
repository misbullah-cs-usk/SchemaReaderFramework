from pathlib import Path
from plot_utils import load_csv, boxplot_metric, barplot_mean_std

def main():
    csv_path = "benchmark_results/benchmark_read_sensor_data_more_features_10k_valid_all_runs.csv"
    out_dir = Path("plots_output/read")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)

    # ---------- READ TIME ----------
    boxplot_metric(
        df,
        metric="read_time_ms",
        title="Read Time Distribution (ms)",
        ylabel="Time (ms)",
        save_path=out_dir / "read_time_boxplot.png"
    )

    barplot_mean_std(
        df,
        metric="read_time_ms",
        title="Read Time Mean ± Std (ms)",
        ylabel="Time (ms)",
        save_path=out_dir / "read_time_bar.png"
    )

    # ---------- CPU USAGE ----------
    boxplot_metric(
        df,
        metric="cpu_avg_proc_percent",
        title="CPU Usage During Read (%)",
        ylabel="CPU (%)",
        save_path=out_dir / "cpu_read_boxplot.png"
    )

    barplot_mean_std(
        df,
        metric="cpu_avg_proc_percent",
        title="CPU Usage Mean ± Std During Read (%)",
        ylabel="CPU (%)",
        save_path=out_dir / "cpu_read_bar.png"
    )

    # ---------- MEMORY DELTA ----------
    barplot_mean_std(
        df,
        metric="mem_delta_mb",
        title="Memory Delta During Read (MB)",
        ylabel="Memory Delta (MB)",
        save_path=out_dir / "read_mem_delta_bar.png"
    )

    # ---------- PEAK MEMORY DELTA ----------
    barplot_mean_std(
        df,
        metric="mem_peak_delta_mb",
        title="Peak Memory Delta During Read (MB)",
        ylabel="Peak Memory Delta (MB)",
        save_path=out_dir / "read_mem_peak_delta_bar.png"
    )

    print("\n[OK] Read benchmark plots generated.")


if __name__ == "__main__":
    main()

