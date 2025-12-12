from plot_utils import load_csv, boxplot_metric, barplot_mean_std
from pathlib import Path

def main():
    csv_path = "benchmark_results/benchmark_write_sensor_data_more_features_10k_valid_all_runs.csv"
    out_dir = Path("plots_output/write")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)

    # ---------- WRITE TIME ----------
    boxplot_metric(
        df,
        metric="write_time_ms",
        title="Write Time Distribution (ms)",
        ylabel="Time (ms)",
        save_path=out_dir / "write_time_boxplot.png"
    )

    barplot_mean_std(
        df,
        metric="write_time_ms",
        title="Write Time Mean ± Std (ms)",
        ylabel="Time (ms)",
        save_path=out_dir / "write_time_bar.png"
    )

    # ---------- FILE SIZE ----------
    barplot_mean_std(
        df,
        metric="file_size_mb",
        title="Output File Size (MB)",
        ylabel="Size (MB)",
        save_path=out_dir / "write_file_size_bar.png"
    )

    # ---------- CPU ----------
    boxplot_metric(
        df,
        metric="cpu_avg_proc_percent",
        title="CPU Usage During Write (%)",
        ylabel="CPU %",
        save_path=out_dir / "write_cpu_boxplot.png"
    )

    # ---------- MEMORY ----------
    barplot_mean_std(
        df,
        metric="mem_peak_delta_mb",
        title="Peak Memory Delta During Write (MB)",
        ylabel="Peak Memory Delta (MB)",
        save_path=out_dir / "write_memory_bar.png"
    )


if __name__ == "__main__":
    main()

