import pandas as pd
import matplotlib.pyplot as plt

def load_csv(csv_path):
    return pd.read_csv(csv_path)


def boxplot_metric(df, metric, title=None, ylabel=None, save_path=None):
    plt.figure(figsize=(9, 5))

    df.boxplot(
        column=metric,
        by="format",
        grid=False,
        showfliers=True
    )

    plt.suptitle("")
    plt.title(title or f"Boxplot of {metric}")
    plt.xlabel("Data Format")
    plt.ylabel(ylabel or metric)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")
    else:
        plt.show()

    plt.close()

def barplot_mean_std(df, metric, title=None, ylabel=None, save_path=None):
    stats = df.groupby("format")[metric].agg(["mean", "std"])

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        stats.index,
        stats["mean"],
        yerr=stats["std"],
        capsize=6
    )

    plt.title(title or f"{metric} (Mean ± Std)")
    plt.xlabel("Data Format")
    plt.ylabel(ylabel or metric)

    # 🔹 Add value labels above bars
    for bar, mean, std in zip(bars, stats["mean"], stats["std"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{mean:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")
    else:
        plt.show()

    plt.close()

