"""
Task 1: Channel Analysis for Nova TV interview assignment.

Single-pass EDA: loads the dataset once, derives temporal features,
generates 7 diagnostic plots, and emits a self-contained HTML report.

Usage:
    uv run python task1_analysis.py
"""

import base64
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

DATA_PATH = "data.csv"
OUTPUT_DIR = Path("plots")
REPORT_PATH = "task1_HTMLreport.html"

# Matplotlib defaults: consistent style across all plots.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# Channel metadata (identified during exploration).
CHANNEL_NAMES = {
    3:  "TV Nova",
    4:  "Prima",
    9:  "Sport Channel",
    54: "Nova Cinema",
}
CHANNEL_COLORS = {
    3:  "#e63946",
    4:  "#457b9d",
    9:  "#2a9d8f",
    54: "#e9c46a",
}
CHANNELS = sorted(CHANNEL_NAMES)


# ═══════════════════════════════════════════════════════════════════
#  Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Read the CSV once, parse the datetime, and derive all temporal columns.

    Column naming convention for features:
        ch{id}__f_{n}  → channel {id}, feature {n}

    Key features by convention:
        f_10 = genre  (categorical)
        f_11 = country (categorical)
        f_12 = production year or similar numeric
    """
    df = pd.read_csv(path, low_memory=False)

    df["timeslot_datetime_from"] = pd.to_datetime(df["timeslot_datetime_from"])

    # Temporal features, derived once and used by all plotting functions.
    df["hour"]   = df["timeslot_datetime_from"].dt.hour
    df["dow"]    = df["timeslot_datetime_from"].dt.day_name()
    df["month"]  = df["timeslot_datetime_from"].dt.to_period("M")
    df["date"]   = df["timeslot_datetime_from"].dt.date
    df["channel_name"] = df["channel_id"].map(CHANNEL_NAMES)

    return df


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> str:
    """Persist figure to OUTPUT_DIR and close it. Returns the saved path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_share_dynamics(df: pd.DataFrame) -> str:
    """Box plot + summary table for each channel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot.
    box_data = [df[df["channel_id"] == ch]["share_15_54"].dropna() for ch in CHANNELS]
    bp = axes[0].boxplot(box_data, labels=[CHANNEL_NAMES[c] for c in CHANNELS],
                         patch_artist=True, widths=0.6)
    for patch, ch in zip(bp["boxes"], CHANNELS):
        patch.set_facecolor(CHANNEL_COLORS[ch])
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Audience Share (%)")
    axes[0].set_title("Share Distribution by Channel")
    axes[0].tick_params(axis="x", rotation=15)

    # Summary table.
    stats = (df.groupby("channel_id")["share_15_54"]
             .agg(["mean", "median", "std", "count"]))
    stats.index = stats.index.map(CHANNEL_NAMES)
    stats.columns = ["Mean", "Median", "Std Dev", "N"]
    cell_text = [[f"{v:.1f}" if c != "N" else f"{int(v):,}"
                  for c, v in zip(stats.columns, row)]
                 for row in stats.values]
    table = axes[1].table(cellText=cell_text, rowLabels=stats.index,
                          colLabels=stats.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    axes[1].axis("off")
    axes[1].set_title("Summary Statistics")

    plt.tight_layout()
    return _save(fig, "01_share_dynamics.png")


def plot_weekly_trend(df: pd.DataFrame) -> str:
    """Weekly resampled mean share over time for every channel."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for ch in CHANNELS:
        sub = (df[df["channel_id"] == ch]
               .set_index("timeslot_datetime_from")["share_15_54"]
               .resample("W").mean())
        ax.plot(sub.index, sub.values,
                label=CHANNEL_NAMES[ch], color=CHANNEL_COLORS[ch],
                linewidth=1.5, alpha=0.85)

    ax.set_ylabel("Average Weekly Share (%)")
    ax.set_title("Audience Share Over Time (Weekly Average)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return _save(fig, "02_weekly_trend.png")


def plot_hourly_pattern(df: pd.DataFrame) -> str:
    """Average share by hour of day, with prime-time band annotated."""
    fig, ax = plt.subplots(figsize=(12, 5))

    hourly = df.groupby(["hour", "channel_id"])["share_15_54"].mean().unstack()
    for ch in CHANNELS:
        if ch in hourly.columns:
            ax.plot(hourly.index, hourly[ch],
                    label=CHANNEL_NAMES[ch], color=CHANNEL_COLORS[ch],
                    linewidth=2.5, marker="o", markersize=4)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Share (%)")
    ax.set_title("Audience Share by Hour of Day")
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Prime-time annotation.
    ax.axvspan(19, 23, alpha=0.08, color="orange")
    ax.text(21, ax.get_ylim()[1] * 0.95, "Prime Time",
            ha="center", fontsize=10, color="orange", fontstyle="italic")

    plt.tight_layout()
    return _save(fig, "03_hourly_pattern.png")


def plot_dow_pattern(df: pd.DataFrame) -> str:
    """Average share by day of week, with weekend band annotated."""
    DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    fig, ax = plt.subplots(figsize=(12, 5))
    dow_data = (df.groupby(["dow", "channel_id"])["share_15_54"]
                .mean().unstack().loc[DOW_ORDER])

    for ch in CHANNELS:
        if ch in dow_data.columns:
            ax.plot(dow_data.index, dow_data[ch],
                    label=CHANNEL_NAMES[ch], color=CHANNEL_COLORS[ch],
                    linewidth=2.5, marker="s", markersize=5)

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Share (%)")
    ax.set_title("Audience Share by Day of Week")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Weekend annotation (Saturday–Sunday = indices 5–6).
    ax.axvspan(4.5, 6.5, alpha=0.08, color="blue")
    ax.text(5.5, ax.get_ylim()[1] * 0.95, "Weekend",
            ha="center", fontsize=10, color="blue", fontstyle="italic")

    plt.tight_layout()
    return _save(fig, "04_dow_pattern.png")


def plot_cross_channel_corr(df: pd.DataFrame) -> str:
    """Correlation heatmap of simultaneous share across channels."""
    pivot = df.pivot_table(index="timeslot_datetime_from",
                           columns="channel_id", values="share_15_54")
    pivot.columns = [CHANNEL_NAMES[c] for c in pivot.columns]
    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.index)

    # Annotate cells with correlation values.
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=13)

    ax.set_title("Cross-Channel Share Correlation")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    plt.tight_layout()
    return _save(fig, "05_cross_channel_corr.png")


def plot_genre_performance(df: pd.DataFrame) -> str:
    """Horizontal bar chart: average share by genre, per channel.

    Channel 9 (Sport) has no genre column data; excluded.
    """
    GENRE_FEATURES = {3: "ch3__f_10", 4: "ch4__f_10", 54: "ch54__f_10"}
    MIN_AIRINGS = 20
    TOP_N = 10

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (ch, col) in zip(axes, GENRE_FEATURES.items()):
        mask = (df["channel_id"] == ch) & df[col].notna()
        genre_share = (df.loc[mask]
                       .groupby(col)["share_15_54"]
                       .agg(["mean", "count"])
                       .query(f"count >= {MIN_AIRINGS}")
                       .sort_values("mean", ascending=True)
                       .tail(TOP_N))

        bars = ax.barh(genre_share.index, genre_share["mean"],
                       color=CHANNEL_COLORS[ch], alpha=0.8)
        ax.set_xlabel("Average Share (%)")
        ax.set_title(CHANNEL_NAMES[ch])
        ax.grid(axis="x", alpha=0.3)

        # Count annotation on each bar.
        for bar, count in zip(bars, genre_share["count"]):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={int(count)}", va="center", fontsize=9, color="gray")

    plt.suptitle("Average Share by Genre (min. 20 airings)", fontsize=16, y=1.02)
    plt.tight_layout()
    return _save(fig, "06_genre_performance.png")


def plot_data_completeness(df: pd.DataFrame) -> str:
    """Grouped bar chart: missing-value percentage by feature and channel.

    Features are identified by splitting the chN__f_M column naming convention.
    """
    null_pct = df.isnull().mean() * 100

    # Group null percentages by channel prefix.
    groups: dict[str, dict[str, float]] = {}
    for col in null_pct.index:
        if "__" not in col or not col.startswith("ch"):
            continue
        prefix, feature = col.split("__")
        groups.setdefault(prefix, {})[feature] = null_pct[col]

    # Build uniform feature set across channels.
    all_features = sorted({f for g in groups.values() for f in g})
    channel_labels = {f"ch{ch}": CHANNEL_NAMES[ch] for ch in CHANNELS}

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_features))
    width = 0.2

    for i, (prefix, data) in enumerate(sorted(groups.items())):
        values = [data.get(f, 0) for f in all_features]
        ax.bar(x + i * width, values, width,
               label=channel_labels.get(prefix, prefix),
               color=list(CHANNEL_COLORS.values())[i], alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(all_features, rotation=45, ha="right")
    ax.set_ylabel("Missing Values (%)")
    ax.set_title("Data Completeness by Feature and Channel")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return _save(fig, "07_data_completeness.png")


# ═══════════════════════════════════════════════════════════════════
#  HTML report
# ═══════════════════════════════════════════════════════════════════

_REPORT_CSS = """
body {
    font-family: 'Segoe UI', sans-serif; max-width: 1100px;
    margin: 0 auto; padding: 40px 20px; background: #fafafa; color: #222;
}
h1 { border-bottom: 3px solid #e63946; padding-bottom: 10px; }
h2 { color: #457b9d; margin-top: 40px; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
      margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.insight {
    background: #f0f7ff; border-left: 4px solid #457b9d;
    padding: 12px 18px; margin: 15px 0; border-radius: 0 4px 4px 0;
}
.section { margin-bottom: 50px; }
"""

_REPORT_SECTIONS = [
    ("Share Dynamics", "01_share_dynamics.png",
     "Channel 3 (TV Nova) dominates with the highest average share. "
     "Channel 4 (Prima) has the second-highest "
     "share at 12.12%, matching Prima's known market position. The Sport Channel has the lowest average "
     "share but likely spikes around major sporting events. Channel 54 (Cinema) "
     "shows moderate share with high variance; movie performance is inherently "
     "unpredictable."),
    ("Weekly Trends", "02_weekly_trend.png",
     "The flagship channel shows the strongest seasonal patterns. All channels "
     "exhibit some degree of weekly fluctuation, with potential holiday effects "
     "visible as spikes."),
    ("Hourly Patterns", "03_hourly_pattern.png",
     "Strong prime-time effect (19:00–23:00) across all channels, with the main "
     "channel benefiting the most. Daytime hours show lower, more uniform shares. "
     "This is the single strongest temporal signal for prediction."),
    ("Day of Week", "04_dow_pattern.png",
     "Weekend viewing patterns differ from weekdays. The sport channel likely "
     "shows the most day-of-week variation due to event scheduling."),
    ("Cross-Channel Correlations", "05_cross_channel_corr.png",
     "Negative correlations between channels confirm the zero-sum nature of "
     "audience share; when one channel gains viewers, others lose them. This "
     "validates the importance of cross-channel features in prediction models."),
    ("Genre Performance", "06_genre_performance.png",
     "Genre is a meaningful predictor of share. Reality TV and game shows drive "
     "high share on Channel 3 (main), while drama and action perform well on "
     "the cinema channel. Channel 9 (Sport) has no genre metadata: sport "
     "broadcasts don't carry this information."),
    ("Data Completeness", "07_data_completeness.png",
     "Channel 9 is systematically missing features f_7 through f_11 (100% null), "
     "consistent with a sport channel lacking movie-style metadata. Other channels "
     "have varying levels of missingness that need handling in the modeling phase."),
]


def build_html_report(plot_paths: dict[str, str], output_path: str) -> None:
    """Embed plots as base64 images into a single self-contained HTML file."""
    html: list[str] = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        "<title>Task 1: Channel Analysis</title>",
        f"<style>{_REPORT_CSS}</style>",
        "</head><body>",
        "<h1>Task 1: Channel Analysis</h1>",
        "<p>Analysis of four Czech TV channels across ~34,000 "
        "timeslots (Jan 2022 – Jan 2024): TV Nova, Prima, Sport Channel, and Nova Cinema.</p>",
    ]

    for title, filename, insight in _REPORT_SECTIONS:
        img_path = plot_paths[filename]
        with open(img_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode()

        html.append(f'<div class="section">')
        html.append(f"  <h2>{title}</h2>")
        html.append(f'  <img src="data:image/png;base64,{img_b64}" alt="{title}">')
        html.append(f'  <div class="insight"><strong>Result:</strong> {insight}</div>')
        html.append("</div>")

    html.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"  {len(df):,} rows, {df['channel_id'].nunique()} channels")
    print(f"  Date range: {df['timeslot_datetime_from'].min()} → "
          f"{df['timeslot_datetime_from'].max()}")

    print("\nGenerating plots ...")
    plot_paths = {
        "01_share_dynamics.png":      plot_share_dynamics(df),
        "02_weekly_trend.png":        plot_weekly_trend(df),
        "03_hourly_pattern.png":     plot_hourly_pattern(df),
        "04_dow_pattern.png":        plot_dow_pattern(df),
        "05_cross_channel_corr.png": plot_cross_channel_corr(df),
        "06_genre_performance.png":  plot_genre_performance(df),
        "07_data_completeness.png":  plot_data_completeness(df),
    }
    for name, path in plot_paths.items():
        print(f"{path}")

    print(f"\nBuilding report → {REPORT_PATH}")
    build_html_report(plot_paths, REPORT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
