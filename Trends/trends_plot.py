import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_trends(buckets: pd.DataFrame, cfg: object, output_path: str = "trend_analysis.png") -> None:
    """
    Generates a plot showing the metric over time and highlights detected trend segments.
    
    Args:
        buckets (pd.DataFrame): DataFrame containing aggregated buckets and trend detection results.
        cfg (object): Configuration object containing metric name and settings.
        output_path (str): Path to save the generated plot.
    """
    if buckets.empty:
        print("[TrendsPlot] No buckets to plot.")
        return

    # Ensure timestamp index for plotting
    if "bucket" in buckets.columns:
        plot_data = buckets.set_index("bucket").sort_index()
    else:
        plot_data = buckets.copy()

    # Show only recent window to keep trend lines readable on large datasets.
    if isinstance(plot_data.index, pd.DatetimeIndex) and not plot_data.empty:
        end_ts = plot_data.index.max()
        start_ts = end_ts - pd.Timedelta(hours=12)
        plot_data = plot_data.loc[start_ts:end_ts]

    plt.figure(figsize=(12, 6))

    # Plot raw metric aggregation
    plt.plot(plot_data.index, plot_data["metric_agg"], label=f"{cfg.metric} ({cfg.agg})", color="blue", alpha=0.6)

    # Plot smoothed metric if available
    if "metric_smooth" in plot_data.columns:
        plt.plot(plot_data.index, plot_data["metric_smooth"], label="Smoothed", color="orange", linestyle="--", alpha=0.8)

    # Highlight detected trends
    if "is_trend" in plot_data.columns:
        # Get segments where is_trend is True
        trend_mask = plot_data["is_trend"].fillna(False)
        
        # Plot markers for trend points
        trend_points = plot_data[trend_mask]
        if not trend_points.empty:
             plt.scatter(trend_points.index, trend_points["metric_agg"], color="red", s=20, label="Detected Trend", zorder=5)

    # Highlight ground truth (TrendWindow) if available
    if "TrendWindow_bucket" in plot_data.columns:
         ground_truth_mask = plot_data["TrendWindow_bucket"].fillna(False).astype(bool)
         gt_points = plot_data[ground_truth_mask]
         if not gt_points.empty:
              # Plot as a different marker or background shading
              # Using scatter with different marker for visibility
              plt.scatter(gt_points.index, gt_points["metric_agg"], color="green", marker="x", s=30, label="Ground Truth (TrendWindow)", alpha=0.5, zorder=4)


    plt.title(f"Trend Analysis: {cfg.source_system} - {cfg.metric}")
    plt.xlabel("Time")
    plt.ylabel(cfg.metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"[TrendsPlot] Plot saved to {output_path}")
    except Exception as e:
        print(f"[TrendsPlot] Failed to save plot: {e}")
    finally:
        plt.close()
