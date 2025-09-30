# Plot viscosity vs step from md_visc_results.csv and save figure.
#!/usr/bin/env python3
"""
plot_md_visc.py

Read md_visc_results.csv and plot viscosity vs step.

Usage:
    python plot_md_visc.py <csvfile> [--out <pngfile>] [--tail N] [--window W]

Example:
    python plot_md_visc.py md_visc_results.csv --out viscosity.png --tail 50 --window 11

The script:
 - automatically finds the viscosity column (last column) if names are not standard,
 - computes a centered moving average (window W),
 - computes tail statistics for last N points,
 - plots raw series and moving average, draws tail mean line and annotates it,
 - saves PNG (default: viscosity.png) and prints summary to stdout.
Dependencies: pandas, matplotlib, numpy
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    # try to read with header; if fails, read without header
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)
    return df

def find_viscosity_series(df):
    # Prefer named column 'viscosity' (case-insensitive), else last numeric column
    cols = list(df.columns)
    lower = [c.lower() if isinstance(c, str) else str(c) for c in cols]
    if 'viscosity' in lower:
        idx = lower.index('viscosity')
        visc = pd.to_numeric(df.iloc[:, idx], errors='coerce')
    else:
        # try to find last column with numeric values
        for c in reversed(cols):
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                visc = s
                break
        else:
            raise RuntimeError("No numeric column found for viscosity.")
    # Try to identify step column (named 'step' or first column)
    step = None
    if 'step' in lower:
        step = pd.to_numeric(df.iloc[:, lower.index('step')], errors='coerce')
    else:
        # if first column numeric and not equal to viscosity column, use it
        first = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        if (first.notna().any()) and not first.equals(visc):
            step = first
        else:
            step = pd.Series(np.arange(len(visc)))
    return step.reset_index(drop=True), visc.reset_index(drop=True)

def moving_average(s, window):
    return s.rolling(window=window, center=True, min_periods=1).mean()

def plot_series(steps, visc, movavg, tail_mean, tail_n, out_png, title="Viscosity vs Step"):
    plt.figure(figsize=(10,5))
    plt.plot(steps, visc, linewidth=0.8, label='raw')
    plt.plot(steps, movavg, linewidth=1.2, label=f'movavg (w={len(movavg.dropna())})' if False else 'moving avg')
    plt.xlabel('Step')
    plt.ylabel('Viscosity [Pa·s]')
    plt.title(title)
    plt.grid(True, linestyle=':', linewidth=0.6)
    plt.axhline(tail_mean, linestyle='--', linewidth=0.9, color='orange')
    # annotate near right end
    x_text = steps.iloc[-1] if len(steps) else 0
    plt.text(x_text, tail_mean, f'  tail mean ({tail_n}) = {tail_mean:.6e} Pa·s',
             va='bottom', ha='right', color='orange', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Plot viscosity vs step from md_visc CSV")
    p.add_argument('csvfile', help='md_visc_results.csv')
    p.add_argument('--out', default='viscosity.png', help='Output PNG filename')
    p.add_argument('--tail', type=int, default=50, help='Number of last points for tail statistics')
    p.add_argument('--window', type=int, default=11, help='Window size for moving average (odd recommended)')
    p.add_argument('--save-processed', default=None, help='Optional CSV to save processed series (step,viscosity,movavg)')
    args = p.parse_args()

    if not os.path.exists(args.csvfile):
        print("CSV file not found:", args.csvfile, file=sys.stderr)
        sys.exit(2)

    df = load_csv(args.csvfile)
    try:
        steps, visc = find_viscosity_series(df)
    except Exception as e:
        print("Failed to locate viscosity series in CSV:", e, file=sys.stderr)
        sys.exit(3)

    # Drop NaNs at front/back to make stats robust
    valid_mask = visc.notna()
    steps = steps[valid_mask].reset_index(drop=True)
    visc = visc[valid_mask].reset_index(drop=True)

    if len(visc) == 0:
        print("No valid viscosity data found.", file=sys.stderr)
        sys.exit(4)

    movavg = moving_average(visc, args.window)

    tail_n = min(args.tail, len(visc))
    tail_vals = visc.iloc[-tail_n:]
    tail_mean = float(tail_vals.mean())
    tail_std = float(tail_vals.std(ddof=1)) if len(tail_vals) > 1 else 0.0

    # plot
    plot_series(steps, visc, movavg, tail_mean, tail_n, args.out)

    # optional save processed
    if args.save_processed:
        outdf = pd.DataFrame({'step': steps, 'viscosity': visc, 'movavg': movavg})
        outdf.to_csv(args.save_processed, index=False)

    # print summary
    print("Output PNG saved to:", args.out)
    print("Data points:", len(visc))
    print("Last:", float(visc.iloc[-1]))
    print(f"Tail mean ({tail_n}): {tail_mean:.6e} Pa.s  tail std: {tail_std:.6e}")

if __name__ == "__main__":
    main()

