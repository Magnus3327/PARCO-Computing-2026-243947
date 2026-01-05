import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_weak_metrics.py <json_path> <output_directory>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for r in data.get('results', []):
        if r['matrix']['name'] == "Generated" and not r.get('errors'):
            rows.append({
                'p': r['mpi']['processes'],
                't_kernel': r['statistics']['kernel_p90_ms'],
                't_total': r['statistics']['kernel_p90_ms'] + r['timings_ms']['communication']
            })

    if not rows:
        print("No valid 'Generated' matrix results found.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values('p')
    p_vals = df['p'].values

    t1_total = df.iloc[0]['t_total']
    t1_kernel = df.iloc[0]['t_kernel']

    eff_total = t1_total / df['t_total'].values
    eff_kernel = t1_kernel / df['t_kernel'].values

    s_total = p_vals * eff_total
    s_kernel = p_vals * eff_kernel

    # FIGURE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- SCALED SPEEDUP ---
    ax1.set_title("Weak Scaling: Scaled Speedup (Gustafson)", fontsize=14)
    ax1.set_xlabel("Number of Processes (p)")
    ax1.set_ylabel(r"Scaled Speedup ($p \cdot T_1 / T_p$)")

    l1, = ax1.plot(
        p_vals, s_total,
        marker='o', linestyle='-', linewidth=2,
        label='Total (Communication + Kernel)'
    )
    l2, = ax1.plot(
        p_vals, s_kernel,
        marker='s', linestyle='--', linewidth=2,
        alpha=0.7, label='Kernel'
    )
    l3, = ax1.plot(
        p_vals, p_vals,
        linestyle=':', linewidth=2,
        color='grey', label='Ideal for plot'
    )

    ax2.plot(
        p_vals, eff_total,
        marker='o', linestyle='-', linewidth=2,
        label='Total Efficiency'
    )
    ax2.plot(
        p_vals, eff_kernel,
        marker='s', linestyle='--', linewidth=2,
        alpha=0.7, label='Kernel Efficiency'
    )
    ax2.axhline(
        y=1.0, linestyle=':', linewidth=2,
        color='grey', label='Ideal Efficiency (E = 1)'
    )


    # --- WEAK EFFICIENCY ---
    ax2.set_title("Weak Scaling: Efficiency", fontsize=14)
    ax2.set_xlabel("Number of Processes (p)")
    ax2.set_ylabel(r"Efficiency ($T_1 / T_p$)")

    for ax in (ax1, ax2):
        ax.set_xscale('log', base=2)
        ax.set_xticks(p_vals)
        ax.set_xticklabels(p_vals)
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)

    # SHARED LEGEND (TOP)
    handles = [l1, l2, l3]
    labels = [h.get_label() for h in handles]

    fig.legend(handles, labels,
               loc='upper center',
               ncol=3,
               fontsize='medium')

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    output_path = os.path.join(output_dir, 'weak_scaling_metrics.pdf')
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Weak scaling plots saved to: {output_path}")

if __name__ == "__main__":
    main()