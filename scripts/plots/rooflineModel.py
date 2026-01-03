import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main():
    # 1. Check arguments (now 5: script, json, output_dir, peak_bw, peak_gflops)
    if len(sys.argv) < 5:
        print(f"Usage: python {sys.argv[0]} <json_path> <output_dir> <peak_bw> <peak_gflops>")
        print(f"Example: python {sys.argv[0]} data.json ./plots 512.0 2650.0")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]
    # Carichiamo i valori numerici dai parametri
    try:
        peak_bw = float(sys.argv[3])
        peak_gflops = float(sys.argv[4])
    except ValueError:
        print("Error: peak_bw and peak_gflops must be numbers (float).")
        sys.exit(1)
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    # 2. Load data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)

    # 3. Extract points using the Inverse Formula (I = GFLOPS / BW)
    plot_points = []
    for r in data.get('results', []):
        if r.get('errors') or 'statistics' not in r: 
            continue
        
        stats = r['statistics']
        gflops = stats.get('gflops', 0)
        bandwidth = stats.get('bandwidth_gbps', 0)
        
        if bandwidth > 0:
            intensity = gflops / bandwidth
            plot_points.append({
                'matrix': r['matrix']['name'].split('/')[-1],
                'intensity': intensity,
                'gflops': gflops
            })

    if not plot_points:
        print("No valid data points found to plot.")
        sys.exit(1)

    # 4. Plotting
    plt.figure(figsize=(11, 7))
    
    # Define range for the x-axis (Operational Intensity)
    x_roof = np.logspace(-3, 1, 500)
    # Roofline function: min(BW * I, Peak GFLOPS)
    y_roof = np.minimum(peak_bw * x_roof, peak_gflops)
    
    # Draw the theoretical ceiling
    plt.loglog(x_roof, y_roof, 'k-', alpha=0.7, linewidth=2, label=f'Theoretical Ceiling ({peak_gflops} GFLOPS, {peak_bw} GB/s)')

    # Scatter points for each matrix
    matrices = sorted(list(set(pt['matrix'] for pt in plot_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))
    
    for i, m_name in enumerate(matrices):
        pts = [p for p in plot_points if p['matrix'] == m_name]
        plt.scatter([p['intensity'] for p in pts], [p['gflops'] for p in pts], 
                    color=colors[i], label=m_name, s=70, edgecolors='white', zorder=10)

    # Labels and Aesthetics
    plt.xlabel('Operational Intensity (FLOP/Byte)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('Roofline Model Analysis (Manual Peaks)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Annotation for the two regimes
    plt.text(x_roof[10], y_roof[10] * 1.5, 'Memory Bound', color='grey', rotation=35)
    plt.text(x_roof[-50], peak_gflops * 0.8, 'Compute Bound', color='grey')

    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'roofline_manual_params.pdf')
    plt.savefig(output_file)
    plt.close()
    print(f"Roofline plot saved to: {output_file}")

if __name__ == "__main__":
    main()