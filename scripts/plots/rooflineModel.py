import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python roofline_inverse.py <json_path> <output_dir>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Hardware Ceiling Calculation (Flexible from JSON nodes)
    node_specs = []
    for node in data.get('nodes', []):
        total_cores = node['total_cores']
        freq_ghz = node['cpu_mhz'] / 1000.0 
        
        # Gold 6xxx series: 32 DP FLOPs/cycle (AVX-512)
        # Memory: 6 channels/socket * 8 bytes * 2.666 GHz
        flops_per_cycle = 32 
        peak_bw_per_socket = 6 * 2.666 * 8 
        
        node_peak_gflops = freq_ghz * total_cores * flops_per_cycle
        node_peak_bw = node['sockets'] * peak_bw_per_socket
        
        node_specs.append({'gflops': node_peak_gflops, 'bw': node_peak_bw})

    # Use max/min for the variance range
    peak_bw = max(n['bw'] for n in node_specs) 
    peak_flops_min = min(n['gflops'] for n in node_specs)
    peak_flops_max = max(n['gflops'] for n in node_specs)

    # 2. Extract points using the Inverse Formula
    plot_points = []
    for r in data.get('results', []):
        if r.get('errors') or 'statistics' not in r: continue
        
        stats = r['statistics']
        gflops = stats['gflops']
        bandwidth = stats['bandwidth_gbps']
        
        if bandwidth > 0:
            # Operational Intensity (I) = GFLOPS / BW
            intensity = gflops / bandwidth
            
            plot_points.append({
                'matrix': r['matrix']['name'].split('/')[-1],
                'intensity': intensity,
                'gflops': gflops
            })

    # 3. Plotting
    plt.figure(figsize=(11, 7))
    x_roof = np.logspace(-2.5, 1, 500)
    
    y_slow = np.minimum(peak_bw * x_roof, peak_flops_min)
    y_fast = np.minimum(peak_bw * x_roof, peak_flops_max)
    
    plt.fill_between(x_roof, y_slow, y_fast, color='grey', alpha=0.15, label='Node Hardware Variance')
    plt.loglog(x_roof, y_slow, 'k--', alpha=0.3, linewidth=1)
    plt.loglog(x_roof, y_fast, 'k-', alpha=0.6, linewidth=1.5, label='Theoretical Ceiling')

    matrices = sorted(list(set(pt['matrix'] for pt in plot_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))
    for i, m_name in enumerate(matrices):
        pts = [p for p in plot_points if p['matrix'] == m_name]
        plt.scatter([p['intensity'] for p in pts], [p['gflops'] for p in pts], 
                    color=colors[i], label=m_name, s=70, edgecolors='white', zorder=10)

    plt.xlabel('Operational Intensity (FLOP/Byte)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('Roofline Model (Inverse Formula Analysis)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'roofline_inverse_analysis.pdf')
    plt.savefig(output_file)
    plt.close()
    print(f"Inverse-based Roofline saved to: {output_file}")

if __name__ == "__main__":
    main()