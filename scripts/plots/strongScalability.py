import json
import matplotlib.pyplot as plt
import os
import sys

# Strong scalability plot (log-log axes): duration (ms) vs number of threads.
# Real best performance data for each matrix shown in color.
# A single dashed gray line represents theoretical scalability (T_seq/p) in legend.

def load_json(filename):
    # Load and parse a JSON file
    with open(filename, 'r') as f:
        return json.load(f)

def extract_sequential_durations(data):
    # Extracts sequential duration (ms) for each matrix
    seq_durations = {}
    for result in data['results']:
        name = result['matrix']['name']
        duration = result['statistics90']['duration_ms']  # 90th percentile duration
        seq_durations[name] = duration
    return seq_durations

def extract_parallel_best_duration_by_threads(data):
    # Extracts the best (minimum) duration per thread count for each matrix
    parallel_best_duration = {}
    for result in data['results']:
        name = result['matrix']['name']
        scenario = result['scenario']
        threads = scenario['threads']
        duration = result['statistics90']['duration_ms']
        if name not in parallel_best_duration:
            parallel_best_duration[name] = {}
        # Save only the minimum duration for each thread count
        if threads not in parallel_best_duration[name]:
            parallel_best_duration[name][threads] = duration
        elif duration < parallel_best_duration[name][threads]:
            parallel_best_duration[name][threads] = duration
    return parallel_best_duration

def plot_strong_scalability_with_single_theory(sequential_file, parallel_file, output_folder):
    # Main plotting routine: draws one colored line per matrix for performance,
    # plus a single dashed gray line for theoretical scalability in the legend.
    seq_json = load_json(sequential_file)
    par_json = load_json(parallel_file)

    seq_durations = extract_sequential_durations(seq_json)
    parallel_best_durations = extract_parallel_best_duration_by_threads(par_json)

    plt.figure(figsize=(12, 8))
    # Use a colorblind-friendly palette and distinct marker shapes
    color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3']
    markers = ['o', 's', '^', 'D', 'v']

    theory_plotted = False  # Ensure theoretical line has only one legend entry

    for idx, (matrix, durations_by_threads) in enumerate(parallel_best_durations.items()):
        # Get sorted thread counts, always include 1 (sequential) first
        threads_list = sorted([t for t in durations_by_threads if t != 1])
        x_vals = [1] + threads_list
        y_vals = [seq_durations[matrix] if matrix in seq_durations else 0]
        # Add parallel durations
        for t in threads_list:
            y_vals.append(durations_by_threads[t])
        color = color_palette[idx % len(color_palette)]
        marker = markers[idx % len(markers)]
        # Plot measured performance
        plt.plot(x_vals, y_vals, marker=marker, color=color, label=matrix, linewidth=2)
        # Compute theoretical: T_seq/p for each thread count,
        # plot single dashed gray curve with legend label only once
        theoretical = [y_vals[0] / t for t in x_vals]
        if not theory_plotted:
            plt.plot(x_vals, theoretical, linestyle='--', color='gray', linewidth=2,
                     label="theoretical scalability (T_seq/p)")
            theory_plotted = True
        else:
            plt.plot(x_vals, theoretical, linestyle='--', color='gray', linewidth=2)

    plt.xlabel('Number of Threads (1 = Sequential)')
    plt.ylabel('Duration Time (ms)')
    plt.title('Strong Scalability of SpMV: \nDuration vs Threads per Matrix (Log-Log)')
    plt.grid(True, which='both', linestyle=':')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "spmv_strong_scalability.png"))

if __name__ == "__main__":
    # Usage: python plot_scalability.py sequential.json parallel.json output_folder
    if len(sys.argv) != 4:
        print("Usage: python plot_scalability.py <sequential.json> <parallel.json> <output_folder>")
        sys.exit(1)
    sequential_file = sys.argv[1]
    parallel_file = sys.argv[2]
    output_folder = sys.argv[3]
    plot_strong_scalability_with_single_theory(sequential_file, parallel_file, output_folder)