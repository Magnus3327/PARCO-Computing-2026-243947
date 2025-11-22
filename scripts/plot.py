import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def load_json_data(folder):
    """
    Load all .json files from the folder into a single DataFrame.
    Expects files like parallel.json and sequential.json with 'results' key.
    """
    all_data = []
    for json_file in glob.glob(os.path.join(folder, "*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            hw = data.get("hardware", {})
            for entry in data.get("results", []):
                matrix = entry.get("matrix", {})
                stats = entry.get("statistics90", {})
                scenario = entry.get("scenario", {})
                # Build a flat dictionary with both hw, scenario, matrix, and stats info
                record = {
                    "matrix_name": matrix.get("name"),
                    "rows": matrix.get("rows"),
                    "cols": matrix.get("cols"),
                    "nnz": matrix.get("nnz"),
                    "threads": scenario.get("threads", None),
                    "scheduling": scenario.get("scheduling_type", None),
                    "chunk_size": scenario.get("chunk_size", None),
                    "duration_ms": stats.get("duration_ms"),
                    "FLOPs": stats.get("FLOPs"),
                    "GFLOPS": stats.get("GFLOPS"),
                    "bandwidth_GBps": stats.get("Bandwidth_GBps"),
                    "arithmetic_intensity": stats.get("arithmetic_intensity"),
                    "warmup_time_ms": entry.get("warmUp_time_ms"),
                    "source_file": os.path.basename(json_file)
                }
                all_data.append(record)
    df = pd.DataFrame(all_data)
    return df

def plot_execution_time(df, output_dir):
    """
    Plot execution time (duration_ms) vs different OpenMP thread counts and scheduling policies.
    Purpose: Show scaling behavior and effect of scheduling on performance.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="threads", y="duration_ms", hue="scheduling", style="chunk_size",
                 markers=True, dashes=False)
    plt.title("Execution time (90th percentile) vs Threads and Scheduling")
    plt.xlabel("Number of Threads")
    plt.ylabel("Duration (ms)")
    plt.legend(title="Scheduling / Chunk Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_time_vs_threads.png"))
    plt.close()

def plot_gflops(df, output_dir):
    """
    Plot GFLOPS performance vs number of threads and scheduling types.
    Purpose: Measure computational throughput and parallel efficiency.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="threads", y="GFLOPS", hue="scheduling", style="chunk_size",
                 markers=True, dashes=False)
    plt.title("GFLOPS vs Threads and Scheduling")
    plt.xlabel("Number of Threads")
    plt.ylabel("GFLOPS")
    plt.legend(title="Scheduling / Chunk Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gflops_vs_threads.png"))
    plt.close()

def plot_cache_misses(df_perf, output_dir):
    """
    Plot cache misses metrics (L1, LLC) if available in dataframe.
    Since perf outputs are not JSON, user must preprocess to a compatible df.
    Purpose: Analyze bottlenecks due to cache inefficiency.
    This function assumes df_perf contains columns 'config', 'L1_miss_percent' and 'LLC_miss_percent'.
    """
    # This function is placeholder, requires preprocessing of perf txt outputs into dataframe.
    pass

def plot_bandwidth(df, output_dir):
    """
    Plot memory bandwidth usage (GB/s) vs threads and scheduling.
    Purpose: Interpret data movement bottlenecks.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="threads", y="bandwidth_GBps", hue="scheduling", style="chunk_size",
                 markers=True, dashes=False)
    plt.title("Memory Bandwidth (GB/s) vs Threads and Scheduling")
    plt.xlabel("Number of Threads")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend(title="Scheduling / Chunk Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bandwidth_vs_threads.png"))
    plt.close()

def plot_speedup(df_parallel, df_sequential, output_dir):
    """
    Compute and plot speedup = sequential_time / parallel_time per matrix and thread count.
    Purpose: Show scalability and parallel efficiency.
    """
    speedup_records = []
    for matrix in df_sequential["matrix_name"].unique():
        seq_times = df_sequential[df_sequential["matrix_name"] == matrix][["duration_ms"]].mean().values[0]
        for threads in df_parallel["threads"].unique():
            par_times_df = df_parallel[(df_parallel["matrix_name"] == matrix) & (df_parallel["threads"] == threads)]
            if par_times_df.empty:
                continue
            par_time = par_times_df["duration_ms"].mean()
            speedup = seq_times / par_time
            speedup_records.append({"matrix_name": matrix, "threads": threads, "speedup": speedup})
    df_speedup = pd.DataFrame(speedup_records)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_speedup, x="threads", y="speedup", hue="matrix_name", marker="o")
    plt.title("Speedup (sequential_time / parallel_time) vs Number of Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(title="Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_threads.png"))
    plt.close()

def plot_iteration_time_variability(df, output_dir):
    """
    Boxplot of iteration times variability for all iterations available.
    Purpose: Visualize consistency and variability across iterations.
    This requires the all_iteration_times_ms stored as list in original JSON.
    This function assumes those lists have been expanded per iteration as rows.
    """
    # Placeholder: Needs JSON preprocessing to expand iterations per record.
    pass

def main():
    parser = argparse.ArgumentParser(description="Plot SpMV performance metrics from JSON results folder.")
    parser.add_argument("input_folder", help="Folder containing parallel.json and sequential.json")
    parser.add_argument("--output_folder", default="plots", help="Folder to save plots")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    df = load_json_data(args.input_folder)

    # Split data into parallel and sequential for speedup
    df_parallel = df[df["source_file"].str.contains("parallel", case=False, na=False)]
    df_sequential = df[df["source_file"].str.contains("sequential", case=False, na=False)]

    # Generate all meaningful plots
    plot_execution_time(df_parallel, args.output_folder)
    plot_gflops(df_parallel, args.output_folder)
    plot_bandwidth(df_parallel, args.output_folder)
    plot_speedup(df_parallel, df_sequential, args.output_folder)

    print(f"Plots generated in folder '{args.output_folder}'")

if __name__ == "__main__":
    main()
