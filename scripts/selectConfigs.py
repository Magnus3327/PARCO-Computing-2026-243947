"""
    scripts/selectConfigs.py
    Select best, worst, and median configurations from JSON results.
    It's used after running the full parameter sweep to identify key configurations
    for further analysis, like perf profiling.
"""
import json
import sys

def median_item(items, key_func):
    sorted_items = sorted(items, key=key_func)
    n = len(sorted_items)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return sorted_items[mid]
    else:
        return sorted_items[mid-1]

def main(json_file, output_file):
    with open(json_file) as f:
        data = json.load(f)

    by_matrix = {}
    for entry in data["results"]:
        m = entry["matrix"]["name"]
        by_matrix.setdefault(m, []).append(entry)

    with open(output_file, "w") as out:
        for m, entries in by_matrix.items():
            best = max(entries, key=lambda e: e["statistics90"]["GFLOPS"])
            worst = min(entries, key=lambda e: e["statistics90"]["GFLOPS"])
            median = median_item(entries, key_func=lambda e: e["statistics90"]["GFLOPS"])
            out.write(f"{m} best {best['scenario']['threads']} {best['scenario']['scheduling_type']} {best['scenario']['chunk_size']}\n")
            out.write(f"{m} worst {worst['scenario']['threads']} {worst['scenario']['scheduling_type']} {worst['scenario']['chunk_size']}\n")
            if median:
                out.write(f"{m} median {median['scenario']['threads']} {median['scenario']['scheduling_type']} {median['scenario']['chunk_size']}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 select_configs.py <input_json> <output_selected_configs>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
