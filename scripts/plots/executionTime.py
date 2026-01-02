import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def main():
    # 1. Check number of arguments
    # Expecting: script_name, json_file, matrix_name, output_directory
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} <json_file> <matrix_name> <output_dir>")
        print(f"Example: python {sys.argv[0]} distributedSPMV.json Flan_1565.mtx ./plots")
        sys.exit(1)

    json_path = sys.argv[1]
    target_matrix = sys.argv[2]
    output_dir = sys.argv[3]

    # 2. Verify file existence and create output directory
    if not os.path.isfile(json_path):
        print(f"Error: File '{json_path}' not found.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Load data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)

    # 4. Extract data
    plot_data = []
    available_matrices = []

    results = data.get('results', [])
    for r in results:
        m_full_name = r.get('matrix', {}).get('name', '')
        available_matrices.append(m_full_name)
        
        # Check if the target matrix name is a substring of the record's matrix name
        if target_matrix in m_full_name:
            try:
                plot_data.append({
                    'Processes': r['mpi']['processes'],
                    'Setup': r['timings_ms']['setup'],
                    'Communication': r['timings_ms']['communication'],
                    'Kernel': r['statistics']['kernel_p90_ms']
                })
            except KeyError as e:
                print(f"Warning: Malformed record for {m_full_name}, missing key: {e}")

    # 5. Handle case where matrix is not found
    if not plot_data:
        print(f"Error: Matrix '{target_matrix}' not found in the file.")
        print("Available matrices:")
        for m in sorted(list(set(available_matrices))):
            print(f" - {m}")
        sys.exit(1)

    # 6. Process DataFrame
    df = pd.DataFrame(plot_data).sort_values('Processes')

    # 7. Plotting (Stacked Bar Chart)
    plt.figure(figsize=(10, 6))
    p_labels = [str(p) for p in df['Processes']]

    plt.bar(p_labels, df['Setup'], label='Setup', color='#2ecc71')
    plt.bar(p_labels, df['Communication'], bottom=df['Setup'], label='Communication', color='#e74c3c')
    plt.bar(p_labels, df['Kernel'], bottom=df['Setup'] + df['Communication'], label='Kernel (p90)', color='#3498db')

    plt.xlabel('Number of Processes (P)')
    plt.ylabel('Execution Time (ms)')
    plt.title(f'Execution Breakdown: {target_matrix}')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()

    # 8. Save the plot without opening it
    # Replace characters that might be problematic for filenames
    clean_name = target_matrix.replace("/", "_").replace(".", "_")
    output_path = os.path.join(output_dir, f"breakdown_{clean_name}.pdf")
    
    plt.savefig(output_path)
    plt.close() # Free up memory
    
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()