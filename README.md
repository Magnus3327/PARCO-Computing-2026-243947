# PARCO-Computing-2026-243947
Project for the course **Fundamentals of Parallel Programming**  
University of Trento – A.Y. 2025/2026

---

## Overview
This project implements **Sparse Matrix-Vector Multiplication (SpMV)** using:

- Compressed Sparse Row (CSR) storage format
- Sequential execution (baseline)
- Parallel execution on shared-memory systems using **OpenMP**

The program collects detailed performance metrics and exports them in **JSON format**, enabling benchmarking and plotting.

---

## Assignment Requirements
✅ Read a matrix in Matrix Market (.mtx) format and convert it to CSR  
✅ Implement sequential SpMV  
✅ Implement parallel SpMV (OpenMP)  
✅ Benchmark:

- At least 10 timed runs, reporting the **90th percentile**
- At least 5 matrices with different sparsity levels
- Compare sequential vs parallel performance
- Vary number of threads
- Evaluate scheduling strategies
- Identify bottlenecks

---

## Repository Structure
```
repo/
    src/
    matrices/
    scripts/
        pbs/
        plots/
    results/
    plots/
    bin/
    obj/
```

---

## Requirements
This project is designed to be built with:

```
g++ 9.1+
OpenMP support
```

Optional (for plotting):

```
Python 3.x
matplotlib
numpy
```

---

## Workflow
1. Load a `.mtx` sparse matrix
2. Convert to **CSR**
3. Generate a random vector
4. Warm‑up run (not timed)
5. Perform `N` timed SpMV iterations
6. Collect timings and metadata
7. Compute:

- 90th percentile execution time
- FLOPs
- GFLOPS
- Memory Bandwidth (GB/s)
- Arithmetic Intensity (FLOPs/Byte)

8. Export results as JSON

---

## Source Files Overview
The project is organized into modular components to improve readability, maintainability and testing.

Core modules:
- **CSR/** — CSR data structure and build
- **MTX/** — Matrix Market (.mtx) loader and validation
- **ResultsManager/** — JSON output, metadata aggregation, percentile computation
- **Utils/** — helpers random vector generation

Main executables:
- `spmvSequential.cpp` — sequential baseline implementation
- `spmvParallel.cpp` — OpenMP parallel implementation

## Output Format (JSON)
```json
{
  "matrix": {
    "name": "...",
    "rows": 0,
    "cols": 0,
    "nnz": 0
  },
  "scenario": {
    "threads": 0,
    "scheduling_type": "...",
    "chunk_size": 0
  },
  "statistics90": {
    "duration_ms": 0.0,
    "FLOPs": 0.0,
    "GFLOPS": 0.0,
    "Bandwidth_GB/s": 0.0,
    "arithmetic_intensity": 0.0
  },
  "warmUp_time_ms": 0.0,
  "all_iteration_times_ms": [...],
  "errors": [...]
}
```

> The **sequential version** does not include OpenMP scenario information.

---

## Download
```
git clone https://github.com/Magnus3327/PARCO-Computing-2026-243947
```

---

## Compilation
Using Makefile:

```
make parallel
make sequential
```

Compiler flags:

```
-O3
-fopenmp (parallel)
```

---

## Command Line Arguments
Parallel execution
```
matrix_path          (required) Path to .mtx file
-T=<int>             OpenMP threads
-S=<string>          Scheduling: static | dynamic | guided
-C=<int>             Chunk size
-I=<int>             Timed iterations
```

Sequential have only
```
matrix_path          (required) Path to .mtx file
-I=<int>             Timed iterations
```

Automatic handling:

- Input validation
- Thread count capped to system maximum
- Runtime OpenMP scheduling configuration
- Memory management via `unique_ptr`

---


## Running Locally
Single run Sequential: 

```
bin/spmvSequential matrices/<matrix>.mtx -I=10
```

Redirect output to JSON file:

```
bin/spmvSequential matrices/<matrix>.mtx -I=10 > output.json
```

Single run parallel:

```
bin/spmvParallel matrices/<matrix>.mtx -T=4 -S=static -C=4 -I=10
```

Redirect output to JSON file:

```
bin/spmvParallel matrices/<matrix>.mtx -T=4 -S=static -C=4 -I=10 > output.json
```

Changing OpenMP scheduling or chunk size does **not** require recompilation.  
All runtime configuration is controlled via CLI arguments.

---

## Matrices
The repository contains several pre-installed matrices (<100 MB due to GitHub limits).  
Additional `.mtx` files may be downloaded and placed in:

The following matrices are used in benchmarking, spanning a range of sizes and sparsity:

| Matrix name            | Rows     | Cols     | NNZ         | Sparsity degree |
|----------------------- |--------- |--------- |------------ |----------------|
| Journals.mtx           | 124      | 124      | 6,096       | 0.6037         |
| dendrimer.mtx          | 730      | 730      | 31,877      | 0.9403         |
| heart1.mtx             | 3,557    | 3,557    | 1,387,773   | 0.8904         |
| TSOPF_FS_b300.mtx      | 29,214   | 29,214   | 2,203,949   | 0.9974         |
| rgg_n_2_21_s0.mtx      | 2,097,152| 2,097,152| 14,487,995  | 0.999997       |

*Sparsity is defined as* \( 1 - \frac{nnz}{rows \times cols} \).

These matrices were selected to illustrate the effect of sparsity and size on SpMV parallel performance and scalability. The largest matrix (rgg_n_2_21_s0.mtx) is automatically downloaded as part of the PBS HPC job.

---

## HPC Execution (PBS)
To run a full benchmark suite on an HPC system:

```
qsub scripts/pbs/simulation.pbs
```

This job will:

- download a large matrix if needed
- load required modules
- run multiple configurations (threads / schedule / chunk)
- collect performance metrics
- perform `perf` profiling:

    - L1-dcache-loads
    - L1-dcache-load-misses
    - LLC-loads
    - LLC-load-misses
    - cache-references
    - cache-misses

Results are stored in:

```
results/
```

---

## Plotting
After job completion:

Locally or on cluster (depending on environment)

```
python scripts/plots/<plot_script>.py
```

The plotting scripts allow selecting output folders dynamically.

Ensure required Python libraries are installed:

```
matplotlib
numpy
```

On macOS, a virtual environment is recommended.

---

 A set of benchmark results (JSON + perf profiling) is included in the `results/` directory for quick verification and comparison.

## Notes
Some `.DS_Store` metadata files may appear due to macOS filesystem behavior.
They do not affect the project.