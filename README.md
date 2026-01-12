# PARCO-Computing-2026-243947 — Deliverable 2 (MPI)

Project for the course **Fundamentals of Parallel Programming**  
University of Trento – A.Y. 2025/2026  

**Student:** Matteo Miglio  
**Student ID:** 243947  
**Email:** matteo.miglio@studenti.unitn.it  

---

## Overview
This deliverable implements **Sparse Matrix–Vector Multiplication (SpMV)** on **distributed-memory systems** using **MPI**.

The project focuses on:
- Data distribution strategies
- Communication overhead analysis
- Strong and weak scalability
- Load balancing and memory footprint behavior

Two MPI-based implementations are provided:

1. **Ghost-based SpMV (main implementation)**  
   Uses explicit ghost exchange to minimize communication during the kernel.
2. **Broadcast-based SpMV (reference implementation)**  
   Uses full vector broadcast for comparison with a simpler communication pattern.

All executions produce **structured JSON output**, later processed by Python scripts to generate performance and scalability plots.

---

## Key Features
- Sparse matrix storage in **Compressed Sparse Row (CSR)** format
- Matrix input via **Matrix Market (.mtx)** or synthetic generation
- **1D cyclic row partitioning** across MPI ranks
- Explicit **ghost zone detection and exchange**
- Pre-indexing (LUT) to remove conditionals inside the SpMV kernel
- Detailed timing separation:
  - setup
  - communication
  - kernel execution
- Strong and weak scaling experiments
- Fully automated HPC execution via PBS

---

## Repository Structure
```
├── D1_Parallel_VS_Sequential/    # Deliverable 1
├── Makefile                      # Global build configuration
├── README.md                     # Project documentation
├── results/                      # Results
│   └── plots/                    # plots
│   └── distributedSpmV.json.     # Results data  
├── bin/                          # Compiled binaries
├── matrices/                     # Matrix Market (.mtx) files (downloaded at runtime)
├── obj/                          # Object files
├── scripts/                      # Job scripts and analysis tools
│   ├── distributed.pbs           # PBS script for ghost-based SpMV
│   ├── distributedBcast.pbs     # PBS script for broadcast-based SpMV
│   ├── plot.sh                   # Auxiliary script used during local development (not used on HPC)
│   └── plots/                    # Python scripts for performance analysis
│       ├── breakdown.py
│       ├── loadBalancing.py
│       ├── memoryFootprintScaling.py
│       ├── performance.py
│       ├── strongScaling.py
│       ├── strongScalingSE.py
│       ├── structVSunstruct.py
│       ├── weakScaling.py
│       └── weakScalingSE.py
└── src/
    ├── distributedSpMV.cpp       # Ghost-based distributed SpMV implementation
    ├── distributedBcastSpMV.cpp  # Broadcast-based distributed SpMV implementation
    ├── CSR/                      # Compressed Sparse Row data structures
    ├── GhostManager/             # Custom ghost exchange management
    ├── MTX/                      # Matrix Market parsing utilities
    ├── ResultsManager/           # Timing and results aggregation
    └── Utils/                    # Shared helper utilities
```
---

## Requirements & Reproducibility
This project is optimized for the UNITN HPC Cluster. To ensure strict reproducibility, the environment is managed via modules and automated scripts.

Software Stack:
- Compiler: gcc91 (specifically g++ 9.1.0) 
- MPI: mpich-3.2.1--gcc-9.1.0 
- Python: 3.10.14_gcc91 for performance analysis

To run local mpi is needed, on MacOS (system used for development) use brew to install it
```
brew install open-mpi
```

**HPC commands**
```
git clone https://github.com/Magnus3327/PARCO-Computing-2026-243947
cd PARCO-Computing-2026-243947
qsub scripts/distributed.pbs
```
>qsub scripts/distributedBcast.pbs to start simulation with broadcasting version

**Simulation result** including plots are into results directory

**Local commands**
```
git clone https://github.com/Magnus3327/PARCO-Computing-2026-243947
cd PARCO-Computing-2026-243947
make distributed
mpirun -np 4 bin/distributedSpMV "-VM=10000;10000;0.01" -I=10
```

result printed into cli

**Script note**
The provided PBS scripts implements a robust reproducibility pipeline:

- Module Loading: Automatically loads the required GCC, MPI, and Python modules. 
- Virtual Environment: Creates a temporary Python venv to install dependencies (matplotlib, numpy, pandas) without affecting the system environment. 
- Execution & Cleanup: After generating all performance plots, the venv directory is automatically removed to maintain a clean workspace.

---

## Implementations

### 1. Ghost-based Distributed SpMV (Main)
**Executable:** `bin/distributedSpMV`

- Matrix rows distributed using **1D cyclic partitioning**
- Input vector distributed cyclically
- Each rank:
  - Detects off-rank column dependencies
  - Exchanges only required ghost values
- Kernel uses **precomputed lookup tables**:
  - `xIndex[]`
  - `isGhost[]`
- No communication inside the kernel loop

This version is the **primary subject of performance analysis**.

---

### 2. Broadcast-based Distributed SpMV (Comparison)
**Executable:** `bin/distributedBcastSpMV`

- Matrix partitioning identical to ghost-based version
- Entire input vector is broadcast to all ranks
- Simpler communication pattern
- Higher communication volume

Used **only for comparison** to highlight the cost/benefit of ghost exchange.

---

## Matrices and Scaling Experiments

### Strong Scaling
Strong scaling experiments were conducted using **real-world sparse matrices** from the **SuiteSparse Matrix Collection**, chosen to represent different sparsity structures and application domains.

The following matrices were used:

| Matrix name        | Sparsity Type   | Notes                    |
|--------------------|-----------------|--------------------------|
| `cit-Patents`      | Unstructured    | Citation graph adjacency |
| `soc-LiveJournal1` | Unstructured    | Social network adjacency |
| `thermal2`         | Structured      | Finite element mesh      |
| `Flan_1565`        | Structured      | Engineering matrix       |

**Notes:**  

- **Structured matrices** (`thermal2`, `Flan_1565`) typically lead to more predictable load balancing and better vectorization.  
- **Unstructured matrices** (`cit-Patents`, `soc-LiveJournal1`) stress the communication system due to irregular ghost exchanges and uneven NNZ distribution.  
- Performance differences between ghost-based and broadcast-based implementations are more pronounced on unstructured matrices.

All matrices are automatically downloaded during the PBS job execution from the official SuiteSparse repository.

For strong scaling, the **input matrix is fixed** while the number of MPI processes is varied:
P = {1, 4, 16, 64, 128, 256}

### Weak Scaling
Weak scaling experiments were performed using **synthetically generated sparse matrices** in order to maintain a **constant computational workload per MPI rank**.

Matrices are generated at runtime using the `-VM=<rows>;<cols>;<density>` option, with a fixed density of `0.01`.  
The target workload is approximately **1 million non-zero elements (NNZ) per MPI rank**.

| MPI Ranks | Matrix Size (N × N) | Total NNZ | Avg expected NNZ per Rank |
|----------:|--------------------:|----------:|--------------------------:|
| 1         | 10,000 × 10,000     | 1M        |                        1M |
| 4         | 20,000 × 20,000     | 4M        |                        1M |
| 16        | 40,000 × 40,000     | 16M       |                        1M |
| 64        | 80,000 × 80,000     | 64M       |                        1M |
| 128       | 113,137 × 113,137   | 128M      |                        1M |
| 256       | 160,000 × 160,000   | 256M      |                        1M |

This configuration isolates **communication and synchronization overheads**, allowing a clear evaluation of weak scalability.

---

### Experimental Rationale
- **Strong scaling** evaluates performance improvements obtained by increasing the number of MPI processes for a fixed problem size.
- **Weak scaling** evaluates the ability of the implementation to maintain constant execution time as both problem size and number of MPI processes increase proportionally.

Real-world matrices highlight the effects of irregular sparsity patterns and load imbalance, while synthetic matrices provide controlled and reproducible scaling conditions.

---

## Compilation
MPI compiler is required.

# Compiler flags:O3

```
make distributed     # Ghost-based implementation
make bcast           # Broadcast-based implementation
```

output bin are:
- distributedSpMV
- distributedBcastSpMV
> stored in bin folder

---

## Command Line Arguments

The distributed executables accept the following command line options:

- `-M=`  
  Path to a Matrix Market (`.mtx`) file.

- `-VM=rows;cols;density` 
  Generate a synthetic sparse matrix with the specified dimensions and density. 

- `-I=`  
  Number of timed SpMV iterations.

Examples:
```
mpirun -np 16 bin/distributedSpMV -M=matrices/heart1.mtx -I=10
mpirun -np 8  bin/distributedSpMV "-VM=10000;10000;0.01" -I=10
```

>-VM option requires quotes in the shell for correct parsing (future improvement: adjust CLI parser)

---

## Execution Workflow
1. MPI initialization
2. Rank 0:
   - Parses CLI
   - Loads or generates matrix
3. Broadcast problem metadata
4. Distribute matrix entries (cyclic rows)
5. Distribute input vector
6. Ghost discovery phase
7. Ghost exchange
8. Pre-indexing (LUT preparation)
9. Warm-up SpMV (not timed)
10. Timed SpMV iterations
11. Reduction of timing statistics
12. JSON output (rank 0)
13. MPI finalize

---

## Output Format (JSON)
Each run produces a JSON object containing:

- Hardware info
- Matrix metadata
- MPI configuration
- Load-balancing statistics (NNZ per rank)
- Ghost statistics per rank and overall
- Timing breakdown:
  - setup (matrix creation, vector creation, matrix distribution)
  - communication (ghost entries or broadcasting vetctor)
  - kernel
- Derived metrics:
  - GFLOPS
  - bandwidth
  - memory footprint per rank

All results are aggregated into a single JSON file during batch execution.

---

## HPC Execution (PBS)

### Job Script

scripts/distributed.pbs

The script:
- Loads required modules (gcc, MPI, Python)
- Downloads matrices from SuiteSparse
- Compiles the project
- Runs:
  - Strong scaling (real matrices)
  - Weak scaling (synthetic matrices)
- Collects hardware metadata
- Aggregates all results into: results/distributedSPMV.json

> To avoid errors, `.out` and `.err` files are written directly in the submission directory.

---

## Plotting
All plots are generated automatically at the end of the PBS job.

Each script takes:
<json_file> <output_dir>

Except:
breakdown.py <json_file> <matrix_name> <output_dir>

Generated plots include:
- Strong scaling
- Weak scaling
- Speedup & efficiency
- Load balancing
- Memory footprint scaling
- Communication vs computation breakdown
- Structured vs unstructured sparsity behavior

Plots are saved in:
results/plots/**bcast or distributed**/

---

## Notes
- Output vector is not gathered, the main focus is on performances
- Ghost-based implementation is the reference solution
- Broadcast version is included for communication comparison only
- Designed for execution on HPC clusters with MPI

---

A complete set of results and plots is included for reproducibility and evaluation.

Some `.DS_Store` metadata files may appear due to macOS filesystem behavior.
They do not affect the project.

---  