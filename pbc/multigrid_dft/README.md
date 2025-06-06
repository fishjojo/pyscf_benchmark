# Multigrid DFT

## Introduction

Benchmark results for single-point energy and nuclear gradient calculations using
the Multigrid DFT implementation in PySCF.
All simulations were performed on a node equipped with Intel(R) Xeon(R) Platinum 8276 CPUs running at 2.20 GHz.

## PySCF Configuration

PySCF was manually compiled with support for FFTW and Intel MKL using the following `cmake` configuration:

```bash
cmake -DENABLE_FFTW=ON -DBUILD_FFTW=ON -DBLA_VENDOR=Intel10_64lp_seq
```

In the PySCF configuration file, the runtime configuration was adjusted to enable
the FFTW backend and set detailed timing output:

```python
pbc_tools_pbc_fft_engine="FFTW"
TIMER_LEVEL=4
```

## Benchmark Results

### Water Clusters

Gamma-point restricted KS-DFT calculations for water molecules in cubic cells.
The configurations were taken from [CP2K](https://github.com/cp2k/cp2k/tree/master/benchmarks/QS),
which were generated by classical MD equilibration.
All calculations use the GTH-TZV2P basis set, the GTH-PBE pseudopotential, and the PBE density functional.
The planewave cutoff of 200 a.u. and the integral precision of $10^{-6}$ a.u. were used.
Reported timings represent the best observed results including:
the Fock build time, the total time for one SCF cycle, and the nuclear gradient time.

| System  | Date      | Git Commit | No. of Cores | Fock Build Time (s) | SCF Time (s) | Nuclear Gradient Time (s) |
|---------|-----------|------------|--------------|---------------------|--------------|----------------------------
| H2O-64  | 2025-5-30 | 623f6f0    | 28           | 1.43                | 2.85         | 4.83                      |
| H2O-128 | 2025-5-30 | 623f6f0    | 28           | 3.29                | 13.44        | 14.76                     |
| H2O-256 | 2025-5-30 | 623f6f0    | 28           | 7.98                | 98.46        | 46.01                     |
| H2O-512 | 2025-5-30 | 623f6f0    | 28           | 18.62               | 489.89       | 153.95                    |

### Bulk Silicon

Gamma-point restricted KS-DFT calculations for bulk silicon with conventional supercells.
The configurations were taken from the [Materials Project](https://next-gen.materialsproject.org/materials/mp-149).
All calculations use the GTH-DZVP basis set, the GTH-PBE pseudopotential, and the PBE density functional.
The planewave cutoff of 140 a.u. and the integral precision of $10^{-8}$ a.u. were used.
Reported timings represent the best observed results including:
the Fock build time, the total time for one SCF cycle, and the nuclear gradient time.

| System   | Date     | Git Commit | No. of Cores | Fock Build Time (s) | SCF Time (s) | Nuclear Gradient Time (s) |
|----------|----------|------------|--------------|---------------------|--------------|----------------------------
| Si       | 2025-6-3 | 623f6f0    | 28           |  0.36               |   0.71       |   0.55                    |
| Si 2x2x2 | 2025-6-3 | 623f6f0    | 28           |  1.30               |   1.57       |   4.35                    |
| Si 3x3x3 | 2025-6-3 | 623f6f0    | 28           |  5.31               |   7.31       |  21.71                    |
| Si 4x4x4 | 2025-6-3 | 623f6f0    | 28           | 15.03               |  39.80       |  97.28                    |
| Si 5x5x5 | 2025-6-3 | 623f6f0    | 28           | 30.24               | 151.69       | 255.05                    |
| Si 6x6x6 | 2025-6-3 | 623f6f0    | 28           | 54.29               | 722.43       | 611.84                    |
