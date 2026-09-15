# Budgeted Depth Rewiring

Reference C++ implementation of budgeted depth rewiring (BDR) for maximizing
the algebraic connectivity of a connected, simple graph with fixed numbers of
vertices and edges. The implementation includes the comparison methods used by
the experiment drivers and numerical tests for the proposal, screening, depth,
connectivity, and eigensolve-accounting logic.

## Requirements

- A C++20 compiler
- OpenMP support for parallel candidate scans
- GNU Make
- Optional: `geng` from nauty for exhaustive graph enumeration

No external linear-algebra library is required.

On macOS, install GCC with Homebrew and select the installed executable, for
example `make CXX=g++-16`. To build without OpenMP, use `OMPFLAGS=`; the same
code then runs sequentially.

## Build and test

```sh
make -j
make test
```

The build produces the following command-line programs in `build/`:

- `sweep_m`: compare methods across edge counts at a fixed eigensolve budget
- `sweep_budget`: compare methods across eigensolve budgets at fixed density
- `enum_opt`: stream `graph6` records and compute per-edge-count maxima
- `compare`: compare generated CSV result tapes
- `audit_oracle`: check inertia decisions against dense eigensolves
- `start_lambda`: reproduce the shared initial-graph values

Generated outputs belong under `results/`, which is intentionally not tracked.

## Examples

Run the full edge-count sweep used for the small-graph comparison:

```sh
mkdir -p results
./build/sweep_m --n 12 --stride 1 --seeds 20 --budget 1000 \
  --threads 8 --out results/n12.csv
```

Run BDR alone at a fixed density:

```sh
./build/sweep_budget --n 64 --rho 0.2 --seeds 20 --budget 1000000 \
  --threads 8 --workers 1 --methods ours --out results/bdr_n64.csv
```

The `sweep_m` driver also exposes the paper ablations through
`--methods ours_local`, `ours_local_no_inertia`, `ours_qhalf`, and
`ours_qdouble`.

Audit the numerical screening decisions:

```sh
./build/audit_oracle --n 24 --trials 40 --seed 4242
```

For exhaustive enumeration, pipe nauty's canonical connected graphs into the
streaming evaluator:

```sh
geng -cq 12 | ./build/enum_opt --n 12 --out results/optimal_n12.csv
```

The exhaustive `n=12` command is computationally intensive and is intended for
parallel execution using nauty's `res/mod` partitioning.

## Source layout

- `src/methods/OURS.*`: BDR implementation
- `src/methods/`: comparison methods
- `src/methods/detail/`: shared graph-search primitives
- `src/app/`: reproducibility and verification drivers
- `tests/test_all.cpp`: deterministic unit and integration tests
