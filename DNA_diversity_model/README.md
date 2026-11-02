# DNA diversity model

Estimate library sequence richness with the first-order jackknife and infer cellular sequence richness under a molecular capture model. Abundance sensitivity uses a shifted symmetric Dirichlet distribution constrained by raw-abundance Gini, with a lognormal distribution as a mean sensitivity comparison.

## Installation

Requires Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python run_analysis.py --help
```

## Input

Sample metadata is a tab-separated file with a header: `id`, `file1`, and optional `file2` and `file3` columns. Single-library metadata can use `id` and `file`. Files in the same row are pooled by exact sequence identity. Filenames must match exactly, and each file must belong to one sample. Relative paths are resolved against `--data-dir`; absolute paths are also accepted.

Sequence files contain one read per line, with the nucleotide sequence in the last tab-separated column. An `ID\tSeq` header, single-column sequence files, and gzip compression are supported. Aggregated sequence/count tables are not supported.

Sequences retain case, length differences, and singleton identities. Accepted characters are `ACGTNacgtn`. Empty sequences are skipped; invalid characters raise an error.

## Configuration

Settings are read from `config.json`. To use another file, place `--config path/to/config.json` before the subcommand. Each subcommand requires `--output`. Use separate output directories for separate runs.

| Physical setting | Default |
| --- | --- |
| Initial cells, `initial_cells` | 25,000,000 |
| Editing efficiency, `editing_efficiency` | 0.2 |
| Expansion, `expansion` | 8 |
| Target copies per cell, `copies_per_cell` | 1 |
| Effective capture coefficient, `effective_efficiency` | 1 |
| Pre-library sampling fraction, `pre_library_fraction` | 1/30 |

The model derives surviving cells as `Ns = N0 * p`, total target molecules as `M = expansion * C * Ns`, and capture probability as `beta = eta * pre_library_fraction`. Editing efficiency is treated as the fraction entering expansion after selection. For pooled libraries, `eta` describes effective capture into the tested library union.

Random seeds, sample-specific seed overrides, sampling fractions, Gini values, simulation counts, and bin resolution are configurable. Fixed-seed results also depend on input sequence order, file pooling order, and simulation chunk size.

## Usage

### Library richness

```bash
python run_analysis.py richness --meta metadata.tsv --data-dir data --output results/richness
```

Runs 20 independent repetitions of nested read subsampling without replacement, from 10% to 100% depth in 10% increments. The default grid requires at least 10 reads per sample.

Outputs:

- `D_library_replicates.csv`: estimates for each repetition and sampling fraction.
- `D_library_summary.csv`: mean estimates, technical standard deviations, and empirical quantiles.
- `full_depth_estimators.csv`: full-depth jackknife estimates for cellular richness inference.

### Analytical capture curves

```bash
python run_analysis.py curves --output results/curves
```

Writes `mean_curves.csv`. Structural bounds with a minimum abundance of one molecule per type are available through `CaptureModel.structural_bounds(D_library)` and in the inference output.

### Capture simulations

```bash
python run_analysis.py simulate --output results/predictions
```

Defaults are Gini values from 0 to 0.8 in increments of 0.1, 2,048 probability bins, and 43 cellular richness positions per Gini value. Each position uses 4,000 repetitions; positions at 1, 2.5, and 5 million types use 10,000.

Writes `prediction_intervals.csv`, including analytical and simulated means and prediction quantiles. Replicate arrays are saved only with `--keep-replicates`.

For local refinement, provide a CSV with `D_cells` and `target_gini` columns:

```bash
python run_analysis.py simulate --nodes local_nodes.csv --output results/refinement
```

### Cellular richness inference

```bash
python run_analysis.py infer --library-table results/richness/full_depth_estimators.csv --predictions results/predictions/prediction_intervals.csv results/refinement/prediction_intervals.csv --output results/inference
```

The library table requires `sample_id` and either `D_library` or `D_library_mean`, with one full-depth jackknife estimate per sample. If present, `estimator` is filtered to `jack1` and `fraction` to 1.

Pass the main prediction grid first, followed by any refinement tables. The refinement path can be omitted when no refinement table is used. Later tables take precedence at duplicate nodes. Prediction tables must use the same `M` and `beta` as the model and cover the cell-count ceiling. Provide nodes spanning the relevant inverse range: interpolation holds the nearest quantile offset constant outside the supplied grid.

Without `--predictions`, the command computes point estimates and equal-abundance intervals. Nonzero-Gini intervals are marked `prediction_grid_not_supplied`.

Outputs:

- `sample_D_cells_by_gini_with_intervals.csv`: point estimates, 95% conditional inverse intervals, structural bounds, and feasibility flags.
- `D_cells_ranges.csv`: minimum, maximum, and median feasible point estimates, sample counts, and interval envelopes for each Gini value.

`D_cells` counts distinct sequence types in the initial cell population. The 95% intervals describe random abundance allocation and capture conditional on the specified model. They exclude uncertainty in the jackknife input, physical parameters, target Gini, and distribution family. Structural bounds and cross-sample envelopes are reported separately.

### Abundance distribution sensitivity

```bash
python run_analysis.py sensitivity --output results/sensitivity
```

Calibrates lognormal weights using 512 abundance realisations and compares means using 4,000 independent simulations at each configured position. Writes `lognormal_mean_sensitivity.csv`.

Every command writes `run_config.json` with its settings, input arguments, dependency versions, and run status.

## Code

| File | Purpose |
| --- | --- |
| `richness.py` | Sequence counting, jackknife estimation, and nested read subsampling |
| `capture_model.py` | Molecular budget, structural bounds, Gini calibration, and analytical capture means |
| `simulation.py` | Grouped abundance and capture simulations |
| `inference.py` | Conditional inversion and sample summaries |
| `run_analysis.py` | Command-line interface |
| `source_map.json` | Source implementation identifiers and validation records |
