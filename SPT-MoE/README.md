# SPT-MoE

Signal Peptide Transfer Mixture of Experts (SPT-MoE) combines HuSP pretraining with DMS adaptation and HuSP replay. Predictions use the mean of two task-head percentiles calculated against pooled training references.

## Installation

Use Python 3.11 and a PyTorch build compatible with your CUDA installation.

```bash
pip install -r requirements.txt
```

ESM feature extraction requires an ESM2 3B checkpoint. Model weights and ESM features are not included.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/` | Data preparation, ESM extraction, model, training and evaluation |
| `config.json` | Model and training parameters |
| `metadata/` | Dataset membership, labels, groups, splits and batch-alignment parameters |
| `check.py` | Basic implementation checks |
| `environment_reference.json` | Dependency versions used for verification |
| `METHODS_MAP.md` / `SOURCE_MAP.json` | Method and source-code references |
| `verification.json` | Verification results |

Run the commands below from the repository root. Use new output directories and filenames.

## Data preparation

Dataset manifests are provided in `metadata/{dms,dpf}_members.csv`. The code uses `dpf` for HuSP. Each manifest includes `sample_id`, `input_sequence`, `group_id`, `split` and class labels. HuSP also has protein entity groups. Preserve the supplied membership and splits.

DMS inputs contain the first 20 amino acids. HuSP inputs contain the selected N-terminal fragments.

### HuSP negative fragments

Place input FASTA files in `T1/` and `T2/` beneath the input root.

```bash
python src/prepare.py truncate --root INPUT_FASTA_ROOT --output NEW_NEGATIVE_DIR
```

The script samples an integer length uniformly from `[8, min(30, L)]` and retains that N-terminal fragment. Sequences shorter than eight residues are excluded. The random seed is 42; files are sorted by name and records retain their FASTA order. Use the supplied manifests to reproduce the study's sequence membership.

`select_husp_positives` selects exact matches between human reviewed SIGNAL peptides and the coding library, with ATG, reading-frame and sequence-deduplication filters. `match_husp_negatives` removes accession and sequence conflicts, then applies two-stage length matching with category-stratified sampling. Negative source entries are human proteins without SIGNAL annotations.

### DMS labels

`data_methods.process_scheme_b` takes a table containing `exp_id` and `theta_total`. It normalizes positive and negative enrichment values separately within each experiment, then applies iterative sigma clipping to the pooled values with `p=1e-6` and `k=norm.ppf(1-p/2)`. Values at or above the threshold form the positive class; an equal number of the lowest values form the negative class.

## ESM features

Create a FASTA file from the manifest's `sample_id` and `input_sequence` columns. FASTA IDs must be unique.

```bash
python src/extract_esm.py --fasta INPUT.fasta --checkpoint ESM2_3B.pt --output NEW_ESM_DIR
python src/prepare.py pack --manifest metadata/dms_members.csv --features DMS_ESM_DIR --output DATA/dms
python src/prepare.py pack --manifest metadata/dpf_members.csv --features HUSP_ESM_DIR --output DATA/dpf
```

Extraction uses layer 33 of ESM2 3B and one sequence per forward pass. Each `<SHA256(sample_id)>.npy` file contains an `(L, 2560)` array of residue features, excluding BOS and EOS tokens.

Packing produces a CSV and NPZ for each split:

| NPZ key | Shape | Contents |
| --- | --- | --- |
| `ids` | `(N,)` | Sample IDs in CSV row order |
| `esm` | `(N, 2560)` | Mean residue embedding |
| `tokens` | `(N, 34, 8)` | Residue features; channels 0 and 1 contain the embedding mean and maximum |
| `mask` | `(N, 34)` | True for actual residues |

The model consumes the first two token channels. The remaining channels are zero. Pooling excludes padding.

## Training

```bash
python src/train.py --data DATA --output NEW_MODEL_DIR --device cuda
```

SPT-MoE has six experts and a 64-dimensional supplementary-feature branch. The scaler assigns equal total weight to each domain and equal weight to unique sequences within each domain.

HuSP pretraining runs for 12 epochs. DMS adaptation starts from the pretrained weights and copies the HuSP head to initialize the DMS head. It updates the gate, FFN, final expert and DMS head while freezing the HuSP head and remaining modules. Each DMS batch is paired with an equally sized HuSP replay batch; the two classification losses have equal weight. Adaptation runs for 39 epochs. The cosine learning-rate schedules span 60 and 40 epochs, respectively.

The training entry point uses fixed epoch counts and reads only training members. `train_stage(validation=...)` also supports training-fold checkpoint selection; historical validation, test and NLUC data must remain outside that selection.

Outputs include `P.pth`, `T1.pth`, `scaler.pkl`, training histories, configuration and `pooled_reference.npy`. The percentile reference pools 1,492 unique DMS and 4,552 unique HuSP training sequences.

## Prediction

```bash
python src/evaluate.py --model MODEL_DIR --features EXTERNAL.npz --metadata EXTERNAL.csv --output NEW_PREDICTIONS.csv
```

The metadata CSV must contain `sample_id` values that match NPZ `ids` in row order. Prediction averages 64 Bernoulli-attention draws with ordinary dropout disabled. Use `--sampling-group independent` for a second fixed sampling stream.

For each head, the percentile is `(n_< + n_≤ + 1)/(2(N+1))`, where `N` is the pooled reference size. The final score is the mean of the two percentiles. It is a relative score, not a calibrated probability or an activity measurement.

## NLUC evaluation

`evaluate.merge_external` accepts `sample_id`, `input_sequence`, `batch` (`REF` or `Recom`), `activity`, `score` and `signalp_sp`.

Exclude B2M, P61769 and the four REF entries listed in `metadata/nluc_ref_exclusions.csv` before evaluation. Average duplicate sequences within Recom only, giving REF38, Recom30 and All68. For pooled evaluation, multiply Recom activity by `recom_to_ref_multiplier` in `metadata/batch_alignment.json`. Use original activity for within-batch correlations and aligned activity for pooled correlations.

`evaluate.correlations` returns Pearson and Spearman coefficients. The SignalP comparator is the Eukarya SP(Sec/SPI) score. B2M may be displayed as a reference point but is excluded from correlations and density estimates.

## Checks

```bash
python check.py
```

The checks cover imports, N-terminal truncation, ISC, feature shapes, repeated MC predictions, gradients and percentile fusion.
