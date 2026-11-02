# C140

CDR3 enrichment analysis for two antigen-selection experiments, followed by 1-aa sequence clustering, antigen-selection origin purity (ASOP) visualization and representative family trees with sequence logos.

## Installation

Requires Python 3.9 or later, NumPy and Matplotlib. Keep `C140/` and `ESSI/` as sibling directories and run commands from the repository root:

```bash
python -m pip install -r C140/requirements.txt
python C140/test_c140.py
```

## Input and configuration

DNA files contain tab-separated `ID` and `Seq` columns, with one read per row. Sample filenames follow `{sample}_45.txt`.

| File | Contents |
| --- | --- |
| `config/c140.json` | Processing, selection and plotting parameters |
| `config/meta.csv` | Sample assignments for exp1 and exp2 |
| `config/rule1.txt` | Ordered post/pre comparisons |

The default configuration translates frame 0, excludes stops, selects positives using right-tail ISC at nominal alpha 0.0001, and plots families with at least 10 nodes. Families are connected components of the Hamming-distance-1 sequence graph.

## Usage

Calculate ESSI scores from the DNA files:

```bash
python C140/run.py score --data /path/to/C140/data --output results/essi
```

Select positive sequences, cluster families and generate figures:

```bash
python C140/run.py analyze --exp1 results/essi/exp1/rule1/analysis.xlsx --exp2 results/essi/exp2/rule1/analysis.xlsx --output results/c140
```

Existing complete `exp1_rule1.xlsx` and `exp2_rule1.xlsx` files can be supplied directly to `--exp1` and `--exp2`. Each must contain a complete `results` sheet with `label` and `Z_total` columns.

Both commands accept `--config` to select another configuration file. Metadata and rules paths are relative to that file. Use an empty output directory. Add `--check-only` to `score` to inspect the sample plan.

## Output

| File | Contents |
| --- | --- |
| `exp1_positive.xlsx`, `exp2_positive.xlsx` | Positive sequences and joint Z scores |
| `c140_summary.xlsx` | Thresholds, family membership, ASOP, plot coordinates and example tree/logo data |
| `family_size_signed_asop_annotated.png/.pdf` | Family size versus signed ASOP |
| `selected_families/*_tree_with_logo.png/.pdf` | Representative family trees and information logos |
| `run.json` | Configuration, input paths, run status and summary statistics |

Tables are saved as XLSX and figures as PNG/PDF.
