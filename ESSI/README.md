# ESSI

Enrichment Score Statistical Inference (ESSI) calculates enrichment Z scores and shrinkage theta scores from sequence counts across one or more comparisons.

## Installation

Requires Python 3.9 or later. The scoring pipeline uses the standard library; `isc.py` requires NumPy.

Run commands from the repository root:

```bash
python -m pip install -r ESSI/requirements.txt
python ESSI/test_essi.py
```

## Input

Sample files contain one sequence or label per line, or tab-separated `ID` and `Seq` columns with a header. Each row counts as one read. The default filename pattern is `{sample}_45.txt`.

The metadata file assigns sample IDs to batches and comparison groups. CSV, TSV and XLSX are supported:

```text
batch,pre pool,pos lib2,pos lib3
exp1,Q0,Q1,Q2
```

Each rules file lists post versus pre comparisons in analysis order:

```text
pos lib2 vs pre pool
pos lib3 vs pos lib2
```

## Usage

```bash
python ESSI/multi_round_pipeline.py --meta meta.csv --data /path/to/data --rules rule1.txt rule2.txt --input-type dna --reading-frame 0 --stop-policy exclude --dual-absent legacy --output results/essi
```

| Option | Description |
| --- | --- |
| `--rules` | One or more rules files |
| `--batch exp1,exp2` | Batches to analyze; default: all |
| `--input-type` | `dna`, `aa` or `label` |
| `--file-template` | Sample filename pattern |
| `--check-only` | Check configuration and list input files |

DNA input is translated using the standard genetic code. Frame 0 and stop exclusion are specified above. Reads with ambiguous bases or incomplete codons are excluded, and synonymous encodings are pooled by amino-acid sequence.

Use an empty output directory. Keep `xlsx_stream.py` and `assets/xlsx_styles.xml` alongside the pipeline.

## Output

| File | Contents |
| --- | --- |
| `<batch>/<rule>/analysis.xlsx` | Joint `results`, individual comparisons and score summaries |
| `<batch>/filtered_by_threshold.xlsx` | Sequences with positive joint Z scores and the specified final-post count cutoffs |
| `statistics.xlsx` | Sample, comparison and combined statistics |
| `manifest.json` | Parameters, input audit and run status |

Joint scores are in the `results` sheet as `Z_total` and `theta_total`. Tables are saved as XLSX.
