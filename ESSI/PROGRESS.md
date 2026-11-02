# ESSI code status

Date: 2026-11-02. Status: ready for repository integration.

Scope: target-sequence input, read counting/translation, single-comparison and joint ESSI, right-tail ISC, XLSX export. No alignment or wet-lab procedures.

Methods: `ESSI_Methods_v1.2_ZH/EN.docx` and `ESSI_Supplementary_Methods_v1.2_ZH/EN.docx`; the ESSI section of the final consolidated PE-Div Methods v6 retains these calculations.

Source mapping:

- `multi_round_pipeline.py`, `xlsx_stream.py`, `assets/xlsx_styles.xml` are byte-identical copies of the final files at the original analysis project root.
- `isc.py::right_isc` extracts the right-tail implementation from `tools/calculate_isc_alpha.py::directional_isc`, with alpha, tolerance and iteration limit exposed as arguments. The stopping iteration's pre-update cutoff is retained.

Validation completed:

- 3 numerical/translation tests passed.
- Copied source hashes match the original source files.
- Extracted ISC cutoff and full iteration history match the existing implementation on deterministic synthetic scores.
- C140 reuses these files and passed a synthetic end-to-end command test.

Not rerun: full sequencing-read processing or fitting of the original complete Z distributions. Historical analysis outputs remain unchanged. No pending decisions.
