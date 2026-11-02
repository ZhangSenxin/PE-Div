# C140 code status

Date: 2026-11-02. Status: ready for repository integration.

Scope: rule1 joint ESSI, right-tail positive calling, 1-aa families, final annotated ASOP panel and its three representative tree/logo examples.

Methods: `C140_Methods_ZH/EN.docx` (v1.1), also incorporated into final consolidated PE-Div Methods v6. The numerical definitions and selected panel remain those documented in the paired C140 trace.

Source mapping (paths relative to the original analysis project):

- `run.py::score`: invokes the unchanged ESSI CLI using the seven rule1 samples. `config/meta.csv` is the named-column mapping extracted from `C140/meta/meta.xlsx`; `config/rule1.txt` is copied unchanged.
- `run.py::analyze`: reads complete results, applies ESSI ISC and exports only the required positive/family/plot tables.
- `analysis.py`: distilled from `overlap_ana/support/cluster_cdr3_union.py`; only 1-aa families retained. Union sorting, component IDs and global Prim MST tie breaks retained.
- `figures.py::spread/panel/colorbar`: final plotting logic from `overlap_ana/support/revise_purity2_panel.py` and terminology from `rename_purity2_to_asop.py`.
- `figures.py::tree_coordinates`: centroid-root and circular layout rules from `overlap_ana/support/plot_cdr3_cluster_trees.py`, expressed locally without zsx or pandas dependencies.
- `figures.py::logo_matrix/draw_logo/draw_tree_logo`: selected rendering functions from `overlap_ana/support/antigen_purity_and_logos.py`; captions now use ASOP only.

Validation completed using existing selected-sequence records:

- All 4328 memberships and all 380 family IDs match the original 1-aa results.
- The complete global MST edge list matches the original; included families=127 and plotted families=126.
- All 127 included ASOP values/counts, all 126 display positions, and all three example selections match.
- All 397 example-node roots/coordinates and all example logo letter heights match within 1e-12.
- XLSX readers passed inline/shared-string, sparse-column and non-first-sheet checks.
- 4 small tests passed; the synthetic full analyze command and XLSX read-back passed.
- The actual ASOP figure and all three example tree/logo PNGs were visually inspected. QA images are not part of the code delivery.

Not rerun: original full DNA scoring or ISC on the complete original XLSX distributions. This delivery does not include raw data, result caches, analysis iterations or manuscript-generation code. No pending decisions.
