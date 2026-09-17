# Matched Multi-Gamma Linkability Pipeline

## Scripts

1. `01_prepare_matched_linkability_dataset.py`
   - Validates the merged 12-pair Ours-QIM JSONL/CSV log.
   - Retains only covers present for every gamma pair and payload.
   - Extracts gamma-stream, image, and explicit cross-channel features.
   - Creates balanced linked and shuffled tuples with the same image, cover,
     and payload in both classes.

2. `02_train_eval_linkability_warden.py`
   - Runs repeated cover-grouped train/test splits.
   - Evaluates image-only, text-only, cross-only, and text+image views.
   - Supports logistic regression and Extra Trees wardens.
   - Reports ACC, BACC, AUC, EER, Pe, 1-2Pe, and MCC.

3. `03_rank_linkability_candidates.py`
   - Ranks all 12 gamma pairs for each held-out stego object.
   - Reports Top-1, Top-3, Top-5, MRR, mean rank, and random baselines.

4. `04_make_linkability_tables_plots.py`
   - Generates LaTeX/CSV tables.
   - Generates PDF, PNG, and SVG AUC, advantage, and ranking figures.

## Dependencies

```bash
pip install numpy pandas pillow scikit-learn matplotlib tqdm joblib
```

## Leakage safeguards

Paths, pair IDs, hashes, cover IDs, and image IDs remain metadata. Model
feature sets are selected only by the prefixes `img_`, `gamma_`, and
`cross_`. `cover_id` is used only for grouped splitting.
