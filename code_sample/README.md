# County health analysis: a compact walkthrough

[Open the executed notebook](code_sample.ipynb) to follow the latest snapshot from raw county CSVs to correlations and mapped profiles. The accompanying [Python script](code_sample.py) contains the same analysis. Short explanations retain the full notebook's step-by-step style, including the decisions that affect interpretation.

The full project compares five CDC PLACES releases. This excerpt focuses on the **2025 release and SVI 2022**, preserving the original analytical rules while keeping the pipeline in six executable code cells.

## Reading the notebook

The opening block imports the libraries and defines the outcomes and candidate factors. Section 1 reads the CSVs with explicit string, integer, and floating-point types. `prepare_county_snapshot()` preserves FIPS leading zeros, removes national aggregates, selects age-adjusted estimates, reshapes measures, treats SVI −999 as missing, and validates the county join. Suppressed estimates remain missing.

Section 2 uses `compare_correlations()` for both diabetes and hypertension. Its pairwise and common-sample policies make the effect of missing data visible: the first uses all available counties for each factor; the second holds the county sample fixed. Section 3 plots four leading diabetes correlates alongside their hypertension associations.

Section 4 standardizes the same 16 features as the full notebook and fits K-Means with `random_state=42` and `n_init=10`. Ordered categorical profile labels make the summary and map consistent. Section 5 renders a static county map from the cached GeoJSON, including multipart polygons and interior holes.

## What the excerpt finds

- **2,956 counties** have both disease outcomes and overall SVI.
- Food insecurity, housing insecurity, and transportation barriers correlate with diabetes at **0.937, 0.921, and 0.914**, respectively, on **2,299 counties**.
- Physical inactivity's diabetes correlation changes from **0.873 on 2,956 counties** to **0.850 on the common 2,299-county sample**.
- The four profiles contain **765 Low, 788 Moderate, 508 High, and 238 Critical** counties. The K=4 silhouette score is **0.203**; these are overlapping descriptive profiles, not validated clinical risk groups.
- Complete-case filtering excludes **657 counties (22.2%) across nine states**. Nine clustered Connecticut planning regions cannot be matched to the cached boundaries. Grey polygons have no matched profile.

The findings describe county-level associations rather than causal relationships. Disease summaries are unweighted county means. PLACES measures have different reference years (2022/2023 here), and SVI is from 2022. The map displays the contiguous U.S., while clustering uses all eligible counties.

## Scope of the excerpt

The latest-release preparation, sampling comparison, feature selection, and clustering reproduce the full analysis. Multi-release charts, extensive schema exploration, the full K search, and interactive map controls remain in [assignment.ipynb](../assignment.ipynb). The heatmap displays four factors, while the feature-selection calculation uses all 16 candidates. The static map uses geographic coordinates with an approximate display aspect and the existing boundaries.

This version is an **executed notebook**. PDF export and verification of a two-page layout are deferred.

## Saved files

| File | Purpose |
|---|---|
| [code_sample.ipynb](code_sample.ipynb) | Executed code, explanations, tables, and figures |
| [code_sample.py](code_sample.py) | Matching standalone analysis source |
| [fig_correlations.png](fig_correlations.png) | Correlations for four factors and both outcomes |
| [fig_risk_tier_map.png](fig_risk_tier_map.png) | Static county-profile map |
| [correlation_summary.csv](correlation_summary.csv) | Unrounded pairwise/common results and sample sizes |
| [cluster_summary.csv](cluster_summary.csv) | County counts and mean outcomes/SVI by profile |
| [_validation/validation.json](_validation/validation.json) | Numerical verification results |
| [_validation/common_correlations.csv](_validation/common_correlations.csv) | Common-sample correlations |
| [_validation/cluster_assignments.csv](_validation/cluster_assignments.csv) | County profiles and the 16 clustering inputs |
| [_validation/protected_files.sha256.json](_validation/protected_files.sha256.json) | Baseline hashes of the original notebook, data, and results |

## Execution and verification

Run from the repository root or `code_sample/`. The analysis reads the cached inputs and writes only inside this folder. The environment used for verification is Python 3.10, pandas 2.0.3, NumPy 1.26.4, scikit-learn 1.7.2, Matplotlib 3.10.9, and seaborn 0.13.2. Notebook execution also requires `nbformat`, `nbclient`, and `ipykernel`; installation instructions are in the [project README](../README.md#reproduce-the-analysis).

From the repository root:

```sh
# Run the standalone analysis and regenerate its figures and result tables.
python code_sample/code_sample.py

# Execute the notebook in a fresh kernel and check against the full analysis.
python code_sample/_validation/execute_and_verify.py
```

The verification compares all selected county values, both outcome correlations, the sampling comparison, feature order, silhouette score, and every county profile with the original notebook and saved results. File hashes confirm that the original inputs and outputs have not changed. Temporary kernel files and caches are ignored by Git.

If editing the script, rebuild the notebook with the retained explanatory text, then execute it to restore real outputs:

```sh
python code_sample/_validation/build_notebook.py
python code_sample/_validation/execute_and_verify.py
```
