Donghang Zou's UChicago ADS Code Sample

[Open the two-page PDF](code_sample.pdf) for the latest-snapshot analysis, or use the [executed notebook](code_sample.ipynb) and [matching Python script](code_sample.py) to run it. The full project is in the [county health notebook](../notebooks/county_health_analysis.ipynb).

The sample reads PLACES 2025 and SVI 2022 CSVs, prepares county records, compares diabetes and hypertension associations, and groups counties into descriptive profiles. All analysis and plotting functions are visible in the PDF. Both pages use a single full-width column, with the repository comment immediately below the title. Code is 8 points with 9.5-point line spacing; both figures have the same width and 9-point labels, including the map legend.

Food insecurity, housing insecurity, and transportation barriers have diabetes correlations of 0.937, 0.921, and 0.914. Physical inactivity changes from 0.873 on 2,956 counties to 0.850 on the 2,299-county common sample. The four profiles contain 765, 788, 508, and 238 counties, with a silhouette score of 0.203. These are overlapping descriptive profiles, not clinical categories or causal effects.

The [correlation chart](figures/fig_correlations.png) includes all 16 factors for both outcomes. The [map](figures/fig_risk_tier_map.png) uses Low, Moderate, High, and Highest; Highest renames the full analysis's Critical profile without changing assignments. The map caption distinguishes missing social-needs measures, missing disease outcomes, and Connecticut boundary mismatches.

The [correlation table](tables/correlation_summary.csv) preserves the unrounded sampling comparison. The [cluster table](tables/cluster_summary.csv) retains county counts and mean disease prevalence/SVI. The PDF prints the full rounded tier summary and silhouette, showing how mean disease prevalence and SVI rise across tiers. The correlation figure starts page 2 immediately after its plotting call at the end of page 1, with its legend above the plots. To retain two pages with more line spacing, setup and plotting calls are compact, prose is shorter, and the map uses a shorter canvas with a single-row legend. Both plotting functions return a figure that is saved by the calling code.

From the repository root, install the dependencies and rebuild the executed notebook and PDF:

```sh
python -m pip install -r requirements.txt
python code_sample/build/build_notebook.py
python code_sample/build/export_pdf.py
```

The notebook builder executes the public source, regenerates the figures and CSVs, and verifies the numerical results against the full analysis. The exporter checks that the script, notebook, and printed PDF code have the same Python syntax tree, confirms two US Letter pages and the minimum code font, and renders page previews for inspection. Reproducible build scripts are versioned; temporary execution files, verification artifacts, and PDF previews are ignored.

The standalone script can also be run with `python code_sample/code_sample.py`. It displays the analysis and regenerates the figures and correlation CSV. The notebook build additionally saves the full cluster summary. No files outside `code_sample/` are changed by these sample commands.
