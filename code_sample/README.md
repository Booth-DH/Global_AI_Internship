Donghang Zou's UChicago ADS Code Sample

[Open the two-page PDF](code_sample.pdf), [executed notebook](code_sample.ipynb), or [matching Python script](code_sample.py). This excerpt comes from the Global AI internship project; the [full analysis](../notebooks/county_health_analysis.ipynb) follows five PLACES releases and two SVI vintages.

The excerpt reads PLACES 2025 and SVI 2022 CSVs, prepares a county snapshot, compares diabetes and hypertension associations, and groups counties into descriptive profiles. All data preparation, analysis, and plotting code is visible. Both Letter pages use one full-width column. Code and printed output are 8.5 points with 9-point line spacing; both figures display 9-point labels at their native 5.6-inch width. The chart call and its output appear together on page 2.

Food insecurity, housing insecurity, and transportation barriers are the strongest diabetes correlates among the 16 tested factors, with correlations of 0.937, 0.921, and 0.914. Physical inactivity changes from 0.873 on 2,956 eligible counties to 0.850 on the 2,299-county common sample. Four profiles contain 765, 788, 508, and 238 counties, with a silhouette score of 0.203. Tier means are unweighted county averages: age-adjusted disease prevalence in percent and SVI percentile rank from 0 to 1. The profiles are ordered by mean standardized diabetes, hypertension, and overall SVI, so increasing tier averages are expected by construction.

The [correlation chart](figures/fig_correlations.png) shows pairwise Pearson correlations for all 16 factors and both outcomes. The [map](figures/fig_risk_tier_map.png) shows the contiguous U.S. using GeoPandas and the cached county boundaries. It retains the light-to-dark blue palette and darker neutral grey. Highest renames the full analysis's Critical profile without changing any assignments.

Grey means measures are missing or geographic identifiers do not match. Nine states lack social-needs measures, and KY/PA lack the two disease outcomes. Connecticut has nine current planning regions with profiles but no matching cached geometry; the eight older county shapes appear grey. The map displays 2,255 of the 2,299 profiles: 30 Alaska counties, five Hawaii counties, and those nine Connecticut regions are omitted. These coverage checks run during the notebook build.

These associations can inform follow-up, but they are descriptive, not causal or individual clinical risk estimates. Measurement years differ and profiles overlap. PLACES uses model-based small-area estimates with shared demographic predictors, so part of an observed association may reflect common model inputs. The [CDC methodology](https://www.cdc.gov/places/methodology/) describes those inputs; this is a reason for caution, not a measured correction to the correlations reported here.

From the repository root, install the dependencies and rebuild:

```sh
python -m pip install -r requirements.txt
python code_sample/build/build_notebook.py
python code_sample/build/export_pdf.py
```

The source uses Python 3.10 or later and GeoPandas 1.1.x. The notebook builder executes it in a fresh kernel, checks both outcomes, all county assignments, sampling counts, silhouette, and map coverage against the existing data, and writes the unrounded [correlation](tables/correlation_summary.csv) and [cluster](tables/cluster_summary.csv) tables. The standalone `python code_sample/code_sample.py` displays the analysis and regenerates the figures; CSV export and reference-data verification belong to the notebook build.

The PDF exporter checks that the script, notebook, and code sent to the PDF renderer have the same syntax tree, confirms two Letter pages and code font size, and renders page previews for inspection. Build scripts are versioned; temporary execution files, verification artifacts, and page previews are ignored. Rebuild commands change files only within `code_sample/`.
