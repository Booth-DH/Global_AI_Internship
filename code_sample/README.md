Donghang Zou's UChicago ADS Code Sample

This is an excerpt from a larger project: https://github.com/Booth-DH/Global_AI_Internship.

[Open the two-page PDF](code_sample.pdf) for the compact analysis, or [the executed notebook](code_sample.ipynb) for all code and output. The [Python script](code_sample.py) runs the same pipeline. The full project combines CDC PLACES releases from 2021 to 2025 with CDC SVI 2020 and 2022 to study county-level chronic disease and social conditions.

The excerpt starts from PLACES 2025 and SVI 2022 CSVs, prepares a county snapshot, compares diabetes and hypertension associations under explicit sampling policies, and groups counties into descriptive profiles. Its custom preparation and correlation functions retain the decisions that matter for interpretation, including identifiers, missing estimates, and county coverage. The PDF prints the analytical code; the script and notebook also contain the complete plotting helpers and conventional plotting imports.

The [correlation figure](fig_correlations.png) shows all 16 factors, ranked by diabetes correlation, with hypertension alongside. The [county map](fig_risk_tier_map.png) shows Low, Moderate, High, and Highest profiles. Highest replaces the original notebook's Critical label; the assignments have not changed. The unrounded [correlation summary](correlation_summary.csv) and [cluster summary](cluster_summary.csv) are saved directly by the code.

Food insecurity, housing insecurity, and transportation barriers correlate with diabetes at 0.937, 0.921, and 0.914. Physical inactivity's correlation is 0.873 on 2,956 counties and 0.850 on the 2,299-county common sample. The profiles contain 765, 788, 508, and 238 counties, with a silhouette score of 0.203. These are overlapping, descriptive county profiles. Associations are not causal, and the measures have different reference years.

Grey map areas reflect unavailable measures. CO, FL, OR, SD, TN, TX, VT, WA, and WY lack the four social-needs measures, excluding 657 counties from the initial 2,956-county sample. KY and PA have just five measures and no diabetes or hypertension estimates, so their counties never enter that sample. Loving County, TX has suppressed disease estimates, accounting for the difference between 658 raw counties in the nine states and 657 eligible counties. Nine Connecticut planning regions lack matching cached geometry.

Run the notebook from the repository root or from this folder, using a Python environment with pandas, NumPy, Matplotlib, and scikit-learn. For the standalone script, run this command from the repository root:

```sh
python code_sample/code_sample.py
```

The analysis reads the cached source CSVs and county boundaries and writes only inside this folder. Notebook execution additionally requires Jupyter and ipykernel. The verified environment used Python 3.10, pandas 2.0.3, NumPy 1.26.4, Matplotlib 3.10.9, and scikit-learn 1.7.2.

The PDF contains exactly two US Letter pages, with 8.5-point code and 8-point tabular output. Both pages were rendered and visually checked. All original numerical results and county assignments match after the label rename, and the original notebook, source data, processed data, and results remain unchanged. Local build scripts, verification records, and page previews are excluded from version control.
