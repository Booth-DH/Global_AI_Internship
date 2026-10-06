# County Health and Social Vulnerability

This Global AI internship project examines how county-level social conditions accompany diabetes and hypertension. It combines five CDC PLACES releases from 2021 to 2025 with CDC SVI 2020 and 2022, compares associations, and maps descriptive county profiles.

Read the [full analysis](notebooks/county_health_analysis.ipynb) for the investigation or the [two-page code sample](code_sample/code_sample.pdf) for a compact account with executed code and output.

## Structure

```text
requirements.txt
notebooks/
  county_health_analysis.ipynb
data/
  raw/
  processed/
outputs/
  figures/
  maps/
  tables/
reports/
  progress/
  final_report.html
  correlation_analysis.pdf
  policy_brief/
code_sample/
  code_sample.pdf
  code_sample.ipynb
  code_sample.py
  figures/
  tables/
  build/
```

The data folders contain cached source files, the merged county panel, measurement-year records, and county boundaries. Outputs contain the full notebook's results. Reports preserve internship write-ups; the policy brief folder contains both the LaTeX brief and the earlier internship health brief.

## Run

From the repository root, install the dependencies and execute the full analysis:

```sh
python -m pip install -r requirements.txt
jupyter nbconvert --execute --inplace --to notebook notebooks/county_health_analysis.ipynb
```

The notebook resolves the repository root whether launched there or inside `notebooks/`. It reads cached inputs and regenerates processed tables, figures, and interactive maps. Open the HTML files under `outputs/maps/` locally in a browser.

To regenerate the sample notebook, run its checks, and rebuild the PDF:

```sh
python code_sample/build/build_notebook.py
python code_sample/build/export_pdf.py
```

The [sample README](code_sample/README.md) explains the excerpt and its outputs.

## Main findings

In the latest snapshot, food insecurity, housing insecurity, and transportation barriers have diabetes correlations of 0.937, 0.921, and 0.914. Physical inactivity's correlation changes from 0.873 on 2,956 counties to 0.850 on the 2,299-county common sample.

The four profiles contain 765, 788, 508, and 238 counties. The sample calls the highest-burden profile Highest; the full analysis retains its original Critical label. Assignments are identical. The silhouette score is 0.203, so these are overlapping descriptive profiles, not validated clinical categories.

These are associations rather than causal effects or individual risk estimates. County means are unweighted, measurement years vary, and missing measures affect coverage. The latest PLACES release is from 2025, with 2022/2023 measurement years and SVI 2022. Complete-case clustering excludes 657 counties from the initial sample, while KY and PA lack the disease outcomes needed to enter it.
