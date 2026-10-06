"""Build, execute, and verify the sample notebook from the public Python source."""
from pathlib import Path
import json
import os
import re
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code_sample'
TITLE = "Donghang Zou's UChicago ADS Code Sample"
ABSTRACT = (
    'This sample is an excerpt from my Global AI internship project; the full code and analysis are available at the GitHub link above. '
    'Chronic disease burden varies across U.S. counties, and its social correlates can guide public health resources. '
    'The project joins CDC PLACES 2021 to 2025 with CDC SVI 2020 and 2022 in a county panel, '
    'compares diabetes and hypertension correlates, and maps K-means profiles. '
    'Food, housing, and transportation insecurity are the strongest diabetes correlates (r of 0.937, 0.921, and 0.914).'
   
)

CAPTION = (
    'Grey marks nine states without social-needs measures (CO, FL, OR, SD, TN, TX, VT, WA, WY), KY and PA '
    'without diabetes or hypertension estimates, and nine CT planning regions without cached geometry.'
)
CLUSTER_NOTE = 'Unweighted county means: prevalence (%), SVI percentile rank (0 to 1).'
FINDINGS = (
    'On the same 2,299 counties, social-needs factors stay the strongest diabetes correlates while inactivity weakens, '
    'and higher-burden profiles cluster in the Deep South. These patterns could help prioritize county follow-up, '
    'but they are descriptive, not causal: PLACES values are model-based estimates with shared demographic inputs '
    'that can inflate correlations, measurement years differ, and profiles overlap.'
)

NOTES = {
    'Building the county snapshot': ('3. Building the county snapshot', ''),
    'Measuring associations': ('4. Measuring associations',
        'Pairwise r uses every available county; the common sample holds counties fixed across factors.'),
    'Drawing the associations': ('', ''),
    'Grouping county profiles': ('5. Grouping and mapping counties',
        'K-means groups counties with similar standardized profiles; the table shows unweighted county means (prevalence %, SVI rank).'),
    'Mapping the profiles': ('', ''),
}


def build():
    parts = re.split(r'^# %% (.+)\n', (OUT / 'code_sample.py').read_text(), flags=re.M)
    nb = nbformat.v4.new_notebook()
    first_line, setup = parts[0].split('\n', 1)
    url_cell = nbformat.v4.new_code_cell(first_line)
    url_cell.metadata['section'] = 'Repository link'
    setup_cell = nbformat.v4.new_code_cell(setup.strip())
    setup_cell.metadata['section'] = 'Setup and loading the data'
    nb.cells = [nbformat.v4.new_markdown_cell('# ' + TITLE), url_cell,
                nbformat.v4.new_markdown_cell('### 1. Motivation and data\n\n' + ABSTRACT),
                nbformat.v4.new_markdown_cell('### 2. Setup and loading the data'), setup_cell]

    for title, body in zip(parts[1::2], parts[2::2]):
        heading, note = NOTES[title]
        if heading:
            nb.cells.append(nbformat.v4.new_markdown_cell('### ' + heading + '\n\n' + note))
        cell = nbformat.v4.new_code_cell(body.strip())
        cell.metadata['section'] = title
        nb.cells.append(cell)
    nb.cells.append(nbformat.v4.new_markdown_cell(CAPTION + '\n\n### 6. Findings and limits\n\n' + FINDINGS))
    nb.metadata.update(kernelspec={'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                       language_info={'name': 'python'})
    return nb


VERIFY = '''
from pandas.testing import assert_frame_equal
panel = pd.read_csv(ROOT / 'data/processed/panel_county_health.csv', dtype={'FIPS': 'string'})
latest = panel.loc[panel.release_year.eq(2025)].set_index('FIPS')
assert_frame_equal(counties[features].sort_index(), latest[features].sort_index(),
                   check_names=False, check_dtype=False)
old = pd.read_csv(ROOT / 'outputs/tables/clustered_counties_2023.csv', dtype={'FIPS': 'string'})
expected = old.set_index('FIPS').risk_tier.replace({'Critical': 'Highest'}).astype('string')
assert complete.tier.astype('string').sort_index().equals(expected.sort_index())
sampling = pd.read_csv(ROOT / 'outputs/tables/correlation_sampling_comparison.csv', index_col='factor')
for actual, saved in [('r_pair', 'r_pairwise'), ('n_pair', 'n_pairwise'),
                      ('r_common', 'r_common'), ('n_common', 'n_common')]:
    assert np.allclose(comparison[actual], sampling.loc[shown, saved], atol=1e-12, rtol=0)
assert np.allclose(correlations, latest[factors + outcomes].corr().loc[factors, outcomes], atol=1e-12, rtol=0)
assert common.n.eq(2299).all()
assert common.r.nlargest(3).index.tolist() == ['FOODINSECU', 'HOUSINSECU', 'LACKTRPT']
assert len(features) == 16 and len(counties) == 2956 and len(complete) == 2299
assert summary['counties'].tolist() == [765, 788, 508, 238]
assert round(silhouette_score(scaled, model.labels_), 3) == .203
assert not places.loc[places.stateabbr.isin(['CO','FL','OR','SD','TN','TX','VT','WA','WY'])].measureid.isin(
    ['FOODINSECU','HOUSINSECU','LACKTRPT','LONELINESS']).any()
for state in ['KY', 'PA']:
    measures = set(places.loc[places.stateabbr.eq(state)].measureid)
    assert len(measures) == 5 and not measures.intersection(outcomes)
summary.to_csv(OUT / 'tables/cluster_summary.csv')
comparison.to_csv(OUT / 'tables/correlation_summary.csv')
print('Verified both outcomes, sampling comparison, all county profiles, and silhouette.')
'''


def main():
    nb = build()
    nb.cells.append(nbformat.v4.new_code_cell(VERIFY))
    with tempfile.TemporaryDirectory(prefix='county-sample-') as tmp:
        work = Path(tmp)
        for env in ['MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR']:
            directory = work / env.lower()
            directory.mkdir()
            os.environ[env] = str(directory)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        kernel_dir = work / 'kernels' / 'sample'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Sample', 'language': 'python'}))
        km = KernelManager(kernel_name='sample', kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[str(work / 'kernels')]))
        NotebookClient(nb, km=km, timeout=600, resources={'metadata': {'path': str(ROOT)}}).execute()
    print(nb.cells[-1].outputs[0].text)
    nb.cells.pop()
    nbformat.write(nb, OUT / 'code_sample.ipynb')
    print('Saved executed code_sample.ipynb')


if __name__ == '__main__':
    main()
