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
    'County social conditions help identify where chronic disease and unmet needs coincide. '
    'The larger project uses five CDC PLACES releases from 2021 to 2025 and CDC SVI 2020 and 2022. '
    'It cleans and stacks a county panel, correlates social and behavioral factors with diabetes and '
    'hypertension, and maps K-means profiles. This excerpt rebuilds the latest snapshot from raw CSVs; '
    'the full project is on GitHub, linked in the first code line. '
    'Food, housing, and transportation insecurity are the strongest diabetes correlates '
    '(r = 0.937, 0.921, and 0.914).'
)
CAPTION = (
    'Grey includes 657 eligible counties in CO, FL, OR, SD, TN, TX, VT, WA, and WY missing social-needs '
    'measures, plus KY and PA, which lack diabetes and hypertension estimates. '
    'Nine CT planning regions lack cached geometry.'
)
FINDINGS = (
    'The social-needs associations exceed physical inactivity, whose diabetes r changes from 0.873 '
    'on 2,956 counties to 0.850 on the 2,299-county common sample. Higher-burden profiles concentrate '
    'in the Deep South. These associations are not causal, and measurement years differ '
    '(PLACES 2022/2023 and SVI 2022). Profiles are descriptive and overlapping (silhouette 0.203).'
)
NOTES = {
    'Loading and typing the data': ('2. Loading and typing the data', ''),
    'Building the county snapshot': ('3. Building the county snapshot',
                                     'Age-adjusted measures form one row per county.'),
    'Measuring associations': ('4. Measuring associations',
        'The table compares diabetes samples; the chart ranks all 16 factors for both outcomes.'),
    'Drawing the associations': ('', ''),
    'Grouping county profiles': ('5. Grouping and mapping counties',
        'Mean absolute r selects factors; profile means are unweighted disease percentages and SVI ranks.'),
    'Mapping the profiles': ('', ''),
}


def build():
    parts = re.split(r'^# %% (.+)\n', (OUT / 'code_sample.py').read_text(), flags=re.M)
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell('# ' + TITLE + '\n\n### 1. Motivation and data\n\n' + ABSTRACT),
                nbformat.v4.new_code_cell(parts[0].strip())]
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
assert np.allclose(r, latest[factors + yvars].corr().loc[factors, yvars], atol=1e-12, rtol=0)
assert len(features) == 16 and len(counties) == 2956 and len(complete) == 2299
assert summary['counties'].tolist() == [765, 788, 508, 238]
assert round(silhouette_score(z, model.labels_), 3) == .203
assert not p.loc[p.stateabbr.isin(['CO','FL','OR','SD','TN','TX','VT','WA','WY'])].measureid.isin(
    ['FOODINSECU','HOUSINSECU','LACKTRPT','LONELINESS']).any()
for state in ['KY', 'PA']:
    measures = set(p.loc[p.stateabbr.eq(state)].measureid)
    assert len(measures) == 5 and not measures.intersection(yvars)
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
