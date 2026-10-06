from pathlib import Path
import hashlib, json, os, sys, time
root=Path(__file__).resolve().parents[2]
out=root/'code_sample'
work=out/'_validation'
# Keep execution caches and all newly written files inside the permitted folder.
for name,sub in [('MPLCONFIGDIR','mpl'),('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','runtime'),('XDG_CACHE_HOME','cache')]:
 folder=work/sub;folder.mkdir(exist_ok=True)
 os.environ[name]=str(folder)
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
kernel_dir=work/'kernels'/'code-sample'
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Code sample validation','language':'python'}))
nb=nbformat.read(out/'code_sample.ipynb',as_version=4)
assert next(c.source for c in nb.cells if c.cell_type=='code').splitlines()[0]=='# GitHub: https://github.com/Booth-DH/Global_AI_Internship'
for c in nb.cells:
 if c.cell_type=='code': compile(c.source,'sample-cell','exec')
verification_code='''
import re
from pandas.testing import assert_frame_equal
blueprint = json.loads((ROOT / 'assignment.ipynb').read_text())
expected_panel = pd.read_csv(ROOT / 'processed_data/panel_county_health.csv', dtype={'FIPS': 'string'})
expected_panel = expected_panel.loc[expected_panel.release_year.eq(2025)].set_index('FIPS')
assert_frame_equal(counties[features].sort_index(), expected_panel[features].sort_index(), check_names=False, check_dtype=False)
expected_clusters = pd.read_csv(ROOT / 'results/clustered_counties_2023.csv', dtype={'FIPS': 'string'}).set_index('FIPS')
assert complete.tier.astype('string').sort_index().equals(expected_clusters.risk_tier.astype('string').sort_index())
expected_sampling = pd.read_csv(ROOT / 'results/correlation_sampling_comparison.csv', index_col='factor')
for factor in shown:
    assert np.isclose(correlations['DIABETES'].loc[factor, 'r'], expected_sampling.loc[factor, 'r_pairwise'], atol=1e-12, rtol=0)
    assert np.isclose(common.loc[factor, 'r'], expected_sampling.loc[factor, 'r_common'], atol=1e-12, rtol=0)
    assert correlations['DIABETES'].loc[factor, 'n'] == expected_sampling.loc[factor, 'n_pairwise']
    assert common.loc[factor, 'n'] == expected_sampling.loc[factor, 'n_common']
feature_cell = next(c for c in blueprint['cells'] if 'THRESHOLD = 0.40' in ''.join(c['source']))
feature_output = ''.join(''.join(o.get('text', [])) for o in feature_cell['outputs'])
expected_features = re.findall(r'^  ([A-Z][A-Z0-9_]+)\\s+.+\\((?:avg \\|r\\||outcome variable)', feature_output, flags=re.M)
assert features == expected_features, (features, expected_features)
# Compare both outcomes with the blueprint's actual displayed correlation table.
expected_full = expected_panel[FACTORS + TARGETS].corr().loc[FACTORS, TARGETS]
assert np.allclose(r.loc[FACTORS, TARGETS], expected_full, atol=1e-12, rtol=0)
validation_cell = next(c for c in blueprint['cells'] if 'K_RANGE    = range(2, 9)' in ''.join(c['source']))
validation_text = ''.join(''.join(o.get('text', [])) for o in validation_cell['outputs'])
expected_silhouette = float(re.search(r'^4\\s+[\\d,]+\\s+([0-9.]+)', validation_text, flags=re.M).group(1))
score = metrics.silhouette_score(z, model.labels_)
assert round(score, 4) == expected_silhouette
excluded = counties.index.difference(complete.index)
assert len(excluded) == 657
assert len(set(f[:2] for f in excluded)) == 9
assert counties.index.str.fullmatch(r'\\d{5}').all()
assert counties.index.is_unique and complete.index.is_unique
assert (comparison.n == common.n).sum() == 3
# Every map polygon's hole must have opposite winding for compound paths.
def signed_area(ring):
    xy = np.array(ring)
    return np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])
for feature in geo['features']:
    geometry = feature['geometry']
    polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
    for polygon in polygons:
        assert all(signed_area(polygon[0]) * signed_area(hole) < 0 for hole in polygon[1:])
# Persist actual computed tables; no outputs are hand-entered into the sample.
comparison.to_csv(OUT / 'correlation_summary.csv')
summary.to_csv(OUT / 'cluster_summary.csv')
common.to_csv(OUT / '_validation/common_correlations.csv')
complete[['tier', *features]].to_csv(OUT / '_validation/cluster_assignments.csv')
report = {'status': 'passed', 'blueprint_cells': len(blueprint['cells']),
          'counties': len(counties), 'clustered_counties': len(complete), 'excluded_counties': len(excluded),
          'excluded_states': len(set(f[:2] for f in excluded)), 'features': features,
          'silhouette_k4': score, 'tier_counts': summary['counties'].to_dict(),
          'contiguous_counties_without_geometry': len(unmapped),
          'correlations': comparison.to_dict(orient='index'),
          'checks': ['full 16-feature order matches saved notebook output',
                     'all selected county values match saved latest panel',
                     'all county tier assignments match saved results',
                     'both outcome correlations match the blueprint panel',
                     'sampling comparison matches saved output to 1e-12',
                     'K=4 silhouette matches notebook output to 4 decimal places',
                     'excluded counties and states verified', 'county polygon holes preserved'],
          'pdf_pages': None, 'pdf_note': 'Executed notebook only, as requested.'}
(OUT / '_validation/validation.json').write_text(json.dumps(report, indent=2) + '\\n')
print('Verification passed against the read-only blueprint and saved outputs.')
'''
nb.cells.append(nbformat.v4.new_code_cell(verification_code))
km=KernelManager(kernel_name='code-sample',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(work/'kernels')]))
start=time.monotonic()
def progress(cell,cell_index,**kwargs):
 if cell.cell_type=='code':print(f'Executing cell {cell_index+1}/{len(nb.cells)}: {cell.source.splitlines()[0][:85]}',flush=True)
client=NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(root)}},on_cell_start=progress)
try:
 client.execute()
except Exception:
 nbformat.write(nb,work/'sample.failed.ipynb')
 raise
nb.cells.pop()  # Keep verification outside the analysis excerpt; retain its report.
nbformat.write(nb,out/'code_sample.ipynb')
manifest=json.loads((work/'protected_files.sha256.json').read_text())
# Finder metadata is included in the local baseline but absent from a fresh clone.
manifest={name: digest for name,digest in manifest.items()
          if Path(name).name != '.DS_Store' or (root/name).exists()}
changed=[name for name,expected in manifest.items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected]
assert not changed, f'Protected files changed: {changed}'
report=json.loads((work/'validation.json').read_text())
report.update(protected_files_unchanged=len(manifest),execution_seconds=round(time.monotonic()-start,2),
              code_cells=sum(c.cell_type=='code' for c in nb.cells))
(work/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'COMPLETE: 6 code cells executed; {len(manifest)} protected files unchanged; {time.monotonic()-start:.1f}s',flush=True)
