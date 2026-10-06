# GitHub: https://github.com/Booth-DH/Global_AI_Internship
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches, path
from sklearn import cluster, preprocessing, metrics
ROOT = Path.cwd() if Path('raw_data').is_dir() else Path.cwd().parent
OUT = ROOT / 'code_sample'
outcomes = ['DIABETES', 'BPHIGH']
factors = ('LPA OBESITY CSMOKING ACCESS2 DEPRESSION SLEEP BINGE RPL_THEMES '
           'RPL_THEME1 RPL_THEME2 RPL_THEME3 RPL_THEME4 FOODINSECU '
           'HOUSINSECU LACKTRPT LONELINESS').split()
svi_cols = [f for f in factors if f.startswith('RPL')]
# 1. Preserve identifiers and missing values before the county-level join.
places = pd.read_csv(ROOT / 'raw_data/places_county_2025.csv', low_memory=False,
    dtype={'locationid': 'string', 'year': 'int16', 'data_value': 'float64'})
svi = pd.read_csv(ROOT / 'raw_data/svi_county_2022.csv', dtype={'FIPS': 'string'},
                  usecols=['FIPS', *svi_cols])
def prepare_county_snapshot(places, svi):
    health = places.loc[(places.stateabbr != 'US') &
                       (places.data_value_type == 'Age-adjusted prevalence')].copy()
    health['FIPS'] = health.locationid.str.zfill(5)
    # Latest observations stay intact: do not backfill suppressed estimates.
    health = health.sort_values('year').drop_duplicates(['FIPS', 'measureid'], keep='last')
    wide = health.pivot(index='FIPS', columns='measureid', values='data_value')
    svi = svi.assign(FIPS=svi.FIPS.str.zfill(5)).replace(-999, np.nan).set_index('FIPS')
    return wide.join(svi, how='inner', validate='one_to_one').dropna(
        subset=[*outcomes, 'RPL_THEMES'])
counties = prepare_county_snapshot(places, svi)
# 2. Common samples hold geography fixed; pairwise samples retain available data.
def compare_correlations(data, outcome, factors, sample_policy='pairwise'):
    assert sample_policy in {'pairwise', 'common'}
    values = data[[outcome, *factors]]
    if sample_policy == 'common':
        values = values.dropna()
    rows = []
    for factor in factors:
        pairs = values[[outcome, factor]].dropna()
        rows.append((factor, pairs[factor].corr(pairs[outcome]), len(pairs)))
    return pd.DataFrame(rows, columns=['factor', 'r', 'n']).set_index('factor')
results = {y: compare_correlations(counties, y, factors) for y in outcomes}
r = pd.DataFrame({y: result.r for y, result in results.items()})
shown = ['FOODINSECU', 'HOUSINSECU', 'LACKTRPT', 'LPA']
common = compare_correlations(counties, 'DIABETES', shown, sample_policy='common')
comparison = r.loc[shown].join(results['DIABETES'].n).join(
    common.rename(columns={'r': 'common_r', 'n': 'common_n'}))
print(f'{len(counties):,} counties; pairwise r / n; common_r and common_n for diabetes:')
print(comparison.round(3).to_string())
# %% PAGE 2
# 3. Scale the original 16 features; order clusters by disease and SVI anchors.
strength = r.abs().mean(axis=1).sort_values(ascending=False)
features = outcomes + strength[strength >= 0.40].index.tolist()
complete = counties.dropna(subset=features).copy()
z = preprocessing.StandardScaler().fit_transform(complete[features])
model = cluster.KMeans(n_clusters=4, random_state=42, n_init=10).fit(z)
anchors = [features.index(f) for f in [*outcomes, 'RPL_THEMES']]
order = np.argsort(model.cluster_centers_[:, anchors].mean(axis=1))
tiers = ['Low', 'Moderate', 'High', 'Critical']
complete['tier'] = pd.Categorical(pd.Series(model.labels_, index=complete.index)
    .map(dict(zip(order, tiers))), categories=tiers, ordered=True)
summary = complete.groupby('tier', observed=True).agg(counties=('DIABETES', 'size'),
    diabetes=('DIABETES', 'mean'), hypertension=('BPHIGH', 'mean'), SVI=('RPL_THEMES', 'mean'))
print(f'{len(features)} features; {len(complete):,}/{len(counties):,} counties; '
      f'K=4 silhouette = {metrics.silhouette_score(z, model.labels_):.3f}')
print(summary.round(3).to_string())
# 4. Draw contiguous U.S. counties; grey means no matched profile.
colors = dict(zip(tiers, ['#27ae60', '#f39c12', '#e67e22', '#c0392b']))
lookup = complete.tier.astype('string').map(colors).to_dict()
geo = json.loads((ROOT / 'processed_data/us_counties.geojson').read_text())
fig, ax = plt.subplots(figsize=(7, 3.5))
for county in geo['features']:
    fips, g = county['id'], county['geometry']
    if fips[:2] in {'02', '15', '72'}:
        continue
    polygons = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
    for polygon in polygons:
        shape = path.Path.make_compound_path(*(path.Path(ring, closed=True) for ring in polygon))
        ax.add_patch(patches.PathPatch(shape, facecolor=lookup.get(fips, '#d3d3d3'),
                                       edgecolor='white', linewidth=0.12))
ax.set(xlim=(-125, -66), ylim=(24, 50), aspect=1.25)
ax.axis('off')
ax.legend(handles=[patches.Patch(color=c, label=t) for t, c in
    {**colors, 'No matched tier': '#d3d3d3'}.items()], loc='upper center',
    bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=12, frameon=False)
fig.savefig(OUT / 'pdf_preview/county_map.png', dpi=220, bbox_inches='tight')
