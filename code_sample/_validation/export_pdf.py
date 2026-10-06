"""Execute the condensed source, verify equivalence, and render a two-page PDF."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import textwrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'code_sample'
PREVIEW = OUT / 'pdf_preview'
PREVIEW.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '_validation/mpl')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.chdir(ROOT)

import pandas as pd
import pymupdf
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, String, Number
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth

source = (OUT / 'pdf_source.py').read_text()
sections = source.split('# %% PAGE 2\n')
namespace = {'__name__': '__main__'}
outputs = []
for section in sections:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(section, str(OUT / 'pdf_source.py'), 'exec'), namespace)
    outputs.append('\n'.join(row.rstrip() for row in captured.getvalue().strip().splitlines()))
(PREVIEW / 'executed_output.txt').write_text('\n\n'.join(outputs) + '\n')

# Compare unrounded results, so compact printing cannot hide a changed analysis.
for name, filename in [('comparison', 'correlation_summary.csv'),
                       ('summary', 'cluster_summary.csv')]:
    expected = pd.read_csv(OUT / filename, index_col=0)
    actual = namespace[name].copy()
    actual.index = actual.index.astype('object')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                  check_exact=False, atol=1e-12, rtol=0)
expected = pd.read_csv(ROOT / 'results/clustered_counties_2023.csv',
                       dtype={'FIPS': 'string'}).set_index('FIPS').risk_tier
assert namespace['complete'].tier.astype('string').sort_index().equals(
    expected.astype('string').sort_index())
manifest = json.loads((OUT / '_validation/protected_files.sha256.json').read_text())
for name, digest in manifest.items():
    file = ROOT / name
    if file.name == '.DS_Store' and not file.exists():
        continue
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, name

PDF = OUT / 'code_sample.pdf'
c = canvas.Canvas(str(PDF), pagesize=(612, 792), pageCompression=1)
c.setTitle('County Health and Social Vulnerability | Donghang Zou')
c.setAuthor('Donghang Zou')
INK = '#173047'
TEAL = '#17676a'
MARGIN = 30
WIDTH = 552
CODE_SIZE = 9
LEADING = 10


def line(text, y, size=9.5, color=INK, font='Helvetica', x=MARGIN):
    assert stringWidth(text, font, size) <= WIDTH + .01, text
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    c.drawString(x, y, text)


def paragraph(text, y, width=110, size=9.3, leading=12):
    for row in textwrap.wrap(text, width=width):
        line(row, y, size)
        y -= leading
    return y


def code(text, y):
    for row in text.rstrip().splitlines():
        assert stringWidth(row, 'Courier', CODE_SIZE) <= WIDTH, row
        x = MARGIN
        for token, value in lex(row, PythonLexer()):
            value = value.rstrip('\n')
            color = '#33424f'
            if token in Comment:
                color = '#326651'
            elif token in Keyword:
                color = '#1b578e'
            elif token in String:
                color = '#864a25'
            elif token in Number:
                color = '#72558e'
            c.setFillColor(HexColor(color))
            c.setFont('Courier', CODE_SIZE)
            c.drawString(x, y, value)
            x += stringWidth(value, 'Courier', CODE_SIZE)
        y -= LEADING
    return y


def output(text, y):
    rows = text.splitlines()
    height = len(rows) * 10.8 + 16
    c.setFillColor(HexColor('#f0f4f6'))
    c.roundRect(MARGIN - 6, y - height + 7, WIDTH + 12, height, 5, fill=1, stroke=0)
    for row in rows:
        line(row, y - 5, size=9, font='Courier')
        y -= 10.8
    return y - 13


def footer(page):
    c.setStrokeColor(HexColor('#d5dfe4'))
    c.line(MARGIN, 27, 582, 27)
    line('GLOBAL AI INTERNSHIP  |  County-level population health', 15, 8, '#596a76')
    c.drawRightString(582, 15, f'{page} / 2')


line('County health and social vulnerability', 762, 17, font='Helvetica-Bold')
line('Donghang Zou  |  Global AI Internship  |  Python', 744, 10, TEAL)
y = paragraph('Which social conditions accompany diabetes and hypertension? The full project combines five CDC '
              'PLACES releases (2021-2025) with SVI 2020/2022. This excerpt rebuilds the latest snapshot.',
              726, width=112, size=9.5)
y = code(sections[0], y - 8)
y = output(outputs[0], y - 3)
y = paragraph('FOODINSECU = food insecurity; HOUSINSECU = housing insecurity; LACKTRPT = transportation '
              'barriers; LPA = physical inactivity. BPHIGH is hypertension. The three social-needs factors '
              'share 2,299 counties; restricting inactivity to those same counties changes its correlation.',
              y - 3, width=112)
assert y >= 35, ('page 1 overflow', y)
footer(1)
c.showPage()

line('From associations to descriptive county profiles', 762, 13, font='Helvetica-Bold')
y = code(sections[1], 743)
y = output(outputs[1], y - 1)
line('Unweighted county means: disease prevalence (%); SVI rank (0-1).', y, 9)
y -= 7
map_image = ImageReader(str(PREVIEW / 'county_map.png'))
w, h = map_image.getSize()
map_height = 200
map_width = map_height * w / h
c.drawImage(map_image, (612 - map_width) / 2, y - map_height,
            width=map_width, height=map_height, mask='auto')
y -= map_height + 11
# These final comments remain selectable text alongside executable code and output.
conclusions = [
    '# Social needs have the strongest associations; highest-burden profiles cluster in the Deep South.',
    '# Associations are not causal; PLACES measures span 2022/2023 and SVI is from 2022.',
    '# K=4 describes overlapping profiles, not clinical risk. Complete cases exclude 657 counties',
    '# across nine states; nine clustered CT planning regions lack matching cached map geometry.',
]
for row in conclusions:
    line(row, y, 9, '#326651', 'Courier')
    y -= 10.5
assert y >= 30, ('page 2 overflow', y)
footer(2)
c.save()

doc = pymupdf.open(PDF)
assert len(doc) == 2
min_font = 100
for number, page in enumerate(doc, 1):
    assert tuple(page.rect) == (0, 0, 612, 792)
    for block in page.get_text('dict')['blocks']:
        for row in block.get('lines', []):
            for span in row['spans']:
                if span['font'].startswith('Courier'):
                    min_font = min(min_font, span['size'])
                x0, y0, x1, y1 = span['bbox']
                assert x0 >= 0 and y0 >= 0 and x1 <= 612 and y1 <= 792, span
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(PREVIEW / f'page-{number}.png')
assert min_font >= 9
report = {'pages': len(doc), 'page_size': 'US Letter (612 x 792 pt)',
          'minimum_code_font_pt': min_font, 'results_match_notebook': True,
          'county_assignments_match_original': True, 'protected_files_unchanged': True,
          'source': 'pdf_source.py', 'rendered_previews': ['page-1.png', 'page-2.png']}
(PREVIEW / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print(f'Saved {PDF}')
