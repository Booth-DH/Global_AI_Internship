"""Export all sample code and real outputs in reading order to two Letter pages."""
from pathlib import Path
import ast
import json
import nbformat
import pymupdf
from pypdf import PdfReader
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, String, Number
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from build_notebook import ROOT, OUT, TITLE, ABSTRACT, CAPTION, FINDINGS, NOTES

NB = nbformat.read(OUT / 'code_sample.ipynb', as_version=4)
CELLS = {c.metadata.get('section', 'Imports'): c for c in NB.cells if c.cell_type == 'code'}
assert all(c.execution_count for c in CELLS.values())
assert not any(o.output_type == 'error' for c in CELLS.values() for o in c.outputs)
PREVIEW = OUT / 'pdf_preview'
PREVIEW.mkdir(exist_ok=True)
PDF = OUT / 'code_sample.pdf'
c = canvas.Canvas(str(PDF), pagesize=(612, 792), pageCompression=1)
c.setTitle(TITLE)
c.setAuthor('Donghang Zou')
MARGIN, WIDTH, FIGURE_WIDTH = 28, 556, 403.2
FONT, LEADING = 8.0, 9.5
INK = '#213b4a'
positions = []
printed_code = []


def text(value, x, y, size=8.5, font='Helvetica', color=INK):
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    c.drawString(x, y, value)


def paragraph(value, x, y, width, size=8.5, leading=9.5):
    words, row = value.split(), ''
    for word in words:
        trial = (row + ' ' + word).strip()
        if stringWidth(trial, 'Helvetica', size) > width:
            text(row, x, y, size)
            y -= leading
            row = word
        else:
            row = trial
    if row:
        text(row, x, y, size)
        y -= leading
    return y


def heading(value, x, y):
    text(value, x, y - 8, size=10.4, color='#135c70')
    return y - 21


def code(value, x, y, width):
    printed_code.append(value)
    for row in value.strip().splitlines():
        assert stringWidth(row, 'Courier', FONT) <= width, (len(row), row)
        xx = x
        for token, piece in lex(row, PythonLexer()):
            piece = piece.rstrip('\n')
            color = '#263a48'
            if token in Comment:
                color = '#376650'
            elif token in Keyword:
                color = '#245982'
            elif token in String:
                color = '#88522e'
            elif token in Number:
                color = '#72558e'
            text(piece, xx, y, FONT, 'Courier', color)
            xx += stringWidth(piece, 'Courier', FONT)
        y -= LEADING if row else 4
    return y


def stream(cell):
    return ''.join(o.text for o in cell.outputs if o.output_type == 'stream' and o.name == 'stdout').strip('\n')


def output(value, x, y, width):
    rows = value.splitlines()
    height = len(rows) * 8.5 + 8
    c.setFillColor(HexColor('#f0f4f6'))
    c.roundRect(x - 4, y - height + 5, width + 8, height, 3, fill=1, stroke=0)
    for row in rows:
        assert stringWidth(row.rstrip(), 'Courier', 8) <= width, row
        text(row.rstrip(), x, y - 4, 8, 'Courier')
        y -= 8.5
    return y - 12


def figure(name, x, y, width):
    img = ImageReader(str(OUT / 'figures' / name))
    w, h = img.getSize()
    height = width * h / w
    c.drawImage(img, x, y - height, width=width, height=height, mask='auto')
    return y - height


def footer(page):
    c.setStrokeColor(HexColor('#d5dfe4'))
    c.line(28, 24, 584, 24)
    text(f'Donghang Zou | UChicago ADS Code Sample | page {page} / 2', 28, 12, 8)


# One consistent full-width text and code frame on both pages.
text(TITLE, MARGIN, 765, 16)
x, y = MARGIN, 748
y = code(CELLS['Repository link'].source, x, y, WIDTH)
y = heading('1. Motivation and data', x, y)
y = paragraph(ABSTRACT, x, y, WIDTH)
y = heading('2. Setup and loading the data', x, y)
y = code(CELLS['Setup and loading the data'].source, x, y, WIDTH)
y = output(stream(CELLS['Setup and loading the data']), x, y - 2, WIDTH)
y = heading('3. Building the county snapshot', x, y)
y = code(CELLS['Building the county snapshot'].source, x, y, WIDTH)
y = heading('4. Measuring associations', x, y)
y = paragraph(NOTES['Measuring associations'][1], x, y, WIDTH)
y = code(CELLS['Measuring associations'].source, x, y, WIDTH)
y = output(stream(CELLS['Measuring associations']), x, y - 2, WIDTH)
y = code(CELLS['Drawing the associations'].source, x, y, WIDTH)
positions.append(('page1', y))
footer(1)
c.showPage()

x, y = MARGIN, 765
y = figure('fig_correlations.png', (612 - FIGURE_WIDTH) / 2, y - 2, FIGURE_WIDTH)
y = heading('5. Grouping and mapping counties', x, y)
y = paragraph(NOTES['Grouping county profiles'][1], x, y, WIDTH)
y = code(CELLS['Grouping county profiles'].source, x, y, WIDTH)
y = output(stream(CELLS['Grouping county profiles']), x, y - 2, WIDTH)
y = code(CELLS['Mapping the profiles'].source, x, y, WIDTH)
y = figure('fig_risk_tier_map.png', (612 - FIGURE_WIDTH) / 2, y, FIGURE_WIDTH)
y = paragraph(CAPTION, x, y - 3, WIDTH)
y = heading('6. Findings and limits', x, y)
y = paragraph(FINDINGS, x, y, WIDTH)
positions.append(('page2', y))
print(positions)
assert all(y >= 28 for _, y in positions), positions
footer(2)
c.save()

script_ast = ast.dump(ast.parse((OUT / 'code_sample.py').read_text()))
notebook_ast = ast.dump(ast.parse('\n\n'.join(cell.source for cell in CELLS.values())))
pdf_ast = ast.dump(ast.parse('\n\n'.join(printed_code)))
assert script_ast == notebook_ast == pdf_ast, 'Script, notebook and printed code differ'

reader = PdfReader(PDF)
assert len(reader.pages) == 2
assert all(tuple(p.mediabox) == (0, 0, 612, 792) for p in reader.pages)
full_text = '\n'.join(p.extract_text(extraction_mode='layout') for p in reader.pages)
assert full_text.count('https://github.com/Booth-DH/Global_AI_Internship') == 1
assert full_text.count(TITLE) == 1
assert 'def plot_correlations' in full_text and 'def plot_county_profiles' in full_text
assert full_text.index('def plot_correlations') < full_text.index('5. Grouping')
font_min = 100
for i, page in enumerate(pymupdf.open(PDF), 1):
    for block in page.get_text('dict')['blocks']:
        for row in block.get('lines', []):
            for span in row['spans']:
                if span['font'] == 'Courier':
                    font_min = min(font_min, span['size'])
                x0, y0, x1, y1 = span['bbox']
                assert x0 >= 20 and y0 >= 0 and x1 <= 592 and y1 <= 792, span
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(PREVIEW / f'page-{i}.png')
assert font_min >= 8
(PREVIEW / 'verification.json').write_text(json.dumps({
    'pages': 2, 'page_size': 'US Letter', 'minimum_code_font_pt': font_min,
    'code_leading_pt': LEADING, 'heading_space_above_pt': 8,
    'github_url_count': 1, 'title_count': 1, 'plotting_functions_visible': True, 'single_column': True, 'map_legend_font_pt': 9,
    'figure_width_pt': FIGURE_WIDTH, 'script_notebook_pdf_code_match': True, 'bottom_positions': positions}, indent=2) + '\n')
print(f'Saved {PDF}')
