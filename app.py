"""
Streamlit Dynamic Label Generator
--------------------------------
Drop-in Streamlit app that:
- Accepts an Excel/CSV with one row per item
- Accepts side/top images (multiple file upload)
- Accepts a base template image (the label design)
- Auto-generates Code128 numeric barcode and DataMatrix (or QR fallback)
- Renders and lets you download generated labels (PNG and optionally a combined PDF)

How to run:
1. Create and activate a Python environment (recommended):
    python -m venv venv
    # Windows: venv\Scripts\activate
    # mac/linux: source venv/bin/activate
2. Install requirements:
    pip install streamlit pandas pillow python-barcode[images] segno pylibdmtx reportlab openpyxl

Notes on libraries:
- python-barcode (with ImageWriter) is used to create Code128 numeric barcodes.
- pylibdmtx is used for DataMatrix. If pylibdmtx is unavailable on your system you can remove it and the app will fallback to QR code generation using 'segno'.
- reportlab is used to combine PNGs into a PDF.

Template expectations:
- The template image should be a full label (PNG) where dynamic elements will be placed.
- The app uses coordinates defined in `LAYOUT` to place elements; you can tweak them to match your template.

Columns expected in the uploaded Excel/CSV (you can have any column names; map them in the UI):
- One column per dynamic field. Typical fields (example): TopLeft1, TopLeft2, MainTitle, FormType, ProductCode,
  GermanDescription, EnglishDescription, LotNumber, Quantity, BarcodeValue, DataMatrixValue, SideImage, TopImage, ColorHex

"""

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import os
import barcode
from barcode.writer import ImageWriter
import segno

# Try import pylibdmtx for DataMatrix generation
try:
    from pylibdmtx import pylibdmtx
    HAS_PYLIBDMTX = True
except Exception:
    HAS_PYLIBDMTX = False

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

st.set_page_config(layout="wide")
st.title("Dynamic Label Generator — Streamlit")

st.markdown("Upload your label template (PNG). Then upload Excel/CSV and images. Map columns to fields and generate labels.")

# Layout constants (pixels) — tweak as per your template
# These coordinates are example values for placing elements; you will likely need to adjust them.
LAYOUT = {
    'top_left_box': (40, 40, 380, 320),   # x1,y1,x2,y2
    'main_title_center': (540, 80),       # x,y center
    'form_type_center': (540, 150),
    'product_code_center': (540, 210),
    'desc_block': (120, 260, 1080, 360),  # x1,y1,x2,y2
    'lot_text': (120, 380),
    'qty_box_center': (980, 380),
    'side_image_pos': (120, 420),
    'top_image_pos': (320, 420),
    'qr_pos': (900, 420),
    'barcode_pos': (1160, 40),
}

# Fonts — you can change to local ttf files if available
try:
    FONT_BOLD = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    FONT_REG = ImageFont.truetype("DejaVuSans.ttf", 26)
    FONT_LARGE = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
except Exception:
    FONT_BOLD = ImageFont.load_default()
    FONT_REG = ImageFont.load_default()
    FONT_LARGE = ImageFont.load_default()


def generate_code128(value):
    """Generate a Code128 barcode PIL image"""
    CODE128 = barcode.get_barcode_class('code128')
    rv = CODE128(str(value), writer=ImageWriter())
    fp = io.BytesIO()
    rv.write(fp, {'module_width':0.2, 'module_height':15.0, 'font_size': 10, 'text_distance': 1})
    fp.seek(0)
    img = Image.open(fp)
    return img


def generate_datamatrix(value, size=200):
    """Generate a DataMatrix (using pylibdmtx) or fallback to QR (segno)
    Returns a PIL image.
    """
    v = str(value)
    if HAS_PYLIBDMTX:
        encoded = pylibdmtx.encode(v.encode('utf8'))
        img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
        img = img.resize((size, size), Image.NEAREST)
        return img
    else:
        # fallback to QR using segno
        qr = segno.make(v)
        buf = io.BytesIO()
        qr.save(buf, kind='png', scale=6)
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert('RGB')
        img = img.resize((size, size), Image.NEAREST)
        return img


def place_text(draw, text, pos, font, anchor="la", fill=(0,0,0)):
    draw.text(pos, text, font=font, fill=fill)


def generate_label_from_row(template_img, row, side_img=None, top_img=None, color_hex="#00a080"):
    # Copy template
    img = template_img.copy().convert('RGBA')
    draw = ImageDraw.Draw(img)

    # Color left box
    x1,y1,x2,y2 = LAYOUT['top_left_box']
    try:
        draw.rectangle([x1,y1,x2,y2], fill=color_hex)
    except Exception:
        draw.rectangle([x1,y1,x2,y2], fill="#00a080")

    # Place texts — using safe access to row
    def getcol(k):
        return str(row.get(k, '')) if pd.notna(row.get(k, '')) else ''

    # Top-left lines (two lines)
    tl1 = getcol('TopLeft1') or getcol('top_left_1') or getcol('TopLeft')
    tl2 = getcol('TopLeft2') or getcol('top_left_2')
    place_text(draw, tl1, (x1+20, y1+20), FONT_LARGE)
    place_text(draw, tl2, (x1+20, y1+80), FONT_LARGE)

    # Main center text
    main = getcol('MainTitle') or getcol('Main')
    place_text(draw, main, LAYOUT['main_title_center'], FONT_LARGE, )

    # Form/Type and product code
    form = getcol('FormType')
    code = getcol('ProductCode')
    place_text(draw, form, LAYOUT['form_type_center'], FONT_REG)
    place_text(draw, code, LAYOUT['product_code_center'], FONT_BOLD)

    # Descriptions
    gd = getcol('GermanDescription')
    ed = getcol('EnglishDescription')
    desc_block = LAYOUT['desc_block']
    w = desc_block[0]
    h = desc_block[1]
    place_text(draw, gd, (w+20, h+10), FONT_REG)
    place_text(draw, ed, (w+20, h+40), FONT_REG)

    # Lot and Qty
    lot = getcol('LotNumber')
    qty = getcol('Quantity')
    place_text(draw, lot, LAYOUT['lot_text'], FONT_REG)
    place_text(draw, str(qty), LAYOUT['qty_box_center'], FONT_LARGE)

    # Paste side & top images if provided
    if side_img:
        s = side_img.copy().convert('RGBA')
        s.thumbnail((180,180))
        img.paste(s, LAYOUT['side_image_pos'], s)
    if top_img:
        t = top_img.copy().convert('RGBA')
        t.thumbnail((180,180))
        img.paste(t, LAYOUT['top_image_pos'], t)

    # Generate datamatrix/qr
    dmval = getcol('DataMatrixValue') or getcol('QRCodeValue') or getcol('QR')
    if dmval:
        dm = generate_datamatrix(dmval, size=200)
        img.paste(dm, LAYOUT['qr_pos'])

    # Generate barcode
    bval = getcol('BarcodeValue') or getcol('barcode') or getcol('EAN')
    if bval:
        try:
            bc = generate_code128(bval)
            bc = bc.convert('RGBA')
            bc.thumbnail((220, 500))
            img.paste(bc, (LAYOUT['barcode_pos'][0]-bc.width//2, LAYOUT['barcode_pos'][1]), bc)
        except Exception as e:
            st.warning(f"Barcode generation failed: {e}")

    return img.convert('RGB')


# --- Streamlit UI ---
with st.sidebar:
    st.header('Uploads & Settings')
    template_file = st.file_uploader('Template PNG (your label template)', type=['png','jpg','jpeg'])
    sheet_file = st.file_uploader('Excel/CSV with data (one row per label)', type=['xlsx','csv'])
    side_images = st.file_uploader('Side view images (upload multiple)', type=['png','jpg','jpeg'], accept_multiple_files=True)
    top_images = st.file_uploader('Top view images (upload multiple)', type=['png','jpg','jpeg'], accept_multiple_files=True)
    out_format = st.selectbox('Output format', ['PNG', 'PDF', 'Both'])
    map_order = st.radio('Map images to rows by', ['Filename column in sheet', 'Upload order matches rows'])

if not template_file:
    st.info('Upload the template PNG to begin')
    st.stop()

if not sheet_file:
    st.info('Upload Excel/CSV with your data')
    st.stop()

# Load template
template_img = Image.open(template_file).convert('RGBA')

# Read sheet
if sheet_file.name.endswith('.csv'):
    df = pd.read_csv(sheet_file)
else:
    df = pd.read_excel(sheet_file)

st.write('Preview of data (first 5 rows)')
st.dataframe(df.head())

# Image mapping
if map_order == 'Filename column in sheet':
    st.markdown('Make sure your sheet has `SideImage` and `TopImage` columns that match uploaded filenames or paths.')

# Show uploaded images counts
st.write(f'Uploaded {len(side_images)} side images and {len(top_images)} top images')

# Let user map columns interactively
st.header('Map your sheet columns to label fields')
cols = df.columns.tolist()
col_map = {}
fields = ['TopLeft1','TopLeft2','MainTitle','FormType','ProductCode','GermanDescription','EnglishDescription','LotNumber','Quantity','BarcodeValue','DataMatrixValue','SideImage','TopImage','ColorHex']
for f in fields:
    col_map[f] = st.selectbox(f, options=['(none)'] + cols, index=0 if f not in cols else cols.index(f)+1)

if st.button('Generate labels'):
    st.info('Generating labels...')
    generated = []

    # Prepare image lists
    side_bufs = [Image.open(x) for x in side_images] if side_images else []
    top_bufs = [Image.open(x) for x in top_images] if top_images else []

    for idx, row in df.iterrows():
        # Build a dictionary for row with mapped keys
        mapped = {}
        for k,v in col_map.items():
            if v and v != '(none)':
                mapped[k] = row[v]
            else:
                mapped[k] = ''

        # Find images
        side_img = None
        top_img = None
        if map_order == 'Upload order matches rows':
            if idx < len(side_bufs):
                side_img = side_bufs[idx]
            if idx < len(top_bufs):
                top_img = top_bufs[idx]
        else:
            # match by filename in sheet
            sname = str(mapped.get('SideImage',''))
            tname = str(mapped.get('TopImage',''))
            for f in side_images:
                if f.name == sname:
                    side_img = Image.open(f)
                    break
            for f in top_images:
                if f.name == tname:
                    top_img = Image.open(f)
                    break

        # Color hex
        color_hex = mapped.get('ColorHex') or '#00a080'

        label_img = generate_label_from_row(template_img, mapped, side_img=side_img, top_img=top_img, color_hex=color_hex)
        generated.append(label_img)

    st.success(f'Generated {len(generated)} labels')

    # Show previews
    cols_display = st.columns(2)
    for i,img in enumerate(generated[:4]):
        with cols_display[i%2]:
            st.image(img, caption=f'Label {i+1}', use_column_width=True)

    # Prepare downloads
    if out_format in ('PNG','Both'):
        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            for i,img in enumerate(generated):
                b = io.BytesIO()
                img.save(b, format='PNG')
                b.seek(0)
                zf.writestr(f'label_{i+1:03d}.png', b.read())
        zip_buf.seek(0)
        st.download_button('Download PNGs (zip)', zip_buf, file_name='labels_pngs.zip', mime='application/zip')

    if out_format in ('PDF','Both'):
        # Create a multi-page PDF using reportlab
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        for img in generated:
            # Resize image to fit page
            w,h = img.size
            # convert PIL to temp buffer
            b = io.BytesIO()
            img.save(b, format='PNG')
            b.seek(0)
            # Draw at full page center
            from reportlab.lib.utils import ImageReader
            ir = ImageReader(b)
            page_w, page_h = letter
            # scale to fit within margins
            scale = min(page_w / w * 0.9, page_h / h * 0.9)
            draw_w = w * scale
            draw_h = h * scale
            x = (page_w - draw_w) / 2
            y = (page_h - draw_h) / 2
            c.drawImage(ir, x, y, draw_w, draw_h)
            c.showPage()
        c.save()
        pdf_buf.seek(0)
        st.download_button('Download PDF', pdf_buf, file_name='labels.pdf', mime='application/pdf')

    st.balloons()
