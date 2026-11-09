"""
Build the 12-slide project deck: presentations/ML_Antenna_Array_Presentation.pptx

Design spec
-----------
* 16:9 widescreen (13.333 x 7.5 in)
* dark theme   background #0f172a, panels #1e293b, text #ffffff, accent #38bdf8
* Calibri throughout
* slide number bottom-right on every slide
* title slide 40pt bold, content body 20-24pt
* every figure embedded (not linked)

The builder is parameterised by THEME/LIGHT_FIGURES so the same script can be
re-run after the figures get their dark-theme upgrade (see Phase 3 of the
polish pass); only the two constants at the top change.

Usage
-----
    python presentations/build_deck.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
BG = RGBColor(0x0F, 0x17, 0x2A)          # dark navy
PANEL = RGBColor(0x1E, 0x29, 0x3B)       # slate panel
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x38, 0xBD, 0xF8)      # sky blue
MUTED = RGBColor(0x94, 0xA3, 0xB8)       # slate-400 secondary text
BORDER = RGBColor(0x33, 0x41, 0x55)

# Figures currently on disk are light-background PNGs, so they get a thin accent
# frame to read as intentional insets on the dark canvas. Flip to False once the
# dark-theme figures are in place.
LIGHT_FIGURES = True
FRAME_COLOR = ACCENT if LIGHT_FIGURES else BORDER

FONT = "Calibri"
DECK_W, DECK_H = 13.333, 7.5

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
OUT_DIR = REPO_ROOT / "presentations"
OUT_PATH = OUT_DIR / "ML_Antenna_Array_Presentation.pptx"


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def set_bg(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def add_text(
    slide,
    left,
    top,
    width,
    height,
    text: str,
    size: int = 20,
    bold: bool = False,
    color: RGBColor = WHITE,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    italic: bool = False,
    line_spacing: float = 1.0,
    space_after: int = 0,
):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return box


def add_bullets(slide, left, top, width, height, items, size=22, gap=18, color=WHITE):
    """Bulleted list with an accent marker run followed by the body run."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.12
        p.space_after = Pt(gap)
        marker = p.add_run()
        marker.text = "\u25b8  "
        marker.font.name = FONT
        marker.font.size = Pt(size)
        marker.font.bold = True
        marker.font.color.rgb = ACCENT
        body = p.add_run()
        body.text = item
        body.font.name = FONT
        body.font.size = Pt(size)
        body.font.color.rgb = color
    return box


def add_rect(
    slide, left, top, width, height, fill=None, line=None, line_w=1.0, rounded=False, radius=0.08
):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(
        shape_type, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if rounded:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def add_callout(
    slide, left, top, width, height, text, size=21, bold=True, color=ACCENT, align=PP_ALIGN.CENTER
):
    add_rect(slide, left, top, width, height, fill=PANEL, line=ACCENT, line_w=1.25, rounded=True)
    add_text(
        slide, left + 0.22, top + 0.10, width - 0.44, height - 0.20,
        text, size=size, bold=bold, color=color, align=align,
        anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05,
    )


def image_aspect(path: Path) -> float:
    with Image.open(path) as im:
        return im.width / im.height


def add_figure(slide, filename: str, left, top, max_w, max_h, frame=True):
    """Fit an image inside the given box, preserving aspect ratio, centred."""
    path = FIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing figure: {path}")
    ar = image_aspect(path)
    w = max_w
    h = w / ar
    if h > max_h:
        h = max_h
        w = h * ar
    l = left + (max_w - w) / 2
    t = top + (max_h - h) / 2
    pic = slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))
    if frame:
        add_rect(slide, l, t, w, h, fill=None, line=FRAME_COLOR, line_w=0.75)
    return pic, (l, t, w, h)


def add_table(slide, left, top, width, height, data, col_widths, font_size=16,
              header_fill=PANEL, header_color=ACCENT, body_fill=BG):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(
        rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    table = shape.table
    table.first_row = False          # turn off the built-in style's header banding
    table.horz_banding = False
    total = sum(col_widths)
    for i, cw in enumerate(col_widths):
        table.columns[i].width = Emu(int(Inches(width) * (cw / total)))
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = body_fill
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(data[r][c])
            run.font.name = FONT
            run.font.size = Pt(font_size)
            run.font.bold = r == 0
            run.font.color.rgb = header_color if r == 0 else WHITE
    return shape


def add_slide_number(slide, number: int) -> None:
    add_text(
        slide, DECK_W - 1.05, DECK_H - 0.52, 0.75, 0.32,
        str(number), size=12, color=MUTED, align=PP_ALIGN.RIGHT,
    )


def new_slide(prs, number: int, title: str | None = None, title_size: int = 34):
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    set_bg(slide)
    add_rect(slide, 0, 0, 0.14, DECK_H, fill=ACCENT)     # accent spine
    if title:
        add_text(slide, 0.62, 0.38, 12.05, 0.92, title, size=title_size, bold=True)
        add_rect(slide, 0.62, 1.30, 1.55, 0.055, fill=ACCENT)
    add_slide_number(slide, number)
    return slide


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #
def slide_title(prs):
    s = new_slide(prs, 1)
    add_rect(s, 0, 0, 0.14, DECK_H, fill=ACCENT)
    add_text(
        s, 0.90, 1.55, 11.55, 2.35,
        "ML-Based Prediction of Antenna Array Gain and Bandwidth "
        "to Avoid Grating Lobes",
        size=40, bold=True, line_spacing=1.06,
    )
    add_rect(s, 0.92, 4.18, 2.30, 0.07, fill=ACCENT)
    add_text(
        s, 0.90, 4.45, 11.2, 0.75,
        "A Machine Learning Surrogate for Antenna Array Design",
        size=24, color=ACCENT,
    )
    add_text(s, 0.90, 5.60, 9.0, 0.42, "Sreekar Balagoni", size=18, bold=True)
    add_text(
        s, 0.90, 6.02, 9.0, 0.40,
        "B.E. ECE, Vasavi College of Engineering", size=16, color=MUTED,
    )
    add_text(s, 0.90, 6.44, 9.0, 0.38, "November 2026", size=16, color=MUTED)
    return s


def slide_problem(prs):
    s = new_slide(prs, 2, "Why ML for Antenna Arrays?")
    add_bullets(
        s, 0.85, 1.75, 11.65, 3.30,
        [
            "Traditional EM simulation (ADS, CST) requires hours per design iteration",
            "Antenna array gain and bandwidth depend on N, spacing, and scan angle "
            "in complex ways",
            "Grating lobes are a critical failure mode \u2014 spatial aliasing destroys "
            "beam quality",
        ],
        size=22, gap=20,
    )
    add_callout(
        s, 0.85, 5.38, 11.65, 1.22,
        "Goal: Train ML models to predict gain, bandwidth, and grating lobe onset \u2014 instantly",
        size=21,
    )
    return s


def slide_physics(prs):
    s = new_slide(prs, 3, "Grating Lobe Physics")

    # prominent threshold equation
    add_rect(s, 0.85, 1.58, 7.35, 1.15, fill=PANEL, line=ACCENT, line_w=1.25, rounded=True)
    add_text(
        s, 0.95, 1.68, 7.15, 0.95,
        "d  \u2264  \u03bb / (1 + |sin \u03b8max|)",
        size=32, bold=True, color=ACCENT, align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )

    add_table(
        s, 0.85, 3.02, 7.35, 2.00,
        [
            ["Condition", "Maximum spacing"],
            ["Broadside only (\u03b8 = 0\u00b0)", "d < \u03bb"],
            ["Full scan (\u03b8 = 90\u00b0)", "d < \u03bb/2"],
            ["\u00b150\u00b0 scan", "d < 0.566\u03bb"],
        ],
        col_widths=[1.45, 1.0], font_size=17,
    )

    add_text(
        s, 0.85, 5.35, 11.6, 0.70,
        "This equation defines the binary classification boundary used in Phase 4.",
        size=20, color=MUTED, italic=True,
    )
    return s


def slide_dataset(prs):
    s = new_slide(prs, 4, "Dataset Generation \u2014 ULA Analytical Model")

    add_table(
        s, 0.85, 1.68, 7.30, 2.45,
        [
            ["Parameter", "Swept values"],
            ["N (elements)", "2, 4, 8, 16, 32"],
            ["d/\u03bb (spacing)", "0.3 \u2192 1.2, step 0.1"],
            ["\u03b8 (scan angle)", "0\u00b0, 15\u00b0, 30\u00b0, 45\u00b0, 60\u00b0"],
        ],
        col_widths=[1.15, 1.55], font_size=17,
    )

    add_callout(s, 8.55, 1.68, 3.95, 1.15, "250 total\ncombinations", size=21)
    add_callout(s, 8.55, 2.98, 3.95, 1.15, "5 \u00d7 10 \u00d7 5\nfull factorial", size=19)

    add_text(
        s, 0.85, 4.45, 11.65, 0.40,
        "Feature engineering", size=20, bold=True, color=ACCENT,
    )
    add_bullets(
        s, 0.85, 4.92, 11.65, 1.10,
        [
            "log_N = log\u2082(N)   \u2014 linearises the aperture term",
            "gl_margin = threshold \u2212 d/\u03bb   \u2014 signed distance to the grating-lobe boundary",
        ],
        size=19, gap=8,
    )

    add_text(
        s, 0.85, 6.28, 11.65, 0.55,
        "Generated analytically from ULA physics \u2014 no EM simulator required.",
        size=18, italic=True, color=MUTED,
    )
    return s


def slide_eda(prs):
    s = new_slide(prs, 5, "Exploratory Data Analysis")

    add_figure(s, "gain_vs_spacing.png", 0.55, 1.52, 5.95, 3.95)
    add_figure(s, "grating_lobe_distribution.png", 6.85, 1.52, 5.95, 3.95)

    add_text(
        s, 0.55, 5.62, 5.95, 1.10,
        "Gain rises ~3 dB per doubling of N; a \u22120.5 dB/\u03bb penalty\n"
        "appears beyond d/\u03bb = 0.5 (mutual coupling).",
        size=16, color=MUTED, line_spacing=1.12,
    )
    add_text(
        s, 6.85, 5.62, 5.95, 1.10,
        "Class balance 115 / 135 (1.17 : 1) \u2014\n"
        "mild enough for accuracy to be a meaningful metric.",
        size=16, color=MUTED, line_spacing=1.12,
    )
    return s


def slide_models(prs):
    s = new_slide(prs, 6, "Machine Learning Pipeline")

    # left column - regression
    add_text(s, 0.70, 1.55, 5.85, 0.40, "Regression (9 models)", size=22, bold=True, color=ACCENT)
    add_rect(s, 0.70, 1.98, 1.25, 0.045, fill=ACCENT)
    add_text(
        s, 0.70, 2.18, 5.85, 3.40,
        "Linear Regression\nRidge Regression\nDecision Tree\nRandom Forest\n"
        "Extra Trees\nGradient Boosting\nXGBoost\nKNN (k=5)\nGaussian Process",
        size=18, color=WHITE, line_spacing=1.15,
    )

    # right column - classification
    add_text(s, 6.85, 1.55, 5.85, 0.40, "Classification (6 models)", size=22, bold=True, color=ACCENT)
    add_rect(s, 6.85, 1.98, 1.25, 0.045, fill=ACCENT)
    add_text(
        s, 6.85, 2.20, 5.85, 2.40,
        "Logistic Regression\nRandom Forest\nGradient Boosting\nXGBoost\n"
        "SVM (RBF, probability=True)\nKNN (k=5)",
        size=18, color=WHITE, line_spacing=1.22,
    )

    # metrics band
    add_rect(s, 0.70, 5.72, 6.00, 1.12, fill=PANEL, line=BORDER, line_w=0.75, rounded=True)
    add_text(s, 0.90, 5.86, 5.60, 0.34, "Regression metrics", size=15, bold=True, color=ACCENT)
    add_text(s, 0.90, 6.22, 5.60, 0.45, "R\u00b2  \u00b7  Variance Score  \u00b7  MAE  \u00b7  MSE  \u00b7  RMSE", size=15, color=WHITE)

    add_rect(s, 6.85, 5.72, 6.00, 1.12, fill=PANEL, line=BORDER, line_w=0.75, rounded=True)
    add_text(s, 7.05, 5.86, 5.60, 0.34, "Classification metrics", size=15, bold=True, color=ACCENT)
    add_text(s, 7.05, 6.22, 5.60, 0.45, "Accuracy  \u00b7  F1  \u00b7  ROC-AUC  \u00b7  Confusion matrix", size=15, color=WHITE)

    add_text(
        s, 0.70, 6.98, 11.9, 0.34,
        "All models: 80/20 split (random_state=42) \u00b7 StandardScaler fitted on train only",
        size=13, color=MUTED,
    )
    return s


def slide_gain(prs):
    s = new_slide(prs, 7, "Gain Prediction Results")
    add_figure(s, "model_comparison_gain.png", 0.45, 1.50, 7.95, 4.62)

    add_callout(s, 8.62, 1.62, 4.25, 1.30,
                "Best: Linear Regression\nR\u00b2 = 0.9997", size=20)
    add_text(
        s, 8.62, 3.18, 4.25, 2.90,
        "Gain follows 10\u00b7log\u2081\u2080(N) \u2014 a near-linear relationship \u2014 "
        "so linear regression is not merely adequate but optimal.",
        size=18, color=WHITE, line_spacing=1.20,
    )
    add_text(
        s, 8.62, 6.05, 4.25, 0.80,
        "RMSE 0.071 dBi \u00b7 MAE 0.054 dBi",
        size=15, color=MUTED, italic=True,
    )
    return s


def slide_bw(prs):
    s = new_slide(prs, 8, "Bandwidth Prediction Results")
    add_figure(s, "model_comparison_bw.png", 0.45, 1.50, 7.95, 4.62)

    add_callout(s, 8.62, 1.62, 4.25, 1.30,
                "Best: Gaussian Process\nR\u00b2 = 0.8075", size=20)
    add_text(
        s, 8.62, 3.18, 4.25, 2.90,
        "Bandwidth carries 20 MHz of irreducible noise, so 0.81 is the "
        "theoretical ceiling \u2014 not a modelling shortfall.",
        size=18, color=WHITE, line_spacing=1.20,
    )
    add_text(
        s, 8.62, 6.05, 4.25, 0.80,
        "RMSE 19.6 MHz \u00b7 clears the 0.70 spec bar",
        size=15, color=MUTED, italic=True,
    )
    return s


def slide_validation(prs):
    s = new_slide(prs, 9, "Model Validation \u2014 Predicted vs. Simulated")
    add_figure(s, "predicted_vs_actual_gain.png", 0.55, 1.58, 4.85, 4.32)
    add_figure(s, "feature_importance.png", 6.30, 1.58, 6.85, 4.32)

    add_text(
        s, 0.55, 6.05, 4.85, 0.85,
        "Test configurations track y = x\nacross 4.7 \u2013 17.3 dBi.",
        size=16, color=MUTED, line_spacing=1.12,
    )
    add_text(
        s, 6.30, 6.05, 6.85, 0.85,
        "Gain is driven by N / log_N; bandwidth by gl_margin (0.85 importance) \u2014 "
        "the physics is recovered.",
        size=16, color=MUTED, line_spacing=1.12,
    )
    return s


def slide_classifier(prs):
    s = new_slide(prs, 10, "Grating Lobe Classification Results")
    add_figure(s, "grating_lobe_boundary.png", 0.50, 1.50, 6.05, 4.10)
    add_figure(s, "confusion_matrix.png", 6.85, 1.50, 3.30, 2.72)

    add_table(
        s, 10.35, 1.50, 2.55, 1.60,
        [["Metric", "RF"], ["Accuracy", "1.0000"], ["ROC-AUC", "1.0000"], ["F1", "1.0000"]],
        col_widths=[1.25, 1.0], font_size=12,
    )

    add_text(
        s, 6.85, 4.42, 6.05, 1.15,
        "Deterministic boundary learned perfectly \u2014 the classifier "
        "rediscovers the physics equation.",
        size=17, color=WHITE, line_spacing=1.15,
    )
    add_text(
        s, 0.50, 5.85, 6.05, 0.85,
        "Orange = grating lobe, green = safe.\nLabels hug the theoretical curve.",
        size=15, color=MUTED, line_spacing=1.12,
    )
    add_text(
        s, 0.50, 6.72, 12.4, 0.35,
        "Random Forest \u00b7 Gradient Boosting \u00b7 XGBoost all score 1.000 \u00b7 "
        "SVM 0.960 \u00b7 KNN 0.960 \u00b7 Logistic 0.880",
        size=13, color=MUTED,
    )
    return s


def slide_summary(prs):
    s = new_slide(prs, 11, "Summary")

    add_table(
        s, 0.70, 1.72, 6.45, 2.10,
        [
            ["Target", "Best model", "Score"],
            ["Gain (R\u00b2)", "Linear Regression", "0.9997"],
            ["Bandwidth (R\u00b2)", "Gaussian Process", "0.8075"],
            ["Grating lobe (acc.)", "Random Forest", "1.0000"],
        ],
        col_widths=[1.25, 1.45, 0.75], font_size=15,
    )

    add_text(s, 7.55, 1.68, 5.20, 0.40, "Conclusions", size=21, bold=True, color=ACCENT)
    add_rect(s, 7.55, 2.10, 1.15, 0.045, fill=ACCENT)
    add_bullets(
        s, 7.55, 2.34, 5.20, 3.80,
        [
            "ML can replace repeated ADS sweeps for array design exploration",
            "Grating lobe onset is learnable from (N, d/\u03bb, \u03b8) with perfect accuracy",
            "gl_margin is the most informative feature \u2014 it encodes the physics directly",
        ],
        size=17, gap=18,
    )

    add_text(
        s, 0.70, 4.30, 6.45, 1.00,
        "37/37 verification checks passed",
        size=19, bold=True, color=ACCENT,
    )
    add_text(
        s, 0.70, 4.72, 6.45, 1.10,
        "Labels re-derived from the analytical formula independently\n"
        "of the generator; results reproducible from a clean clone.",
        size=15, color=MUTED, line_spacing=1.12,
    )
    return s


def slide_future(prs):
    s = new_slide(prs, 12, "Future Work & References")

    add_text(s, 0.70, 1.52, 5.70, 0.38, "Future work", size=20, bold=True, color=ACCENT)
    add_rect(s, 0.70, 1.94, 1.05, 0.045, fill=ACCENT)
    add_bullets(
        s, 0.70, 2.16, 5.70, 3.90,
        [
            "Replace analytical model with full-wave ADS/CST EM simulation data",
            "Extend to 2D planar arrays and mutual coupling effects",
            "Hide gl_margin to force classifier to learn boundary from raw parameters",
        ],
        size=17, gap=18,
    )

    add_text(s, 6.85, 1.52, 6.00, 0.38, "References", size=20, bold=True, color=ACCENT)
    add_rect(s, 6.85, 1.94, 1.05, 0.045, fill=ACCENT)

    refs = [
        "[1]  M. A. Haque et al., \u201cMachine learning-based technique for gain "
        "prediction of mm-wave miniaturized 5G MIMO slotted antenna array,\u201d "
        "Scientific Reports, 2025.  DOI: 10.1038/s41598-024-84182-w",
        "[2]  M. A. Haque et al., \u201cMachine learning-based technique for gain and "
        "resonance prediction of mid band 5G Yagi antenna,\u201d Scientific Reports, "
        "13:12590, 2023.  PMC10400634",
        "[3]  P. Jain, P. K. Sahoo, A. D. Khaleel, A. J. A. Al-Gburi, \u201cEnhanced "
        "prediction of metamaterial antenna parameters using advanced machine learning "
        "regression models,\u201d Progress In Electromagnetics Research C, 146, 1\u201312, "
        "2024.  DOI: 10.2528/PIERC24060901",
    ]
    top = 2.18
    for ref in refs:
        add_text(s, 6.85, top, 6.00, 1.15, ref, size=12, color=WHITE, line_spacing=1.16)
        top += 1.28

    add_text(
        s, 0.70, 6.30, 12.15, 0.45,
        "Repository: github.com/Sreekarji/ml-array  \u00b7  All code seeded to 42 for reproducibility",
        size=14, color=MUTED, italic=True,
    )
    return s


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(DECK_W)
    prs.slide_height = Inches(DECK_H)

    builders = [
        slide_title,
        slide_problem,
        slide_physics,
        slide_dataset,
        slide_eda,
        slide_models,
        slide_gain,
        slide_bw,
        slide_validation,
        slide_classifier,
        slide_summary,
        slide_future,
    ]
    for fn in builders:
        fn(prs)
        print(f"  built slide {len(prs.slides.__iter__.__self__._sldIdLst):>2} - {fn.__name__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PATH))
    print(f"\n[saved] {OUT_PATH}")
    print(f"[info ] {len(prs.slides)} slides, {DECK_W}x{DECK_H} in (16:9)")
    return OUT_PATH


if __name__ == "__main__":
    build()
