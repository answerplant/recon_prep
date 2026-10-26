"""Render the NBS fine-tuning time estimates as a PDF (reportlab, Calibri from C:/Windows/Fonts).

    python docs/make_estimates_pdf.py docs/NBS_fine-tuning_estimates.pdf
"""

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = sys.argv[1]
pdfmetrics.registerFont(TTFont("Calibri", "C:/Windows/Fonts/calibri.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Bold", "C:/Windows/Fonts/calibrib.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Italic", "C:/Windows/Fonts/calibrii.ttf"))
pdfmetrics.registerFontFamily("Calibri", normal="Calibri", bold="Calibri-Bold", italic="Calibri-Italic", boldItalic="Calibri-Bold")

INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
ACCENT = colors.HexColor("#1F4E79")
RULE = colors.HexColor("#CBD2D9")
HEAD_BG = colors.HexColor("#E4EBF2")
NOTE_BG = colors.HexColor("#F5F7FA")

base = ParagraphStyle("base", fontName="Calibri", fontSize=10.5, leading=14, textColor=INK, alignment=TA_LEFT)
title = ParagraphStyle("title", parent=base, fontName="Calibri-Bold", fontSize=20, leading=24, textColor=ACCENT, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", parent=base, fontSize=11, leading=15, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=base, fontName="Calibri-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("body", parent=base, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=base, leftIndent=12, bulletIndent=2, spaceAfter=3)
cell = ParagraphStyle("cell", parent=base, fontSize=9.5, leading=12)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Calibri-Bold")
small = ParagraphStyle("small", parent=base, fontSize=8.5, leading=11, textColor=MUTED)


def P(text, style=body):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths, bold_first_col=True, highlight_rows=()):
    data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(c), cell_b if r == 0 or (bold_first_col and i == 0) else cell) for i, c in enumerate(row)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), NOTE_BG))
    t.setStyle(TableStyle(style))
    return t


def note_box(paragraphs):
    t = Table([[p] for p in paragraphs], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Calibri", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "GLiNER2.5-Decide fine-tuning on the National Bibliography of Scotland: time estimates")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


story = [
    P("Fine-tuning GLiNER2.5-Decide on the National Bibliography of Scotland", title),
    P("Time estimates for Getty AAT / ULAN / TGN reconciliation &nbsp;·&nbsp; 26 October 2026 &nbsp;·&nbsp; "
      "Planning estimates: not yet measured on NBS data or EIDF hardware", subtitle),

    P("Summary", h1),
    *bullets([
        "<b>Training a Yale-scale model</b> (about 25,000 labelled decisions, 2 epochs) takes about "
        "<b>1.5–2.5 hours on one EIDF H100 or H200</b>, 4–5 hours on an EIDF A100, and "
        "<b>12–20 hours on an RTX 5060 Ti 16 GB</b>.",
        "<b>Finding candidates is no longer a bottleneck.</b> The local Getty index answers in 5–30 ms per unique "
        "term (measured), so even the whole bibliography takes under 3 hours on a laptop. The live Getty service "
        "would take over a week.",
        "<b>Labelling is the real constraint.</b> 25,000 decisions is too many to review by hand; plan on "
        "language-model labels with a human-checked validation set, as the Yale team did.",
        "<b>Getty only partly covers a bibliography.</b> Places, genres and physical-description terms match well. "
        "Most authors are not in ULAN, so person terms will mostly be <i>none</i>.",
    ]),

    P("Basis of the estimates", h1),
    *bullets([
        "<b>Reference run:</b> the Yale model GLiNER2.5-Decide-linkedart-recon trained on 24,318 decisions for "
        "2 epochs (3,040 steps, effective batch 16, bf16, gradient checkpointing) in 283 minutes on one NVIDIA L40S. "
        "That is about <b>0.35 s per decision per epoch</b>, including periodic evaluation, or about 12 minutes per "
        "1,000 decisions over 2 epochs.",
        "<b>Other GPUs</b> are scaled from that run by their bf16 compute and memory bandwidth relative to the L40S. "
        "These are not benchmarks; run <font face='Courier'>train.py --preflight-only</font> on the target "
        "hardware to confirm memory and speed.",
        "<b>NBS size:</b> 507,089 records (version 6, CC-BY 4.0, MARCXML). Terms per record are assumed at about 4 "
        "(place of publication, subjects, genre, names, physical description) and have not been measured.",
        "<b>Likely upper bound:</b> bibliographic records are shorter than museum descriptions, and Getty-only "
        "candidate lists are shorter (12.8 candidates on average in the sample, against about 22 in the Yale data), "
        "so each NBS example should cost less than a Yale example.",
    ]),

    KeepTogether([
        P("Hardware", h1),
        table([
            ["GPU", "Memory", "Speed vs L40S", "Batch setting", "Notes"],
            ["NVIDIA L40S (reference)", "48 GB", "1×", "8 × 2 accumulation", "Yale's run: 21 GiB peak on the longest batch"],
            ["EIDF H200", "141 GB", "~2.2–3.3×", "8 × 2", "16 GPUs on the service"],
            ["EIDF H100", "80 GB", "~2–3×", "8 × 2", "152 GPUs; usually the longest queue"],
            ["EIDF A100 80 GB / 40 GB", "80 / 40 GB", "~1.0–1.3×", "8 × 2", "80 + 56 GPUs"],
            ["EIDF A100 MIG slice 3g.20gb", "20 GB", "~0.35–0.55×", "4 × 4", "About half an A100 40 GB"],
            ["EIDF A100 MIG slice 1g.5gb", "5 GB", "Not viable", "—", "A full fine-tune needs ~5.4 GB before any data"],
            ["RTX 5060 Ti 16 GB", "16 GB", "~0.25–0.4×", "2–4 × 4–8", "448 GB/s; ~47 bf16 TFLOPS (third-party estimate)"],
        ], [42, 20, 24, 27, 57], highlight_rows=(1,)),
    ]),

    KeepTogether([
        P("Training time (2 epochs, one GPU)", h1),
        table([
            ["Labelled decisions", "Roughly", "L40S (reference)", "EIDF H100 / H200", "EIDF A100", "EIDF MIG 3g.20gb", "RTX 5060 Ti 16 GB"],
            ["2,000", "Pilot, ~500 records", "~25 min", "~7–12 min", "~18–23 min", "~45–65 min", "~1–1.5 h"],
            ["25,000", "Yale scale, ~6–8k records", "~5 h", "<b>~1.5–2.5 h</b>", "~4–5 h", "~9–14 h", "<b>~12–20 h</b>"],
            ["100,000", "Large, ~25–30k records", "~20 h", "~6–10 h", "~15–20 h", "~1.5–2.3 days", "~2–3.3 days"],
            ["~2,000,000", "Whole NBS at ~4 terms/record", "~16 days", "~5–8 days", "~12–16 days", "~4–6.5 weeks", "~6–9 weeks"],
        ], [24, 33, 22, 23, 21, 24, 23]),
        Spacer(1, 4),
        P("Training on the whole bibliography is not recommended: the Yale model reached 0.926 accuracy from 24,000 "
          "decisions, and gains beyond about 100,000 are likely to be small. If needed, EIDF's default quota of 12 GPUs "
          "allows multi-GPU training (supported by gliner2's trainer); 8 H100s would bring the whole-NBS row to about a day.", small),
    ]),

    KeepTogether([
        P("Data preparation and running the model", h1),
        table([
            ["Stage", "Estimate", "Basis"],
            ["Build the local Getty index (one-off)", "~17 min", "Measured: AAT 1 min, ULAN 2.5 min, TGN 12 min, search index 1.5 min"],
            ["Find candidates: local index", "5–30 ms per unique term<br/>25k sample: under 5 min<br/>Whole NBS: ~25 min–2.5 h",
             "Measured on 47 test terms; whole-NBS figure assumes ~300,000 unique terms"],
            ["Find candidates: live Getty service", "2–3 s per unique term<br/>25k sample: ~6–8 h<br/>Whole NBS: over a week",
             "Polite request rate; repeated terms are free from the cache"],
            ["Labelling", "The main constraint", "LLM-assisted labels plus a human-checked validation set (1,000–2,000 decisions)"],
            ["Running the trained model", "~0.09 s per decision<br/>Whole NBS: ~2 days on one data-centre GPU, 1–2 weeks on a 5060 Ti",
             "Yale: 1,224 decisions in 108 s"],
        ], [44, 68, 58], highlight_rows=(2,)),
    ]),

    P("Practical notes", h1),
    *bullets([
        "<b>EIDF GPU Service:</b> jobs run as Kubernetes jobs queued by Kueue. The default project quota is 12 GPUs, "
        "100 CPU cores and 1 TiB memory. Access is through the EIDF portal for University of Edinburgh researchers "
        "and DDI Programme partners.",
        "<b>RTX 5060 Ti:</b> use Linux or WSL2 with a PyTorch build for CUDA 12.8 or later (needed for Blackwell "
        "cards). Run the training script's preflight check first to find the largest batch that fits in 16 GB.",
        "<b>Getty snapshot:</b> the local index holds AAT from August 2026 and ULAN and TGN from January 2026. Against "
        "the live service it gave the same top candidate for all 47 test terms; the few other differences were "
        "records Getty edited after January 2026.",
        "<b>Starting model:</b> fastino/GLiNER2.5-Decide is Apache-2.0. Starting from the Yale model gives a head "
        "start, but the result inherits its CC BY-NC 4.0 licence.",
    ]),

    KeepTogether([P("Sources", h1), P("Yale Cultural Heritage, GLiNER2.5-Decide-linkedart-recon model card and training code (Hugging Face) · National Library of Scotland Data Foundry, National Bibliography of Scotland v6 · EIDF GPU Service website and documentation · RTX 5060 Ti specifications (flopper.io, Guru3D) · GPU comparisons (Spheron, CUDO Compute) · Local measurements of the recon_prep Getty index, 23–26 October 2026.", small)]),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                        title="NBS fine-tuning time estimates", author="Alastair Plant",
                        subject="GLiNER2.5-Decide fine-tuning on the National Bibliography of Scotland")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
