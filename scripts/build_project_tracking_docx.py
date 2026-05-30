from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "01_final_report"
OUT_PATH = OUT_DIR / "WasteWise_Project_Tracking_Report.docx"


BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"
GREEN = "1F7A3A"
GOLD = "7A5A00"
RED = "9B1C1C"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_width(table, widths_dxa: list[int]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = tbl.tblGrid
    if grid is None:
        grid = OxmlElement("w:tblGrid")
        tbl.insert(0, grid)
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            width = widths_dxa[min(idx, len(widths_dxa) - 1)]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_text(cell, text: str, bold=False, color=None, size=9.5) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def style_table(table, widths_dxa: list[int], header_fill=LIGHT_BLUE) -> None:
    table.style = "Table Grid"
    set_table_width(table, widths_dxa)
    set_repeat_table_header(table.rows[0])
    for row_i, row in enumerate(table.rows):
        for cell in row.cells:
            if row_i == 0:
                set_cell_shading(cell, header_fill)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.bold = True
                        r.font.color.rgb = RGBColor.from_string(INK)
            elif row_i % 2 == 0:
                set_cell_shading(cell, "FBFCFD")


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    for i, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], header, bold=True, color=INK, size=9)
    for row_values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_values):
            set_cell_text(cells[i], str(value), size=9)
    style_table(table, widths)
    doc.add_paragraph()


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.add_run(item)


def add_callout(doc: Document, title: str, body: str, fill=CALLOUT) -> None:
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_margins(cell, top=130, bottom=130, start=160, end=160)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(DARK_BLUE)
    r.font.size = Pt(11)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.paragraph_format.line_spacing = 1.15
    p2.add_run(body).font.size = Pt(10)
    style_table(table, [9360], header_fill=fill)
    doc.add_paragraph()


def add_figure(doc: Document, path: Path, caption: str, width_in=5.9) -> None:
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string("555555")


def configure_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    title = styles["Title"]
    title.font.name = "Calibri"
    title.font.size = Pt(24)
    title.font.color.rgb = RGBColor.from_string(INK)
    title.paragraph_format.space_after = Pt(6)

    subtitle = styles["Subtitle"]
    subtitle.font.name = "Calibri"
    subtitle.font.size = Pt(12)
    subtitle.font.color.rgb = RGBColor.from_string("555555")
    subtitle.paragraph_format.space_after = Pt(12)

    for style_name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 14, 7),
        ("Heading 3", 12, DARK_BLUE, 10, 5),
    ]:
        st = styles[style_name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    for style_name in ["List Bullet", "List Number"]:
        st = styles[style_name]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        st.paragraph_format.left_indent = Inches(0.375)
        st.paragraph_format.first_line_indent = Inches(-0.188)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.line_spacing = 1.25


def add_header_footer(doc: Document) -> None:
    section = doc.sections[0]
    header_p = section.header.paragraphs[0]
    header_p.text = "WasteWise FYP Project Tracking Report"
    header_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in header_p.runs:
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor.from_string("666666")

    footer_p = section.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer_p.add_run("Generated: 2026-05-30")
    for r in footer_p.runs:
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor.from_string("666666")


def build_doc() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_styles(doc)
    add_header_footer(doc)

    doc.add_paragraph("WasteWise Project Tracking Report", style="Title")
    doc.add_paragraph("Final Year Project: automated waste classification, feature-based ML, and classification-first localization", style="Subtitle")
    add_callout(
        doc,
        "Current project position",
        "Current datasets are `external_datasets/super_yolo_dataset` for YOLO localization and `data/merged_dataset_v5` for classification. Older `merged_dataset_v3` results remain useful historical ML evidence, but they should not be described as the newest dataset.",
        fill="EAF4EE",
    )

    add_table(
        doc,
        ["Area", "Current status", "Main evidence"],
        [
            ["Dataset", "Use newest datasets for current tracking.", "super_yolo_dataset: 23,929 images / 102,777 boxes; merged_dataset_v5: 29,639 classification images."],
            ["ML", "Keep as main explainable pipeline, but rerun if all final metrics must use newest data.", "Legacy/stable lecturer artifact: 637 handcrafted features; XGBoost accuracy 0.6742, F1-macro 0.6506."],
            ["PCA", "Completed but final claim must match artifact.", "637 -> 128 components keeps 99.90% variance; current artifact shows 4.53 pp accuracy drop."],
            ["DL classification", "Use as comparison / gate, not final localization metric.", "EfficientNetB0 crop classifier 94.29% in architecture comparison."],
            ["DL localization", "Improved after Stage 2 rework.", "YOLO localization-only conf=0.35: precision 0.7614, recall 0.5134, mean IoU 0.9004."],
            ["Mobile", "Implementation exists, physical validation still required.", "Expo app, TFLite model assets, scan/history/settings UI."],
        ],
        [1500, 4200, 3660],
    )

    doc.add_paragraph("1. Project Goal And Scope", style="Heading 1")
    doc.add_paragraph(
        "WasteWise is an FYP system for automated waste recognition and sorting. The work now separates explainable Machine Learning evidence from Deep Learning localization evidence, so each branch can be reported honestly and evaluated with the right metric."
    )
    add_bullets(
        doc,
        [
            "Primary ML goal: classify waste object crops using lecturer-explainable handcrafted features.",
            "Primary DL goal after rework: perform classification first, then localize objects as a localization-only task.",
            "Deployment direction: mobile/edge-ready inference with sorting guidance, history, and settings.",
        ],
    )

    doc.add_paragraph("2. Dataset Tracking", style="Heading 1")
    doc.add_paragraph(
        "Newest datasets detected in the workspace are listed below. These should be treated as the current dataset sources for the report. Older dataset names are retained only when a specific saved experiment was produced from them."
    )
    add_table(
        doc,
        ["YOLO class", "Train boxes", "Val boxes", "Test boxes", "Total boxes"],
        [
            ["plastic", "18,418", "1,568", "1,668", "21,654"],
            ["glass", "7,186", "2,488", "9", "9,683"],
            ["metal", "8,785", "1,800", "542", "11,127"],
            ["paper", "4,731", "188", "1,347", "6,266"],
            ["cardboard", "7,433", "1,614", "35", "9,082"],
            ["organic", "32,171", "12,748", "46", "44,965"],
            ["Total boxes", "78,724", "20,406", "3,647", "102,777"],
        ],
        [1700, 1900, 1900, 1900, 1960],
    )
    add_table(
        doc,
        ["Dataset", "Split", "Images", "Notes"],
        [
            ["external_datasets/super_yolo_dataset", "train", "19,559", "YOLO-format images with label files."],
            ["external_datasets/super_yolo_dataset", "val", "3,266", "YOLO-format images with label files."],
            ["external_datasets/super_yolo_dataset", "test", "1,104", "YOLO-format images with label files."],
            ["data/merged_dataset_v5", "train", "24,039", "Classification-folder dataset; includes Background."],
            ["data/merged_dataset_v5", "test", "5,600", "Balanced 800 images per class."],
        ],
        [3100, 1100, 1200, 3960],
    )
    add_table(
        doc,
        ["merged_dataset_v5 class", "Train images", "Test images", "Total"],
        [
            ["Background", "3,500", "800", "4,300"],
            ["cardboard", "3,425", "800", "4,225"],
            ["glass", "3,500", "800", "4,300"],
            ["metal", "3,230", "800", "4,030"],
            ["organic", "3,500", "800", "4,300"],
            ["paper", "3,500", "800", "4,300"],
            ["plastic", "3,384", "800", "4,184"],
            ["Total", "24,039", "5,600", "29,639"],
        ],
        [3100, 2000, 2000, 2260],
    )
    add_callout(
        doc,
        "Reporting note",
        "The newest YOLO dataset is imbalanced, especially in validation/test class distribution. The newest classification dataset is much more balanced and includes a Background class. Legacy ML metrics from `merged_dataset_v3` should be labelled as legacy unless rerun on the newest dataset.",
        fill="FFF7DF",
    )

    doc.add_paragraph("3. Machine Learning Pipeline", style="Heading 1")
    doc.add_paragraph(
        "The ML branch is the most explainable and stable part of the project. It uses object crops from YOLO labels, resizes crops internally, extracts a fixed 637-D vector, then compares classical ML models."
    )
    add_callout(
        doc,
        "Dataset alignment note",
        "The model scores below come from saved lecturer-facing artifacts, not a fresh rerun on `super_yolo_dataset` or `merged_dataset_v5`. To make the final thesis fully newest-dataset-aligned, rerun feature extraction and ML model comparison on the newest selected dataset.",
        fill="FFF7DF",
    )
    add_table(
        doc,
        ["Feature group", "Count", "Purpose"],
        [
            ["Spatial", "8", "Intensity, gradients, edge density."],
            ["Frequency / FFT", "9", "Radial frequency energy and high-frequency texture."],
            ["Color", "44", "HSV histograms plus BGR/HSV mean and standard deviation."],
            ["HOG", "576", "Local shape and gradient-orientation texture."],
            ["Total", "637", "Fixed handcrafted representation for explainable ML."],
        ],
        [2200, 1200, 5960],
    )
    add_table(
        doc,
        ["Model", "Accuracy", "F1-macro", "Decision"],
        [
            ["XGBoost", "0.6742", "0.6506", "Best lecturer-facing ML result."],
            ["Random Forest", "0.6317", "0.6111", "Useful for feature importance."],
            ["ExtraTrees", "0.6312", "0.6113", "Strong high-dimensional tree baseline."],
            ["Linear SVM", "0.5960", "0.5642", "Margin-based baseline."],
            ["Logistic Regression", "0.5864", "0.5558", "Linear standardized baseline."],
            ["Decision Tree", "0.5115", "0.4883", "Simple interpretable baseline."],
        ],
        [2100, 1500, 1500, 4260],
    )
    add_table(
        doc,
        ["Feature importance group", "Importance"],
        [
            ["HOG", "59.5808%"],
            ["Color", "29.2090%"],
            ["Frequency", "5.6582%"],
            ["Spatial", "5.5520%"],
        ],
        [5200, 4160],
    )
    add_figure(doc, ROOT / "runs" / "ml" / "feature_ml_lecturer_6class_4k" / "chart_model_comparison.png", "Figure 1. Classical ML model comparison.", 5.8)
    add_figure(doc, ROOT / "runs" / "ml" / "feature_ml_lecturer_6class_4k" / "chart_domain_importance.png", "Figure 2. Feature group / domain importance.", 5.8)

    doc.add_paragraph("4. PCA Dimensionality Reduction", style="Heading 1")
    add_table(
        doc,
        ["Components", "Explained variance", "Accuracy", "Weighted F1", "Latency"],
        [
            ["637", "100.00%", "73.24%", "0.7319", "0.0533 ms"],
            ["64", "99.78%", "67.48%", "0.6736", "0.0284 ms"],
            ["128", "99.90%", "68.71%", "0.6863", "0.0314 ms"],
            ["256", "99.97%", "66.95%", "0.6691", "0.0296 ms"],
        ],
        [1700, 2300, 1600, 1800, 1960],
    )
    add_callout(
        doc,
        "PCA claim control",
        "Current artifact shows 637 -> 128 drops accuracy from 73.24% to 68.71%, a 4.53 percentage-point drop. If slides need to say around 2%, rerun PCA or replace with the matching artifact.",
        fill="FFF7DF",
    )
    add_figure(doc, ROOT / "runs" / "dl" / "pca_experiments" / "pca_dimensionality_chart.png", "Figure 3. PCA component count vs accuracy and latency.", 5.8)

    doc.add_paragraph("5. Deep Learning Experiments", style="Heading 1")
    add_table(
        doc,
        ["Experiment", "Result", "Use in final report"],
        [
            ["ANN/CNN crop baselines", "Tuned ANN accuracy 0.4057; tuned CNN accuracy 0.4413.", "Baseline evidence only."],
            ["CNN + ANN soft voting", "50/50 ensemble accuracy 78.69%, macro F1 78.12%.", "Shows feature complementarity."],
            ["Architecture comparison", "EfficientNetB0 94.29%; ResNet50 89.76%; MobileNetV2 85.43%.", "Supports EfficientNetB0 as classifier/gate."],
            ["Old 2-stage pipeline", "YOLO localization -> EfficientNetB0 verification; 100-image sweep had 295 accepted from 348 proposals.", "Experimental evidence; not final direction."],
        ],
        [2500, 3700, 3160],
    )
    add_table(
        doc,
        ["Deep model", "Accuracy", "Size", "Latency"],
        [
            ["MobileNetV2", "85.43%", "20.07 MB", "253.5 ms"],
            ["ResNet50", "89.76%", "161.52 MB", "163.1 ms"],
            ["EfficientNetB0", "94.29%", "29.21 MB", "288.6 ms"],
        ],
        [2500, 2100, 2300, 2460],
    )
    add_figure(doc, ROOT / "runs" / "dl" / "comparison_models" / "confusion_matrix_grid.png", "Figure 4. Deep model confusion matrix grid.", 5.8)

    doc.add_paragraph("6. Final DL Rework: Classification First, Localization Second", style="Heading 1")
    doc.add_paragraph(
        "The new DL requirement is reversed from the old YOLO-first pipeline. Stage 1 classifies or gates the image first. Stage 2 performs localization only. The class decision and box localization are tracked separately."
    )
    add_table(
        doc,
        ["Stage", "Role", "Current implementation"],
        [
            ["Stage 1", "Classification / image-level gate", "EfficientNetB0 classifier predicts image/crop class and diagnostic confidence."],
            ["Stage 2", "Localization only", "YOLO model is used only to output bounding boxes; YOLO class is not the final classifier."],
            ["Evaluation", "Localization metrics", "IoU@0.5 matching, precision, recall, mean matched IoU."],
        ],
        [1500, 2900, 4960],
    )
    add_table(
        doc,
        ["Stage 2 localizer", "Precision", "Recall", "Mean IoU", "TP", "FP", "FN"],
        [
            ["Grad-CAM baseline", "0.2568", "0.0728", "0.7127", "19", "55", "242"],
            ["YOLO localization-only, conf=0.25", "0.6352", "0.5670", "0.9012", "148", "85", "113"],
            ["YOLO localization-only, conf=0.35", "0.7614", "0.5134", "0.9004", "134", "42", "127"],
        ],
        [3000, 1200, 1200, 1400, 800, 800, 960],
    )
    add_callout(
        doc,
        "Recommended DL localization setting",
        "`--localizer yolo --yolo-conf 0.35` gives the best quick-check balance: higher precision, still useful recall, and high localization IoU.",
        fill="EAF4EE",
    )
    add_figure(
        doc,
        ROOT / "runs" / "dl" / "localization_rework" / "yolo_conf025_stratified60" / "visuals" / "rf_garbage_metal391_jpg.rf.d2d79150c42df8cd64bea8d65acc58ab_yolo.jpg",
        "Figure 5. Example classification-first / YOLO localization-only output.",
        5.8,
    )

    doc.add_paragraph("7. Mobile Application Tracking", style="Heading 1")
    add_table(
        doc,
        ["Component", "Status"],
        [
            ["Framework", "Expo Dev Client / React Native mobile app in `mobile/`."],
            ["Inference", "Vision Camera + `react-native-fast-tflite` + YOLO post-processing."],
            ["Model assets", "`best_float16.tflite`, `best_float32.tflite`, `best_metadata.json`."],
            ["User features", "Live scan, result page, history, settings, sorting guidance, leaderboard skeleton."],
            ["Validation gap", "Physical Android device or emulator runtime validation still required."],
        ],
        [2400, 6960],
    )

    doc.add_paragraph("8. Reproducible Commands", style="Heading 1")
    add_table(
        doc,
        ["Task", "Command / path"],
        [
            ["Feature ML run", r".\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data merged_dataset_v3\data.yaml --out runs\ml\feature_ml_lecturer_6class_4k --exclude-classes other --max-per-class-train 4000 --max-per-class-test 800"],
            ["Current YOLO dataset", r"external_datasets\super_yolo_dataset\data.yaml"],
            ["Current classification dataset", r"data\merged_dataset_v5\data.yaml"],
            ["PCA experiment", r".\.venv311\Scripts\python.exe scripts\train_pca_ann.py"],
            ["DL localization rework", r".\.venv311\Scripts\python.exe scripts\classification_to_localization_pipeline.py --max-images 60 --sample-mode stratified --seed 42 --localizer yolo --yolo-conf 0.35 --out-dir runs\dl\localization_rework\yolo_conf035_stratified60_final"],
            ["Mobile static check", r"cd mobile && node .\node_modules\typescript\bin\tsc --noEmit"],
        ],
        [2200, 7160],
    )

    doc.add_paragraph("9. Artifact Index", style="Heading 1")
    add_table(
        doc,
        ["Area", "Key artifact"],
        [
            ["Workflow summary", r"docs\01_final_report\WORKFLOW_APPROACHES_AND_DL_REWORK.md"],
            ["Current workflow/report notes", r"docs\01_final_report\WORKFLOW_APPROACHES_AND_DL_REWORK.md"],
            ["Newest YOLO dataset", r"external_datasets\super_yolo_dataset"],
            ["Newest classification dataset", r"data\merged_dataset_v5"],
            ["ML lecturer run", r"runs\ml\feature_ml_lecturer_6class_4k\REPORT.md"],
            ["PCA report", r"runs\dl\pca_experiments\PCA_Dimensionality_Report.md"],
            ["ML vs DL comparison", r"runs\comparisons\model_comparison\REPORT.md"],
            ["DL architecture comparison", r"runs\dl\comparison_models\model_comparison_report.md"],
            ["DL localization improved", r"runs\dl\localization_rework\yolo_conf035_stratified60_final\REPORT.md"],
            ["Mobile model assets", r"mobile\assets\model"],
        ],
        [2700, 6660],
    )

    doc.add_paragraph("10. Next Actions", style="Heading 1")
    add_numbered(
        doc,
        [
            "Keep ML pipeline as the main explainable result and avoid changing its artifact paths unless rerunning all tables.",
            "Decide whether to keep PCA artifact as-is or rerun if the final presentation must claim about 2% accuracy loss.",
            "Use the classification-first YOLO-localization run for DL localization evidence.",
            "Run mobile app on a physical Android device or emulator and record FPS, detection stability, and model toggle behavior.",
            "Add final thesis text explaining why classification accuracy and localization metrics are reported separately.",
        ],
    )

    doc.save(OUT_PATH)


if __name__ == "__main__":
    build_doc()
    print(OUT_PATH)
