"""
Script to generate publication-quality PDF report for Component C (Group 5) Week 2.
Uses ReportLab to build 'Group_5_Component_C_Week_2_Report.pdf'.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54, 750, "CSIT332 Machine Learning · Component C (Group 5) · Week 2 Report"
            )
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, text)
        self.drawString(
            54, 36, "Group 5 · Component C: The Statistical Evaluation Layer"
        )
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf_report(filename="Week 2/Group_5_Component_C_Week_2_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate blue
    accent_color = colors.HexColor("#C53030")    # Deep red / alert
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal
    light_bg = colors.HexColor("#F7FAFC")        # Soft off-white

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=secondary_color,
        spaceAfter=10,
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        spaceAfter=5,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2C5282"),
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=dark_neutral,
        alignment=1, # Center
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=table_cell,
        alignment=0, # Left
    )

    story = []

    # Title Banner
    story.append(Paragraph("CSIT332 · PRINCIPLES OF MACHINE LEARNING", subtitle_style))
    story.append(Paragraph("Component C: The Statistical Evaluation Layer", title_style))
    story.append(Paragraph("<b>Week 2 Deliverable:</b> Benchmark Evaluation on Reference Data & The Dual Nature of Truth<br/><b>Group:</b> Group 5 &nbsp;|&nbsp; <b>Date:</b> October 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

    # Executive Summary / Callout
    summary_text = (
        "<b>Executive Summary:</b> Week 2 resolves a critical methodological anomaly identified in Week 1: "
        "evaluating models trained on small samples (n=50) against abundant-data (asymptotic) performance produced "
        "a negative optimism value (-0.035). By reforming ground truth to be measured honestly at each sample size "
        "(Version A) rather than asymptotically (Version B), this report isolates pure model selection bias from finite-sample "
        "skill deficits. Across 12,000 reference evaluations, Version A yields strictly positive and shrinking optimism "
        "(+0.106 to +0.022) and rising winner accuracy (43% to 69%), confirming theoretical predictions."
    )
    summary_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 1, secondary_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # SECTION 1: Why the Two Versions Give Different Answers
    story.append(Paragraph("1. Why the Two Versions of Truth Give Different Answers", h1_style))
    p1 = (
        "<b>The Fundamental Question:</b> A clinical or industrial researcher who draws a sample of 50 patients "
        "and runs 5-fold cross-validation asks a practical question: <i>Does her empirical selection procedure support "
        "naming a winner, or would a different sample of 50 patients have produced a different decision?</i> Her sample size "
        "is already an immutable constraint of her study; she is not asking which model would win with 12,600 patients, "
        "because she will never possess 12,600 patients. The purpose of our benchmark is to quantify the uncertainty and "
        "optimism she genuinely faces, rather than penalizing her for an outcome she cannot alter."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "<b>What Version A Measures (Selection Bias at Sample Size n):</b> In Version A, truth is measured by fitting each "
        "classifier on the exact CV training folds (4/5 of n rows) and evaluating it on a frozen held-out test set of 6,000 "
        "independent rows. The declared winner's cross-validated score is then compared against that same model's honest score "
        "at sample size n. Because cross-validation estimates are noisy random variables, selecting the empirical maximum over "
        "six classifiers naturally capitalizes on positive sampling error. This yields an <b>optimism</b> that is strictly "
        "positive (+0.1059 at n=50, +0.0559 at n=100, +0.0406 at n=200, and +0.0220 at n=500). As sample size grows, fold "
        "variance contracts, so the winner's inflation shrinks smoothly toward zero. The <b>winner-correct rate</b> generally "
        "improves as sample size increases, rising from 43% at n = 50 to 69% at n = 500, dipping slightly to 41% at n = 100 before "
        "increasing at the larger sample sizes (47% at n=200, 69% at n=500). Version A isolates pure selection bias."
    )
    story.append(Paragraph(p2, body_style))

    p3 = (
        "<b>What Version B Conflates (The Anomaly of Negative Optimism):</b> In Version B, the winner's CV score is compared "
        "against its asymptotic performance with abundant data (12,600 rows). This adds two entirely separate phenomena together "
        "and misreports them as a single number:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Observed(n) - True(&infin;) = [Observed(n) - True(n)] + [True(n) - True(&infin;)]</b><br/>"
        "The first bracket is <i>selection bias</i> (+0.1059 at n=50), which pushes the score upward. The second bracket is the "
        "<i>finite-sample skill deficit</i> (-0.1411 at n=50), representing how much lower a model's true performance is when trained on "
        "40 rows compared to 12,600 rows. At small sample sizes, the skill deficit dwarfs the selection bias (+0.1059 - 0.1411 = -0.0352), "
        "swamping it into a negative composite difference (-0.0352) and creating the illusion of negative selection optimism."
    )
    story.append(Paragraph(p3, body_style))

    p4 = (
        "<b>Penalizing Correct Scientific Decisions:</b> Crucially, classifiers possess vastly different learning curves. Simple "
        "linear models like logistic regression have high inductive bias and excel with minimal data (true AUC = 0.690 at n=50). "
        "Data-hungry models like gradient boosting have high variance; at n=50, gradient boosting severely overfits (true AUC = 0.636), "
        "yet reaches 0.837 asymptotically. A researcher with 50 samples who chooses logistic regression has made the <i>objectively "
        "correct scientific choice</i>. Version A credits her decision (43% correct rate). Version B insists that gradient boosting "
        "is the 'truth', falsely penalizing her for picking the model that actually performed best on her data. Consequently, Version B's "
        "winner-correct rate collapses to 6%–15% and fails to improve with sample size."
    )
    story.append(Paragraph(p4, body_style))
    story.append(Spacer(1, 6))

    # SECTION 2: Table 1 - Four Quantities x Four Sample Sizes
    story.append(Paragraph("2. Experimental Results: Four Quantities × Four Sample Sizes", h1_style))
    story.append(Paragraph(
        "Table 1 presents the complete evaluation of the 12,000-row reference dataset (100 replicates × 4 sample sizes × 6 classifiers × 5 folds) "
        "comparing Version A (truth at sample size n) against Version B (asymptotic truth).", body_style
    ))

    t1_data = [
        [
            Paragraph("<b>Sample Size (n)</b>", table_cell_bold),
            Paragraph("<b>Winner-Correct<br/>Ver A / Ver B</b>", table_cell_bold),
            Paragraph("<b>Kendall Tau (&tau;)<br/>Ver A / Ver B*</b>", table_cell_bold),
            Paragraph("<b>Mean Optimism<br/>Ver A / Ver B</b>", table_cell_bold),
            Paragraph("<b>True Best Model (n)<br/>[True AUC]</b>", table_cell_bold),
            Paragraph("<b>Empirical Winner Distribution<br/>(100 Replicates)</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>n = 50</b>", table_cell_bold),
            Paragraph("<b>0.4300</b> / 0.1500", table_cell),
            Paragraph("<b>+0.3166</b> / +0.0177", table_cell),
            Paragraph("<b>+0.1059</b> / -0.0352", table_cell),
            Paragraph("logreg [0.6903]", table_cell),
            Paragraph("logreg 43%, gb 15%, svm 13%,<br/>rf 13%, knn 12%, dtree 4%", table_cell_left),
        ],
        [
            Paragraph("<b>n = 100</b>", table_cell_bold),
            Paragraph("<b>0.4100</b> / 0.1200", table_cell),
            Paragraph("<b>+0.5091</b> / +0.2008", table_cell),
            Paragraph("<b>+0.0559</b> / -0.0439", table_cell),
            Paragraph("logreg [0.7232]", table_cell),
            Paragraph("logreg 41%, svm 32%, gb 12%,<br/>rf 10%, knn 5%", table_cell_left),
        ],
        [
            Paragraph("<b>n = 200</b>", table_cell_bold),
            Paragraph("<b>0.4700</b> / 0.0600", table_cell),
            Paragraph("<b>+0.6908</b> / +0.3358", table_cell),
            Paragraph("<b>+0.0406</b> / -0.0216", table_cell),
            Paragraph("svm [0.7544]", table_cell),
            Paragraph("svm 47%, logreg 32%, rf 13%,<br/>gb 6%, dtree 1%, knn 1%", table_cell_left),
        ],
        [
            Paragraph("<b>n = 500</b>", table_cell_bold),
            Paragraph("<b>0.6900</b> / 0.0800", table_cell),
            Paragraph("<b>+0.8063</b> / +0.4728", table_cell),
            Paragraph("<b>+0.0220</b> / -0.0091", table_cell),
            Paragraph("svm [0.7856]", table_cell),
            Paragraph("svm 69%, logreg 23%, gb 8%", table_cell_left),
        ],
    ]

    t1_table = Table(t1_data, colWidths=[65, 85, 80, 85, 95, 94])
    t1_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1_table)
    story.append(Paragraph(
        "<font size='7' color='#718096'>*Note on Kendall's &tau;: Version A (against true AUC at n) is the primary finite-sample metric. "
        "Version B (against asymptotic AUC) is included as supplementary context to illustrate ordinal divergence at small samples. "
        "In Version B, the asymptotic target winner is gb (0.836803).</font>", body_style
    ))
    story.append(Spacer(1, 8))

    # SECTION 3: Task 3 - Paired t-Test Power Analysis
    story.append(Paragraph("3. Statistical Power & Sign-Error Analysis: Close vs. Wide Pairs", h1_style))
    story.append(Paragraph(
        "To examine why sample size and gap size must be evaluated jointly, we perform 5-fold paired t-tests (&alpha;=0.05) "
        "across all 100 replicates for the two required pairs: a <b>Close Pair</b> (&Delta; &lt; 0.01 in true AUC at n=50) and a "
        "<b>Wide Pair</b> (&Delta; &gt; 0.05 in true AUC across all n). Both Version A (truth at n) and Version B (asymptotic truth) are reported.", body_style
    ))

    t2_data = [
        [
            Paragraph("<b>Pair & Classification</b>", table_cell_bold),
            Paragraph("<b>Sample Size</b>", table_cell_bold),
            Paragraph("<b>True Gap at n<br/>(&Delta; AUC)</b>", table_cell_bold),
            Paragraph("<b>Power (Ver A)<br/>[Correct Detection]</b>", table_cell_bold),
            Paragraph("<b>Sign-Error (Ver A)<br/>[Wrong Direction]</b>", table_cell_bold),
            Paragraph("<b>Power (Ver B)*<br/>[Asymptotic Ref]</b>", table_cell_bold),
            Paragraph("<b>Sign-Error (Ver B)*<br/>[Asymptotic Ref]</b>", table_cell_bold),
        ],
        # Close pair: gb vs knn
        [
            Paragraph("<b>gb vs knn</b><br/>(Close Pair:<br/>&Delta; &lt; 0.01 at n=50)", table_cell_left),
            Paragraph("n = 50<br/>n = 100<br/>n = 200<br/>n = 500", table_cell),
            Paragraph("+0.0016<br/>+0.0183<br/>+0.0406<br/>+0.0634", table_cell),
            Paragraph("0.0500 (5%)<br/>0.0800 (8%)<br/>0.1400 (14%)<br/>0.3800 (38%)", table_cell),
            Paragraph("<b>0.0400 (4%)</b><br/>0.0100 (1%)<br/>0.0000 (0%)<br/>0.0000 (0%)", table_cell),
            Paragraph("0.0500 (5%)<br/>0.0800 (8%)<br/>0.1400 (14%)<br/>0.3800 (38%)", table_cell),
            Paragraph("0.0400 (4%)<br/>0.0100 (1%)<br/>0.0000 (0%)<br/>0.0000 (0%)", table_cell),
        ],
        # Wide pair: logreg vs dtree
        [
            Paragraph("<b>logreg vs dtree</b><br/>(Wide Pair:<br/>&Delta; &gt; 0.12 at all n)", table_cell_left),
            Paragraph("n = 50<br/>n = 100<br/>n = 200<br/>n = 500", table_cell),
            Paragraph("+0.1290<br/>+0.1494<br/>+0.1558<br/>+0.1415", table_cell),
            Paragraph("0.1800 (18%)<br/>0.4200 (42%)<br/>0.5200 (52%)<br/><b>0.8700 (87%)</b>", table_cell),
            Paragraph("0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%)", table_cell),
            Paragraph("0.1800 (18%)<br/>0.4200 (42%)<br/>0.5200 (52%)<br/><b>0.8700 (87%)</b>", table_cell),
            Paragraph("0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%)<br/>0.0000 (0%)", table_cell),
        ],
    ]

    t2_table = Table(t2_data, colWidths=[95, 55, 65, 75, 75, 70, 69])
    t2_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<font size='7' color='#718096'>*Note: For gb vs knn and logreg vs dtree, both Version A and Version B agree on the superior classifier direction (gb > knn and logreg > dtree), so their numerical rates match.</font>", body_style
    ))
    story.append(Spacer(1, 6))

    # Supplementary Reversal Pair
    story.append(Paragraph("<b>Supplementary Reversal Analysis: svm vs logreg</b>", h2_style))
    story.append(Paragraph(
        "To illustrate how truth direction inverts between small and large samples, we also evaluated <code>svm vs logreg</code>. "
        "At n=50, logreg is genuinely superior by +0.0981; Version A credits favoring logreg (Power=3%, Sign-Error=1%). "
        "However, Version B considers svm asymptotically superior by +0.0277, inverting the verdict and misclassifying favoring logreg as a Sign Error (3%) and favoring svm as Power (1%)! "
        "At n=200, a near-tie occurs (&Delta; = +0.0021 favoring svm), with Power=3% and Sign-Error=1%.", body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 4: Methodological Insights & Key Takeaways
    story.append(Paragraph("4. Core Methodological Insights & Practical Takeaways", h1_style))
    takeaways = [
        "<b>1. Gap Size and Sample Size Must Be Evaluated Jointly:</b> Statistical power is a function of effect size over standard error "
        "(&Delta; / SE). When the true gap is minute (&Delta; = 0.0016 for gb vs knn at n=50), increasing sample size from 50 to 100 barely "
        "moves power (5% to 8%) because the gap itself is negligible. Conversely, when the gap is large (&Delta; = 0.14 for logreg vs dtree), "
        "increasing sample size from 50 to 500 pushes power from 18% to 87%.",
        
        "<b>2. Sign Errors Are Concentrated in Near-Ties at Small N:</b> In the close pair gb vs knn at n=50, 4 out of 9 significant rejections "
        "were in the wrong direction (sign error rate = 4%). A researcher obtaining p &lt; 0.05 on 50 rows for two evenly matched algorithms has "
        "a ~44% chance of declaring the inferior model superior. Once n reaches 200 or the gap exceeds 0.05, sign errors vanish to 0%.",

        "<b>3. The Inductive Bias Tradeoff:</b> Complex, flexible models (gradient boosting, SVM) only reveal their superiority once sufficient "
        "samples constrain their parameter space. Simple models with strong inductive bias (logistic regression) dominate in data-sparse "
        "regimes. Benchmarking systems must measure finite-sample truth, otherwise researchers will be misinformed into deploying complex models "
        "that underperform in small-sample applications.",

        "<b>4. Complete Verification:</b> The updated module <code>Week 2/component_c.py</code> passed all 18 automated unit tests, validating "
        "lexicographical tie-breaking, fold integrity, non-computable AUC rejection, exact p &ge; 0.05 threshold boundary rejection, truth-ties, and sample-size independence.",
    ]
    for item in takeaways:
        story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 2))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")


if __name__ == "__main__":
    build_pdf_report()
