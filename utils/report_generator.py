"""
PDF report generation for IoT-IDS analysis results.

This module creates professional PDF reports with metrics, visualizations,
and analysis summaries for export and documentation.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Page configuration
PAGE_SIZE = letter
MARGIN = 0.75 * inch

# Styles
STYLES = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    'CustomTitle',
    parent=STYLES['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#2c3e50'),
    spaceAfter=30,
    alignment=TA_CENTER
)
HEADING_STYLE = ParagraphStyle(
    'CustomHeading',
    parent=STYLES['Heading2'],
    fontSize=16,
    textColor=colors.HexColor('#34495e'),
    spaceAfter=12,
    spaceBefore=12
)
BODY_STYLE = ParagraphStyle(
    'CustomBody',
    parent=STYLES['BodyText'],
    fontSize=11,
    leading=14
)

# =============================================================================
# MAIN REPORT GENERATION
# =============================================================================

def generate_analysis_report(
    results_df: pd.DataFrame,
    model_name: str,
    metadata: Dict[str, Any],
    include_plots: bool = True
) -> bytes:
    """
    Generate comprehensive analysis report as PDF.

    Args:
        results_df: DataFrame with analysis results
        model_name: Name of model used
        metadata: Model metadata dictionary
        include_plots: Whether to include visualizations

    Returns:
        PDF file as bytes

    Example:
        >>> pdf_bytes = generate_analysis_report(df, "Synthetic", meta)
        >>> with open("report.pdf", "wb") as f:
        ...     f.write(pdf_bytes)
    """
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    # Build document elements
    story = []

    # Header
    story.extend(_create_header(model_name))

    # Metrics summary
    story.extend(_create_metrics_section(results_df, metadata))

    # Results table
    story.extend(_create_results_table(results_df))

    # Visualizations (if requested)
    if include_plots:
        story.append(PageBreak())
        story.extend(_create_visualizations_section(results_df))

    # Footer
    story.extend(_create_footer())

    # Build PDF
    doc.build(story)

    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

def generate_comparison_report(
    results_synthetic: pd.DataFrame,
    results_real: pd.DataFrame,
    concordance_rate: float
) -> bytes:
    """
    Generate model comparison report.

    Args:
        results_synthetic: Results from synthetic model
        results_real: Results from real model
        concordance_rate: Agreement rate between models (0-100)

    Returns:
        PDF file as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    story = []

    # Header
    title = Paragraph("Model Comparison Report", TITLE_STYLE)
    story.append(title)
    story.append(Spacer(1, 20))

    # Comparison summary
    summary = f"""
    <b>Concordance Rate:</b> {concordance_rate:.2f}%<br/>
    <b>Samples Analyzed:</b> {len(results_synthetic)}<br/>
    <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    story.append(Paragraph(summary, BODY_STYLE))
    story.append(Spacer(1, 20))

    # Side-by-side comparison table
    story.append(Paragraph("Prediction Comparison", HEADING_STYLE))
    story.extend(_create_comparison_table(results_synthetic, results_real))

    # Footer
    story.extend(_create_footer())

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

# =============================================================================
# DOCUMENT SECTIONS
# =============================================================================

def _create_header(model_name: str) -> List[Any]:
    """Create report header."""
    elements = []

    # Title
    title = Paragraph(
        f"IoT Intrusion Detection Report<br/><font size=14>Model: {model_name}</font>",
        TITLE_STYLE
    )
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Metadata
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f"""
    <b>Report Generated:</b> {timestamp}<br/>
    <b>Analysis Type:</b> IoT Network Traffic Classification<br/>
    <b>Threat Categories:</b> 8 (Benign + 7 Attack Types)
    """
    elements.append(Paragraph(info, BODY_STYLE))
    elements.append(Spacer(1, 30))

    return elements

def _create_metrics_section(
    results_df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> List[Any]:
    """Create metrics summary section."""
    elements = []

    elements.append(Paragraph("Performance Metrics", HEADING_STYLE))

    # Extract metrics
    metrics_data = [
        ['Metric', 'Value'],
        ['Model Accuracy', f"{metadata.get('accuracy', 'N/A')}%"],
        ['Precision (Weighted)', f"{metadata.get('precision', 'N/A')}%"],
        ['Recall (Weighted)', f"{metadata.get('recall', 'N/A')}%"],
        ['F1-Score (Weighted)', f"{metadata.get('f1_score', 'N/A')}%"],
        ['Samples Analyzed', str(len(results_df))],
    ]

    # Calculate threat distribution
    if 'prediction' in results_df.columns:
        threat_counts = results_df['prediction'].value_counts()
        benign_count = threat_counts.get('Benign', 0)
        threat_count = len(results_df) - benign_count
        metrics_data.extend([
            ['Benign Samples', str(benign_count)],
            ['Threat Samples', str(threat_count)],
            ['Threat Ratio', f"{(threat_count/len(results_df)*100):.1f}%"]
        ])

    # Create table
    table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    return elements

def _create_results_table(results_df: pd.DataFrame, max_rows: int = 20) -> List[Any]:
    """Create results data table."""
    elements = []

    elements.append(Paragraph(f"Analysis Results (Top {max_rows})", HEADING_STYLE))

    # Prepare data
    display_df = results_df.head(max_rows)

    # Create table data
    table_data = [['Sample', 'Prediction', 'Confidence', 'True Label (if available)']]

    for idx, row in display_df.iterrows():
        table_data.append([
            str(idx + 1),
            str(row.get('prediction', 'N/A')),
            f"{row.get('confidence', 0):.1f}%",
            str(row.get('true_label', 'N/A'))
        ])

    # Create table
    table = Table(table_data, colWidths=[0.8*inch, 1.5*inch, 1.2*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue])
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    return elements

def _create_comparison_table(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    max_rows: int = 15
) -> List[Any]:
    """Create side-by-side comparison table."""
    elements = []

    # Prepare data
    table_data = [['Sample', 'Synthetic Model', 'Conf.', 'Real Model', 'Conf.', 'Match']]

    for idx in range(min(max_rows, len(results_a), len(results_b))):
        pred_a = results_a.iloc[idx].get('prediction', 'N/A')
        conf_a = results_a.iloc[idx].get('confidence', 0)
        pred_b = results_b.iloc[idx].get('prediction', 'N/A')
        conf_b = results_b.iloc[idx].get('confidence', 0)
        match = '✓' if pred_a == pred_b else '✗'

        table_data.append([
            str(idx + 1),
            str(pred_a),
            f"{conf_a:.1f}%",
            str(pred_b),
            f"{conf_b:.1f}%",
            match
        ])

    # Create table
    table = Table(table_data, colWidths=[0.6*inch, 1.3*inch, 0.8*inch, 1.3*inch, 0.8*inch, 0.6*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    return elements

def _create_visualizations_section(results_df: pd.DataFrame) -> List[Any]:
    """Create visualizations section with threat distribution."""
    elements = []

    elements.append(Paragraph("Threat Distribution", HEADING_STYLE))

    # Create simple bar chart
    if 'prediction' in results_df.columns:
        threat_counts = results_df['prediction'].value_counts()

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 4))
        threat_counts.plot(kind='bar', ax=ax, color='#3498db')
        ax.set_title('Detected Threats')
        ax.set_xlabel('Threat Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save to buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        plt.close()

        # Add to PDF
        img = RLImage(img_buffer, width=5*inch, height=2.5*inch)
        elements.append(img)
        elements.append(Spacer(1, 20))

    return elements

def _create_footer() -> List[Any]:
    """Create report footer."""
    elements = []

    elements.append(Spacer(1, 40))

    footer_text = """
    <para alignment="center">
    <font size=9 color="grey">
    Generated by IoT-IDS Dashboard | Universidad Señor de Sipán<br/>
    This is an automated analysis report for academic demonstration purposes.
    </font>
    </para>
    """
    elements.append(Paragraph(footer_text, BODY_STYLE))

    return elements

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_report_to_file(pdf_bytes: bytes, filename: str) -> None:
    """
    Save PDF bytes to file.

    Args:
        pdf_bytes: PDF content as bytes
        filename: Output filename
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        f.write(pdf_bytes)
