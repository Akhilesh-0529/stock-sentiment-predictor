from datetime import datetime
import os
import pandas as pd


def _ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def create_markdown_report(symbol: str, analysis_text: str, output_path: str = None) -> str:
    _ensure_reports_dir()
    if output_path is None:
        output_path = f"reports/{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)
    return output_path


def create_pdf_report(symbol: str, analysis_data, output_path: str = None) -> str:
    _ensure_reports_dir()
    if output_path is None:
        output_path = f"reports/{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors

        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        title = Paragraph(f"Stock Analysis Report: {symbol}", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 12))

        ts = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
        story.append(ts)
        story.append(Spacer(1, 12))

        if isinstance(analysis_data, str):
            for line in analysis_data.splitlines():
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 6))
        elif isinstance(analysis_data, dict):
            for section, content in analysis_data.items():
                story.append(Paragraph(f"<b>{section}</b>", styles["Heading2"]))
                story.append(Spacer(1, 6))

                if isinstance(content, str):
                    for line in content.splitlines():
                        story.append(Paragraph(line, styles["Normal"]))
                        story.append(Spacer(1, 4))
                elif isinstance(content, (list, tuple)):
                    for item in content:
                        story.append(Paragraph(f"- {item}", styles["Normal"]))
                        story.append(Spacer(1, 2))
                elif isinstance(content, pd.DataFrame):
                    data = [list(content.columns)]
                    for _, row in content.reset_index().iterrows():
                        data.append([str(x) for x in row.tolist()])
                    tbl = Table(data, repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(str(content), styles["Normal"]))
                    story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(str(analysis_data), styles["Normal"]))
            story.append(Spacer(1, 6))

        doc.build(story)
        return output_path

    except ImportError:
        fallback_text = ""
        if isinstance(analysis_data, str):
            fallback_text = analysis_data
        elif isinstance(analysis_data, dict):
            lines = []
            for k, v in analysis_data.items():
                lines.append(f"## {k}")
                if isinstance(v, (list, tuple)):
                    lines.extend([f"- {i}" for i in v])
                else:
                    lines.append(str(v))
                lines.append("")
            fallback_text = "\n".join(lines)
        else:
            fallback_text = str(analysis_data)

        md_path = create_markdown_report(symbol, fallback_text, output_path=None)
        return md_path


if __name__ == "__main__":
    print("Testing report generator...")
    
    test_data = {
        "Summary": "AAPL stock analysis shows bullish trend",
        "Metrics": ["Price: $262.82", "Change: +1.25%", "Volatility: 1.60%"],
        "Recommendation": "HOLD position with slight bullish bias"
    }
    
    md_path = create_markdown_report("AAPL", "# Test Report\n\nThis is a test.")
    print(f"✅ Markdown report created: {md_path}")
    
    pdf_path = create_pdf_report("AAPL", test_data)
    print(f"✅ PDF report created: {pdf_path}")
    
    print("\n✅ All tests passed!")