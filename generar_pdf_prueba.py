from pathlib import Path
from scripts.generate_pdf_report import generate_pdf

generate_pdf(
    report_json_path=Path("reporte_consolidado_wh2023.json"),
    output_pdf_path=Path("reporte_wh2023.pdf"),
)
print("PDF generado en reporte_wh2023.pdf")
