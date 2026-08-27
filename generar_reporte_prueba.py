import pandas as pd
from pathlib import Path
from core.pipeline import ImputationPipeline
from imputers.diagnostico_runner import DiagnosticoRunner
from core.report_builder import build_report_json

df = pd.read_csv("wh2023_limpio.csv")

diagnostico = DiagnosticoRunner().run(df)
result = ImputationPipeline().run(df, goal="prediction")

report = build_report_json(
    diagnostico=diagnostico,
    pipeline_result=result,
    dataset_name="World Happiness 2023",
    output_path=Path("reporte_consolidado_wh2023.json"),
    df_original=df,
)
print("JSON consolidado guardado en reporte_consolidado_wh2023.json")
