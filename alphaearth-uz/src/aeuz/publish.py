from pathlib import Path
from .utils import load_config

def run():
    # TODO: format report and export PDF (placeholder creates md)
    Path("alphaearth-uz/reports/AlphaEarth_Uzbekistan_Report.md").write_text("# AlphaEarth Uzbekistan Report\n\nCompiled draft (placeholder).")
    return {"status":"ok","artifacts":["reports/AlphaEarth_Uzbekistan_Report.md"]}
