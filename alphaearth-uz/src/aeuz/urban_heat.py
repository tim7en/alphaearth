from pathlib import Path
import pandas as pd
from .utils import load_config, ensure_dir

def run():
    cfg = load_config()
    figs = cfg["paths"]["figs"]
    tables = cfg["paths"]["tables"]
    ensure_dir(figs); ensure_dir(tables)
    # TODO: implement LST downscaling using embeddings + NDVI
    Path(f"{figs}/urban_heat_tashkent.png").write_text("placeholder")
    df = pd.DataFrame({"district":["Example"],"heat_risk":["High"]})
    Path(f"{tables}/urban_heat_scores.csv").write_text(df.to_csv(index=False))
    return {"status":"ok","artifacts":["figs/urban_heat_tashkent.png","tables/urban_heat_scores.csv"]}
