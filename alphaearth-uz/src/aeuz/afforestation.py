from pathlib import Path
import json
from .utils import load_config, ensure_dir

def run():
    cfg = load_config()
    final = cfg["paths"]["final"]
    ensure_dir(final)
    # TODO: implement clustering-based suitability
    Path(f"{final}/afforestation_candidates.geojson").write_text(json.dumps({"type":"FeatureCollection","features":[]}))
    return {"status":"ok","artifacts":["data_final/afforestation_candidates.geojson"]}
