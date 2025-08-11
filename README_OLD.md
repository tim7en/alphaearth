# Uzbekistan AlphaEarth Research Explorer

Discover ongoing geospatial research into Uzbekistan’s landscapes, ecosystems, and urban environments,
powered by **AlphaEarth embeddings** (radar + optical). This repository contains a reproducible pipeline,
QA plan, and research-grade reporting for:

- Soil Moisture Analysis
- Afforestation & Reforestation Planning
- Land Degradation Trends
- Riverbank Disturbance Analysis
- Protected Area Disturbance Analysis
- Biodiversity Disturbance Analysis
- Urban Heat Analysis

## Quick Start

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run orchestrator (dry run prints planned tasks)
python -m aeuz.orchestrator --dry-run

# 3) Execute a module
python -m aeuz.orchestrator --run soil_moisture
```

See `RUNBOOK.md` for details. CI/CD is configured in `.github/workflows/alphaearth.yml`.
