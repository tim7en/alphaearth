# RUNBOOK

**Country/Region:** Uzbekistan (Karakalpakstan, Tashkent, Samarkand, Bukhara, Namangan)  
**Time Window:** 2017–2025  
**CRS:** EPSG:4326 for storage; reproject as needed for analysis  
**Last updated:** 2025-08-10

## Environment
- Python 3.10+
- Install dependencies via `pip install -r requirements.txt`

## Data Locations
- Raw: `alphaearth-uz/data_raw/`
- Working: `alphaearth-uz/data_work/`
- Final: `alphaearth-uz/data_final/`

## Tasks
Orchestrated by `python -m aeuz.orchestrator`:
- soil_moisture
- afforestation
- degradation
- riverbank
- protected_areas
- biodiversity
- urban_heat
- synthesis
- qa
- publish

## Reproducibility
- All tasks are deterministic given seeded configs (`alphaearth-uz/config.json`).
- Each model produces a `model_card.md` in `models/`.
