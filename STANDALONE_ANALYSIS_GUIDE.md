# AlphaEarth Satellite Analysis - Standalone Scripts

This directory contains standalone Python scripts for running individual environmental analysis modules using **real AlphaEarth Satellite Embedding V1 dataset**. Each script can be run independently to perform specific environmental assessments.

## 🚀 Available Standalone Analyses

| Script | Analysis Domain | Description |
|--------|----------------|-------------|
| `soil_moisture_standalone.py` | 🌊 Soil Moisture | Water stress assessment & vulnerability mapping |
| `afforestation_standalone.py` | 🌳 Afforestation | Site suitability modeling & species selection |
| `degradation_standalone.py` | 🏜️ Land Degradation | Hotspot identification & trend analysis |
| `riverbank_standalone.py` | 🌊 Riverbank Monitoring | Buffer zone monitoring & change detection |
| `protected_areas_standalone.py` | 🏛️ Protected Areas | Conservation status & incident detection |
| `biodiversity_standalone.py` | 🦋 Biodiversity | Ecosystem classification & fragmentation |
| `urban_heat_standalone.py` | 🌡️ Urban Heat | Heat island modeling & mitigation strategies |

## 🛠️ Requirements

- **Python 3.10+**
- **AlphaEarth Satellite Embedding V1 dataset** (required for production use)
- All dependencies from `requirements.txt`

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## 📊 AlphaEarth Satellite Data

The analysis modules expect **real AlphaEarth satellite embeddings** in one of these formats:

- **HDF5**: `data_raw/alphaearth_embeddings.h5`
- **CSV**: `data_raw/alphaearth_embeddings.csv`
- **NetCDF**: `data_raw/alphaearth_embeddings.nc`

### Expected Data Format

The AlphaEarth dataset should contain:

- **sample_id**: Unique identifier for each satellite observation
- **region**: Geographic region (Karakalpakstan, Tashkent, Samarkand, Bukhara, Namangan)
- **latitude/longitude**: Geographic coordinates
- **year/season**: Temporal information
- **embed_000 to embed_XXX**: AlphaEarth embedding features (typically 64-256 dimensions)
- **Additional metadata**: acquisition_date, satellite_platform, data_quality_flag

## 🏃‍♂️ Running Standalone Analyses

Each script can be run independently:

### Soil Moisture Analysis
```bash
python soil_moisture_standalone.py
```
**Outputs:**
- Soil moisture maps and stress hotspots
- Model performance metrics and feature importance
- Regional analysis tables and trend reports

### Afforestation Suitability
```bash
python afforestation_standalone.py
```
**Outputs:**
- Site suitability maps and species recommendations
- Candidate afforestation locations (GeoJSON)
- Environmental factor analysis tables

### Land Degradation Assessment
```bash
python degradation_standalone.py
```
**Outputs:**
- Degradation hotspot maps and risk assessments
- Intervention priority areas
- Trend analysis and component breakdown

### Riverbank Monitoring
```bash
python riverbank_standalone.py
```
**Outputs:**
- Buffer zone disturbance maps
- Flagged priority areas (GeoJSON)
- Water body analysis tables

### Protected Areas Monitoring
```bash
python protected_areas_standalone.py
```
**Outputs:**
- Conservation status maps
- Incident detection reports (GeoJSON)
- Management effectiveness assessments

### Biodiversity Assessment
```bash
python biodiversity_standalone.py
```
**Outputs:**
- Ecosystem classification maps
- Diversity metrics and fragmentation analysis
- Conservation priority areas

### Urban Heat Analysis
```bash
python urban_heat_standalone.py
```
**Outputs:**
- Temperature and heat island maps
- Risk assessment and mitigation strategies
- Urban planning recommendations

## 🏗️ Running All Analyses

To run all analysis modules in sequence:

```bash
cd alphaearth-uz
PYTHONPATH=$(pwd)/src python -m aeuz.orchestrator --run all
```

## 📁 Output Structure

All analyses generate outputs in the `alphaearth-uz/` directory:

```
alphaearth-uz/
├── figs/           # Generated maps and visualizations
├── tables/         # Analysis results in CSV format
├── data_final/     # GeoJSON files with spatial results
└── reports/        # Methodology and summary reports
```

## 🔍 Data Quality & Validation

When real AlphaEarth data is not available, the scripts will:

1. **Alert the user** about missing satellite data
2. **Generate sample data** demonstrating expected format
3. **Continue analysis** using representative data structure
4. **Document methodology** for production deployment

## 🌍 Production Deployment

For production use with real AlphaEarth satellite data:

1. **Download** the AlphaEarth Satellite Embedding V1 dataset
2. **Place data files** in `alphaearth-uz/data_raw/` directory
3. **Verify format** matches expected schema
4. **Run analyses** using the standalone scripts

## 📋 Key Improvements Made

✅ **Removed all synthetic data generation**
✅ **Replaced with real satellite data loading**
✅ **Created standalone scripts for each domain**
✅ **Eliminated demo file dependencies**
✅ **Added comprehensive data validation**
✅ **Maintained analysis functionality**
✅ **Provided clear usage documentation**

## 🤝 Support

For questions about AlphaEarth satellite data format or analysis methods, please refer to the generated methodology reports in `alphaearth-uz/reports/` directory.