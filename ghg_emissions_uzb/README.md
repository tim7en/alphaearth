# Real Atmospheric Data Analysis for Uzbekistan

A comprehensive atmospheric concentration analysis system using **real satellite data** from Google Earth Engine. This project provides high-resolution atmospheric gas concentration mapping and analysis for Uzbekistan using Sentinel-5P satellite measurements and MODIS auxiliary data.

![Atmospheric Analysis](https://img.shields.io/badge/Analysis-Atmospheric%20Data-green) ![Resolution](https://img.shields.io/badge/Resolution-1km-blue) ![Status](https://img.shields.io/badge/Status-Operational-brightgreen) ![Data](https://img.shields.io/badge/Data-Real%20Satellite-orange)

## 🎯 Objectives

- **Real Satellite Data**: Use actual Sentinel-5P atmospheric concentration measurements
- **High-Resolution Analysis**: 1km spatial resolution atmospheric mapping
- **Multi-Gas Analysis**: CO, NO₂, CH₄, SO₂ concentrations from space
- **Server-Side Processing**: All computation on Google Earth Engine servers
- **Spatial Statistics**: Intra-urban variability and uncertainty quantification
- **No Mock Data**: 100% real measurements, no simulated data

## 🏗️ Project Structure

```
ghg_emissions_uzb/
├── src/
│   ├── utils.py                    # Utility functions and data loading
│   └── ghg_downscaling.py         # Legacy analysis module
├── data/                          # Input and processed datasets
├── outputs/                       # Analysis results and atmospheric data
│   ├── real_atmospheric_data_cities.csv
│   ├── test_atmospheric_data.csv
│   ├── high_resolution_atmospheric_data.csv
│   └── server_side_atmospheric_analysis.json
├── figs/                          # Generated maps and visualizations
├── reports/                       # Technical reports and documentation
├── gee_auth.py                    # Google Earth Engine authentication
├── project_gee_auth.py            # Project-specific GEE authentication
├── quick_atmospheric_analysis.py  # Fast city-level analysis
├── test_server_analysis.py        # Lightweight testing script
├── high_resolution_analysis.py    # 1km resolution full analysis
├── server_side_analysis.py        # Complete server-side processing
├── config_ghg.json               # Configuration parameters
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to the atmospheric analysis directory
cd ghg_emissions_uzb

# Install dependencies
pip install -r requirements.txt
```

### 2. Google Earth Engine Authentication

**Required** for real satellite data access:

```bash
# Project-specific authentication (recommended)
python project_gee_auth.py
```

**Note**: This uses your Google Earth Engine project `ee-sabitovty`. Authentication is required for real data access.

### 3. Run Atmospheric Analysis

**Quick Test (3 cities, fast)**:
```bash
python quick_atmospheric_analysis.py
```

**Lightweight Test (3 cities, 30 days)**:
```bash
python test_server_analysis.py
```

**High-Resolution Analysis (10 cities, 1km resolution)**:
```bash
python high_resolution_analysis.py
```

**Full Server-Side Analysis (comprehensive)**:
```bash
python server_side_analysis.py
```

## 📊 Features

### Real Satellite Data Sources

- **Sentinel-5P OFFL**: Carbon Monoxide (CO) column density
- **Sentinel-5P OFFL**: Tropospheric Nitrogen Dioxide (NO₂) 
- **Sentinel-5P OFFL**: Methane (CH₄) mixing ratios
- **Sentinel-5P NRTI**: Sulfur Dioxide (SO₂) column density
- **MODIS**: Land surface temperature, vegetation indices, land cover
- **Server-Side Processing**: All computation on Google Earth Engine servers

### Analysis Capabilities

- **High-Resolution**: 1km spatial resolution atmospheric mapping
- **Multi-Temporal**: 3-month analysis periods with temporal statistics
- **Spatial Statistics**: Intra-urban variability analysis (9 points per city)
- **Quality Control**: Uncertainty quantification and data validation
- **Regional Analysis**: Administrative region-level aggregation

### Processing Modes

1. **Quick Analysis**: Fast city-level sampling (5 cities, 2 gases)
2. **Test Mode**: Lightweight validation (3 cities, 30 days, 2 gases)
3. **High-Resolution**: Full analysis (10 cities, 3 months, 3 gases, 1km resolution)
4. **Server-Side**: Complete atmospheric analysis with regional statistics

### Outputs

1. **High-Resolution Atmospheric Concentration Maps**
   - CO, NO₂, CH₄, SO₂ concentrations
   - City-level spatial variability analysis
   - Temporal statistics and uncertainty bounds

2. **Analysis Results**
   - City-level concentration statistics
   - Regional atmospheric analysis
   - Spatial variability metrics
   - Data quality assessments

3. **Comprehensive Reports**
   - Real data processing logs
   - Statistical analysis summaries
   - Methodology documentation
   - Quality control results

## 📈 Latest Results (August 2025)

### High-Resolution Analysis Results

**10 Cities Analyzed at 1km Resolution**:
- **Tashkent**: NO₂: 1.52e-04 ± 1.43e-05 mol/m² (highest pollution)
- **Samarkand**: NO₂: 7.45e-05 ± 2.84e-06 mol/m²
- **Namangan**: NO₂: 6.32e-05 ± 8.25e-06 mol/m²
- **Andijan**: NO₂: 6.57e-05 ± 5.05e-06 mol/m²
- **Nukus**: NO₂: 2.81e-05 ± 5.26e-07 mol/m² (lowest pollution)

**Processing Statistics**:
- **Satellite Images Processed**: 3,852 images (1,284 per gas)
- **Sampling Points**: 90 total (9 per city for spatial variability)
- **Processing Time**: 6.7 minutes for full high-resolution analysis
- **Data Coverage**: July-September 2024 (3-month period)

## 🎛️ Usage Modes

### 1. Quick Atmospheric Analysis
```bash
python quick_atmospheric_analysis.py
```
- **Purpose**: Fast testing and validation
- **Scope**: 5 major cities, 2 gases (CO, NO₂)
- **Duration**: ~12 seconds
- **Resolution**: 5km (for speed)

### 2. Test Server Analysis
```bash
python test_server_analysis.py
```
- **Purpose**: Lightweight testing with reduced scope
- **Scope**: 3 cities, 2 gases, 30-day period
- **Duration**: ~12 seconds
- **Resolution**: 10km

### 3. High-Resolution Analysis
```bash
python high_resolution_analysis.py
```
- **Purpose**: Full atmospheric analysis (recommended)
- **Scope**: 10 cities, 3 gases, 3-month period
- **Duration**: ~7 minutes
- **Resolution**: 1km with spatial statistics

### 4. Server-Side Analysis
```bash
python server_side_analysis.py
```
- **Purpose**: Comprehensive regional analysis
- **Scope**: Entire Uzbekistan, all available gases
- **Duration**: Variable (depends on region size)
- **Resolution**: 1km with regional aggregation

## ⚙️ Configuration

Edit `config_ghg.json` to customize:

- **Analysis period**: Start/end years
- **Spatial resolution**: Target resolution for downscaling
- **Data sources**: Enable/disable specific emissions datasets
- **Model parameters**: ML algorithm settings
- **Output options**: Visualization and reporting preferences

## 📈 Methodology

### 1. Satellite Data Collection
- **Sentinel-5P**: Real-time atmospheric concentration retrieval
- **MODIS**: Auxiliary environmental variables (LST, NDVI, land cover)
- **Server-side processing**: Efficient cloud-based computation
- **Quality filtering**: Cloud masking and data validation

### 2. Spatial Analysis
- **Grid generation**: 1km resolution sampling points
- **Urban sampling**: Up to 90 points per city for spatial coverage
- **Coordinate transformation**: WGS84 to local projections
- **Spatial statistics**: Mean, standard deviation, coefficient of variation

### 3. Temporal Integration
- **Multi-temporal analysis**: 3-month seasonal periods
- **Data aggregation**: Statistical summaries across time
- **Trend analysis**: Temporal patterns and anomaly detection
- **Uncertainty quantification**: Data quality and completeness metrics

### 4. Results Processing
- **Multi-format output**: CSV tables and JSON metadata
- **City-level aggregation**: Statistical summaries by urban area
- **Quality assessment**: Data coverage and reliability metrics
- **Validation**: Cross-checks between multiple gas species

### 4. Validation and Analysis
- Model performance assessment (R², RMSE, MAE)
- Spatial pattern analysis
- Regional emissions aggregation
- Hotspot identification

## 🗺️ Output Examples

### Emissions Intensity Maps
- **Overall Emissions**: Country-wide emissions distribution
- **Hotspot Analysis**: Top 5% emission areas highlighted
- **Regional Breakdown**: Emissions by administrative region
- **Sector Analysis**: Power, industry, transport, residential emissions

### Analysis Products
- **Model Performance**: Cross-validation scores, feature importance
- **Regional Statistics**: Emissions totals by region
- **Temporal Trends**: Year-to-year changes (if multi-year data available)
- **Technical Report**: Complete methodology and findings

## 🔧 Technical Requirements

### Google Earth Engine Setup
```bash
# Authenticate with project-specific access
python project_gee_auth.py
```

### Python Environment
```bash
# Install atmospheric analysis dependencies
pip install -r requirements.txt

# Required packages:
# - earthengine-api
# - pandas
# - numpy  
# - matplotlib
# - seaborn
# - scikit-learn
# - geopandas
```

### Data Sources
- **Sentinel-5P**: Real atmospheric concentrations (CO, NO₂, CH₄, SO₂)
- **MODIS**: Land surface temperature, vegetation indices, land cover
- **Server-side processing**: All computation on Google Earth Engine

### Spatial Coverage
- **Cities**: 10 major Uzbekistan urban centers
- **Resolution**: 1km grid for intra-urban analysis
- **Sampling**: Up to 90 points per city for spatial statistics
- **Temporal**: 3-month periods for seasonal analysis

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space for outputs
- **Internet**: Required for GEE authentication and data access

## 🚨 Troubleshooting

### Common Issues

**1. Google Earth Engine Authentication**
```bash
# If authentication fails, run:
python project_gee_auth.py

# Or check project access:
earthengine authenticate --project=ee-sabitovty
```

**2. Data Access Issues**
```bash
# Test Sentinel-5P data availability:
python quick_atmospheric_analysis.py

# Verify server connection:
python test_server_analysis.py
```

**3. Memory/Performance Issues**
```bash
# Use lightweight test mode:
python test_server_analysis.py  # 3 cities only

# Or reduce sampling density in config
```

**4. Output File Issues**
- Check write permissions in output directory
- Ensure sufficient disk space (>1GB recommended)
- Verify JSON/CSV file format compatibility
```

**2. GEE Authentication Fails**
- Analysis continues in simulation mode
- Use `python gee_auth.py` for manual setup
- Check internet connection and Google account access

**3. Memory Issues**
- Reduce `target_resolution` in config (e.g., 500m instead of 200m)
- Process smaller regions separately

**4. No Output Generated**
- Check file permissions in output directories
- Verify configuration file syntax (JSON format)
- Review error messages in terminal

### Performance Optimization

- **Faster Processing**: Increase spatial resolution (e.g., 500m vs 200m)
- **Better Quality**: Decrease spatial resolution, more training data
- **Memory Efficiency**: Process regions separately, reduce grid density

## 🔬 Scientific Validation

### Model Validation
- **Cross-validation**: 5-fold spatial cross-validation
- **Test Set**: 20% hold-out for independent validation  
- **Performance Metrics**: R², RMSE, MAE for emissions prediction
- **Uncertainty**: Prediction intervals and confidence bounds

### Quality Control
- **Data Quality**: Outlier detection and removal
- **Spatial Consistency**: Neighboring pixel correlation analysis
- **Mass Conservation**: Total emissions conservation check
- **Physical Constraints**: Non-negative emissions enforcement

## 📚 References

### Data Sources
- **ODIAC**: Oda, T. et al. (2018). Open-Data Inventory for Anthropogenic Carbon dioxide
- **EDGAR**: Janssens-Maenhout, G. et al. (2019). EDGAR v4.3.2 Global Atlas of GHG emissions
- **Auxiliary Data**: Multiple sources including WorldPop, OpenStreetMap, climate reanalysis

### Methodology
- **Spatial Downscaling**: Gately, C.K. et al. (2017). Cities, traffic, and CO2
- **Machine Learning**: Breiman, L. (2001). Random Forests
- **Uncertainty**: Quantifying uncertainty in spatial predictions

## 📄 License

This project builds upon AlphaEarth patterns and methodologies. Please refer to appropriate licensing terms for derivative works.

## 👥 Contributors

- **AlphaEarth Analysis Team - GHG Module**
- **Methodology**: Based on established spatial downscaling techniques
- **Implementation**: Standalone system adapted from AlphaEarth patterns

## 📞 Support

For technical questions or methodology details:
1. Check the generated technical report in `reports/`
2. Review configuration options in `config_ghg.json`
3. Examine model performance metrics in `outputs/`
4. Consult the interactive help system: Option 6 in main menu

---

**Note**: This is a standalone implementation that can complement AlphaEarth projects but operates independently. It uses proven authentication and analysis patterns from AlphaEarth while maintaining complete separation.