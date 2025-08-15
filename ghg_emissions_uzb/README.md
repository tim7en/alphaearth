# GHG Emissions Downscaling for Uzbekistan

A comprehensive, standalone greenhouse gas (GHG) emissions downscaling system for Uzbekistan using machine learning and geospatial data. This project provides high-resolution (200m) emissions mapping by downscaling coarse-resolution (1km) satellite and inventory data.

![GHG Emissions Analysis](https://img.shields.io/badge/Analysis-GHG%20Emissions-green) ![Resolution](https://img.shields.io/badge/Resolution-200m-blue) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🎯 Objectives

- **Spatial Downscaling**: Transform 1km resolution emissions data to 200m high-resolution maps
- **Multi-source Integration**: Combine ODIAC, EDGAR, and other emissions datasets
- **Machine Learning**: Use advanced ML algorithms for spatial prediction
- **Comprehensive Analysis**: Sector-specific emissions, uncertainty quantification, and validation
- **Standalone Operation**: Complete separation from AlphaEarth project while leveraging proven patterns

## 🏗️ Project Structure

```
ghg_emissions_uzb/
├── src/
│   ├── utils.py              # Utility functions and data loading
│   └── ghg_downscaling.py    # Main analysis module
├── data/                     # Input and processed datasets
├── outputs/                  # Analysis results and model outputs
├── figs/                     # Generated maps and visualizations
├── reports/                  # Technical reports and documentation
├── gee_auth.py              # Google Earth Engine authentication
├── ghg_downscaling_uzb.py   # Main executable script
├── config_ghg.json          # Configuration parameters
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or copy the ghg_emissions_uzb folder
cd ghg_emissions_uzb

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Google Earth Engine Authentication

For real satellite data (recommended but not required):

```bash
python gee_auth.py
```

If GEE authentication fails, the system will automatically run in simulation mode with realistic synthetic data.

### 3. Run Analysis

```bash
python ghg_downscaling_uzb.py
```

Follow the interactive menu to select analysis options.

## 📊 Features

### Data Sources

- **ODIAC**: Fossil fuel CO2 emissions from GEE
- **EDGAR**: Sectoral emissions inventory (when available)
- **Auxiliary Data**: Population, land use, infrastructure, climate variables
- **Simulation Mode**: Realistic synthetic data when satellite data unavailable

### Machine Learning Models

- **Random Forest**: Ensemble method for robust predictions
- **Gradient Boosting**: Advanced boosting for complex patterns
- **Cross-validation**: 5-fold CV for model validation
- **Feature Importance**: Analysis of predictor variable contributions

### Spatial Downscaling

- **Input Resolution**: 1km (ODIAC/EDGAR native resolution)
- **Output Resolution**: 200m (configurable)
- **Coverage**: Complete Uzbekistan territory
- **Grid Points**: ~50,000 high-resolution prediction points

### Outputs

1. **High-Resolution Emissions Maps**
   - CO2, CH4, N2O emissions intensity
   - Sector-specific breakdowns
   - Emissions hotspot identification

2. **Analysis Results**
   - Regional emissions summaries
   - Model performance metrics
   - Uncertainty quantification

3. **Technical Reports**
   - Comprehensive methodology documentation
   - Model validation results
   - Key findings and recommendations

## 🎛️ Usage Modes

### 1. Complete Analysis (Recommended)
Runs end-to-end analysis including data loading, model training, prediction, and visualization.

### 2. Authentication Setup
Configure Google Earth Engine access for real satellite data.

### 3. Data Exploration
Load and examine emissions datasets without model training.

### 4. Model Training
Train machine learning models using loaded data.

### 5. Mapping Only
Generate visualizations from existing analysis results.

## ⚙️ Configuration

Edit `config_ghg.json` to customize:

- **Analysis period**: Start/end years
- **Spatial resolution**: Target resolution for downscaling
- **Data sources**: Enable/disable specific emissions datasets
- **Model parameters**: ML algorithm settings
- **Output options**: Visualization and reporting preferences

## 📈 Methodology

### 1. Data Integration
- Load emissions data from multiple sources (ODIAC, EDGAR)
- Generate auxiliary geospatial predictors (population, land use, infrastructure)
- Spatial matching between emissions points and predictor variables

### 2. Model Training
- Feature engineering and preprocessing
- Multiple ML algorithm comparison (Random Forest, Gradient Boosting)
- Cross-validation and hyperparameter optimization
- Feature importance analysis

### 3. Spatial Prediction
- Create high-resolution prediction grid (200m spacing)
- Interpolate auxiliary variables to prediction grid
- Apply trained models for emissions prediction
- Uncertainty quantification

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

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space for outputs
- **Internet**: Required for GEE authentication (optional)

### Dependencies
- **Core**: numpy, pandas, scikit-learn, matplotlib, seaborn
- **Geospatial**: geopandas, rasterio, shapely (optional)
- **Earth Engine**: earthengine-api, geemap (optional)

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install -r requirements.txt
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