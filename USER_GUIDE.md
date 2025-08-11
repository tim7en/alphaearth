# AlphaEarth Uzbekistan Analysis - User Guide

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- 4GB available disk space
- Internet connection for dependency installation

### Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/tim7en/alphaearth.git
cd alphaearth
```

2. **Set up Python environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the analysis:**

**Option A: Use the Jupyter Notebook (Recommended)**
```bash
jupyter notebook AlphaEarth_Uzbekistan_Comprehensive_Analysis.ipynb
```
Then run all cells sequentially or use "Run All".

**Option B: Use the Command Line Interface**
```bash
cd alphaearth-uz
PYTHONPATH=$(pwd)/src python -m aeuz.orchestrator --run all
```

**Option C: Use the Makefile**
```bash
make setup  # Install dependencies
make run    # Run all analyses
make qa     # Run quality assurance
```

## Analysis Modules

The AlphaEarth Uzbekistan analysis includes 7 core environmental assessment modules:

### 1. Soil Moisture Analysis (`soil_moisture`)
- **Purpose**: Water stress assessment and vulnerability mapping
- **Techniques**: Random Forest regression, statistical trend analysis
- **Outputs**: Regional moisture maps, stress hotspots, prediction models
- **Key Metrics**: National average moisture, water stress areas, model performance

### 2. Afforestation Planning (`afforestation`)
- **Purpose**: Site suitability modeling and species selection
- **Techniques**: Binary classification, regression modeling, species matching
- **Outputs**: Suitability maps, species recommendations, cost estimates
- **Key Metrics**: Suitable sites, suitability scores, survival probabilities

### 3. Land Degradation Assessment (`degradation`)
- **Purpose**: Degradation hotspot identification and trend analysis
- **Techniques**: Anomaly detection, temporal trend analysis, risk assessment
- **Outputs**: Degradation maps, hotspot locations, intervention priorities
- **Key Metrics**: Degradation scores, hotspot counts, priority areas

### 4. Riverbank Disturbance Analysis (`riverbank`)
- **Purpose**: Buffer zone monitoring and disturbance mapping
- **Techniques**: Clustering analysis, buffer zone assessment, change detection
- **Outputs**: Disturbance maps, buffer integrity analysis, intervention sites
- **Key Metrics**: Disturbance scores, buffer widths, priority interventions

### 5. Protected Area Monitoring (`protected_areas`)
- **Purpose**: Conservation status assessment and incident detection
- **Techniques**: Anomaly detection, conservation status classification
- **Outputs**: Conservation status maps, incident reports, management effectiveness
- **Key Metrics**: Conservation status, incident counts, threat levels

### 6. Biodiversity Analysis (`biodiversity`)
- **Purpose**: Ecosystem classification and fragmentation assessment
- **Techniques**: Clustering, PCA, fragmentation metrics
- **Outputs**: Ecosystem maps, diversity metrics, conservation priorities
- **Key Metrics**: Habitat quality, ecosystem types, fragmentation indices

### 7. Urban Heat Analysis (`urban_heat`)
- **Purpose**: Heat island modeling and mitigation strategies
- **Techniques**: Random Forest regression, spatial analysis, cooling potential
- **Outputs**: Temperature maps, heat risk assessment, mitigation strategies
- **Key Metrics**: Land surface temperature, UHI intensity, cooling potential

## Understanding the Outputs

### Directory Structure
```
alphaearth-uz/
├── tables/          # CSV data files
├── figs/           # PNG visualizations
├── data_final/     # GeoJSON spatial data
├── reports/        # Executive summaries
├── qa/             # Quality assurance reports
└── src/aeuz/       # Source code modules
```

### File Types

**CSV Tables** (`tables/`)
- `*_regional_analysis.csv`: Regional summary statistics
- `*_model_performance.csv`: Machine learning model metrics
- `*_hotspots.csv`: Identified problem areas
- `*_priorities.csv`: Action priority rankings

**PNG Figures** (`figs/`)
- `*_overview_analysis.png`: Comprehensive overview plots
- `*_spatial_analysis.png`: Geographic distribution maps
- `*_feature_importance.png`: Model feature rankings

**GeoJSON Files** (`data_final/`)
- `*_candidates.geojson`: Recommended intervention sites
- `*_flags.geojson`: Problem area markers
- `*_incidents.geojson`: Detected environmental incidents

**Reports** (`reports/`)
- `executive_summary.md`: High-level findings and recommendations
- `integrated_analysis_summary.json`: Machine-readable results
- `AlphaEarth_Uzbekistan_Report.md`: Comprehensive technical report

## Quality Assurance

The analysis includes comprehensive quality assurance (QA):

### QA Metrics
- **File Completion**: Tracks generation of all expected outputs
- **Data Quality**: Validates completeness, consistency, outliers
- **Statistical Validation**: Checks regional coverage, value ranges
- **Model Performance**: Validates ML model metrics

### Interpreting QA Results
- **PASS**: All critical checks passed, high confidence in results
- **WARNING**: Minor issues detected, results generally reliable
- **FAIL**: Critical issues found, results require investigation

### QA Report Location
View detailed QA results at: `alphaearth-uz/qa/qa_report.md`

## Regional Analysis

The analysis covers 5 key regions of Uzbekistan:

1. **Karakalpakstan**: Aral Sea region, high water stress
2. **Tashkent**: Capital region, urban heat focus
3. **Samarkand**: Historical region, balanced development
4. **Bukhara**: Desert edge, irrigation-dependent
5. **Namangan**: Fergana Valley, agricultural focus

### Regional Priorities
Each region receives customized recommendations based on:
- Environmental vulnerabilities
- Economic development patterns
- Population density
- Existing infrastructure
- Conservation needs

## Advanced Usage

### Running Individual Modules
```bash
cd alphaearth-uz
PYTHONPATH=$(pwd)/src python -m aeuz.orchestrator --run soil_moisture
PYTHONPATH=$(pwd)/src python -m aeuz.orchestrator --run afforestation
# ... etc for other modules
```

### Customizing Analysis
Edit `config.json` to modify:
- Time window
- Random seeds (for reproducibility)
- Output paths
- Regional focus

### Dry Run (Preview)
```bash
PYTHONPATH=$(pwd)/src python -m aeuz.orchestrator --dry-run
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/path/to/alphaearth/alphaearth-uz/src
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

**Memory Issues**
- Reduce sample sizes in synthetic data generation
- Run modules individually instead of all at once
- Ensure 4GB+ RAM available

**File Permission Errors**
```bash
# Ensure write permissions for output directories
chmod -R 755 alphaearth-uz/
```

### Getting Help

1. Check the QA report for data quality issues
2. Review module logs for error messages  
3. Ensure all dependencies are installed
4. Verify Python version (3.10+ required)
5. Check available disk space (500MB+ recommended)

## Research Applications

### Environmental Monitoring
- Baseline environmental assessments
- Change detection and trend analysis
- Impact assessment for development projects
- Climate change vulnerability assessment

### Policy Support
- Land use planning
- Conservation priority setting
- Infrastructure development guidance
- Environmental regulation enforcement

### Academic Research
- Multi-temporal environmental analysis
- Machine learning applications in environmental science
- Regional environmental comparison studies
- Method validation and benchmarking

## Data Sources & Methodology

### AlphaEarth Embeddings
- Synthetic satellite-derived environmental indicators
- Multi-spectral and radar data fusion
- 128-256 dimensional feature vectors
- Regional characteristic patterns

### Machine Learning Models
- **Random Forest**: Soil moisture, urban heat prediction
- **Gradient Boosting**: Afforestation suitability
- **XGBoost**: Regression tasks
- **Clustering**: Biodiversity, anomaly detection
- **Isolation Forest**: Degradation hotspots

### Statistical Methods
- Mann-Kendall trend tests
- Confidence interval estimation
- Outlier detection
- Regional summary statistics

## Citation & Acknowledgments

When using this analysis, please cite:

```
AlphaEarth Uzbekistan Environmental Analysis (2025).
Comprehensive environmental assessment using satellite embeddings and machine learning.
Available at: https://github.com/tim7en/alphaearth
```

### Academic References
See `CITATIONS.bib` for complete reference list.

### License
This analysis is provided for research and educational purposes. 
Check repository license for usage terms.

---

## Quick Reference Commands

```bash
# Complete analysis pipeline
make run

# Quality assurance only  
make qa

# Individual module
PYTHONPATH=$(pwd)/src python -m aeuz.orchestrator --run <module_name>

# Start Jupyter notebook
jupyter notebook AlphaEarth_Uzbekistan_Comprehensive_Analysis.ipynb
```

**Modules**: `soil_moisture`, `afforestation`, `degradation`, `riverbank`, `protected_areas`, `biodiversity`, `urban_heat`, `synthesis`, `qa_module`, `publish`

---

*For technical support, please review the documentation files in the repository or check the QA reports for detailed diagnostics.*