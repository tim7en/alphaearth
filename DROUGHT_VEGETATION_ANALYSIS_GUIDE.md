# Drought & Vegetation Anomaly Analysis Documentation

## Overview

This module implements a comprehensive drought and vegetation anomaly analysis system for Uzbekistan's agro-districts, addressing the research question:

**"Where and when have agro-districts experienced the deepest vegetation deficits since 2000? How did 2021 & 2023 compare?"**

## Features

### 🛰️ Mock Data Infrastructure
- **MOD13Q1 NDVI/EVI**: Mock 250m resolution, 16-day composites (2000-2023)
- **CHIRPS Precipitation**: Mock 5.5km resolution, daily data (2000-2023)
- Realistic seasonal patterns and drought signatures
- Production-ready replacement for Google Earth Engine data

### 📊 Analysis Components
1. **Z-Score Analysis**: NDVI/EVI anomalies vs 2001-2020 baseline
2. **SPI Calculation**: Standardized Precipitation Index (90-day window)
3. **Mann-Kendall Trends**: Pixel-wise vegetation trend analysis
4. **Hotspot Identification**: Districts with severe drought conditions
5. **Temporal Comparison**: 2021 vs 2023 drought analysis

### 🏭 Production Features
- **Memory Optimization**: Efficient processing of large datasets
- **Rate Limiting**: API-ready for future GEE integration
- **Data Validation**: Quality checks and validation scoring
- **Progress Tracking**: Real-time monitoring of analysis progress
- **Chunked Processing**: Memory-efficient data handling

## Quick Start

### Run Standalone Analysis
```bash
python drought_vegetation_standalone.py
```

### Integration with AlphaEarth Framework
```python
from aeuz import drought_vegetation_analysis

# Run complete analysis
results = drought_vegetation_analysis.run()
```

## Output Products

### 📈 Visualizations
- `drought_atlas_comprehensive.png`: Multi-panel drought atlas
- `drought_timeseries_analysis.png`: Time series for top affected districts

### 📊 Data Tables
- `drought_hotspots_ranking.csv`: Districts ranked by severity
- `drought_2021_vs_2023_comparison.csv`: Year-to-year comparison
- `vegetation_trend_analysis.csv`: Mann-Kendall trend results
- `drought_analysis_summary.json`: Comprehensive summary report

## Methodology

### Data Specifications
- **Temporal Coverage**: 2000-2023 (24 years)
- **Baseline Period**: 2001-2020 for z-score calculation
- **Spatial Coverage**: 13 Uzbekistan agro-districts
- **Growing Season**: April to September focus
- **Comparison Years**: 2021 vs 2023 detailed analysis

### Analysis Methods
1. **Vegetation Anomalies**:
   - Monthly z-scores calculated against 2001-2020 baseline
   - Seasonal adjustment for natural vegetation cycles
   - Threshold: -1.5 for severe drought classification

2. **Precipitation Analysis**:
   - SPI calculated using 90-day rolling windows
   - Standardized against 2001-2020 baseline period
   - Exponential distribution fitting for precipitation data

3. **Trend Analysis**:
   - Mann-Kendall test for monotonic trends
   - Significance testing at p < 0.05 level
   - Kendall's tau for trend strength measurement

4. **Severity Scoring**:
   - Combined NDVI, EVI, and SPI severe events
   - Weighted by frequency and magnitude
   - District ranking by total severity score

## Key Findings (Example Results)

### Most Affected Districts
1. **Samarkand**: Highest severity score (52.0)
2. **Jizzakh**: Second highest severity (50.0)
3. **Tashkent**: Third highest severity (50.0)

### Trend Analysis
- **0%** of districts show significant declining vegetation trends
- Long-term vegetation generally stable or improving
- No statistically significant degradation detected

### 2021 vs 2023 Comparison
- **2021**: More severe drought conditions in 12 districts
- **2023**: Improved conditions in most areas
- **1 district**: Worse drought in 2023 vs 2021

## Production Considerations

### Memory Management
- **Monitoring**: Real-time memory usage tracking
- **Optimization**: Automatic DataFrame memory optimization
- **Chunking**: Large dataset processing in manageable chunks
- **Thresholds**: Alerts for high memory usage (>1GB)

### API Rate Limiting
- **Simulation**: 5 calls/second rate limiting for GEE compatibility
- **Monitoring**: Call count tracking and logging
- **Scalability**: Ready for real GEE API integration

### Data Quality
- **Validation**: Automated quality scoring (0-100%)
- **Missing Data**: Detection and impact assessment
- **Outliers**: Statistical outlier identification
- **Reporting**: Quality metrics for each district/variable

## File Structure

```
drought_vegetation_analysis/
├── drought_vegetation_analysis.py   # Main analysis module
├── mock_gee_data.py                # Mock data generation
├── production_features.py          # Production optimizations
└── drought_vegetation_standalone.py # Standalone script
```

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
psutil>=5.8.0
```

## Configuration

### Analysis Parameters
```python
BASELINE_PERIOD = (2001, 2020)
COMPARISON_YEARS = [2021, 2023]
GROWING_SEASON = [4, 5, 6, 7, 8, 9]  # Apr-Sep
SEVERITY_THRESHOLD = -1.5
SPI_WINDOW_DAYS = 90
```

### Production Settings
```python
MEMORY_THRESHOLD_MB = 1000
API_RATE_LIMIT = 5.0  # calls/second
CHUNK_SIZE = 1000
ENABLE_CACHING = True
MAX_RETRIES = 3
TIMEOUT_SECONDS = 300
```

## Future Enhancements

### Real GEE Integration
1. Replace mock data with actual MOD13Q1 and CHIRPS
2. Implement authentication handling
3. Add retry logic for API failures
4. Optimize for GEE rate limits

### Additional Analysis
1. Sub-seasonal drought analysis
2. Compound drought-heat events
3. Agricultural impact assessment
4. Climate change attribution

### Visualization Improvements
1. Interactive web maps
2. Animation of drought progression
3. District-level dashboards
4. Real-time monitoring interface

## Support and Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce chunk size or increase system memory
2. **Mock Data**: Regenerate if corrupted using `force_recreate=True`
3. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Validation
- Run quality checks on generated data
- Verify output file generation
- Check analysis consistency across runs
- Validate against known drought events

## Citation

When using this analysis in research:

```bibtex
@software{drought_vegetation_analysis_2025,
  title={Drought & Vegetation Anomaly Analysis for Uzbekistan Agro-Districts},
  author={AlphaEarth Research Team},
  year={2025},
  url={https://github.com/tim7en/alphaearth},
  note={Mock GEE data implementation for production-ready drought analysis}
}
```

## License

This software is part of the AlphaEarth environmental analysis framework and follows the project's licensing terms.