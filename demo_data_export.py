#!/usr/bin/env python3
"""
Data Extraction Demo for Uzbekistan Urban Expansion Analysis
============================================================
This script demonstrates how to extract and download the original data
used in the urban expansion analysis.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

def create_demo_data_export():
    """Create a demonstration of data export with sample data"""
    print("💾 DEMO: EXTRACTING ORIGINAL DATA FROM URBAN EXPANSION ANALYSIS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/tmp"
    
    # 1. Demo satellite data structure
    print("📊 Creating sample satellite data structure...")
    
    # Sample cities from the configuration
    demo_cities = ["Tashkent", "Samarkand", "Bukhara"]
    demo_periods = ["period_2016", "period_2020", "period_2025"]
    
    demo_satellite_data = {}
    
    for period in demo_periods:
        period_data = []
        for city in demo_cities:
            # Create sample data points for each city
            for i in range(10):  # 10 sample points per city
                sample_point = {
                    'City': city,
                    'Period': period,
                    'Year_Range': period.split('_')[1],
                    'Sample_Longitude': 69.2401 + np.random.normal(0, 0.1),  # Around Tashkent
                    'Sample_Latitude': 41.2995 + np.random.normal(0, 0.1),
                    'LST_Day': np.random.normal(28, 5),  # Temperature in Celsius
                    'LST_Night': np.random.normal(18, 3),
                    'Built_Probability': np.random.uniform(0.1, 0.8),
                    'Green_Probability': np.random.uniform(0.2, 0.7),
                    'Water_Probability': np.random.uniform(0, 0.1),
                    'NDVI': np.random.uniform(0.1, 0.6),
                    'NDBI': np.random.uniform(0, 0.4),
                    'NDWI': np.random.uniform(-0.2, 0.3),
                    'EUI': np.random.uniform(0, 0.3),
                    'UHI_Intensity': np.random.uniform(1, 4),
                    'Green_Connectivity': np.random.uniform(0.2, 0.8),
                    'Impervious_Surface': np.random.uniform(0.1, 0.7),
                    'Nighttime_Lights': np.random.uniform(0, 2)
                }
                period_data.append(sample_point)
        
        demo_satellite_data[period] = pd.DataFrame(period_data)
        
        # Export each period
        csv_path = f"{output_dir}/DEMO_uzbekistan_satellite_data_{period}_{timestamp}.csv"
        demo_satellite_data[period].to_csv(csv_path, index=False)
        print(f"   ✅ {period}: {csv_path}")
    
    # 2. Demo city impacts data
    print("📈 Creating sample city impacts data...")
    
    impacts_data = []
    for city in demo_cities:
        impact = {
            'city': city,
            'temp_day_change_10yr': np.random.normal(2.0, 0.5),
            'temp_night_change_10yr': np.random.normal(3.0, 0.8),
            'uhi_change_10yr': np.random.normal(0.5, 0.2),
            'built_change_10yr': np.random.normal(0.05, 0.02),
            'green_change_10yr': np.random.normal(-0.02, 0.01),
            'water_change_10yr': np.random.normal(0, 0.005),
            'ndvi_change_10yr': np.random.normal(-0.01, 0.01),
            'connectivity_change_10yr': np.random.normal(-0.03, 0.01),
            'temp_day_rate_per_year': np.random.normal(0.2, 0.05),
            'built_expansion_rate_per_year': np.random.normal(0.005, 0.002),
            'samples_baseline': 50,
            'samples_latest': 50
        }
        impacts_data.append(impact)
    
    impacts_df = pd.DataFrame(impacts_data)
    impacts_df.set_index('city', inplace=True)
    
    impacts_csv = f"{output_dir}/DEMO_uzbekistan_city_impacts_{timestamp}.csv"
    impacts_df.to_csv(impacts_csv)
    print(f"   ✅ City impacts: {impacts_csv}")
    
    # 3. Demo regional statistics
    print("🌍 Creating sample regional statistics...")
    
    regional_stats = {
        'temp_day_change_10yr_mean': impacts_df['temp_day_change_10yr'].mean(),
        'temp_day_change_10yr_std': impacts_df['temp_day_change_10yr'].std(),
        'temp_night_change_10yr_mean': impacts_df['temp_night_change_10yr'].mean(),
        'uhi_change_10yr_mean': impacts_df['uhi_change_10yr'].mean(),
        'built_expansion_10yr_mean': impacts_df['built_change_10yr'].mean(),
        'green_change_10yr_mean': impacts_df['green_change_10yr'].mean(),
        'analysis_span_years': 10,
        'cities_analyzed': len(demo_cities),
        'total_samples': sum(len(df) for df in demo_satellite_data.values())
    }
    
    regional_json = f"{output_dir}/DEMO_uzbekistan_regional_stats_{timestamp}.json"
    with open(regional_json, 'w') as f:
        json.dump(regional_stats, f, indent=2)
    print(f"   ✅ Regional stats: {regional_json}")
    
    # 4. Demo combined dataset
    print("🔗 Creating combined analysis dataset...")
    
    # Combine latest satellite data with impact metrics
    latest_data = demo_satellite_data['period_2025']
    combined_data = []
    
    for _, row in latest_data.iterrows():
        city_name = row['City']
        if city_name in impacts_df.index:
            combined_row = row.to_dict()
            impact_row = impacts_df.loc[city_name]
            
            # Add impact metrics with prefix
            for col in impact_row.index:
                combined_row[f'impact_{col}'] = impact_row[col]
            
            combined_data.append(combined_row)
    
    combined_df = pd.DataFrame(combined_data)
    combined_csv = f"{output_dir}/DEMO_uzbekistan_combined_dataset_{timestamp}.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"   ✅ Combined dataset: {combined_csv}")
    
    # 5. Create comprehensive data dictionary
    print("📖 Creating data dictionary...")
    
    data_dictionary = {
        "dataset_info": {
            "title": "DEMO: Uzbekistan Urban Expansion Impact Analysis Dataset",
            "description": "Sample data structure for urban expansion analysis",
            "temporal_range": "2016-2025",
            "spatial_resolution": "100m",
            "cities_analyzed": len(demo_cities),
            "analysis_date": timestamp,
            "data_type": "DEMONSTRATION",
            "data_sources": [
                "Google Earth Engine (simulated)",
                "MODIS Land Surface Temperature",
                "Dynamic World Land Cover", 
                "Landsat 8/9 Surface Reflectance",
                "VIIRS Nighttime Lights"
            ]
        },
        "file_structure": {
            "satellite_data_files": {
                "pattern": "uzbekistan_satellite_data_{period}_{timestamp}.csv",
                "description": "Raw satellite observations by time period",
                "variables": {
                    "City": "Urban center name",
                    "Period": "Analysis time period",
                    "Sample_Longitude": "Sample point longitude (degrees)",
                    "Sample_Latitude": "Sample point latitude (degrees)",
                    "LST_Day": "Land Surface Temperature Day (°C)",
                    "LST_Night": "Land Surface Temperature Night (°C)",
                    "Built_Probability": "Built-up area probability (0-1)",
                    "Green_Probability": "Green space probability (0-1)",
                    "Water_Probability": "Water body probability (0-1)",
                    "NDVI": "Normalized Difference Vegetation Index",
                    "NDBI": "Normalized Difference Built-up Index",
                    "NDWI": "Normalized Difference Water Index",
                    "EUI": "Enhanced Urban Index",
                    "UHI_Intensity": "Urban Heat Island Intensity (°C)",
                    "Green_Connectivity": "Green space connectivity index",
                    "Impervious_Surface": "Impervious surface fraction",
                    "Nighttime_Lights": "VIIRS nighttime lights radiance"
                }
            },
            "impact_analysis_file": {
                "filename": "uzbekistan_city_impacts_{timestamp}.csv",
                "description": "City-level impact analysis results",
                "variables": {
                    "temp_day_change_10yr": "10-year day temperature change (°C)",
                    "temp_night_change_10yr": "10-year night temperature change (°C)",
                    "uhi_change_10yr": "10-year UHI intensity change (°C)",
                    "built_change_10yr": "10-year built-up expansion",
                    "green_change_10yr": "10-year green space change",
                    "water_change_10yr": "10-year water body change",
                    "ndvi_change_10yr": "10-year vegetation health change",
                    "connectivity_change_10yr": "10-year green connectivity change",
                    "temp_day_rate_per_year": "Annual temperature change rate",
                    "built_expansion_rate_per_year": "Annual urban expansion rate"
                }
            },
            "combined_dataset_file": {
                "filename": "uzbekistan_combined_dataset_{timestamp}.csv",
                "description": "Merged satellite data + impact analysis",
                "note": "Contains both raw observations and calculated impacts"
            }
        },
        "usage_examples": {
            "load_satellite_data": """
# Load satellite data for specific period
import pandas as pd
data_2025 = pd.read_csv('uzbekistan_satellite_data_period_2025_{timestamp}.csv')
print(data_2025.head())
            """,
            "load_impact_analysis": """
# Load city impact analysis
impacts = pd.read_csv('uzbekistan_city_impacts_{timestamp}.csv', index_col=0)
print(impacts.describe())
            """,
            "spatial_analysis": """
# Create spatial plots
import matplotlib.pyplot as plt
plt.scatter(data_2025['Sample_Longitude'], data_2025['Sample_Latitude'], 
           c=data_2025['LST_Day'], cmap='RdYlBu_r')
plt.colorbar(label='Day Temperature (°C)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Temperature Distribution')
plt.show()
            """,
            "time_series_analysis": """
# Analyze temporal changes
periods = ['period_2016', 'period_2020', 'period_2025']
for period in periods:
    df = pd.read_csv(f'uzbekistan_satellite_data_{period}_{timestamp}.csv')
    print(f'{period}: Mean Temperature = {df["LST_Day"].mean():.1f}°C')
            """
        }
    }
    
    dict_json = f"{output_dir}/DEMO_uzbekistan_data_dictionary_{timestamp}.json"
    with open(dict_json, 'w') as f:
        json.dump(data_dictionary, f, indent=2)
    print(f"   ✅ Data dictionary: {dict_json}")
    
    # 6. Create usage guide
    print("📋 Creating usage guide...")
    
    usage_guide = f"""
# DEMO: UZBEKISTAN URBAN EXPANSION DATA USAGE GUIDE

## 📊 EXTRACTED DATA OVERVIEW

**Dataset ID**: {timestamp}
**Data Type**: DEMONSTRATION (Sample structure)
**Cities**: {', '.join(demo_cities)}
**Time Periods**: 2016, 2020, 2025
**Total Observations**: {sum(len(df) for df in demo_satellite_data.values())} sample points

## 📁 AVAILABLE FILES

### 1. Satellite Data (by Period)
- `DEMO_uzbekistan_satellite_data_period_2016_{timestamp}.csv`
- `DEMO_uzbekistan_satellite_data_period_2020_{timestamp}.csv` 
- `DEMO_uzbekistan_satellite_data_period_2025_{timestamp}.csv`

**Contains**: Temperature, land cover, vegetation indices, urban indicators

### 2. City Impact Analysis
- `DEMO_uzbekistan_city_impacts_{timestamp}.csv`

**Contains**: 10-year changes, expansion rates, environmental impacts

### 3. Combined Dataset
- `DEMO_uzbekistan_combined_dataset_{timestamp}.csv`

**Contains**: Latest satellite data + impact metrics for each sample point

### 4. Regional Statistics
- `DEMO_uzbekistan_regional_stats_{timestamp}.json`

**Contains**: Regional averages, standard deviations, analysis metadata

### 5. Documentation
- `DEMO_uzbekistan_data_dictionary_{timestamp}.json`

**Contains**: Complete variable definitions and usage examples

## 🔍 QUICK START ANALYSIS

### Load and Explore Data
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load latest satellite data
data_2025 = pd.read_csv('DEMO_uzbekistan_satellite_data_period_2025_{timestamp}.csv')
print(data_2025.describe())

# Load city impacts
impacts = pd.read_csv('DEMO_uzbekistan_city_impacts_{timestamp}.csv', index_col=0)
print(impacts[['temp_day_change_10yr', 'built_change_10yr']].head())
```

### Temperature Analysis
```python
# Compare temperatures across cities
city_temps = data_2025.groupby('City')['LST_Day'].mean()
print(city_temps)

# Plot temperature distribution
plt.boxplot([data_2025[data_2025['City']==city]['LST_Day'] 
            for city in data_2025['City'].unique()])
plt.title('Temperature Distribution by City')
plt.show()
```

### Urban Expansion Analysis
```python
# Analyze built-up vs green space
plt.scatter(data_2025['Built_Probability'], data_2025['Green_Probability'],
           c=data_2025['LST_Day'], alpha=0.6)
plt.xlabel('Built-up Probability')
plt.ylabel('Green Space Probability') 
plt.colorbar(label='Day Temperature (°C)')
plt.title('Urban vs Green Space Relationship')
plt.show()
```

### Impact Assessment
```python
# City ranking by temperature change
temp_ranking = impacts.sort_values('temp_day_change_10yr', ascending=False)
print("Cities ranked by temperature increase:")
print(temp_ranking[['temp_day_change_10yr', 'built_change_10yr']])
```

## 📈 ANALYSIS CAPABILITIES

1. **Temporal Analysis**: Compare indicators across 2016-2025
2. **Spatial Analysis**: Map variables using coordinates
3. **Urban Metrics**: Built-up expansion, green space loss
4. **Climate Impacts**: Temperature changes, UHI intensity
5. **Correlation Studies**: Relationships between variables

## 🔧 DATA NOTES

- **Quality**: Demonstration data with realistic patterns
- **Resolution**: 100m spatial analysis
- **Coverage**: Urban core areas (reduced buffer zones)
- **Sampling**: ~10 points per city per period
- **Processing**: Simulated Google Earth Engine workflow

## 📞 SUPPORT

For real data extraction from the full analysis:
1. Run the complete `urban_expansion_impact_analysis.py` script
2. Use the `export_original_data()` function
3. Check `/tmp/` directory for exported files

**Note**: This is demonstration data showing the structure.
For actual analysis, use the full pipeline with Google Earth Engine.
"""
    
    guide_path = f"{output_dir}/DEMO_UZBEKISTAN_DATA_USAGE_GUIDE_{timestamp}.md"
    with open(guide_path, 'w') as f:
        f.write(usage_guide)
    print(f"   ✅ Usage guide: {guide_path}")
    
    # Summary
    files_created = [f for f in os.listdir(output_dir) if f'DEMO_{timestamp}' in f or f'{timestamp}' in f]
    
    print("\n" + "="*70)
    print("✅ DEMO DATA EXPORT COMPLETE!")
    print(f"📁 Files created: {len(files_created)}")
    print(f"💾 Location: {output_dir}")
    print(f"🆔 Dataset ID: {timestamp}")
    print("\n📋 Created Files:")
    for file in sorted(files_created):
        print(f"   - {file}")
    print("\n🚀 Next Steps:")
    print("   1. Open the usage guide for analysis examples")
    print("   2. Load CSV files in Python/R for analysis")
    print("   3. Use data dictionary for variable definitions")
    print("   4. Run full pipeline for real satellite data")
    print("="*70)
    
    return timestamp

if __name__ == "__main__":
    create_demo_data_export()
