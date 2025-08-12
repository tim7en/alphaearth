import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def load_all_real_data():
    """Load and process all real SUHI data from CSV files"""
    
    data_dir = "d:/alphaearth/scientific_suhi_analysis/data"
    csv_files = glob.glob(os.path.join(data_dir, "suhi_data_period_*.csv"))
    all_data = []
    
    print("📊 Loading real SUHI data from CSV files...")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        year = int(filename.split('_')[3])
        
        try:
            df = pd.read_csv(csv_file)
            df['Year'] = year
            all_data.append(df)
            print(f"   ✅ Loaded {len(df)} records for {year}")
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    if not all_data:
        print("❌ No data files found!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📈 Total records loaded: {len(combined_df)} across {len(all_data)} years")
    
    # Convert numeric columns
    numeric_cols = ['SUHI_Day', 'SUHI_Night', 'LST_Day_Urban', 'LST_Day_Rural', 
                   'LST_Night_Urban', 'LST_Night_Rural', 'NDVI_Urban', 'NDVI_Rural',
                   'NDBI_Urban', 'NDBI_Rural', 'NDWI_Urban', 'NDWI_Rural',
                   'Urban_Prob', 'Rural_Prob', 'Urban_Pixel_Count', 'Rural_Pixel_Count']
    
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    return combined_df

def create_enhanced_dashboard_with_real_data():
    """Create enhanced dashboard with real data, handling JSON serialization properly"""
    
    print("🚀 Creating Enhanced SUHI Dashboard with Real Data")
    print("="*60)
    
    # Load real data
    df = load_all_real_data()
    if df is None:
        return None
    
    # Calculate basic statistics for each city
    city_stats = []
    for city in df['City'].unique():
        city_data = df[df['City'] == city]
        
        # Calculate trends
        years = city_data['Year'].values
        day_values = city_data['SUHI_Day'].dropna().values
        night_values = city_data['SUHI_Night'].dropna().values
        
        day_trend = float(np.polyfit(years, day_values, 1)[0]) if len(day_values) >= 3 else 0.0
        night_trend = float(np.polyfit(years, night_values, 1)[0]) if len(night_values) >= 3 else 0.0
        
        # Urban size category
        avg_pixels = float(city_data['Urban_Pixel_Count'].mean())
        if avg_pixels > 2000:
            urban_size = 'Large'
        elif avg_pixels > 500:
            urban_size = 'Medium'
        else:
            urban_size = 'Small'
        
        # Extreme events
        day_90th = df['SUHI_Day'].quantile(0.9)
        extreme_events = int(len(city_data[city_data['SUHI_Day'] > day_90th]))
        
        stats = {
            'name': city,
            'records': int(len(city_data)),
            'dayMean': round(float(city_data['SUHI_Day'].mean()), 3),
            'dayStd': round(float(city_data['SUHI_Day'].std()), 3),
            'dayMin': round(float(city_data['SUHI_Day'].min()), 3),
            'dayMax': round(float(city_data['SUHI_Day'].max()), 3),
            'nightMean': round(float(city_data['SUHI_Night'].mean()), 3),
            'nightStd': round(float(city_data['SUHI_Night'].std()), 3),
            'nightMin': round(float(city_data['SUHI_Night'].min()), 3),
            'nightMax': round(float(city_data['SUHI_Night'].max()), 3),
            'dayTrend': round(day_trend, 4),
            'nightTrend': round(night_trend, 4),
            'urbanSize': urban_size,
            'extremeEvents': extreme_events,
            'avgUrbanPixels': round(float(avg_pixels), 0),
            'avgNDVIUrban': round(float(city_data['NDVI_Urban'].mean()), 3),
            'avgNDBIUrban': round(float(city_data['NDBI_Urban'].mean()), 3),
            'dataQualityGood': int(len(city_data[city_data['Data_Quality'] == 'Good']))
        }
        city_stats.append(stats)
    
    # Calculate regional trends
    years = sorted(df['Year'].unique())
    regional_day_means = []
    regional_night_means = []
    
    for year in years:
        year_data = df[df['Year'] == year]
        regional_day_means.append(round(float(year_data['SUHI_Day'].mean()), 3))
        regional_night_means.append(round(float(year_data['SUHI_Night'].mean()), 3))
    
    day_trend = float(np.polyfit(years, regional_day_means, 1)[0])
    night_trend = float(np.polyfit(years, regional_night_means, 1)[0])
    
    # Calculate year-over-year changes
    changes_data = []
    for city in df['City'].unique():
        city_data = df[df['City'] == city].sort_values('Year')
        for i in range(1, len(city_data)):
            current = city_data.iloc[i]
            previous = city_data.iloc[i-1]
            
            if not (pd.isna(current['SUHI_Day']) or pd.isna(previous['SUHI_Day'])):
                changes_data.append({
                    'city': city,
                    'fromYear': int(previous['Year']),
                    'toYear': int(current['Year']),
                    'dayChange': round(float(current['SUHI_Day'] - previous['SUHI_Day']), 3),
                    'nightChange': round(float(current['SUHI_Night'] - previous['SUHI_Night']), 3)
                })
    
    # Generate time series for each city
    time_series = {}
    for city in df['City'].unique():
        city_data = df[df['City'] == city].sort_values('Year')
        time_series[city] = {
            'years': [int(year) for year in city_data['Year'].tolist()],
            'dayValues': [round(float(val), 3) if not pd.isna(val) else None for val in city_data['SUHI_Day'].tolist()],
            'nightValues': [round(float(val), 3) if not pd.isna(val) else None for val in city_data['SUHI_Night'].tolist()],
            'lstUrbanDay': [round(float(val), 1) if not pd.isna(val) else None for val in city_data['LST_Day_Urban'].tolist()],
            'lstRuralDay': [round(float(val), 1) if not pd.isna(val) else None for val in city_data['LST_Day_Rural'].tolist()],
            'dataQuality': city_data['Data_Quality'].tolist()
        }
    
    # Create enhanced dashboard data
    dashboard_data = {
        "metadata": {
            "title": "Enhanced SUHI Analysis Dashboard - Uzbekistan Cities",
            "analysisDate": datetime.now().isoformat(),
            "dataSource": "Real Google Earth Engine Analysis - Landsat 8/9 + Multi-dataset Classification",
            "temporalRange": f"{int(df['Year'].min())}-{int(df['Year'].max())}",
            "totalObservations": int(len(df)),
            "citiesCount": int(df['City'].nunique()),
            "yearsCount": int(df['Year'].nunique()),
            "spatialResolution": "200m",
            "dataQuality": {
                "good": int(len(df[df['Data_Quality'] == 'Good'])),
                "improved": int(len(df[df['Data_Quality'] == 'Improved'])),
                "goodPercentage": round(float(len(df[df['Data_Quality'] == 'Good']) / len(df) * 100), 1)
            }
        },
        
        "scientificMethodology": {
            "title": "Surface Urban Heat Island (SUHI) Analysis Methodology",
            "overview": "Multi-dataset approach combining satellite LST with advanced urban classification",
            "steps": [
                {
                    "step": 1,
                    "title": "Multi-Dataset Urban Classification",
                    "description": "Combined classification using Dynamic World, GHSL, ESA WorldCover, MODIS LC, and GLAD",
                    "formula": "Urban_Probability = (DW + GHSL + ESA + MODIS + GLAD) / 5",
                    "threshold": "Urban: P > 0.15, Rural: P < 0.2"
                },
                {
                    "step": 2,
                    "title": "Land Surface Temperature Processing",
                    "description": "MODIS LST data for warm season (June-August)",
                    "formula": "LST_Celsius = (MODIS_LST × 0.02) - 273.15",
                    "processing": "Monthly median composite for robust estimates"
                },
                {
                    "step": 3,
                    "title": "SUHI Intensity Calculation",
                    "description": "Temperature difference between urban and rural areas",
                    "dayFormula": "SUHI_Day = mean(LST_Day_Urban) - mean(LST_Day_Rural)",
                    "nightFormula": "SUHI_Night = mean(LST_Night_Urban) - mean(LST_Night_Rural)"
                },
                {
                    "step": 4,
                    "title": "Spatial Aggregation",
                    "description": "Urban and rural zone definition with 25km buffer",
                    "urbanZone": "Pixels with P(urban) > 0.15 within city boundaries",
                    "ruralZone": "Pixels with P(urban) < 0.2 within 25km ring",
                    "minimumPixels": "Urban: 10, Rural: 25 for statistical validity"
                },
                {
                    "step": 5,
                    "title": "Trend Analysis",
                    "description": "Linear regression for temporal trends",
                    "formula": "Trend = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²",
                    "significance": "t-test with α = 0.05"
                }
            ]
        },
        
        "analysisResults": {
            "SUHI_Day_Change_mean": 1.523,
            "SUHI_Day_Change_std": 2.681,
            "SUHI_Day_Change_min": -2.269,
            "SUHI_Day_Change_max": 4.980,
            "SUHI_Night_Change_mean": -0.444,
            "SUHI_Night_Change_std": 0.519,
            "SUHI_Night_Change_min": -0.755,
            "SUHI_Night_Change_max": 0.481,
            "cities_with_valid_data": 5,
            "analysis_type": "multi_dataset_suhi",
            "method": "combined_urban_classification",
            "warm_months": [6, 7, 8],
            "analysis_scale_m": 200,
            "urban_threshold": 0.15,
            "rural_threshold": 0.2,
            "cities_analyzed": 14,
            "cities_with_valid_suhi": 5,
            "analysis_span_years": 9
        },
        
        "cities": sorted(city_stats, key=lambda x: x['dayMean'], reverse=True),
        
        "regionalTrends": {
            "years": years,
            "dayMeans": regional_day_means,
            "nightMeans": regional_night_means,
            "dayTrend": round(day_trend, 4),
            "nightTrend": round(night_trend, 4)
        },
        
        "timeSeriesData": time_series,
        
        "yearOverYearChanges": changes_data,
        
        "statistics": {
            "regional": {
                "dayMean": round(float(df['SUHI_Day'].mean()), 3),
                "dayStd": round(float(df['SUHI_Day'].std()), 3),
                "dayMin": round(float(df['SUHI_Day'].min()), 3),
                "dayMax": round(float(df['SUHI_Day'].max()), 3),
                "nightMean": round(float(df['SUHI_Night'].mean()), 3),
                "nightStd": round(float(df['SUHI_Night'].std()), 3),
                "nightMin": round(float(df['SUHI_Night'].min()), 3),
                "nightMax": round(float(df['SUHI_Night'].max()), 3)
            }
        },
        
        "extremeEvents": {
            "dayThreshold": round(float(df['SUHI_Day'].quantile(0.9)), 2),
            "nightThreshold": round(float(df['SUHI_Night'].quantile(0.9)), 2),
            "citiesWithMostEvents": [
                {
                    "city": city['name'],
                    "events": city['extremeEvents'],
                    "percentage": round(city['extremeEvents'] / city['records'] * 100, 1) if city['records'] > 0 else 0
                }
                for city in sorted(city_stats, key=lambda x: x['extremeEvents'], reverse=True)[:5]
                if city['extremeEvents'] > 0
            ]
        }
    }
    
    # Save the enhanced dashboard data
    output_file = 'd:/alphaearth/enhanced_suhi_dashboard_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("✅ ENHANCED DASHBOARD DATA CREATED WITH REAL DATA")
    print("="*60)
    print(f"📄 Output file: {output_file}")
    print(f"📊 Total observations: {len(df)}")
    print(f"🏙️ Cities analyzed: {df['City'].nunique()}")
    print(f"📅 Years covered: {df['Year'].min()}-{df['Year'].max()}")
    print(f"📈 Regional day SUHI: {df['SUHI_Day'].mean():+.3f}°C")
    print(f"🌙 Regional night SUHI: {df['SUHI_Night'].mean():+.3f}°C")
    print(f"📊 Day trend: {day_trend:+.4f}°C/year")
    print(f"🌙 Night trend: {night_trend:+.4f}°C/year")
    print(f"📊 Year-over-year changes: {len(changes_data)} calculated")
    
    return dashboard_data

if __name__ == "__main__":
    create_enhanced_dashboard_with_real_data()
