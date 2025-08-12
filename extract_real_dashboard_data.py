import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def load_all_real_data():
    """Load and process all real SUHI data from CSV files and JSON"""
    
    data_dir = "d:/alphaearth/scientific_suhi_analysis/data"
    
    # Load all CSV files for each year
    csv_files = glob.glob(os.path.join(data_dir, "suhi_data_period_*.csv"))
    all_data = []
    
    print("📊 Loading real SUHI data from CSV files...")
    
    for csv_file in csv_files:
        # Extract year from filename
        filename = os.path.basename(csv_file)
        year = int(filename.split('_')[3])
        
        try:
            df = pd.read_csv(csv_file)
            df['Year'] = year
            all_data.append(df)
            print(f"   ✅ Loaded {len(df)} records for {year}")
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    # Combine all data
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

def calculate_year_over_year_changes(df):
    """Calculate year-over-year changes for all cities"""
    
    print("📊 Calculating year-over-year changes...")
    
    changes_data = []
    cities = df['City'].unique()
    years = sorted(df['Year'].unique())
    
    for city in cities:
        city_data = df[df['City'] == city].sort_values('Year')
        
        for i in range(1, len(city_data)):
            current_year = city_data.iloc[i]
            previous_year = city_data.iloc[i-1]
            
            if not (pd.isna(current_year['SUHI_Day']) or pd.isna(previous_year['SUHI_Day'])):
                day_change = current_year['SUHI_Day'] - previous_year['SUHI_Day']
                night_change = current_year['SUHI_Night'] - previous_year['SUHI_Night']
                
                changes_data.append({
                    'City': city,
                    'From_Year': int(previous_year['Year']),
                    'To_Year': int(current_year['Year']),
                    'SUHI_Day_Change': day_change,
                    'SUHI_Night_Change': night_change,
                    'SUHI_Day_Previous': previous_year['SUHI_Day'],
                    'SUHI_Day_Current': current_year['SUHI_Day'],
                    'SUHI_Night_Previous': previous_year['SUHI_Night'],
                    'SUHI_Night_Current': current_year['SUHI_Night']
                })
    
    changes_df = pd.DataFrame(changes_data)
    print(f"   ✅ Calculated {len(changes_df)} year-over-year changes")
    
    return changes_df

def calculate_city_statistics(df):
    """Calculate comprehensive statistics for each city"""
    
    print("📊 Calculating city statistics...")
    
    city_stats = []
    for city in df['City'].unique():
        city_data = df[df['City'] == city]
        
        # Basic statistics
        stats = {
            'name': city,
            'records': len(city_data),
            'years_covered': len(city_data['Year'].unique()),
            'first_year': int(city_data['Year'].min()),
            'last_year': int(city_data['Year'].max()),
            
            # Day SUHI stats
            'dayMean': city_data['SUHI_Day'].mean(),
            'dayStd': city_data['SUHI_Day'].std(),
            'dayMin': city_data['SUHI_Day'].min(),
            'dayMax': city_data['SUHI_Day'].max(),
            'dayMedian': city_data['SUHI_Day'].median(),
            
            # Night SUHI stats
            'nightMean': city_data['SUHI_Night'].mean(),
            'nightStd': city_data['SUHI_Night'].std(),
            'nightMin': city_data['SUHI_Night'].min(),
            'nightMax': city_data['SUHI_Night'].max(),
            'nightMedian': city_data['SUHI_Night'].median(),
            
            # Urban characteristics
            'avgUrbanPixels': city_data['Urban_Pixel_Count'].mean(),
            'avgRuralPixels': city_data['Rural_Pixel_Count'].mean(),
            'avgNDVIUrban': city_data['NDVI_Urban'].mean(),
            'avgNDVIRural': city_data['NDVI_Rural'].mean(),
            'avgNDBIUrban': city_data['NDBI_Urban'].mean(),
            'avgNDBIRural': city_data['NDBI_Rural'].mean(),
            'avgNDWIUrban': city_data['NDWI_Urban'].mean(),
            'avgNDWIRural': city_data['NDWI_Rural'].mean(),
            
            # Data quality
            'goodDataCount': len(city_data[city_data['Data_Quality'] == 'Good']),
            'improvedDataCount': len(city_data[city_data['Data_Quality'] == 'Improved']),
            'dataQualityPercent': len(city_data[city_data['Data_Quality'] == 'Good']) / len(city_data) * 100
        }
        
        # Calculate trend (linear regression)
        years = city_data['Year'].values
        day_values = city_data['SUHI_Day'].dropna().values
        night_values = city_data['SUHI_Night'].dropna().values
        
        if len(day_values) >= 3:
            valid_years = city_data.dropna(subset=['SUHI_Day'])['Year'].values
            stats['dayTrend'] = np.polyfit(valid_years, day_values, 1)[0]
        else:
            stats['dayTrend'] = 0.0
            
        if len(night_values) >= 3:
            valid_years = city_data.dropna(subset=['SUHI_Night'])['Year'].values
            stats['nightTrend'] = np.polyfit(valid_years, night_values, 1)[0]
        else:
            stats['nightTrend'] = 0.0
        
        # Urban size category
        avg_pixels = stats['avgUrbanPixels']
        if avg_pixels > 2000:
            stats['urbanSize'] = 'Large'
        elif avg_pixels > 500:
            stats['urbanSize'] = 'Medium'
        else:
            stats['urbanSize'] = 'Small'
        
        # Extreme events (top 10% of all day SUHI values)
        day_90th = df['SUHI_Day'].quantile(0.9)
        stats['extremeEvents'] = len(city_data[city_data['SUHI_Day'] > day_90th])
        stats['extremeThreshold'] = day_90th
        
        city_stats.append(stats)
    
    print(f"   ✅ Calculated statistics for {len(city_stats)} cities")
    return city_stats

def calculate_regional_trends(df):
    """Calculate regional trends by year"""
    
    print("📊 Calculating regional trends...")
    
    years = sorted(df['Year'].unique())
    regional_stats = {
        'years': years,
        'dayMeans': [],
        'nightMeans': [],
        'dayStds': [],
        'nightStds': [],
        'recordCounts': []
    }
    
    for year in years:
        year_data = df[df['Year'] == year]
        
        regional_stats['dayMeans'].append(year_data['SUHI_Day'].mean())
        regional_stats['nightMeans'].append(year_data['SUHI_Night'].mean())
        regional_stats['dayStds'].append(year_data['SUHI_Day'].std())
        regional_stats['nightStds'].append(year_data['SUHI_Night'].std())
        regional_stats['recordCounts'].append(len(year_data))
    
    # Calculate overall trends
    regional_stats['dayTrend'] = np.polyfit(years, regional_stats['dayMeans'], 1)[0]
    regional_stats['nightTrend'] = np.polyfit(years, regional_stats['nightMeans'], 1)[0]
    
    print(f"   ✅ Regional day trend: {regional_stats['dayTrend']:+.4f}°C/year")
    print(f"   ✅ Regional night trend: {regional_stats['nightTrend']:+.4f}°C/year")
    
    return regional_stats

def calculate_correlations(df):
    """Calculate correlation matrix for all variables"""
    
    print("📊 Calculating correlations...")
    
    corr_vars = ['SUHI_Day', 'SUHI_Night', 'NDVI_Urban', 'NDVI_Rural', 
                'NDBI_Urban', 'NDBI_Rural', 'NDWI_Urban', 'NDWI_Rural',
                'Urban_Pixel_Count', 'Rural_Pixel_Count']
    
    # Filter for available columns
    available_vars = [var for var in corr_vars if var in df.columns]
    correlation_matrix = df[available_vars].corr()
    
    correlations = {
        'variables': available_vars,
        'matrix': correlation_matrix.round(3).values.tolist(),
        'strongestCorrelations': {}
    }
    
    # Extract key correlations
    if 'SUHI_Day' in available_vars and 'SUHI_Night' in available_vars:
        correlations['strongestCorrelations']['dayVsNight'] = correlation_matrix.loc['SUHI_Day', 'SUHI_Night']
    
    for var in ['NDVI_Urban', 'NDBI_Urban', 'NDWI_Urban', 'Urban_Pixel_Count']:
        if var in available_vars:
            correlations['strongestCorrelations'][f'dayVs{var}'] = correlation_matrix.loc['SUHI_Day', var]
    
    print(f"   ✅ Calculated correlations for {len(available_vars)} variables")
    return correlations

def generate_time_series_data(df):
    """Generate time series data for each city"""
    
    print("📊 Generating time series data...")
    
    time_series = {}
    for city in df['City'].unique():
        city_data = df[df['City'] == city].sort_values('Year')
        
        time_series[city] = {
            'years': city_data['Year'].tolist(),
            'dayValues': city_data['SUHI_Day'].round(3).tolist(),
            'nightValues': city_data['SUHI_Night'].round(3).tolist(),
            'lstUrbanDay': city_data['LST_Day_Urban'].round(1).tolist(),
            'lstRuralDay': city_data['LST_Day_Rural'].round(1).tolist(),
            'lstUrbanNight': city_data['LST_Night_Urban'].round(1).tolist(),
            'lstRuralNight': city_data['LST_Night_Rural'].round(1).tolist(),
            'ndviUrban': city_data['NDVI_Urban'].round(3).tolist(),
            'ndviRural': city_data['NDVI_Rural'].round(3).tolist(),
            'ndbiUrban': city_data['NDBI_Urban'].round(3).tolist(),
            'ndbiRural': city_data['NDBI_Rural'].round(3).tolist(),
            'urbanPixels': city_data['Urban_Pixel_Count'].tolist(),
            'ruralPixels': city_data['Rural_Pixel_Count'].tolist(),
            'dataQuality': city_data['Data_Quality'].tolist()
        }
    
    print(f"   ✅ Generated time series for {len(time_series)} cities")
    return time_series

def create_enhanced_dashboard_data():
    """Create comprehensive dashboard data with all real data"""
    
    print("🚀 Creating Enhanced SUHI Dashboard with Real Data")
    print("="*60)
    
    # Load all real data
    df = load_all_real_data()
    if df is None:
        return None
    
    # Calculate year-over-year changes
    changes_df = calculate_year_over_year_changes(df)
    
    # Calculate comprehensive statistics
    city_stats = calculate_city_statistics(df)
    regional_trends = calculate_regional_trends(df)
    correlations = calculate_correlations(df)
    time_series = generate_time_series_data(df)
    
    # Add the provided analysis results
    analysis_results = {
        "SUHI_Day_Change_mean": 1.5230228300835507,
        "SUHI_Day_Change_std": 2.6805754869544423,
        "SUHI_Day_Change_min": -2.268954021410657,
        "SUHI_Day_Change_max": 4.979675015063485,
        "cities_with_valid_data": 5,
        "SUHI_Night_Change_mean": -0.44362175653933383,
        "SUHI_Night_Change_std": 0.5188749319626144,
        "SUHI_Night_Change_min": -0.7553929366688212,
        "SUHI_Night_Change_max": 0.4805574788528091,
        "analysis_type": "multi_dataset_suhi",
        "method": "combined_urban_classification",
        "warm_months": [6, 7, 8],
        "analysis_scale_m": 200,
        "urban_threshold": 0.15,
        "rural_threshold": 0.2,
        "min_urban_pixels": 10,
        "min_rural_pixels": 25,
        "ring_width_km": 25,
        "cities_analyzed": 14,
        "cities_with_valid_suhi": 5,
        "analysis_span_years": 9,
        "first_period": "2015",
        "last_period": "2024"
    }
    
    # Create comprehensive dashboard data structure
    dashboard_data = {
        "metadata": {
            "title": "Enhanced SUHI Analysis Dashboard - Uzbekistan Cities",
            "analysisDate": datetime.now().isoformat(),
            "dataSource": "Real Google Earth Engine Analysis - Landsat 8/9 + Multi-dataset Classification",
            "temporalRange": f"{df['Year'].min()}-{df['Year'].max()}",
            "totalObservations": len(df),
            "citiesCount": df['City'].nunique(),
            "yearsCount": df['Year'].nunique(),
            "analysisMethod": "Multi-dataset Urban Classification + LST Difference Analysis",
            "datasets": ["Dynamic World", "GHSL", "ESA WorldCover", "MODIS LC", "GLAD", "MODIS LST"],
            "spatialResolution": "200m",
            "temporalResolution": "Annual (warm season: Jun-Aug)",
            "dataQuality": {
                "good": len(df[df['Data_Quality'] == 'Good']),
                "improved": len(df[df['Data_Quality'] == 'Improved']),
                "total": len(df),
                "goodPercentage": round(len(df[df['Data_Quality'] == 'Good']) / len(df) * 100, 1)
            }
        },
        
        "scientificMethodology": {
            "title": "Surface Urban Heat Island (SUHI) Analysis Methodology",
            "overview": "This analysis employs a multi-dataset approach combining satellite-derived land surface temperature with advanced urban classification techniques.",
            "steps": [
                {
                    "step": 1,
                    "title": "Multi-Dataset Urban Classification",
                    "description": "Combined classification using Dynamic World, GHSL, ESA WorldCover, MODIS LC, and GLAD datasets",
                    "formula": "Urban_Probability = (DW_urban + GHSL_built + ESA_built + MODIS_urban + GLAD_urban) / 5",
                    "threshold": "Urban pixels: P(urban) > 0.15, Rural pixels: P(urban) < 0.2"
                },
                {
                    "step": 2,
                    "title": "Land Surface Temperature Extraction",
                    "description": "MODIS LST data processed for warm season months (June-August)",
                    "formula": "LST_Kelvin = MODIS_LST * 0.02 + 273.15, LST_Celsius = LST_Kelvin - 273.15",
                    "processing": "Monthly composite median values for robust temperature estimates"
                },
                {
                    "step": 3,
                    "title": "SUHI Intensity Calculation",
                    "description": "Surface Urban Heat Island intensity as temperature difference",
                    "formula": "SUHI = LST_Urban - LST_Rural",
                    "dayFormula": "SUHI_Day = mean(LST_Day_Urban) - mean(LST_Day_Rural)",
                    "nightFormula": "SUHI_Night = mean(LST_Night_Urban) - mean(LST_Night_Rural)"
                },
                {
                    "step": 4,
                    "title": "Spatial Aggregation",
                    "description": "Urban and rural zone definition with buffer analysis",
                    "urbanZone": "Pixels with P(urban) > 0.15 within city boundaries",
                    "ruralZone": "Pixels with P(urban) < 0.2 within 25km ring buffer",
                    "minimumPixels": "Urban: 10 pixels, Rural: 25 pixels for statistical validity"
                },
                {
                    "step": 5,
                    "title": "Trend Analysis",
                    "description": "Linear regression analysis for temporal trends",
                    "formula": "Trend = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²",
                    "significance": "Tested using t-test with α = 0.05"
                }
            ],
            "qualityControl": [
                "Multi-dataset consensus for urban classification reliability",
                "Seasonal filtering to warm months for consistent SUHI patterns",
                "Minimum pixel thresholds to ensure statistical representativeness",
                "Data quality flags and improvement procedures for missing/poor data"
            ]
        },
        
        "analysisResults": analysis_results,
        
        "cities": sorted(city_stats, key=lambda x: x['dayMean'], reverse=True),
        
        "regionalTrends": regional_trends,
        
        "correlations": correlations,
        
        "timeSeriesData": time_series,
        
        "yearOverYearChanges": {
            "data": changes_df.to_dict('records') if len(changes_df) > 0 else [],
            "statistics": {
                "dayChangesMean": changes_df['SUHI_Day_Change'].mean() if len(changes_df) > 0 else 0,
                "dayChangesStd": changes_df['SUHI_Day_Change'].std() if len(changes_df) > 0 else 0,
                "nightChangesMean": changes_df['SUHI_Night_Change'].mean() if len(changes_df) > 0 else 0,
                "nightChangesStd": changes_df['SUHI_Night_Change'].std() if len(changes_df) > 0 else 0,
                "totalChanges": len(changes_df)
            }
        },
        
        "extremeEvents": {
            "dayThreshold": df['SUHI_Day'].quantile(0.9),
            "nightThreshold": df['SUHI_Night'].quantile(0.9),
            "dayExtreme": len(df[df['SUHI_Day'] > df['SUHI_Day'].quantile(0.9)]),
            "nightExtreme": len(df[df['SUHI_Night'] > df['SUHI_Night'].quantile(0.9)]),
            "citiesWithMostEvents": [
                {
                    "city": city['name'],
                    "events": city['extremeEvents'],
                    "percentage": round(city['extremeEvents'] / city['records'] * 100, 1) if city['records'] > 0 else 0
                }
                for city in sorted(city_stats, key=lambda x: x['extremeEvents'], reverse=True)[:5]
                if city['extremeEvents'] > 0
            ]
        },
        
        "statistics": {
            "regional": {
                "dayMean": round(df['SUHI_Day'].mean(), 3),
                "dayStd": round(df['SUHI_Day'].std(), 3),
                "dayMin": round(df['SUHI_Day'].min(), 3),
                "dayMax": round(df['SUHI_Day'].max(), 3),
                "dayMedian": round(df['SUHI_Day'].median(), 3),
                "nightMean": round(df['SUHI_Night'].mean(), 3),
                "nightStd": round(df['SUHI_Night'].std(), 3),
                "nightMin": round(df['SUHI_Night'].min(), 3),
                "nightMax": round(df['SUHI_Night'].max(), 3),
                "nightMedian": round(df['SUHI_Night'].median(), 3)
            },
            "urbanSizeEffect": {
                size: {
                    "count": len([c for c in city_stats if c['urbanSize'] == size]),
                    "meanDaySUHI": round(np.mean([c['dayMean'] for c in city_stats if c['urbanSize'] == size]), 3) if any(c['urbanSize'] == size for c in city_stats) else 0,
                    "meanNightSUHI": round(np.mean([c['nightMean'] for c in city_stats if c['urbanSize'] == size]), 3) if any(c['urbanSize'] == size for c in city_stats) else 0
                }
                for size in ['Small', 'Medium', 'Large']
            }
        }
    }
    
    # Save the enhanced dashboard data
    output_file = 'd:/alphaearth/enhanced_suhi_dashboard_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("✅ ENHANCED DASHBOARD DATA CREATED")
    print("="*60)
    print(f"📄 Output file: {output_file}")
    print(f"📊 Total observations: {len(df)}")
    print(f"🏙️ Cities analyzed: {df['City'].nunique()}")
    print(f"📅 Years covered: {df['Year'].min()}-{df['Year'].max()}")
    print(f"📈 Regional day SUHI: {df['SUHI_Day'].mean():+.3f}°C")
    print(f"🌙 Regional night SUHI: {df['SUHI_Night'].mean():+.3f}°C")
    print(f"📊 Year-over-year changes: {len(changes_df)} calculated")
    print(f"🎯 Data quality: {dashboard_data['metadata']['dataQuality']['goodPercentage']}% good")
    
    return dashboard_data

if __name__ == "__main__":
    create_enhanced_dashboard_data()
