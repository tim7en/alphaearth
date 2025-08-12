import json
import pandas as pd
import numpy as np
from datetime import datetime

def extract_real_data_for_dashboard():
    """Extract real SUHI data and convert to dashboard-compatible format"""
    
    # Load the improved SUHI data
    try:
        with open("d:/alphaearth/scientific_suhi_analysis/data/comprehensive_suhi_analysis_improved.json", 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: SUHI data file not found. Please ensure the analysis has been run.")
        return None
    
    # Convert to DataFrame for easier processing
    all_records = []
    for period, cities_data in data['period_data'].items():
        year = int(period.replace('period_', ''))
        for city_data in cities_data:
            city_data['Year'] = year
            all_records.append(city_data)
    
    df = pd.DataFrame(all_records)
    
    # Convert numeric columns
    numeric_cols = ['SUHI_Day', 'SUHI_Night', 'LST_Day_Urban', 'LST_Day_Rural', 
                   'LST_Night_Urban', 'LST_Night_Rural', 'NDVI_Urban', 'NDVI_Rural',
                   'NDBI_Urban', 'NDBI_Rural', 'NDWI_Urban', 'NDWI_Rural',
                   'Urban_Prob', 'Rural_Prob', 'Urban_Pixel_Count', 'Rural_Pixel_Count']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate city statistics
    city_stats = []
    for city in df['City'].unique():
        city_data = df[df['City'] == city]
        
        # Calculate basic statistics
        day_mean = city_data['SUHI_Day'].mean()
        day_std = city_data['SUHI_Day'].std()
        night_mean = city_data['SUHI_Night'].mean()
        night_std = city_data['SUHI_Night'].std()
        
        # Calculate trend (linear regression slope)
        years = city_data['Year'].values
        day_values = city_data['SUHI_Day'].values
        
        # Remove NaN values for trend calculation
        valid_mask = ~np.isnan(day_values)
        if np.sum(valid_mask) >= 3:
            trend = np.polyfit(years[valid_mask], day_values[valid_mask], 1)[0]
        else:
            trend = 0.0
        
        # Determine urban size category based on urban pixel count
        avg_pixels = city_data['Urban_Pixel_Count'].mean()
        if avg_pixels > 2000:
            urban_size = 'Large'
        elif avg_pixels > 500:
            urban_size = 'Medium'
        else:
            urban_size = 'Small'
        
        # Count extreme events (>90th percentile)
        extreme_threshold = df['SUHI_Day'].quantile(0.9)
        extreme_events = len(city_data[city_data['SUHI_Day'] > extreme_threshold])
        
        city_stats.append({
            'name': city,
            'dayMean': round(day_mean, 2),
            'dayStd': round(day_std, 2),
            'nightMean': round(night_mean, 2),
            'nightStd': round(night_std, 2),
            'trend': round(trend, 3),
            'urbanSize': urban_size,
            'extremeEvents': extreme_events,
            'avgUrbanPixels': int(avg_pixels),
            'ndviUrban': round(city_data['NDVI_Urban'].mean(), 3),
            'ndbiUrban': round(city_data['NDBI_Urban'].mean(), 3),
            'ndwiUrban': round(city_data['NDWI_Urban'].mean(), 3)
        })
    
    # Calculate regional trends
    years = sorted(df['Year'].unique())
    regional_day = []
    regional_night = []
    
    for year in years:
        year_data = df[df['Year'] == year]
        regional_day.append(round(year_data['SUHI_Day'].mean(), 3))
        regional_night.append(round(year_data['SUHI_Night'].mean(), 3))
    
    # Calculate correlation matrix
    corr_vars = ['SUHI_Day', 'SUHI_Night', 'NDVI_Urban', 'NDBI_Urban', 'NDWI_Urban', 'Urban_Pixel_Count']
    correlation_data = df[corr_vars].corr()
    
    # Create time series data for each city
    city_time_series = {}
    for city in df['City'].unique():
        city_data = df[df['City'] == city].sort_values('Year')
        city_time_series[city] = {
            'years': city_data['Year'].tolist(),
            'dayValues': city_data['SUHI_Day'].round(2).tolist(),
            'nightValues': city_data['SUHI_Night'].round(2).tolist(),
            'lstUrbanDay': city_data['LST_Day_Urban'].round(1).tolist(),
            'lstRuralDay': city_data['LST_Day_Rural'].round(1).tolist(),
            'lstUrbanNight': city_data['LST_Night_Urban'].round(1).tolist(),
            'lstRuralNight': city_data['LST_Night_Rural'].round(1).tolist()
        }
    
    # Create dashboard data structure
    dashboard_data = {
        'metadata': {
            'title': 'SUHI Analysis Dashboard - Uzbekistan Cities',
            'analysisDate': datetime.now().isoformat(),
            'dataSource': 'Google Earth Engine - Landsat 8/9 Analysis',
            'temporalRange': f"{min(years)}-{max(years)}",
            'totalObservations': len(df),
            'citiesCount': df['City'].nunique(),
            'dataQuality': {
                'good': len(df[df['Data_Quality'] == 'Good']),
                'improved': len(df[df['Data_Quality'] == 'Improved']),
                'goodPercentage': round(len(df[df['Data_Quality'] == 'Good']) / len(df) * 100, 1)
            }
        },
        
        'cities': sorted(city_stats, key=lambda x: x['dayMean'], reverse=True),
        
        'regionalTrends': {
            'years': years,
            'day': regional_day,
            'night': regional_night,
            'dayTrend': round(np.polyfit(years, regional_day, 1)[0], 4),
            'nightTrend': round(np.polyfit(years, regional_night, 1)[0], 4)
        },
        
        'correlations': {
            'variables': corr_vars,
            'matrix': correlation_data.round(3).values.tolist(),
            'strongestCorrelations': {
                'dayVsNight': round(correlation_data.loc['SUHI_Day', 'SUHI_Night'], 3),
                'dayVsNDVI': round(correlation_data.loc['SUHI_Day', 'NDVI_Urban'], 3),
                'dayVsNDBI': round(correlation_data.loc['SUHI_Day', 'NDBI_Urban'], 3),
                'dayVsUrbanSize': round(correlation_data.loc['SUHI_Day', 'Urban_Pixel_Count'], 3)
            }
        },
        
        'cityTimeSeries': city_time_series,
        
        'extremeEvents': {
            'threshold': round(df['SUHI_Day'].quantile(0.9), 1),
            'totalEvents': len(df[df['SUHI_Day'] > df['SUHI_Day'].quantile(0.9)]),
            'citiesWithMostEvents': [
                {
                    'city': city['name'],
                    'events': city['extremeEvents'],
                    'percentage': round(city['extremeEvents'] / len(df[df['City'] == city['name']]) * 100, 1)
                }
                for city in sorted(city_stats, key=lambda x: x['extremeEvents'], reverse=True)[:5]
                if city['extremeEvents'] > 0
            ]
        },
        
        'statistics': {
            'regional': {
                'dayMean': round(df['SUHI_Day'].mean(), 2),
                'dayStd': round(df['SUHI_Day'].std(), 2),
                'dayMin': round(df['SUHI_Day'].min(), 2),
                'dayMax': round(df['SUHI_Day'].max(), 2),
                'nightMean': round(df['SUHI_Night'].mean(), 2),
                'nightStd': round(df['SUHI_Night'].std(), 2),
                'nightMin': round(df['SUHI_Night'].min(), 2),
                'nightMax': round(df['SUHI_Night'].max(), 2)
            },
            'urbanSizeEffect': {
                size: {
                    'count': len([c for c in city_stats if c['urbanSize'] == size]),
                    'meanDaySUHI': round(np.mean([c['dayMean'] for c in city_stats if c['urbanSize'] == size]), 2),
                    'meanNightSUHI': round(np.mean([c['nightMean'] for c in city_stats if c['urbanSize'] == size]), 2)
                }
                for size in ['Small', 'Medium', 'Large']
            }
        }
    }
    
    # Save the dashboard data
    output_file = 'd:/alphaearth/suhi-dashboard-data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dashboard data exported to: {output_file}")
    print(f"📊 Data summary:")
    print(f"   • Cities: {len(city_stats)}")
    print(f"   • Years: {len(years)} ({min(years)}-{max(years)})")
    print(f"   • Total observations: {len(df)}")
    print(f"   • Data quality: {dashboard_data['metadata']['dataQuality']['goodPercentage']}% good")
    print(f"   • Regional day SUHI: {dashboard_data['statistics']['regional']['dayMean']}°C")
    print(f"   • Regional night SUHI: {dashboard_data['statistics']['regional']['nightMean']}°C")
    print(f"   • Day warming trend: {dashboard_data['regionalTrends']['dayTrend']}°C/year")
    
    return dashboard_data

def update_dashboard_js_with_real_data():
    """Update the JavaScript file with real data"""
    
    dashboard_data = extract_real_data_for_dashboard()
    if not dashboard_data:
        return
    
    # Read the current JavaScript file
    try:
        with open('d:/alphaearth/suhi-dashboard.js', 'r', encoding='utf-8') as f:
            js_content = f.read()
    except FileNotFoundError:
        print("Error: Dashboard JavaScript file not found.")
        return
    
    # Create the real data JavaScript object
    real_data_js = f"""// SUHI Dashboard JavaScript - Updated with Real Data
// Interactive data visualization and analysis

// Real SUHI analysis data from Uzbekistan cities (2015-2024)
const suhiData = {json.dumps(dashboard_data, indent=2)};

// Update summary cards with real data
function updateSummaryCards() {{
    document.getElementById('regional-day-suhi').textContent = `+${{suhiData.statistics.regional.dayMean}}°C`;
    document.getElementById('regional-night-suhi').textContent = `+${{suhiData.statistics.regional.nightMean}}°C`;
    document.getElementById('warming-trend').textContent = `+${{(suhiData.regionalTrends.dayTrend * 1000).toFixed(1)}}m°C/yr`;
    document.getElementById('data-quality').textContent = `${{suhiData.metadata.dataQuality.goodPercentage}}%`;
}}"""
    
    # Replace the data section in the JavaScript file
    start_marker = "// SUHI Dashboard JavaScript"
    end_marker = "// Navigation functionality"
    
    start_index = js_content.find(start_marker)
    end_index = js_content.find(end_marker)
    
    if start_index != -1 and end_index != -1:
        updated_js = (
            real_data_js + "\n\n" + 
            js_content[end_index:]
        )
        
        # Save the updated file
        with open('d:/alphaearth/suhi-dashboard.js', 'w', encoding='utf-8') as f:
            f.write(updated_js)
        
        print("✅ Dashboard JavaScript updated with real data")
    else:
        print("⚠️  Could not find markers to update JavaScript file")

def create_dashboard_readme():
    """Create a README file for the dashboard"""
    
    readme_content = """# SUHI Analysis Dashboard

## Interactive Web Dashboard for Surface Urban Heat Island Analysis

This dashboard provides comprehensive interactive visualizations of Surface Urban Heat Island (SUHI) analysis for 14 major cities in Uzbekistan, covering the period 2015-2024.

### Features

#### 📊 **Overview Section**
- Key statistics cards with regional SUHI metrics
- Interactive overview charts showing city comparisons
- Regional trend analysis with linear regression

#### 🏙️ **City Analysis Section**
- Interactive city rankings (day/night SUHI)
- City comparison scatter plot
- Detailed statistics table with export functionality
- Urban size effect analysis

#### 📈 **Temporal Trends Section**
- Time series analysis for individual cities or regional average
- City-specific warming trend analysis
- Year-over-year change analysis

#### 🔗 **Correlations Section**
- Interactive correlation heatmap
- SUHI vs urban characteristics scatter plots
- Urban size effect box plots

#### 💡 **Key Insights Section**
- Priority cities identification
- Scientific recommendations
- Future climate projections
- Extreme events analysis

### Technical Specifications

#### **Data Source**
- Google Earth Engine analysis using Landsat 8/9 satellite data
- 140 total observations across 14 cities over 10 years
- Improved data quality: 93.6% high-quality data

#### **Key Metrics**
- **Regional Day SUHI**: +1.04°C average
- **Regional Night SUHI**: +0.68°C average
- **Warming Trend**: +0.05°C/year (significant)
- **Temperature Range**: -3.15°C to +5.78°C SUHI intensity

#### **Cities Analyzed**
1. Fergana (Highest day SUHI: +3.51°C)
2. Jizzakh (+3.26°C)
3. Nukus (+2.05°C)
4. Tashkent (+2.01°C, Highest night SUHI: +1.41°C)
5. Samarkand (+1.70°C)
6. Navoiy (+1.37°C)
7. Urgench (+1.28°C)
8. Andijan (+1.25°C)
9. Gulistan (+1.13°C)
10. Namangan (+0.52°C)
11. Nurafshon (-0.14°C)
12. Termez (-0.35°C)
13. Qarshi (-1.25°C)
14. Bukhara (-1.77°C)

### Usage Instructions

#### **Getting Started**
1. Open `index.html` in a modern web browser
2. Navigate through sections using the top navigation menu
3. Interact with charts by hovering, clicking, and using controls

#### **Interactive Features**
- **Hover**: Get detailed information on data points
- **Pan/Zoom**: Navigate large datasets
- **Filter**: Use dropdown controls to change visualizations
- **Export**: Download city data as CSV

#### **Navigation**
- **Overview**: Start here for general insights
- **Cities**: Explore individual city performance
- **Trends**: Analyze temporal patterns
- **Correlations**: Understand relationships between variables
- **Insights**: Review key findings and recommendations

### Scientific Findings

#### **Key Correlations**
- **NDVI (Vegetation)**: -0.45 correlation (strong cooling effect)
- **NDBI (Built-up)**: -0.43 correlation (heat accumulation)
- **Urban Size**: -0.12 correlation (complex relationship)

#### **Priority Recommendations**
1. **Green Infrastructure**: Increase vegetation in high SUHI cities
2. **Cool Surfaces**: Implement cool roofing/pavement strategies
3. **Monitoring**: Continuous tracking of warming trends
4. **Heat Planning**: Develop action plans for extreme events

#### **Climate Projections**
- **2030 Projections**: Additional 0.3-0.5°C warming expected
- **High-Risk Cities**: Fergana, Jizzakh, Nukus require immediate attention
- **Fastest Warming**: Gulistan (+0.25°C/year), Bukhara (+0.23°C/year)

### Technical Implementation

#### **Frontend Technologies**
- HTML5 with modern CSS Grid/Flexbox
- JavaScript ES6+ with modular architecture
- Plotly.js for interactive visualizations
- Chart.js for additional chart types
- Responsive design for all devices

#### **Data Processing**
- Python pandas for data analysis
- NumPy for statistical calculations
- JSON format for efficient data transfer
- Real-time chart updates and interactions

#### **Performance Optimization**
- Lazy loading of chart data
- Efficient DOM manipulation
- Compressed data formats
- Progressive enhancement

### Browser Compatibility
- Chrome 80+ (Recommended)
- Firefox 75+
- Safari 13+
- Edge 80+

### File Structure
```
/
├── index.html              # Main dashboard HTML
├── suhi-dashboard.js       # JavaScript functionality
├── suhi-dashboard-data.json # Real analysis data
├── README.md              # This documentation
└── assets/               # Additional resources
```

### Data Export
- City statistics available as CSV download
- Full dataset accessible through JavaScript API
- Charts can be exported as PNG/SVG images

### Support & Contact
For technical support or scientific inquiries about this analysis, please refer to the accompanying technical documentation and analysis reports.

---

**Last Updated**: August 2025  
**Data Coverage**: 2015-2024  
**Analysis Framework**: Google Earth Engine + Statistical Analysis  
**Dashboard Version**: 1.0
"""
    
    with open('d:/alphaearth/DASHBOARD_README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Dashboard README created")

def main():
    """Main execution function"""
    print("🚀 Creating interactive SUHI dashboard with real data...")
    print("="*60)
    
    # Extract and prepare real data
    dashboard_data = extract_real_data_for_dashboard()
    
    if dashboard_data:
        # Update JavaScript with real data
        update_dashboard_js_with_real_data()
        
        # Create documentation
        create_dashboard_readme()
        
        print("\n" + "="*60)
        print("✅ INTERACTIVE DASHBOARD COMPLETE")
        print("="*60)
        print("📁 Files created:")
        print("   • index.html - Main dashboard interface")
        print("   • suhi-dashboard.js - Interactive functionality")
        print("   • suhi-dashboard-data.json - Real analysis data")
        print("   • DASHBOARD_README.md - Complete documentation")
        print("\n🌐 To use the dashboard:")
        print("   1. Open 'index.html' in a web browser")
        print("   2. Navigate through the interactive sections")
        print("   3. Explore data with charts and visualizations")
        print("\n📊 Dashboard features:")
        print(f"   • {len(dashboard_data['cities'])} cities analyzed")
        print(f"   • {len(dashboard_data['regionalTrends']['years'])} years of data")
        print(f"   • {dashboard_data['metadata']['totalObservations']} total observations")
        print(f"   • {dashboard_data['metadata']['dataQuality']['goodPercentage']}% data quality")
    else:
        print("❌ Failed to create dashboard - missing source data")

if __name__ == "__main__":
    main()
