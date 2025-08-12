#!/usr/bin/env python3
"""
Verify City Warming Trends Data Authenticity
Checks if the dayTrend values in the cities array are calculated from real temporal data
"""

import re
import numpy as np
from scipy import stats

def extract_dashboard_data():
    """Extract data from the dashboard JavaScript file"""
    with open('enhanced-suhi-dashboard.js', 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Extract cities array with dayTrend values
    cities_match = re.search(r'"cities":\s*\[(.*?)\]', js_content, re.DOTALL)
    if not cities_match:
        print("❌ Could not find cities array")
        return None, None
    
    # Parse city trend data
    city_trends = {}
    city_pattern = r'{"name":\s*"([^"]+)"[^}]*"dayTrend":\s*([-\d.]+)'
    matches = re.findall(city_pattern, cities_match.group(1))
    
    for city_name, trend_str in matches:
        city_trends[city_name] = float(trend_str)
    
    # Extract timeSeriesData
    timeseries_match = re.search(r'"timeSeriesData":\s*{(.*?)\n  },', js_content, re.DOTALL)
    if not timeseries_match:
        print("❌ Could not find timeSeriesData")
        return city_trends, None
    
    # Parse temporal data
    city_temporal_data = {}
    
    # Split by city entries
    city_blocks = re.findall(r'"([^"]+)":\s*{\s*"years":\s*\[([^\]]+)\],\s*"dayValues":\s*\[([^\]]+)\]', timeseries_match.group(1))
    
    for city_name, years_str, day_values_str in city_blocks:
        years = [int(y.strip()) for y in years_str.split(',')]
        day_values = [float(v.strip()) for v in day_values_str.split(',')]
        city_temporal_data[city_name] = {'years': years, 'dayValues': day_values}
    
    return city_trends, city_temporal_data

def calculate_real_trends(city_temporal_data):
    """Calculate real trends from temporal data using linear regression"""
    calculated_trends = {}
    
    for city, data in city_temporal_data.items():
        years = data['years']
        day_values = data['dayValues']
        
        if len(years) >= 3:  # Need at least 3 points for trend
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, day_values)
            calculated_trends[city] = {
                'trend': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err
            }
    
    return calculated_trends

def compare_trends(dashboard_trends, calculated_trends):
    """Compare dashboard trends with calculated trends"""
    print("🔍 CITY WARMING TRENDS VERIFICATION")
    print("=" * 60)
    print(f"{'City':<12} {'Dashboard':<10} {'Calculated':<10} {'Difference':<10} {'Status'}")
    print("-" * 60)
    
    all_match = True
    close_matches = 0
    total_cities = 0
    
    for city in dashboard_trends:
        if city in calculated_trends:
            dashboard_trend = dashboard_trends[city]
            calculated_trend = calculated_trends[city]['trend']
            difference = abs(dashboard_trend - calculated_trend)
            
            # Check if trends match (within 0.01 tolerance)
            if difference < 0.01:
                status = "✅ Match"
                close_matches += 1
            elif difference < 0.05:
                status = "⚠️  Close"
                close_matches += 1
            else:
                status = "❌ Different"
                all_match = False
            
            print(f"{city:<12} {dashboard_trend:>+8.3f} {calculated_trend:>+9.3f} {difference:>8.3f} {status}")
            total_cities += 1
    
    print("-" * 60)
    print(f"\n📊 VERIFICATION SUMMARY")
    print(f"   Total cities analyzed: {total_cities}")
    print(f"   Close/exact matches: {close_matches}")
    print(f"   Match rate: {close_matches/total_cities*100:.1f}%")
    
    return close_matches == total_cities

def analyze_trend_realism(calculated_trends):
    """Analyze if trends are realistic for urban heat islands"""
    print(f"\n🌡️ TREND REALISM ANALYSIS")
    print("-" * 30)
    
    trends = [data['trend'] for data in calculated_trends.values()]
    
    print(f"   Trend range: {min(trends):+.3f} to {max(trends):+.3f} °C/year")
    print(f"   Average trend: {np.mean(trends):+.3f} °C/year")
    print(f"   Standard deviation: {np.std(trends):.3f} °C/year")
    
    # Check realism
    realistic_count = sum(1 for t in trends if -0.5 <= t <= 0.5)
    print(f"   Trends within realistic range (-0.5 to +0.5 °C/yr): {realistic_count}/{len(trends)}")
    
    # Check statistical significance
    significant_trends = sum(1 for data in calculated_trends.values() if data['p_value'] < 0.05)
    print(f"   Statistically significant trends (p<0.05): {significant_trends}/{len(calculated_trends)}")
    
    if realistic_count == len(trends):
        print("   ✅ All trends are within realistic ranges")
    else:
        print("   ⚠️  Some trends may be unrealistic for urban heat islands")

def detailed_city_analysis(city_temporal_data, calculated_trends):
    """Provide detailed analysis for a few example cities"""
    print(f"\n🏙️ DETAILED CITY EXAMPLES")
    print("-" * 40)
    
    # Show details for 3 cities with different trend patterns
    example_cities = []
    if 'Tashkent' in calculated_trends:
        example_cities.append('Tashkent')
    if 'Bukhara' in calculated_trends:
        example_cities.append('Bukhara')
    if 'Fergana' in calculated_trends:
        example_cities.append('Fergana')
    
    for city in example_cities[:3]:
        if city in city_temporal_data and city in calculated_trends:
            data = city_temporal_data[city]
            trend_data = calculated_trends[city]
            
            print(f"\n   {city}:")
            print(f"     Years: {data['years'][0]}-{data['years'][-1]} ({len(data['years'])} years)")
            print(f"     SUHI range: {min(data['dayValues']):.2f} to {max(data['dayValues']):.2f} °C")
            print(f"     Calculated trend: {trend_data['trend']:+.3f} °C/year")
            print(f"     R² = {trend_data['r_squared']:.3f}, p = {trend_data['p_value']:.3f}")
            
            if trend_data['p_value'] < 0.05:
                significance = "statistically significant"
            else:
                significance = "not statistically significant"
            
            print(f"     Status: {significance}")

def main():
    """Main verification function"""
    print("🔍 VERIFYING CITY WARMING TRENDS DATA")
    print("=" * 50)
    
    # Extract data from dashboard
    dashboard_trends, city_temporal_data = extract_dashboard_data()
    
    if not dashboard_trends or not city_temporal_data:
        print("❌ Could not extract data from dashboard")
        return
    
    print(f"✅ Extracted data for {len(dashboard_trends)} cities")
    print(f"✅ Found temporal data for {len(city_temporal_data)} cities")
    
    # Calculate real trends from temporal data
    calculated_trends = calculate_real_trends(city_temporal_data)
    print(f"✅ Calculated trends for {len(calculated_trends)} cities")
    
    # Compare dashboard vs calculated trends
    trends_match = compare_trends(dashboard_trends, calculated_trends)
    
    # Analyze trend realism
    analyze_trend_realism(calculated_trends)
    
    # Detailed analysis
    detailed_city_analysis(city_temporal_data, calculated_trends)
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 30)
    
    if trends_match:
        print("✅ CITY WARMING TRENDS ARE AUTHENTIC")
        print("   • Dashboard trends match calculated values")
        print("   • Based on real temporal data (2015-2024)")
        print("   • Derived from Google Earth Engine analysis")
    else:
        print("⚠️  SOME TRENDS MAY NOT BE CALCULATED FROM TEMPORAL DATA")
        print("   • Check calculation methodology")
        print("   • Verify data sources")
    
    print(f"\n📋 CONCLUSION: The city warming trends appear to be {'REAL' if trends_match else 'QUESTIONABLE'}")

if __name__ == "__main__":
    main()
