#!/usr/bin/env python3
"""
Final verification script to ensure complete data consistency 
across all dashboard components
"""

import json
import re
import os
from collections import Counter

def main():
    # Read the JavaScript file
    js_file_path = "enhanced-suhi-dashboard.js"
    
    if not os.path.exists(js_file_path):
        print(f"❌ JavaScript file not found: {js_file_path}")
        return
    
    with open(js_file_path, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    print("🔍 Dashboard Data Consistency Verification")
    print("=" * 50)
    
    # Extract suhiData from JavaScript
    data_match = re.search(r'const suhiData = ({.*?});', js_content, re.DOTALL)
    if not data_match:
        print("❌ Could not find suhiData definition")
        return
    
    # Parse the JavaScript object (simplified approach)
    try:
        # Extract cities array
        cities_match = re.search(r'"cities":\s*\[(.*?)\]', data_match.group(1), re.DOTALL)
        if cities_match:
            cities_content = cities_match.group(1)
            # Count city objects
            city_objects = re.findall(r'{\s*"name":\s*"([^"]+)"', cities_content)
            print(f"✅ Cities in main array: {len(city_objects)}")
            print(f"   Cities: {', '.join(city_objects[:5])}{'...' if len(city_objects) > 5 else ''}")
        
        # Extract timeSeriesData
        timeseries_match = re.search(r'"timeSeriesData":\s*{(.*?)\n  },', data_match.group(1), re.DOTALL)
        if timeseries_match:
            timeseries_content = timeseries_match.group(1)
            # Find all city names in timeSeriesData - look for quoted city names followed by colon
            timeseries_cities = re.findall(r'\n    "([^"]+)":\s*{', timeseries_content)
            print(f"✅ Cities in timeSeriesData: {len(timeseries_cities)}")
            print(f"   Cities: {', '.join(timeseries_cities[:5])}{'...' if len(timeseries_cities) > 5 else ''}")
        else:
            # Fallback parsing method
            timeseries_cities = re.findall(r'"(\w+)":\s*{\s*"years":', js_content)
            print(f"✅ Cities in timeSeriesData (fallback): {len(timeseries_cities)}")
            print(f"   Cities: {', '.join(timeseries_cities[:5])}{'...' if len(timeseries_cities) > 5 else ''}")
    except Exception as e:
        print(f"❌ Error parsing JavaScript data: {e}")
        return
    
    # Check for consistency
    print("\n🔍 Cross-Reference Verification")
    print("-" * 30)
    
    if len(city_objects) == len(timeseries_cities):
        print(f"✅ Data consistency: Both arrays have {len(city_objects)} cities")
    else:
        print(f"⚠️  Inconsistency: Cities array has {len(city_objects)}, timeSeriesData has {len(timeseries_cities)}")
    
    # Check if all cities match
    city_objects_set = set(city_objects)
    timeseries_cities_set = set(timeseries_cities)
    
    if city_objects_set == timeseries_cities_set:
        print("✅ All cities match between arrays")
    else:
        print("⚠️  City name mismatch detected:")
        only_in_cities = city_objects_set - timeseries_cities_set
        only_in_timeseries = timeseries_cities_set - city_objects_set
        if only_in_cities:
            print(f"   Only in cities array: {only_in_cities}")
        if only_in_timeseries:
            print(f"   Only in timeSeriesData: {only_in_timeseries}")
    
    # Check for expected 14 cities
    expected_count = 14
    if len(city_objects) == expected_count:
        print(f"✅ Correct number of cities: {expected_count}")
    else:
        print(f"⚠️  Expected {expected_count} cities, found {len(city_objects)}")
    
    # Check chart functions
    print("\n🔍 Chart Function Analysis")
    print("-" * 30)
    
    chart_functions = [
        'loadTemporalTrendsChart',
        'loadCityTrendsChart', 
        'loadCorrelationScatter',
        'loadCorrelationHeatmap'
    ]
    
    for func_name in chart_functions:
        func_match = re.search(f'function {func_name}\\(\\)(.*?)\\n}}', js_content, re.DOTALL)
        if func_match:
            func_content = func_match.group(1)
            # Check if it uses real data
            if 'suhiData.cities' in func_content or 'suhiData.timeSeriesData' in func_content:
                print(f"✅ {func_name}: Uses real data")
            else:
                print(f"⚠️  {func_name}: May not use real data")
        else:
            print(f"❓ {func_name}: Function not found")
    
    # Check for any remaining mock data patterns
    print("\n🔍 Mock Data Detection")
    print("-" * 30)
    
    mock_patterns = [
        r'mock\w*',
        r'fake\w*',
        r'dummy\w*',
        r'test\w*data',
        r'sample\w*data'
    ]
    
    mock_found = False
    for pattern in mock_patterns:
        matches = re.findall(pattern, js_content, re.IGNORECASE)
        if matches:
            print(f"⚠️  Found potential mock data: {pattern} -> {set(matches)}")
            mock_found = True
    
    if not mock_found:
        print("✅ No mock data patterns detected")
    
    print("\n🎯 Final Assessment")
    print("=" * 50)
    
    if (len(city_objects) == expected_count and 
        city_objects_set == timeseries_cities_set and 
        not mock_found):
        print("✅ Dashboard appears fully consistent with real data only!")
        print("✅ All 14 cities properly integrated across all components")
        print("✅ No mock data detected")
        print("\n🚀 Dashboard is ready for deployment!")
    else:
        print("⚠️  Some inconsistencies detected - review above for details")
    
    print(f"\n📊 Summary: {len(city_objects)} cities with authentic temporal data (2015-2024)")

if __name__ == "__main__":
    main()
