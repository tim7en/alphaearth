#!/usr/bin/env python3
"""
Extract real temporal SUHI data from CSV files for dashboard
No mock data - only actual Google Earth Engine analysis results
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def extract_real_temporal_data():
    """Extract real temporal data from CSV files"""
    
    # Directory containing the real SUHI data
    data_dir = Path("scientific_suhi_analysis/data")
    
    # Get all SUHI period CSV files
    csv_files = list(data_dir.glob("suhi_data_period_*_20250812_195437.csv"))
    csv_files.sort()  # Sort by filename (which includes year)
    
    print(f"Found {len(csv_files)} CSV files with real SUHI data")
    
    # Dictionary to store temporal data for each city
    temporal_data = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        # Extract year from filename
        year = int(csv_file.stem.split('_')[3])
        print(f"Processing year {year}: {csv_file.name}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file)
            
            # Process each city in the file
            for _, row in df.iterrows():
                city = row['City']
                
                # Skip if data quality is poor or SUHI values are missing
                if pd.isna(row['SUHI_Day']) or pd.isna(row['SUHI_Night']) or row['Data_Quality'] == 'Poor':
                    continue
                
                # Initialize city data if not exists
                if city not in temporal_data:
                    temporal_data[city] = {
                        'years': [],
                        'dayValues': [],
                        'nightValues': [],
                        'dataQuality': []
                    }
                
                # Add data for this year
                temporal_data[city]['years'].append(year)
                temporal_data[city]['dayValues'].append(float(row['SUHI_Day']))
                temporal_data[city]['nightValues'].append(float(row['SUHI_Night']))
                temporal_data[city]['dataQuality'].append(row['Data_Quality'])
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Sort data by year for each city
    for city in temporal_data:
        city_data = temporal_data[city]
        # Create list of (year, day, night, quality) tuples and sort by year
        combined = list(zip(city_data['years'], city_data['dayValues'], 
                          city_data['nightValues'], city_data['dataQuality']))
        combined.sort(key=lambda x: x[0])  # Sort by year
        
        # Unpack back to separate lists
        temporal_data[city]['years'] = [x[0] for x in combined]
        temporal_data[city]['dayValues'] = [x[1] for x in combined]
        temporal_data[city]['nightValues'] = [x[2] for x in combined]
        temporal_data[city]['dataQuality'] = [x[3] for x in combined]
    
    # Print summary
    print(f"\nReal temporal data extracted for {len(temporal_data)} cities:")
    for city, data in temporal_data.items():
        print(f"  {city}: {len(data['years'])} years of data ({min(data['years'])}-{max(data['years'])})")
        print(f"    Day SUHI range: {min(data['dayValues']):.2f} to {max(data['dayValues']):.2f}°C")
        print(f"    Night SUHI range: {min(data['nightValues']):.2f} to {max(data['nightValues']):.2f}°C")
    
    return temporal_data

def generate_real_dashboard_data():
    """Generate complete dashboard data with real temporal information"""
    
    # Extract real temporal data
    real_temporal_data = extract_real_temporal_data()
    
    # Create JavaScript object string for dashboard
    js_temporal_data = {}
    
    for city, data in real_temporal_data.items():
        js_temporal_data[city] = {
            "years": data['years'],
            "dayValues": [round(val, 3) for val in data['dayValues']],
            "nightValues": [round(val, 3) for val in data['nightValues']],
            "dataQuality": data['dataQuality']
        }
    
    # Convert to JavaScript format
    js_code = '  "timeSeriesData": {\n'
    
    for i, (city, data) in enumerate(js_temporal_data.items()):
        js_code += f'    "{city}": {{\n'
        js_code += f'      "years": {data["years"]},\n'
        js_code += f'      "dayValues": {data["dayValues"]},\n'
        js_code += f'      "nightValues": {data["nightValues"]},\n'
        js_code += f'      "dataQuality": {json.dumps(data["dataQuality"])}\n'
        js_code += '    }'
        if i < len(js_temporal_data) - 1:
            js_code += ','
        js_code += '\n'
    
    js_code += '  },'
    
    print("\n" + "="*60)
    print("REAL TEMPORAL DATA FOR DASHBOARD (JavaScript format):")
    print("="*60)
    print(js_code)
    
    # Save to file
    with open('real_temporal_data.js', 'w') as f:
        f.write(js_code)
    
    print(f"\nReal temporal data saved to: real_temporal_data.js")
    print("Replace the timeSeriesData section in enhanced-suhi-dashboard.js with this data")
    
    return js_temporal_data

if __name__ == "__main__":
    generate_real_dashboard_data()
