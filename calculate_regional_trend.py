#!/usr/bin/env python3
"""
Calculate the real regional warming trend from actual temporal data
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def main():
    print("🔍 Calculating Real Regional Warming Trend")
    print("=" * 50)
    
    # Directory containing the CSV files
    csv_dir = "scientific_suhi_analysis/data/"
    
    # Find all SUHI CSV files
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith('suhi_data_period_') and f.endswith('.csv')]
    csv_files.sort()
    
    if not csv_files:
        print("❌ No SUHI CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Extract year-wise regional averages
    years = []
    regional_day_means = []
    regional_night_means = []
    
    for csv_file in csv_files:
        try:
            # Extract year from filename
            year = int(csv_file.split('_')[3])
            years.append(year)
            
            # Read the CSV file
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            
            # Calculate regional averages for this year
            day_mean = df['SUHI_Day'].mean()
            night_mean = df['SUHI_Night'].mean()
            
            regional_day_means.append(day_mean)
            regional_night_means.append(night_mean)
            
            print(f"  {year}: Day SUHI = {day_mean:.3f}°C, Night SUHI = {night_mean:.3f}°C")
            
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            continue
    
    if len(years) < 3:
        print("❌ Need at least 3 years of data for trend calculation")
        return
    
    # Calculate linear trends using scipy
    day_slope, day_intercept, day_r_value, day_p_value, day_std_err = stats.linregress(years, regional_day_means)
    night_slope, night_intercept, night_r_value, night_p_value, night_std_err = stats.linregress(years, regional_night_means)
    
    print(f"\n📊 Regional Trend Analysis ({min(years)}-{max(years)})")
    print("-" * 40)
    print(f"Day SUHI Trend: {day_slope:+.6f}°C/year")
    print(f"  R² = {day_r_value**2:.4f}, p-value = {day_p_value:.4f}")
    print(f"  Standard error: ±{day_std_err:.6f}°C/year")
    
    print(f"\nNight SUHI Trend: {night_slope:+.6f}°C/year")
    print(f"  R² = {night_r_value**2:.4f}, p-value = {night_p_value:.4f}")
    print(f"  Standard error: ±{night_std_err:.6f}°C/year")
    
    # Determine significance
    day_significant = day_p_value < 0.05
    night_significant = night_p_value < 0.05
    
    print(f"\n🎯 Trend Significance")
    print(f"Day trend {'IS' if day_significant else 'is NOT'} statistically significant (p={'<0.05' if day_significant else f'={day_p_value:.3f}'})")
    print(f"Night trend {'IS' if night_significant else 'is NOT'} statistically significant (p={'<0.05' if night_significant else f'={night_p_value:.3f}'})")
    
    # Compare with current dashboard value
    current_dashboard_trend = 0.0385
    print(f"\n📋 Dashboard Comparison")
    print(f"Current dashboard value: +{current_dashboard_trend:.4f}°C/year")
    print(f"Calculated real value:   {day_slope:+.4f}°C/year")
    print(f"Difference: {abs(day_slope - current_dashboard_trend):.4f}°C/year")
    
    if abs(day_slope - current_dashboard_trend) > 0.001:
        print(f"⚠️  Dashboard value should be updated to {day_slope:+.4f}°C/year")
    else:
        print("✅ Dashboard value is close to calculated trend")
    
    # Show decade projection
    decade_change_day = day_slope * 10
    decade_change_night = night_slope * 10
    
    print(f"\n🔮 10-Year Projection")
    print(f"Day SUHI change by {max(years)+10}: {decade_change_day:+.3f}°C")
    print(f"Night SUHI change by {max(years)+10}: {decade_change_night:+.3f}°C")

if __name__ == "__main__":
    main()
