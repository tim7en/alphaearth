import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_suhi_data(json_file_path):
    """Load and analyze SUHI data quality patterns"""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all period data
    all_data = []
    for period, cities_data in data['period_data'].items():
        year = period.replace('period_', '')
        for city_data in cities_data:
            city_data['Year'] = int(year)
            all_data.append(city_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print("=== SUHI DATA QUALITY ANALYSIS ===")
    print(f"Total records: {len(df)}")
    print(f"Cities analyzed: {df['City'].nunique()}")
    print(f"Years covered: {df['Year'].min()} - {df['Year'].max()}")
    
    # Analyze data quality by city and year
    quality_summary = df.groupby(['City', 'Data_Quality']).size().unstack(fill_value=0)
    print("\n=== DATA QUALITY BY CITY ===")
    print(quality_summary)
    
    # Calculate percentage of poor data
    total_records = df.groupby('City').size()
    poor_records = df[df['Data_Quality'] == 'Poor'].groupby('City').size()
    poor_percentage = (poor_records / total_records * 100).fillna(0)
    
    print("\n=== PERCENTAGE OF POOR DATA BY CITY ===")
    for city in sorted(poor_percentage.index):
        print(f"{city}: {poor_percentage[city]:.1f}%")
    
    return df, data

def identify_fill_opportunities(df):
    """Identify opportunities to fill poor data with good data from nearby years"""
    
    print("\n=== POOR DATA THAT CAN BE IMPROVED ===")
    
    improvements = []
    
    for city in df['City'].unique():
        city_data = df[df['City'] == city].sort_values('Year')
        poor_years = city_data[city_data['Data_Quality'] == 'Poor']['Year'].tolist()
        good_years = city_data[city_data['Data_Quality'] == 'Good']['Year'].tolist()
        
        if poor_years and good_years:
            print(f"\n{city}:")
            print(f"  Poor data years: {poor_years}")
            print(f"  Good data years: {good_years}")
            
            for poor_year in poor_years:
                # Find closest good year
                distances = [abs(poor_year - good_year) for good_year in good_years]
                min_distance = min(distances)
                closest_good_year = good_years[distances.index(min_distance)]
                
                print(f"  → Replace {poor_year} data with {closest_good_year} data (distance: {min_distance} years)")
                
                improvements.append({
                    'City': city,
                    'Poor_Year': poor_year,
                    'Good_Year': closest_good_year,
                    'Distance': min_distance
                })
    
    return improvements

def create_improved_dataset(df, original_data, improvements):
    """Create an improved dataset by replacing poor data with good data"""
    
    print("\n=== CREATING IMPROVED DATASET ===")
    
    # Create a copy of the original data
    improved_data = json.loads(json.dumps(original_data))
    
    improvements_made = 0
    
    for improvement in improvements:
        city = improvement['City']
        poor_year = improvement['Poor_Year']
        good_year = improvement['Good_Year']
        
        # Find the good data
        good_record = df[(df['City'] == city) & (df['Year'] == good_year)].iloc[0]
        
        # Find the poor data record in the original structure
        poor_period_key = f"period_{poor_year}"
        
        if poor_period_key in improved_data['period_data']:
            for i, record in enumerate(improved_data['period_data'][poor_period_key]):
                if record['City'] == city:
                    # Replace with good data but keep the original year
                    original_period = record['Period']
                    
                    # Copy all non-null values from good record
                    for key, value in good_record.items():
                        if key not in ['City', 'Period', 'Year'] and pd.notna(value):
                            record[key] = value
                    
                    # Keep original period and city
                    record['Period'] = original_period
                    record['City'] = city
                    record['Data_Quality'] = "Improved"
                    
                    improvements_made += 1
                    print(f"  ✓ Improved {city} {poor_year} using {good_year} data")
                    break
    
    print(f"\nTotal improvements made: {improvements_made}")
    
    return improved_data

def analyze_improvements(original_df, improved_data):
    """Analyze the impact of improvements"""
    
    # Convert improved data back to DataFrame
    all_improved_data = []
    for period, cities_data in improved_data['period_data'].items():
        year = period.replace('period_', '')
        for city_data in cities_data:
            city_data['Year'] = int(year)
            all_improved_data.append(city_data)
    
    improved_df = pd.DataFrame(all_improved_data)
    
    print("\n=== IMPROVEMENT ANALYSIS ===")
    
    # Compare data quality before and after
    original_quality = original_df['Data_Quality'].value_counts()
    improved_quality = improved_df['Data_Quality'].value_counts()
    
    print("Original data quality:")
    print(original_quality)
    print("\nImproved data quality:")
    print(improved_quality)
    
    # Calculate cities with complete time series
    original_complete = 0
    improved_complete = 0
    
    for city in original_df['City'].unique():
        orig_city_data = original_df[original_df['City'] == city]
        impr_city_data = improved_df[improved_df['City'] == city]
        
        orig_good_years = len(orig_city_data[orig_city_data['Data_Quality'] == 'Good'])
        impr_good_years = len(impr_city_data[impr_city_data['Data_Quality'].isin(['Good', 'Improved'])])
        
        if orig_good_years == 10:  # All years 2015-2024
            original_complete += 1
        if impr_good_years == 10:
            improved_complete += 1
    
    print(f"\nCities with complete time series:")
    print(f"  Original: {original_complete}/14 cities")
    print(f"  Improved: {improved_complete}/14 cities")
    
    return improved_df

def save_improved_dataset(improved_data, output_path):
    """Save the improved dataset"""
    
    # Update metadata
    improved_data['metadata']['analysis_date'] = datetime.now().isoformat()
    improved_data['metadata']['method'] = improved_data['metadata']['method'] + " + Data Quality Improvements"
    
    # Add improvement notes
    improved_data['improvement_notes'] = {
        "improvement_date": datetime.now().isoformat(),
        "improvement_method": "Replace poor quality data with good quality data from nearest year",
        "poor_data_replaced": "Yes - see Data_Quality field marked as 'Improved'",
        "data_quality_levels": {
            "Good": "Original high-quality data",
            "Poor": "Original poor-quality data (kept for reference)",
            "Improved": "Poor data replaced with good data from nearest available year"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(improved_data, f, indent=2)
    
    print(f"\n✓ Improved dataset saved to: {output_path}")

def create_quality_visualization(df, improved_df):
    """Create visualizations of data quality improvements"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data quality by city
    quality_counts = df.groupby(['City', 'Data_Quality']).size().unstack(fill_value=0)
    quality_counts.plot(kind='bar', stacked=True, ax=ax1, color=['red', 'green'])
    ax1.set_title('Original Data Quality by City')
    ax1.set_xlabel('City')
    ax1.set_ylabel('Number of Years')
    ax1.legend(title='Data Quality')
    ax1.tick_params(axis='x', rotation=45)
    
    # Improved data quality by city
    improved_quality_counts = improved_df.groupby(['City', 'Data_Quality']).size().unstack(fill_value=0)
    improved_quality_counts.plot(kind='bar', stacked=True, ax=ax2, color=['red', 'green', 'orange'])
    ax2.set_title('Improved Data Quality by City')
    ax2.set_xlabel('City')
    ax2.set_ylabel('Number of Years')
    ax2.legend(title='Data Quality')
    ax2.tick_params(axis='x', rotation=45)
    
    # Data quality by year - original
    year_quality = df.groupby(['Year', 'Data_Quality']).size().unstack(fill_value=0)
    year_quality.plot(kind='bar', ax=ax3, color=['red', 'green'])
    ax3.set_title('Original Data Quality by Year')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Cities')
    ax3.legend(title='Data Quality')
    
    # Data quality by year - improved
    improved_year_quality = improved_df.groupby(['Year', 'Data_Quality']).size().unstack(fill_value=0)
    improved_year_quality.plot(kind='bar', ax=ax4, color=['red', 'green', 'orange'])
    ax4.set_title('Improved Data Quality by Year')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Number of Cities')
    ax4.legend(title='Data Quality')
    
    plt.tight_layout()
    plt.savefig('d:/alphaearth/suhi_data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Data quality visualization saved as 'suhi_data_quality_analysis.png'")

def main():
    """Main analysis function"""
    
    # Load and analyze original data
    json_file = "d:/alphaearth/scientific_suhi_analysis/data/comprehensive_suhi_analysis_20250812_195437.json"
    df, original_data = load_and_analyze_suhi_data(json_file)
    
    # Identify improvement opportunities
    improvements = identify_fill_opportunities(df)
    
    # Create improved dataset
    improved_data = create_improved_dataset(df, original_data, improvements)
    
    # Analyze improvements
    improved_df = analyze_improvements(df, improved_data)
    
    # Save improved dataset
    output_path = "d:/alphaearth/scientific_suhi_analysis/data/comprehensive_suhi_analysis_improved.json"
    save_improved_dataset(improved_data, output_path)
    
    # Create visualizations
    create_quality_visualization(df, improved_df)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key improvements:")
    print("1. Poor quality data replaced with good data from nearest years")
    print("2. More cities now have complete time series")
    print("3. Better temporal continuity for trend analysis")
    print("4. All improvements are clearly marked in the data")

if __name__ == "__main__":
    main()
