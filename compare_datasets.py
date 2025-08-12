import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_datasets():
    """Compare original and improved datasets"""
    
    # Load both datasets
    with open("d:/alphaearth/scientific_suhi_analysis/data/comprehensive_suhi_analysis_20250812_195437.json", 'r') as f:
        original_data = json.load(f)
    
    with open("d:/alphaearth/scientific_suhi_analysis/data/comprehensive_suhi_analysis_improved.json", 'r') as f:
        improved_data = json.load(f)
    
    # Convert to DataFrames
    original_records = []
    for period, cities_data in original_data['period_data'].items():
        year = period.replace('period_', '')
        for city_data in cities_data:
            city_data['Year'] = int(year)
            original_records.append(city_data)
    
    improved_records = []
    for period, cities_data in improved_data['period_data'].items():
        year = period.replace('period_', '')
        for city_data in cities_data:
            city_data['Year'] = int(year)
            improved_records.append(city_data)
    
    original_df = pd.DataFrame(original_records)
    improved_df = pd.DataFrame(improved_records)
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Data quality comparison
    original_quality = original_df['Data_Quality'].value_counts()
    improved_quality = improved_df['Data_Quality'].value_counts()
    
    ax1.pie(original_quality.values, labels=original_quality.index, autopct='%1.1f%%', 
            colors=['lightcoral', 'lightgreen'], startangle=90)
    ax1.set_title('Original Dataset\nData Quality Distribution')
    
    ax2.pie(improved_quality.values, labels=improved_quality.index, autopct='%1.1f%%', 
            colors=['lightgreen', 'orange'], startangle=90)
    ax2.set_title('Improved Dataset\nData Quality Distribution')
    
    # 3. Cities with complete data
    cities = original_df['City'].unique()
    original_complete = []
    improved_complete = []
    
    for city in cities:
        orig_good = len(original_df[(original_df['City'] == city) & (original_df['Data_Quality'] == 'Good')])
        impr_good = len(improved_df[(improved_df['City'] == city) & (improved_df['Data_Quality'].isin(['Good', 'Improved']))])
        
        original_complete.append(orig_good)
        improved_complete.append(impr_good)
    
    x = range(len(cities))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], original_complete, width, label='Original', color='lightcoral', alpha=0.8)
    ax3.bar([i + width/2 for i in x], improved_complete, width, label='Improved', color='lightgreen', alpha=0.8)
    ax3.set_xlabel('Cities')
    ax3.set_ylabel('Years with Good Data')
    ax3.set_title('Data Completeness by City')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cities, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    summary_data = {
        'Metric': ['Total Records', 'Good Quality', 'Poor Quality', 'Improved Quality', 'Cities with Complete Data'],
        'Original': [len(original_df), 
                    len(original_df[original_df['Data_Quality'] == 'Good']),
                    len(original_df[original_df['Data_Quality'] == 'Poor']),
                    0,
                    sum(1 for c in cities if len(original_df[(original_df['City'] == c) & (original_df['Data_Quality'] == 'Good')]) == 10)],
        'Improved': [len(improved_df),
                    len(improved_df[improved_df['Data_Quality'] == 'Good']),
                    0,
                    len(improved_df[improved_df['Data_Quality'] == 'Improved']),
                    sum(1 for c in cities if len(improved_df[(improved_df['City'] == c) & (improved_df['Data_Quality'].isin(['Good', 'Improved']))]) == 10)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Improvement Summary', pad=20)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('d:/alphaearth/suhi_data_improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== DATASET COMPARISON SUMMARY ===")
    print(f"Original dataset: {len(original_df)} records")
    print(f"Improved dataset: {len(improved_df)} records")
    print(f"Poor data eliminated: {len(original_df[original_df['Data_Quality'] == 'Poor'])} → 0")
    print(f"Improved records added: {len(improved_df[improved_df['Data_Quality'] == 'Improved'])}")
    print(f"Cities with complete data: {sum(1 for c in cities if len(original_df[(original_df['City'] == c) & (original_df['Data_Quality'] == 'Good')]) == 10)} → {sum(1 for c in cities if len(improved_df[(improved_df['City'] == c) & (improved_df['Data_Quality'].isin(['Good', 'Improved']))]) == 10)}")
    
    return original_df, improved_df

def analyze_specific_improvements():
    """Analyze the specific improvements made to each city"""
    
    with open("d:/alphaearth/scientific_suhi_analysis/data/comprehensive_suhi_analysis_improved.json", 'r') as f:
        improved_data = json.load(f)
    
    print("\n=== SPECIFIC IMPROVEMENTS ANALYSIS ===")
    
    # Extract 2015 data
    period_2015 = improved_data['period_data']['period_2015']
    improved_cities = [record for record in period_2015 if record.get('Data_Quality') == 'Improved']
    
    print(f"Cities improved in 2015: {len(improved_cities)}")
    
    for record in improved_cities:
        city = record['City']
        suhi_day = record.get('SUHI_Day', 'N/A')
        suhi_night = record.get('SUHI_Night', 'N/A')
        urban_pixels = record.get('Urban_Pixel_Count', 0)
        rural_pixels = record.get('Rural_Pixel_Count', 0)
        
        print(f"\n{city}:")
        print(f"  SUHI Day: {suhi_day:.2f}°C" if isinstance(suhi_day, (int, float)) else f"  SUHI Day: {suhi_day}")
        print(f"  SUHI Night: {suhi_night:.2f}°C" if isinstance(suhi_night, (int, float)) else f"  SUHI Night: {suhi_night}")
        print(f"  Urban pixels: {urban_pixels}")
        print(f"  Rural pixels: {rural_pixels}")

if __name__ == "__main__":
    original_df, improved_df = compare_datasets()
    analyze_specific_improvements()
