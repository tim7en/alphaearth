"""
Drought & Vegetation Anomaly Analysis Module

This module analyzes drought and vegetation anomalies in Uzbekistan's agro-districts
to answer the research question: "Where and when have agro-districts experienced 
the deepest vegetation deficits since 2000? How did 2021 & 2023 compare?"

Analysis Components:
1. NDVI/EVI z-scores vs 2001–2020 baseline
2. SPI (Standardized Precipitation Index) from CHIRPS
3. Pixel-wise Mann-Kendall trend analysis
4. District drought atlas creation
5. Anomaly time-series analysis
6. Hotspot mapping

Data Sources (Mock):
- MOD13Q1 NDVI/EVI (250m, 16-day composites, 2000-2023)
- CHIRPS daily precipitation (5.5km, daily, 2000-2023)

Output Products:
- District drought atlas
- Anomaly time-series plots
- Hotspot maps
- 2021 vs 2023 comparison analysis
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .utils import (load_config, ensure_dir, setup_plotting, save_plot, 
                   calculate_confidence_interval, perform_trend_analysis,
                   create_summary_statistics, validate_data_quality)
from .mock_gee_data import MockGEEDataGenerator, create_mock_datasets
from .production_features import (memory_monitor, rate_limited, MemoryOptimizer, 
                                DataValidator, ChunkedProcessor, ProgressTracker,
                                log_system_status, PRODUCTION_CONFIG)

class DroughtVegetationAnalyzer:
    """Comprehensive drought and vegetation anomaly analysis"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize analyzer with configuration"""
        self.config = load_config(config_path)
        self.random_seed = self.config.get('random_seed', 42)
        self.output_dir = Path(self.config['paths']['figs'])
        self.tables_dir = Path(self.config['paths']['tables'])
        self.data_dir = Path('data_work')
        
        # Ensure directories exist
        ensure_dir(self.output_dir)
        ensure_dir(self.tables_dir)
        ensure_dir(self.data_dir)
        
        # Set up plotting
        setup_plotting()
        
        # Analysis parameters
        self.baseline_period = (2001, 2020)  # Baseline for z-score calculation
        self.comparison_years = [2021, 2023]  # Years for detailed comparison
        self.growing_season_months = [4, 5, 6, 7, 8, 9]  # April to September
        
        # Initialize data containers
        self.mod13q1_data = None
        self.chirps_data = None
        self.processed_data = {}
        
        # Production features
        self.data_validator = DataValidator()
        self.chunked_processor = ChunkedProcessor(chunk_size=PRODUCTION_CONFIG['chunk_size'])
        self.memory_optimizer = MemoryOptimizer()
        
    @memory_monitor(threshold_mb=PRODUCTION_CONFIG['memory_threshold_mb'])
    def load_or_create_mock_data(self, force_recreate: bool = False) -> None:
        """Load existing mock data or create new datasets"""
        
        mod13q1_file = self.data_dir / 'mock_mod13q1_data.pkl'
        chirps_file = self.data_dir / 'mock_chirps_data.pkl'
        
        if not force_recreate and mod13q1_file.exists() and chirps_file.exists():
            print("📂 Loading existing mock datasets...")
            generator = MockGEEDataGenerator()
            self.mod13q1_data = generator.load_mock_data('mock_mod13q1_data.pkl', str(self.data_dir))
            self.chirps_data = generator.load_mock_data('mock_chirps_data.pkl', str(self.data_dir))
        else:
            print("🛰️ Creating new mock datasets...")
            self.mod13q1_data, self.chirps_data = create_mock_datasets(
                output_dir=str(self.data_dir),
                start_date='2000-01-01',
                end_date='2023-12-31'
            )
            
        print(f"✅ Mock data loaded: {len(self.mod13q1_data['districts'])} districts")
        print(f"   - MOD13Q1 time points: {len(self.mod13q1_data['dates'])}")
        print(f"   - CHIRPS time points: {len(self.chirps_data['dates'])}")
        
        # Validate data quality
        print("🔍 Validating data quality...")
        for district in self.mod13q1_data['districts'][:3]:  # Sample validation
            validation = self.data_validator.validate_district_data(
                self.mod13q1_data['data'][district], district
            )
            print(f"   - {district}: Quality score {validation['overall_quality']:.1f}%")
        
    def calculate_vegetation_zscore(self, district: str, variable: str = 'NDVI') -> pd.Series:
        """Calculate z-scores for vegetation indices vs 2001-2020 baseline"""
        
        dates = pd.to_datetime(self.mod13q1_data['dates'])
        values = self.mod13q1_data['data'][district][variable]
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'year': dates.year,
            'month': dates.month
        })
        
        # Filter baseline period (2001-2020)
        baseline_mask = (df['year'] >= self.baseline_period[0]) & (df['year'] <= self.baseline_period[1])
        baseline_data = df[baseline_mask]
        
        # Calculate baseline statistics by month for seasonal adjustment
        baseline_stats = baseline_data.groupby('month')['value'].agg(['mean', 'std']).reset_index()
        baseline_stats.columns = ['month', 'baseline_mean', 'baseline_std']
        
        # Merge with full dataset
        df_with_baseline = df.merge(baseline_stats, on='month', how='left')
        
        # Calculate z-scores
        df_with_baseline['zscore'] = ((df_with_baseline['value'] - df_with_baseline['baseline_mean']) / 
                                     df_with_baseline['baseline_std'])
        
        # Handle cases where std is 0 (no variability in baseline)
        df_with_baseline['zscore'] = df_with_baseline['zscore'].fillna(0)
        
        return df_with_baseline.set_index('date')['zscore']
    
    def calculate_spi(self, district: str, window_days: int = 90) -> pd.Series:
        """Calculate Standardized Precipitation Index (SPI)"""
        
        dates = pd.to_datetime(self.chirps_data['dates'])
        precipitation = self.chirps_data['data'][district]['precipitation']
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'precipitation': precipitation
        })
        df.set_index('date', inplace=True)
        
        # Calculate rolling sum for the specified window
        df['precip_sum'] = df['precipitation'].rolling(window=window_days, min_periods=30).sum()
        
        # Filter baseline period for SPI calculation
        baseline_start = f"{self.baseline_period[0]}-01-01"
        baseline_end = f"{self.baseline_period[1]}-12-31"
        baseline_data = df.loc[baseline_start:baseline_end, 'precip_sum'].dropna()
        
        # Calculate baseline statistics
        baseline_mean = baseline_data.mean()
        baseline_std = baseline_data.std()
        
        # Calculate SPI
        df['spi'] = (df['precip_sum'] - baseline_mean) / baseline_std
        
        return df['spi']
    
    def perform_mann_kendall_trend(self, district: str, variable: str = 'NDVI') -> Dict[str, float]:
        """Perform Mann-Kendall trend test on vegetation data"""
        
        dates = pd.to_datetime(self.mod13q1_data['dates'])
        values = self.mod13q1_data['data'][district][variable]
        
        # Remove any NaN values
        valid_mask = ~np.isnan(values)
        clean_values = values[valid_mask]
        
        if len(clean_values) < 10:  # Need minimum data points for trend analysis
            return {'tau': np.nan, 'p_value': np.nan, 'trend': 'insufficient_data'}
        
        # Perform Mann-Kendall test
        tau, p_value = stats.kendalltau(np.arange(len(clean_values)), clean_values)
        
        # Determine trend significance
        alpha = 0.05
        if p_value < alpha:
            if tau > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no_trend'
            
        return {
            'tau': tau,
            'p_value': p_value,
            'trend': trend,
            'slope': tau,  # Kendall's tau as trend strength
            'n_observations': len(clean_values)
        }
    
    def identify_drought_hotspots(self, severity_threshold: float = -1.5) -> pd.DataFrame:
        """Identify districts with severe drought conditions"""
        
        hotspots = []
        
        for district in self.mod13q1_data['districts']:
            # Calculate NDVI z-scores
            ndvi_zscore = self.calculate_vegetation_zscore(district, 'NDVI')
            evi_zscore = self.calculate_vegetation_zscore(district, 'EVI')
            spi = self.calculate_spi(district)
            
            # Focus on growing season
            growing_season_mask = ndvi_zscore.index.month.isin(self.growing_season_months)
            
            # Calculate severity metrics
            ndvi_severe_events = (ndvi_zscore[growing_season_mask] < severity_threshold).sum()
            evi_severe_events = (evi_zscore[growing_season_mask] < severity_threshold).sum()
            
            # Align SPI with vegetation data timeframe
            spi_aligned = spi.reindex(ndvi_zscore.index, method='nearest')
            spi_severe_events = (spi_aligned[growing_season_mask] < severity_threshold).sum()
            
            # Calculate mean severity during comparison years
            comparison_mask = ndvi_zscore.index.year.isin(self.comparison_years)
            combined_mask = growing_season_mask & comparison_mask
            
            mean_ndvi_2021_2023 = ndvi_zscore[combined_mask].mean()
            mean_spi_2021_2023 = spi_aligned[combined_mask].mean()
            
            # Mann-Kendall trend analysis
            mk_results = self.perform_mann_kendall_trend(district, 'NDVI')
            
            hotspots.append({
                'district': district,
                'ndvi_severe_events': ndvi_severe_events,
                'evi_severe_events': evi_severe_events,
                'spi_severe_events': spi_severe_events,
                'mean_ndvi_zscore_2021_2023': mean_ndvi_2021_2023,
                'mean_spi_2021_2023': mean_spi_2021_2023,
                'trend_tau': mk_results['tau'],
                'trend_p_value': mk_results['p_value'],
                'trend_direction': mk_results['trend'],
                'total_severity_score': ndvi_severe_events + evi_severe_events + spi_severe_events
            })
            
        df = pd.DataFrame(hotspots)
        df = df.sort_values('total_severity_score', ascending=False)
        
        return df
    
    def compare_drought_years(self, year1: int = 2021, year2: int = 2023) -> pd.DataFrame:
        """Compare drought conditions between two specific years"""
        
        comparison_results = []
        
        for district in self.mod13q1_data['districts']:
            # Calculate z-scores
            ndvi_zscore = self.calculate_vegetation_zscore(district, 'NDVI')
            evi_zscore = self.calculate_vegetation_zscore(district, 'EVI')
            spi = self.calculate_spi(district)
            spi_aligned = spi.reindex(ndvi_zscore.index, method='nearest')
            
            # Filter by years and growing season
            year1_mask = (ndvi_zscore.index.year == year1) & (ndvi_zscore.index.month.isin(self.growing_season_months))
            year2_mask = (ndvi_zscore.index.year == year2) & (ndvi_zscore.index.month.isin(self.growing_season_months))
            
            # Calculate metrics for each year
            year1_ndvi_mean = ndvi_zscore[year1_mask].mean()
            year1_evi_mean = evi_zscore[year1_mask].mean()
            year1_spi_mean = spi_aligned[year1_mask].mean()
            
            year2_ndvi_mean = ndvi_zscore[year2_mask].mean()
            year2_evi_mean = evi_zscore[year2_mask].mean()
            year2_spi_mean = spi_aligned[year2_mask].mean()
            
            # Calculate differences
            ndvi_difference = year2_ndvi_mean - year1_ndvi_mean
            evi_difference = year2_evi_mean - year1_evi_mean
            spi_difference = year2_spi_mean - year1_spi_mean
            
            # Determine which year was worse
            if year1_ndvi_mean < year2_ndvi_mean:
                worse_year = year1
                drought_intensity_change = 'improved'
            else:
                worse_year = year2
                drought_intensity_change = 'worsened'
                
            comparison_results.append({
                'district': district,
                f'{year1}_ndvi_zscore': year1_ndvi_mean,
                f'{year1}_evi_zscore': year1_evi_mean,
                f'{year1}_spi': year1_spi_mean,
                f'{year2}_ndvi_zscore': year2_ndvi_mean,
                f'{year2}_evi_zscore': year2_evi_mean,
                f'{year2}_spi': year2_spi_mean,
                'ndvi_difference': ndvi_difference,
                'evi_difference': evi_difference,
                'spi_difference': spi_difference,
                'worse_drought_year': worse_year,
                'change_direction': drought_intensity_change
            })
            
        return pd.DataFrame(comparison_results)
    
    def create_drought_atlas(self) -> None:
        """Create comprehensive drought atlas visualization"""
        
        # Identify hotspots
        hotspots_df = self.identify_drought_hotspots()
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Uzbekistan Agro-Districts Drought Atlas (2000-2023)', fontsize=16, fontweight='bold')
        
        # Panel 1: Severity scores by district
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(hotspots_df)), hotspots_df['total_severity_score'], 
                      color=plt.cm.Reds(hotspots_df['total_severity_score'] / hotspots_df['total_severity_score'].max()))
        ax1.set_xlabel('Districts (ranked by severity)')
        ax1.set_ylabel('Total Severity Score')
        ax1.set_title('A) Drought Severity Ranking')
        ax1.set_xticks(range(0, len(hotspots_df), 2))
        ax1.set_xticklabels([hotspots_df.iloc[i]['district'][:8] for i in range(0, len(hotspots_df), 2)], 
                           rotation=45, ha='right')
        
        # Panel 2: 2021 vs 2023 NDVI comparison
        ax2 = axes[0, 1]
        comparison_df = self.compare_drought_years(2021, 2023)
        scatter = ax2.scatter(comparison_df['2021_ndvi_zscore'], comparison_df['2023_ndvi_zscore'],
                            c=comparison_df['ndvi_difference'], cmap='RdBu_r', s=80, alpha=0.7)
        ax2.plot([-3, 2], [-3, 2], 'k--', alpha=0.5, label='No change line')
        ax2.set_xlabel('2021 NDVI Z-Score')
        ax2.set_ylabel('2023 NDVI Z-Score')
        ax2.set_title('B) 2021 vs 2023 NDVI Comparison')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='NDVI Difference (2023-2021)')
        
        # Panel 3: Trend analysis
        ax3 = axes[1, 0]
        trend_colors = {'increasing': 'green', 'decreasing': 'red', 'no_trend': 'gray', 'insufficient_data': 'black'}
        trend_counts = hotspots_df['trend_direction'].value_counts()
        wedges, texts, autotexts = ax3.pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%',
                                          colors=[trend_colors.get(label, 'gray') for label in trend_counts.index])
        ax3.set_title('C) Long-term NDVI Trends (Mann-Kendall)')
        
        # Panel 4: Precipitation vs vegetation relationship
        ax4 = axes[1, 1]
        scatter2 = ax4.scatter(hotspots_df['mean_spi_2021_2023'], hotspots_df['mean_ndvi_zscore_2021_2023'],
                             c=hotspots_df['total_severity_score'], cmap='Reds', s=80, alpha=0.7)
        ax4.set_xlabel('Mean SPI (2021-2023)')
        ax4.set_ylabel('Mean NDVI Z-Score (2021-2023)')
        ax4.set_title('D) Precipitation-Vegetation Relationship')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.colorbar(scatter2, ax=ax4, label='Severity Score')
        
        plt.tight_layout()
        save_plot(fig, self.output_dir / "drought_atlas_comprehensive.png")
        plt.close()
        
        print("📊 Drought atlas created and saved")
        
    def create_time_series_analysis(self) -> None:
        """Create time series analysis for top drought-affected districts"""
        
        # Get top 4 most affected districts
        hotspots_df = self.identify_drought_hotspots()
        top_districts = hotspots_df.head(4)['district'].tolist()
        
        # Create time series plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Drought & Vegetation Anomaly Time Series (Top 4 Affected Districts)', 
                    fontsize=14, fontweight='bold')
        
        for i, district in enumerate(top_districts):
            ax = axes[i//2, i%2]
            
            # Calculate z-scores and SPI
            ndvi_zscore = self.calculate_vegetation_zscore(district, 'NDVI')
            spi = self.calculate_spi(district)
            spi_aligned = spi.reindex(ndvi_zscore.index, method='nearest')
            
            # Plot time series
            ax.plot(ndvi_zscore.index, ndvi_zscore.values, 'g-', alpha=0.7, label='NDVI Z-Score')
            ax2 = ax.twinx()
            ax2.plot(spi_aligned.index, spi_aligned.values, 'b-', alpha=0.7, label='SPI (90-day)')
            
            # Highlight drought periods
            drought_mask = ndvi_zscore < -1.5
            ax.fill_between(ndvi_zscore.index, -3, 3, where=drought_mask, alpha=0.2, color='red', label='Severe Drought')
            
            # Highlight comparison years
            for year in self.comparison_years:
                year_mask = ndvi_zscore.index.year == year
                if year_mask.any():
                    ax.axvspan(f'{year}-01-01', f'{year}-12-31', alpha=0.1, color='orange')
            
            ax.set_title(f'{district}')
            ax.set_ylabel('NDVI Z-Score', color='g')
            ax2.set_ylabel('SPI', color='b')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axhline(y=-1.5, color='r', linestyle='--', alpha=0.5)
            
            if i == 0:  # Only show legend on first subplot
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
        plt.tight_layout()
        save_plot(fig, self.output_dir / "drought_timeseries_analysis.png")
        plt.close()
        
        print("📈 Time series analysis created and saved")
        
    def save_analysis_tables(self) -> None:
        """Save analysis results to CSV tables"""
        
        # Drought hotspots table
        hotspots_df = self.identify_drought_hotspots()
        hotspots_df.to_csv(self.tables_dir / "drought_hotspots_ranking.csv", index=False)
        
        # Year comparison table
        comparison_df = self.compare_drought_years(2021, 2023)
        comparison_df.to_csv(self.tables_dir / "drought_2021_vs_2023_comparison.csv", index=False)
        
        # Trend analysis summary
        trend_summary = []
        for district in self.mod13q1_data['districts']:
            mk_results = self.perform_mann_kendall_trend(district, 'NDVI')
            trend_summary.append({
                'district': district,
                'mann_kendall_tau': mk_results['tau'],
                'p_value': mk_results['p_value'],
                'trend_direction': mk_results['trend'],
                'n_observations': mk_results['n_observations']
            })
        
        trend_df = pd.DataFrame(trend_summary)
        trend_df.to_csv(self.tables_dir / "vegetation_trend_analysis.csv", index=False)
        
        print("💾 Analysis tables saved to CSV files")
        
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        hotspots_df = self.identify_drought_hotspots()
        comparison_df = self.compare_drought_years(2021, 2023)
        
        # Key findings
        most_affected_district = hotspots_df.iloc[0]['district']
        worst_severity_score = hotspots_df.iloc[0]['total_severity_score']
        
        # Trend analysis summary
        decreasing_trends = (hotspots_df['trend_direction'] == 'decreasing').sum()
        total_districts = len(hotspots_df)
        
        # 2021 vs 2023 comparison
        worse_in_2023 = (comparison_df['change_direction'] == 'worsened').sum()
        worse_in_2021 = (comparison_df['change_direction'] == 'improved').sum()
        
        summary = {
            'analysis_period': '2000-2023',
            'baseline_period': f'{self.baseline_period[0]}-{self.baseline_period[1]}',
            'total_districts_analyzed': total_districts,
            'most_affected_district': most_affected_district,
            'highest_severity_score': float(worst_severity_score),
            'districts_with_decreasing_vegetation_trend': int(decreasing_trends),
            'percent_districts_declining': f"{(decreasing_trends/total_districts)*100:.1f}%",
            'districts_worse_drought_2023_vs_2021': int(worse_in_2023),
            'districts_worse_drought_2021_vs_2023': int(worse_in_2021),
            'key_findings': [
                f"Most severely affected district: {most_affected_district}",
                f"{decreasing_trends} out of {total_districts} districts show declining vegetation trends",
                f"Drought conditions worsened in {worse_in_2023} districts by 2023 compared to 2021",
                f"2021 had worse conditions in {worse_in_2021} districts compared to 2023"
            ],
            'methodology': {
                'vegetation_indices': ['NDVI', 'EVI'],
                'precipitation_data': 'CHIRPS',
                'z_score_baseline': f'{self.baseline_period[0]}-{self.baseline_period[1]}',
                'spi_window': '90 days',
                'trend_test': 'Mann-Kendall',
                'severity_threshold': -1.5
            }
        }
        
        # Save summary report
        import json
        with open(self.tables_dir / "drought_analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

def run() -> Dict[str, Any]:
    """Main function to run the complete drought and vegetation analysis"""
    
    print("🌾 Starting Drought & Vegetation Anomaly Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DroughtVegetationAnalyzer()
    
    # Load or create mock data
    analyzer.load_or_create_mock_data()
    
    # Run core analysis
    print("\n📊 Running drought hotspot identification...")
    hotspots = analyzer.identify_drought_hotspots()
    print(f"   - Identified {len(hotspots)} districts with varying drought severity")
    
    print("\n📈 Performing 2021 vs 2023 comparison...")
    comparison = analyzer.compare_drought_years(2021, 2023)
    print(f"   - Analyzed drought conditions across {len(comparison)} districts")
    
    # Create visualizations
    print("\n🗺️ Creating drought atlas...")
    analyzer.create_drought_atlas()
    
    print("\n📈 Creating time series analysis...")
    analyzer.create_time_series_analysis()
    
    # Save results
    print("\n💾 Saving analysis tables...")
    analyzer.save_analysis_tables()
    
    # Generate summary report
    print("\n📋 Generating summary report...")
    summary = analyzer.generate_summary_report()
    
    print("\n✅ Drought & Vegetation Analysis Complete!")
    print(f"📊 Key Finding: {summary['most_affected_district']} is the most severely affected district")
    print(f"📈 Trend Analysis: {summary['districts_with_decreasing_vegetation_trend']} districts show declining vegetation")
    print(f"🔄 2021 vs 2023: {summary['districts_worse_drought_2023_vs_2021']} districts had worse drought in 2023")
    
    return summary

if __name__ == "__main__":
    results = run()
    print(f"\n📋 Analysis Summary: {results}")