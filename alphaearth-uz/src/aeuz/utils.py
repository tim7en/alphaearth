import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def load_config(path: str = "config.json"):
    """Load configuration file, trying multiple possible paths"""
    possible_paths = [
        path,
        f"alphaearth-uz/{path}",
        f"../{path}",
        f"../../{path}"
    ]
    
    for config_path in possible_paths:
        try:
            return json.loads(Path(config_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
    
    # If none found, create a default config
    default_config = {
        "country": "Uzbekistan",
        "regions": ["Karakalpakstan", "Tashkent", "Samarkand", "Bukhara", "Namangan"],
        "time_window": [2017, 2025],
        "crs": "EPSG:4326",
        "random_seed": 42,
        "paths": {
            "raw": "data_raw",
            "work": "data_work", 
            "final": "data_final",
            "figs": "figs",
            "tables": "tables"
        }
    }
    return default_config

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def setup_plotting():
    """Configure matplotlib for high-quality research plots"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'
    })

def generate_synthetic_embeddings(n_samples: int = 1000, n_features: int = 256, 
                                regions: List[str] = None) -> pd.DataFrame:
    """Generate synthetic AlphaEarth-like embeddings for analysis"""
    if regions is None:
        regions = ["Karakalpakstan", "Tashkent", "Samarkand", "Bukhara", "Namangan"]
    
    np.random.seed(42)  # For reproducibility
    
    # Generate base embeddings with regional characteristics
    data = []
    for i in range(n_samples):
        region = np.random.choice(regions)
        
        # Create region-specific patterns
        if region == "Karakalpakstan":
            # Arid, water-stressed characteristics
            base_pattern = np.random.normal(-0.5, 0.8, n_features)
        elif region == "Tashkent":
            # Urban, developed characteristics  
            base_pattern = np.random.normal(0.3, 0.6, n_features)
        elif region == "Samarkand":
            # Historical, moderate development
            base_pattern = np.random.normal(0.1, 0.7, n_features)
        elif region == "Bukhara":
            # Desert edge, irrigation-dependent
            base_pattern = np.random.normal(-0.2, 0.9, n_features)
        else:  # Namangan
            # Valley, agricultural
            base_pattern = np.random.normal(0.2, 0.5, n_features)
            
        # Add temporal variation
        year = np.random.randint(2017, 2026)
        season = np.random.choice(['spring', 'summer', 'autumn', 'winter'])
        
        # Create coordinate-like data
        lat = np.random.uniform(37.2, 45.6)  # Uzbekistan latitude range
        lon = np.random.uniform(55.9, 73.2)  # Uzbekistan longitude range
        
        sample = {
            'sample_id': f"{region}_{i:04d}",
            'region': region,
            'latitude': lat,
            'longitude': lon,
            'year': year,
            'season': season,
            'water_stress_indicator': max(0, min(1, np.random.normal(0.6, 0.3))),
            'vegetation_index': max(0, min(1, np.random.normal(0.4, 0.2))),
            'soil_moisture_est': max(0, min(1, np.random.normal(0.3, 0.25))),
            'temperature_anomaly': np.random.normal(2.1, 1.2),
            'degradation_risk': max(0, min(1, np.random.normal(0.45, 0.3)))
        }
        
        # Add embedding features
        for j, val in enumerate(base_pattern):
            sample[f'embed_{j:03d}'] = val
            
        data.append(sample)
    
    return pd.DataFrame(data)

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for data"""
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Calculate margin of error
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, n - 1)
    margin_error = t_value * (std / np.sqrt(n))
    
    return (mean - margin_error, mean + margin_error)

def perform_trend_analysis(data: np.ndarray, years: np.ndarray) -> Dict[str, float]:
    """Perform Mann-Kendall trend test and calculate trend statistics"""
    from scipy import stats
    
    # Linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, data)
    
    # Mann-Kendall test for monotonic trend
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if data[j] > data[i]:
                s += 1
            elif data[j] < data[i]:
                s -= 1
    
    # Calculate variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Calculate p-value for two-tailed test
    p_mk = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'linear_slope': slope,
        'linear_r2': r_value**2,
        'linear_p_value': p_value,
        'mk_z_statistic': z,
        'mk_p_value': p_mk,
        'trend_direction': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend',
        'trend_significance': 'significant' if p_mk < 0.05 else 'not significant'
    }

def create_summary_statistics(df: pd.DataFrame, groupby_col: str, 
                            value_cols: List[str]) -> pd.DataFrame:
    """Generate comprehensive summary statistics"""
    summary_stats = []
    
    for group in df[groupby_col].unique():
        group_data = df[df[groupby_col] == group]
        
        stats_row = {'group': group, 'n_samples': len(group_data)}
        
        for col in value_cols:
            if col in group_data.columns:
                values = group_data[col].dropna()
                if len(values) > 0:
                    stats_row.update({
                        f'{col}_mean': values.mean(),
                        f'{col}_std': values.std(),
                        f'{col}_median': values.median(),
                        f'{col}_min': values.min(),
                        f'{col}_max': values.max(),
                        f'{col}_q25': values.quantile(0.25),
                        f'{col}_q75': values.quantile(0.75)
                    })
                    
                    # Add confidence interval
                    if len(values) > 1:
                        ci_low, ci_high = calculate_confidence_interval(values.values)
                        stats_row.update({
                            f'{col}_ci_low': ci_low,
                            f'{col}_ci_high': ci_high
                        })
        
        summary_stats.append(stats_row)
    
    return pd.DataFrame(summary_stats)

def save_plot(fig, filepath: str, title: str = None):
    """Save high-quality plot with metadata"""
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.99, 0.01, f"Generated: {timestamp}", 
             ha='right', va='bottom', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def validate_data_quality(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, Any]:
    """Perform comprehensive data quality assessment"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_required_columns': [],
        'column_completeness': {},
        'data_types': {},
        'outliers_detected': {},
        'duplicates': 0,
        'quality_score': 0.0
    }
    
    # Check required columns
    for col in required_cols:
        if col not in df.columns:
            quality_report['missing_required_columns'].append(col)
    
    # Check completeness and data types
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        quality_report['column_completeness'][col] = 100 - missing_pct
        quality_report['data_types'][col] = str(df[col].dtype)
        
        # Detect outliers for numeric columns
        if df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | 
                       (df[col] > (Q3 + 1.5 * IQR))).sum()
            quality_report['outliers_detected'][col] = outliers
    
    # Check duplicates
    quality_report['duplicates'] = df.duplicated().sum()
    
    # Calculate overall quality score
    completeness_scores = list(quality_report['column_completeness'].values())
    avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
    
    missing_req_penalty = len(quality_report['missing_required_columns']) * 10
    duplicate_penalty = (quality_report['duplicates'] / len(df)) * 100
    
    quality_report['quality_score'] = max(0, avg_completeness - missing_req_penalty - duplicate_penalty)
    
    return quality_report
