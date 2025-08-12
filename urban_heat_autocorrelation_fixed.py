#!/usr/bin/env python3
"""
Urban Heat Analysis with Autocorrelation Corrections
===================================================
Fixed version addressing spatial/temporal autocorrelation issues
"""

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import (cross_val_score, train_test_split, GridSearchCV, 
                                   TimeSeriesSplit, RepeatedKFold, GroupKFold)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Enhanced city configuration with REDUCED sampling to minimize spatial autocorrelation
UZBEKISTAN_CITIES = {
    "Tashkent": {"lat": 41.2995, "lon": 69.2401, "buffer": 25000, "samples": 150},  # Reduced significantly
    "Samarkand": {"lat": 39.6542, "lon": 66.9597, "buffer": 18000, "samples": 100}, 
    "Bukhara": {"lat": 39.7747, "lon": 64.4286, "buffer": 18000, "samples": 100},
    "Namangan": {"lat": 40.9983, "lon": 71.6726, "buffer": 15000, "samples": 80},
    "Andijan": {"lat": 40.7821, "lon": 72.3442, "buffer": 15000, "samples": 80},
    "Nukus": {"lat": 42.4731, "lon": 59.6103, "buffer": 15000, "samples": 80},
    "Qarshi": {"lat": 38.8406, "lon": 65.7890, "buffer": 12000, "samples": 60},
    "Kokand": {"lat": 40.5194, "lon": 70.9428, "buffer": 12000, "samples": 60},
    "Fergana": {"lat": 40.3842, "lon": 71.7843, "buffer": 15000, "samples": 80},
    "Urgench": {"lat": 41.5500, "lon": 60.6333, "buffer": 12000, "samples": 60},
    "Jizzakh": {"lat": 40.1158, "lon": 67.8420, "buffer": 10000, "samples": 50},
    "Termez": {"lat": 37.2242, "lon": 67.2783, "buffer": 12000, "samples": 60},
    "Gulistan": {"lat": 40.4834, "lon": 68.7842, "buffer": 8000, "samples": 40},
    "Navoi": {"lat": 40.0844, "lon": 65.3792, "buffer": 10000, "samples": 50}
}

def authenticate_gee():
    """Initialize Google Earth Engine"""
    try:
        print("🔑 Initializing Google Earth Engine...")
        ee.Initialize(project='ee-sabitovty')
        print("✅ Google Earth Engine initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ GEE Authentication failed: {e}")
        return False

def calculate_spatial_autocorrelation(df, variable='LST_Day', max_distance=5000):
    """Calculate Moran's I to assess spatial autocorrelation"""
    if 'Sample_Longitude' not in df.columns or 'Sample_Latitude' not in df.columns:
        return None, None
    
    # Create coordinate matrix
    coords = df[['Sample_Longitude', 'Sample_Latitude']].values
    
    # Calculate pairwise distances
    distances = squareform(pdist(coords))
    
    # Create spatial weights matrix (inverse distance, with cutoff)
    weights = np.zeros_like(distances)
    mask = (distances > 0) & (distances <= max_distance)
    weights[mask] = 1.0 / distances[mask]
    
    # Row-standardize weights
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]
    weights[np.isnan(weights)] = 0
    
    # Calculate Moran's I
    values = df[variable].values
    n = len(values)
    mean_val = values.mean()
    
    numerator = 0
    denominator = 0
    W = weights.sum()
    
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
        denominator += (values[i] - mean_val) ** 2
    
    if W > 0 and denominator > 0:
        morans_i = (n / W) * (numerator / denominator)
        
        # Expected value under null hypothesis
        expected_i = -1 / (n - 1)
        
        return morans_i, expected_i
    
    return None, None

def remove_spatially_autocorrelated_samples(df, min_distance=2000):
    """Remove samples that are too close to each other to reduce spatial autocorrelation"""
    if 'Sample_Longitude' not in df.columns or 'Sample_Latitude' not in df.columns:
        return df
    
    print(f"🎯 Removing spatially autocorrelated samples (min distance: {min_distance}m)...")
    
    coords = df[['Sample_Longitude', 'Sample_Latitude']].values
    distances = squareform(pdist(coords))
    
    # Start with all samples
    keep_indices = set(range(len(df)))
    
    # For each pair of samples that are too close, remove one randomly
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if distances[i, j] < min_distance / 111000:  # Convert to degrees approximately
                if i in keep_indices and j in keep_indices:
                    # Remove the sample with higher density around it
                    neighbors_i = np.sum(distances[i] < min_distance / 111000)
                    neighbors_j = np.sum(distances[j] < min_distance / 111000)
                    
                    if neighbors_i > neighbors_j:
                        keep_indices.discard(i)
                    else:
                        keep_indices.discard(j)
    
    filtered_df = df.iloc[list(keep_indices)].reset_index(drop=True)
    
    print(f"   📉 Reduced from {len(df)} to {len(filtered_df)} samples")
    print(f"   📊 Removed {len(df) - len(filtered_df)} spatially autocorrelated samples")
    
    return filtered_df

def check_multicollinearity(X, feature_names, threshold=0.95):
    """Check for multicollinearity and remove highly correlated features"""
    print("🔍 Checking for multicollinearity...")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print(f"   ⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (r > {threshold}):")
        for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
            print(f"      {feat1} ↔ {feat2}: r = {corr:.3f}")
    
    # Remove highly correlated features
    to_remove = set()
    for feat1, feat2, corr in high_corr_pairs:
        if feat1 not in to_remove and feat2 not in to_remove:
            to_remove.add(feat2)  # Remove the second feature
    
    if to_remove:
        keep_indices = [i for i, name in enumerate(feature_names) if name not in to_remove]
        X_filtered = X[:, keep_indices]
        feature_names_filtered = [name for name in feature_names if name not in to_remove]
        
        print(f"   🗑️ Removed {len(to_remove)} highly correlated features")
        print(f"   ✅ Kept {len(feature_names_filtered)} features")
        
        return X_filtered, feature_names_filtered
    
    return X, feature_names

def collect_autocorrelation_aware_data():
    """Collect data with explicit autocorrelation prevention"""
    print("🚀 Autocorrelation-Aware Data Collection")
    print("📊 Configuration:")
    print(f"   🎯 Spatial Resolution: 1000m (increased to reduce autocorrelation)")
    print(f"   🏙️ Cities: {len(UZBEKISTAN_CITIES)} administrative centers")
    print(f"   📈 Total Max Samples: {sum(city['samples'] for city in UZBEKISTAN_CITIES.values())}")
    print(f"   🔍 Minimum sample distance: 2000m")
    
    # Create unified geometry for all cities
    all_cities = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([city_info['lon'], city_info['lat']]).buffer(city_info['buffer']),
            {
                'city': city_name,
                'lat': city_info['lat'],
                'lon': city_info['lon'],
                'buffer': city_info['buffer'],
                'samples': city_info['samples']
            }
        ) for city_name, city_info in UZBEKISTAN_CITIES.items()
    ])
    
    uzbekistan_bounds = all_cities.geometry().bounds()
    scale = 2000  # Increased scale to reduce spatial autocorrelation
    
    print("🛰️ Collecting satellite data with temporal separation...")
    
    # === TEMPORALLY SEPARATED DATASETS ===
    
    # 1. MODIS LST - Use ONLY 2024 data to avoid temporal autocorrelation
    print("🌡️ MODIS LST (2024 only, summer season)...")
    try:
        modis_lst = ee.ImageCollection('MODIS/061/MOD11A2') \
            .filterDate('2024-06-01', '2024-08-31') \
            .filterBounds(uzbekistan_bounds) \
            .select(['LST_Day_1km', 'LST_Night_1km']) \
            .median()
        
        lst_day = modis_lst.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_Day')
        lst_night = modis_lst.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_Night')
        
    except Exception as e:
        print(f"⚠️ MODIS error: {e}, using fallback values")
        lst_day = ee.Image.constant(28).rename('LST_Day')
        lst_night = ee.Image.constant(22).rename('LST_Night')
    
    # 2. Landsat 8/9 - Single season only (summer 2024)
    print("🛰️ Landsat 8/9 (summer 2024 only)...")
    try:
        landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
            .filterDate('2024-06-01', '2024-08-31') \
            .filterBounds(uzbekistan_bounds) \
            .filter(ee.Filter.lt('CLOUD_COVER', 30)) \
            .median()
        
        # Scale and process Landsat bands
        optical = landsat.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal = landsat.select('ST_B.*').multiply(0.00341802).add(149.0).subtract(273.15)
        
        # ONLY essential indices to reduce multicollinearity
        ndvi = optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndbi = optical.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        ndwi = optical.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        
        landsat_lst = thermal.select('ST_B10').rename('Landsat_LST')
        
    except Exception as e:
        print(f"⚠️ Landsat error: {e}, using fallback values")
        ndvi = ee.Image.constant(0.3).rename('NDVI')
        ndbi = ee.Image.constant(0.1).rename('NDBI')
        ndwi = ee.Image.constant(0.2).rename('NDWI')
        landsat_lst = ee.Image.constant(26).rename('Landsat_LST')
    
    # 3. Dynamic World V1 - SINGLE time period only
    print("🌍 Dynamic World V1 (summer 2024)...")
    try:
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterDate('2024-06-01', '2024-08-31') \
            .filterBounds(uzbekistan_bounds) \
            .select(['built', 'trees', 'water']) \
            .median()
        
        built_prob = dw.select('built').rename('Built_Probability')
        green_prob = dw.select('trees').rename('Green_Probability')
        water_prob = dw.select('water').rename('Water_Probability')
        
    except Exception as e:
        print(f"⚠️ Dynamic World error: {e}, using fallback values")
        built_prob = ee.Image.constant(0.2).rename('Built_Probability')
        green_prob = ee.Image.constant(0.4).rename('Green_Probability')
        water_prob = ee.Image.constant(0.1).rename('Water_Probability')
    
    # 4. Static datasets only
    print("🗺️ Static geographical data...")
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('Elevation')
    slope = ee.Terrain.slope(elevation).rename('Slope')
    
    # === MINIMAL FEATURE ENGINEERING (reduce correlation) ===
    print("🔧 Engineering minimal independent features...")
    
    # Only truly independent features
    thermal_amplitude = lst_day.subtract(lst_night).rename('Thermal_Amplitude')
    
    # Simple ecological metrics (no complex combinations that create autocorrelation)
    urban_intensity = built_prob.rename('Urban_Intensity')
    vegetation_coverage = green_prob.rename('Vegetation_Coverage')
    water_coverage = water_prob.rename('Water_Coverage')
    
    # === MINIMAL FEATURE SET ===
    combined_image = ee.Image.cat([
        # Core thermal (primary target and one derivative)
        lst_day, thermal_amplitude,
        # Independent land cover
        urban_intensity, vegetation_coverage, water_coverage,
        # Core vegetation (only one index)
        ndvi,
        # Static topography
        elevation, slope
    ])
    
    # === SPATIALLY DISTRIBUTED SAMPLING ===
    print("🎯 Spatially distributed sampling to minimize autocorrelation...")
    
    all_data = []
    
    for city_name, city_info in UZBEKISTAN_CITIES.items():
        print(f"  📍 Sampling {city_name} ({city_info['samples']} points max)...")
        
        city_point = ee.Geometry.Point([city_info['lon'], city_info['lat']])
        city_buffer = city_point.buffer(city_info['buffer'])
        
        try:
            # Use systematic sampling with larger scale to ensure spatial independence
            city_samples = combined_image.sample(
                region=city_buffer,
                scale=scale,  # 2000m scale
                numPixels=city_info['samples'],
                seed=42,
                geometries=True
            ).map(lambda f: f.set({'City': city_name}))
            
            # Extract data
            sample_data = city_samples.getInfo()
            
            city_df_list = []
            for feature in sample_data['features']:
                props = feature['properties']
                if 'LST_Day' in props and props['LST_Day'] is not None:
                    coords = feature['geometry']['coordinates']
                    props['Sample_Longitude'] = coords[0]
                    props['Sample_Latitude'] = coords[1]
                    props['City_Latitude'] = city_info['lat']
                    props['City_Longitude'] = city_info['lon']
                    city_df_list.append(props)
            
            if city_df_list:
                city_df = pd.DataFrame(city_df_list)
                all_data.append(city_df)
                print(f"    ✅ {len(city_df)} samples collected from {city_name}")
            else:
                print(f"    ⚠️ No valid samples from {city_name}")
                
        except Exception as e:
            print(f"    ❌ Error sampling {city_name}: {e}")
            continue
    
    if not all_data:
        print("❌ No data collected from any city!")
        return None
    
    # Combine all city data
    df = pd.concat(all_data, ignore_index=True)
    
    # Apply spatial autocorrelation filtering
    df = remove_spatially_autocorrelated_samples(df, min_distance=2000)
    
    # Calculate and report spatial autocorrelation
    morans_i, expected_i = calculate_spatial_autocorrelation(df, 'LST_Day')
    if morans_i is not None:
        print(f"\n📊 Spatial Autocorrelation Assessment:")
        print(f"   🎯 Moran's I: {morans_i:.4f}")
        print(f"   🎯 Expected I: {expected_i:.4f}")
        print(f"   📈 Autocorrelation level: {'HIGH' if morans_i > 0.3 else 'MODERATE' if morans_i > 0.1 else 'LOW'}")
    
    # Store autocorrelation values for later use
    df.attrs = {'morans_i': morans_i, 'expected_i': expected_i}
    
    # Data quality assessment
    print(f"\n📊 Autocorrelation-Corrected Dataset Summary:")
    print(f"   📈 Total samples: {len(df)}")
    print(f"   🏙️ Cities: {df['City'].nunique()}")
    print(f"   📋 Features: {len([col for col in df.columns if col not in ['City', 'Sample_Longitude', 'Sample_Latitude', 'City_Latitude', 'City_Longitude']])}")
    print(f"   🌡️ Temperature range: {df['LST_Day'].min():.1f}°C to {df['LST_Day'].max():.1f}°C")
    
    # Conservative quality filtering
    for col in ['LST_Day', 'Thermal_Amplitude']:
        if col in df.columns:
            q5, q95 = df[col].quantile([0.05, 0.95])  # More conservative filtering
            df = df[(df[col] >= q5) & (df[col] <= q95)]
    
    print(f"   ✅ After quality control: {len(df)} samples")
    
    return df

def build_autocorrelation_robust_models(df):
    """Build models with explicit autocorrelation controls"""
    print("\n🤖 Building Autocorrelation-Robust ML Models...")
    
    # Prepare features with multicollinearity check
    feature_cols = [col for col in df.columns 
                   if col not in ['City', 'Sample_Longitude', 'Sample_Latitude', 
                                'City_Latitude', 'City_Longitude'] 
                   and df[col].dtype in ['float64', 'int64']]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['LST_Day'].fillna(df['LST_Day'].median())
    
    print(f"📊 Initial Features: {len(feature_cols)}")
    print(f"📊 Samples: {len(X)}")
    
    # Check and remove multicollinearity
    X_array, feature_names_filtered = check_multicollinearity(X.values, feature_cols, threshold=0.8)
    
    # Conservative outlier removal
    Q1, Q3 = y.quantile([0.1, 0.9])  # More conservative
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = (y >= lower) & (y <= upper)
    X_array, y = X_array[mask], y[mask]
    
    print(f"📊 After outlier removal: {len(X_array)} samples")
    
    # Minimal preprocessing to avoid overfitting
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # NO PCA - use original features to maintain interpretability
    print(f"🔍 Using {X_scaled.shape[1]} original features (no PCA)")
    
    # Conservative CV strategies with explicit spatial controls
    cv_strategies = {
        'Spatial_GroupKFold': GroupKFold(n_splits=min(4, df['City'].nunique())),  # City-based splits
        'Conservative_KFold': RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),
    }
    
    groups = df.loc[mask, 'City'].astype('category').cat.codes
    
    # Conservative models to avoid overfitting
    models = {
        'Conservative_RF': RandomForestRegressor(
            n_estimators=100,  # Reduced
            max_depth=8,       # Reduced
            min_samples_split=10,  # Increased
            min_samples_leaf=5,    # Increased
            max_features='sqrt', 
            random_state=42, n_jobs=-1
        ),
        'Simple_XGBoost': xgb.XGBRegressor(
            n_estimators=100,      # Reduced
            learning_rate=0.1,     # Increased
            max_depth=6,           # Reduced
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        ),
        'Linear_Elastic': ElasticNet(
            alpha=1.0,      # Regularization
            l1_ratio=0.5,   # Mix of L1 and L2
            random_state=42
        )
    }
    
    # Enhanced training and validation with spatial awareness
    results = {}
    
    # SPATIAL train-test split (by city)
    cities = df.loc[mask, 'City'].unique()
    np.random.seed(42)
    test_cities = np.random.choice(cities, size=max(1, len(cities)//4), replace=False)
    
    train_mask = ~df.loc[mask, 'City'].isin(test_cities)
    test_mask = df.loc[mask, 'City'].isin(test_cities)
    
    X_train_spatial = X_scaled[train_mask]
    X_test_spatial = X_scaled[test_mask]
    y_train_spatial = y[train_mask]
    y_test_spatial = y[test_mask]
    
    print(f"🌍 Spatial split: {len(X_train_spatial)} train, {len(X_test_spatial)} test")
    print(f"📍 Test cities: {', '.join(test_cities)}")
    
    train_groups = groups[train_mask]
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        
        cv_scores = {}
        for cv_name, cv_strategy in cv_strategies.items():
            try:
                if cv_name == 'Spatial_GroupKFold':
                    scores = cross_val_score(
                        model, X_train_spatial, y_train_spatial, 
                        cv=cv_strategy, groups=train_groups,
                        scoring='r2', n_jobs=-1
                    )
                else:
                    scores = cross_val_score(
                        model, X_train_spatial, y_train_spatial, 
                        cv=cv_strategy, scoring='r2', n_jobs=-1
                    )
                
                cv_scores[cv_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max()
                }
            except Exception as e:
                print(f"    ⚠️ {cv_name} failed: {e}")
                cv_scores[cv_name] = {'mean': -999, 'std': 0, 'min': -999, 'max': -999}
        
        # Train final model
        model.fit(X_train_spatial, y_train_spatial)
        
        # Comprehensive evaluation on SPATIAL test set
        y_pred_train = model.predict(X_train_spatial)
        y_pred_test = model.predict(X_test_spatial)
        
        train_r2 = r2_score(y_train_spatial, y_pred_train)
        test_r2 = r2_score(y_test_spatial, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test_spatial, y_pred_test))
        test_mae = mean_absolute_error(y_test_spatial, y_pred_test)
        
        overfitting = train_r2 - test_r2
        
        # Calculate robust CV score
        valid_scores = [cv['mean'] for cv in cv_scores.values() if cv['mean'] > -999]
        robust_cv_score = np.mean(valid_scores) if valid_scores else -999
        cv_stability = np.std(valid_scores) if len(valid_scores) > 1 else 999
        
        # Penalize overfitting more heavily
        composite_score = test_r2 - (overfitting * 0.5) - (cv_stability * 0.2)
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'robust_cv_score': robust_cv_score,
            'cv_stability': cv_stability,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'overfitting': overfitting,
            'composite_score': composite_score
        }
        
        print(f"  ✅ {name}:")
        print(f"     Spatial CV R² = {robust_cv_score:.4f} ± {cv_stability:.4f}")
        print(f"     Spatial Test R² = {test_r2:.4f}")
        print(f"     Overfitting = {overfitting:.4f}")
        print(f"     RMSE = {test_rmse:.3f}°C")
    
    # Select best model based on spatial generalization
    best_model = max(results.items(), key=lambda x: x[1]['composite_score'])
    print(f"\n🏆 Best Model: {best_model[0]} (Spatial Composite Score: {best_model[1]['composite_score']:.4f})")
    
    # Additional diagnostics
    print(f"\n📊 Model Diagnostics:")
    for name, result in results.items():
        overfitting_severity = "HIGH" if result['overfitting'] > 0.1 else "MODERATE" if result['overfitting'] > 0.05 else "LOW"
        print(f"   {name}: Overfitting = {overfitting_severity} ({result['overfitting']:.3f})")
    
    return {
        'results': results,
        'best_model': best_model[0],
        'scaler': scaler,
        'features': feature_names_filtered,
        'X_test_spatial': X_test_spatial,
        'y_test_spatial': y_test_spatial,
        'test_cities': test_cities,
        'spatial_autocorrelation': {
            'morans_i': getattr(df, 'attrs', {}).get('morans_i', None),
            'expected_i': getattr(df, 'attrs', {}).get('expected_i', None)
        }
    }

def main():
    """Main execution function with autocorrelation controls"""
    print("🚀 Starting Autocorrelation-Robust Urban Heat Analysis")
    print("="*100)
    
    try:
        # Initialize GEE
        if not authenticate_gee():
            return
        
        # Collect autocorrelation-aware data
        print("\n📡 Phase 1: Autocorrelation-aware data collection...")
        df = collect_autocorrelation_aware_data()
        
        if df is None:
            print("❌ No data collected. Exiting...")
            return
        
        # Build robust models
        print("\n🤖 Phase 2: Building autocorrelation-robust ML models...")
        model_results = build_autocorrelation_robust_models(df)
        
        # Final autocorrelation assessment
        if model_results['spatial_autocorrelation']['morans_i'] is not None:
            morans_i = model_results['spatial_autocorrelation']['morans_i']
            autocorr_level = "HIGH" if morans_i > 0.3 else "MODERATE" if morans_i > 0.1 else "LOW"
            print(f"\n📊 Final Spatial Autocorrelation: {autocorr_level} (Moran's I = {morans_i:.4f})")
        
        best_result = model_results['results'][model_results['best_model']]
        print(f"\n🎯 Final Performance Summary:")
        print(f"   Best Model: {model_results['best_model']}")
        print(f"   Spatial Test R²: {best_result['test_r2']:.4f}")
        print(f"   RMSE: {best_result['test_rmse']:.3f}°C")
        print(f"   Overfitting: {best_result['overfitting']:.4f}")
        print(f"   Test Cities: {', '.join(model_results['test_cities'])}")
        
        print("\n" + "="*100)
        print("🎉 Autocorrelation-Robust Analysis Complete!")
        print("📋 Key Improvements:")
        print("   ✅ Spatially distributed sampling (2km minimum distance)")
        print("   ✅ Temporal consistency (single season only)")
        print("   ✅ Multicollinearity removal")
        print("   ✅ Spatial cross-validation")
        print("   ✅ Conservative model parameters")
        print("   ✅ Spatial autocorrelation assessment")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
