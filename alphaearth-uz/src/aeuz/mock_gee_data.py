"""
Mock Google Earth Engine Data Generator

This module creates mock datasets that resemble real Google Earth Engine data
including MOD13Q1 NDVI/EVI and CHIRPS precipitation data. This allows for
development and testing without requiring GEE authentication, and can be
replaced with real GEE data in production.

Mock Data Specifications:
- MOD13Q1 NDVI/EVI: 250m resolution, 16-day composites, 2000-2023
- CHIRPS: 5.5km resolution, daily precipitation, 2000-2023
- Spatial coverage: Uzbekistan agricultural districts
- Realistic value ranges and seasonal patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Mock data specifications based on real GEE datasets
MOD13Q1_SPECS = {
    'spatial_resolution': 250,  # meters
    'temporal_resolution': 16,  # days
    'ndvi_range': (-2000, 10000),  # scaled by 10000
    'evi_range': (-2000, 10000),   # scaled by 10000
    'bands': ['NDVI', 'EVI', 'VI_Quality', 'red_reflectance', 'nir_reflectance']
}

CHIRPS_SPECS = {
    'spatial_resolution': 5566,  # meters (~5.5km)
    'temporal_resolution': 1,    # daily
    'precipitation_range': (0, 200),  # mm/day (max extreme events)
    'bands': ['precipitation']
}

# Uzbekistan agricultural districts (simplified representation)
UZBEKISTAN_AGRO_DISTRICTS = {
    'Karakalpakstan': {'lat': 43.8, 'lon': 59.4, 'area_km2': 166590},
    'Khorezm': {'lat': 41.5, 'lon': 60.6, 'area_km2': 6464},
    'Navoi': {'lat': 40.1, 'lon': 65.4, 'area_km2': 110800},
    'Bukhara': {'lat': 39.8, 'lon': 64.4, 'area_km2': 39400},
    'Kashkadarya': {'lat': 38.9, 'lon': 65.8, 'area_km2': 28570},
    'Surkhandarya': {'lat': 37.9, 'lon': 67.6, 'area_km2': 20800},
    'Samarkand': {'lat': 39.7, 'lon': 66.9, 'area_km2': 16773},
    'Jizzakh': {'lat': 40.1, 'lon': 67.8, 'area_km2': 21179},
    'Syrdarya': {'lat': 40.8, 'lon': 68.7, 'area_km2': 5120},
    'Tashkent': {'lat': 41.3, 'lon': 69.2, 'area_km2': 15300},
    'Namangan': {'lat': 41.0, 'lon': 71.7, 'area_km2': 7900},
    'Andijan': {'lat': 40.8, 'lon': 72.3, 'area_km2': 4303},
    'Fergana': {'lat': 40.4, 'lon': 71.8, 'area_km2': 6760}
}

class MockGEEDataGenerator:
    """Generate mock Google Earth Engine datasets with realistic properties"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize mock data generator with random seed for reproducibility"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_time_series(self, start_date: str = '2000-01-01', 
                           end_date: str = '2023-12-31',
                           temporal_resolution_days: int = 16) -> pd.DatetimeIndex:
        """Generate time series for mock data"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=temporal_resolution_days)
            
        return pd.DatetimeIndex(dates)
    
    def _add_seasonal_pattern(self, base_values: np.ndarray, 
                            dates: pd.DatetimeIndex,
                            amplitude: float = 0.3) -> np.ndarray:
        """Add realistic seasonal patterns to mock data"""
        # Create seasonal pattern (higher values in growing season)
        day_of_year = dates.dayofyear.values  # Convert to numpy array
        seasonal = amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        if len(seasonal) != len(base_values):
            seasonal = np.tile(seasonal, len(base_values) // len(seasonal) + 1)[:len(base_values)]
            
        return base_values + seasonal
    
    def _add_interannual_variability(self, values: np.ndarray,
                                   dates: pd.DatetimeIndex,
                                   drought_years: List[int] = [2000, 2001, 2008, 2012, 2021],
                                   amplitude: float = 0.4) -> np.ndarray:
        """Add interannual variability including drought years"""
        modified_values = values.copy()
        
        for i, date in enumerate(dates):
            if i >= len(modified_values):
                break
                
            year = date.year
            
            # Add drought stress in specific years
            if year in drought_years:
                if year == 2021:  # Severe drought year
                    stress_factor = -0.6
                elif year in [2000, 2001, 2008, 2012]:  # Moderate drought years
                    stress_factor = -0.3
                else:
                    stress_factor = -0.2
                    
                # Add some spatial variability
                regional_factor = np.random.normal(1.0, 0.2)
                modified_values[i] += stress_factor * amplitude * regional_factor
                
            # Add random year-to-year variability
            yearly_noise = np.random.normal(0, 0.1)
            modified_values[i] += yearly_noise
            
        return modified_values
    
    def generate_mod13q1_mock_data(self, districts: Optional[List[str]] = None,
                                 start_date: str = '2000-01-01',
                                 end_date: str = '2023-12-31') -> Dict:
        """Generate mock MOD13Q1 NDVI/EVI data"""
        
        if districts is None:
            districts = list(UZBEKISTAN_AGRO_DISTRICTS.keys())
            
        # Generate time series (16-day composites)
        dates = self.generate_time_series(start_date, end_date, 16)
        
        mock_data = {
            'dates': dates,
            'districts': districts,
            'metadata': MOD13Q1_SPECS.copy(),
            'data': {}
        }
        
        for district in districts:
            district_info = UZBEKISTAN_AGRO_DISTRICTS[district]
            n_pixels = max(10, int(district_info['area_km2'] / 100))  # Approximate pixel count
            
            district_data = {}
            
            # Generate NDVI data
            base_ndvi = np.random.normal(6000, 1500, len(dates))  # Base NDVI around 0.6
            base_ndvi = np.clip(base_ndvi, MOD13Q1_SPECS['ndvi_range'][0], 
                              MOD13Q1_SPECS['ndvi_range'][1])
            
            # Add seasonal patterns
            ndvi_seasonal = self._add_seasonal_pattern(base_ndvi, dates, amplitude=2000)
            
            # Add interannual variability and drought impacts
            ndvi_final = self._add_interannual_variability(ndvi_seasonal, dates, amplitude=2000)
            ndvi_final = np.clip(ndvi_final, MOD13Q1_SPECS['ndvi_range'][0], 
                               MOD13Q1_SPECS['ndvi_range'][1])
            
            district_data['NDVI'] = ndvi_final
            
            # Generate EVI data (correlated with NDVI but different range)
            evi_base = ndvi_final * 0.8 + np.random.normal(0, 300, len(dates))
            evi_final = np.clip(evi_base, MOD13Q1_SPECS['evi_range'][0], 
                              MOD13Q1_SPECS['evi_range'][1])
            district_data['EVI'] = evi_final
            
            # Generate quality flags (simplified)
            district_data['VI_Quality'] = np.random.randint(0, 3, len(dates))
            
            # Generate reflectance data
            district_data['red_reflectance'] = np.random.normal(2000, 500, len(dates))
            district_data['nir_reflectance'] = np.random.normal(4000, 800, len(dates))
            
            mock_data['data'][district] = district_data
            
        return mock_data
    
    def generate_chirps_mock_data(self, districts: Optional[List[str]] = None,
                                start_date: str = '2000-01-01',
                                end_date: str = '2023-12-31') -> Dict:
        """Generate mock CHIRPS precipitation data"""
        
        if districts is None:
            districts = list(UZBEKISTAN_AGRO_DISTRICTS.keys())
            
        # Generate daily time series
        dates = self.generate_time_series(start_date, end_date, 1)
        
        mock_data = {
            'dates': dates,
            'districts': districts,
            'metadata': CHIRPS_SPECS.copy(),
            'data': {}
        }
        
        for district in districts:
            # Base precipitation pattern for Central Asia
            # Higher in spring/early summer, lower in summer/fall
            base_precip = np.random.exponential(2.0, len(dates))  # Exponential distribution for precip
            
            # Add seasonal pattern
            seasonal_precip = self._add_seasonal_pattern(base_precip, dates, amplitude=1.5)
            seasonal_precip = np.maximum(seasonal_precip, 0)  # No negative precipitation
            
            # Add drought year impacts (reduced precipitation)
            drought_years = [2000, 2001, 2008, 2012, 2021, 2023]
            precip_modified = self._add_interannual_variability(
                seasonal_precip, dates, drought_years, amplitude=3.0)
            precip_modified = np.maximum(precip_modified, 0)
            
            # Clip to realistic range
            precip_final = np.clip(precip_modified, 0, CHIRPS_SPECS['precipitation_range'][1])
            
            mock_data['data'][district] = {
                'precipitation': precip_final
            }
            
        return mock_data
    
    def save_mock_data(self, data: Dict, filename: str, output_dir: str = 'data_work'):
        """Save mock data to file for reuse"""
        from pathlib import Path
        import pickle
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"📁 Mock data saved to {filepath}")
        
    def load_mock_data(self, filename: str, data_dir: str = 'data_work') -> Dict:
        """Load previously saved mock data"""
        from pathlib import Path
        import pickle
        
        filepath = Path(data_dir) / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Mock data file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"📂 Mock data loaded from {filepath}")
        return data

def create_mock_datasets(output_dir: str = 'data_work', 
                        start_date: str = '2000-01-01',
                        end_date: str = '2023-12-31') -> Tuple[Dict, Dict]:
    """Create and save both MOD13Q1 and CHIRPS mock datasets"""
    
    generator = MockGEEDataGenerator()
    
    print("🛰️ Generating mock MOD13Q1 NDVI/EVI data...")
    mod13q1_data = generator.generate_mod13q1_mock_data(start_date=start_date, end_date=end_date)
    generator.save_mock_data(mod13q1_data, 'mock_mod13q1_data.pkl', output_dir)
    
    print("🌧️ Generating mock CHIRPS precipitation data...")
    chirps_data = generator.generate_chirps_mock_data(start_date=start_date, end_date=end_date)
    generator.save_mock_data(chirps_data, 'mock_chirps_data.pkl', output_dir)
    
    print(f"✅ Mock datasets created for {len(mod13q1_data['districts'])} districts")
    print(f"   - Time range: {start_date} to {end_date}")
    print(f"   - MOD13Q1: {len(mod13q1_data['dates'])} time points")
    print(f"   - CHIRPS: {len(chirps_data['dates'])} time points")
    
    return mod13q1_data, chirps_data

if __name__ == "__main__":
    # Test the mock data generator
    mod13q1, chirps = create_mock_datasets()
    
    # Display sample data
    print("\n📊 Sample MOD13Q1 data (Tashkent district):")
    sample_ndvi = mod13q1['data']['Tashkent']['NDVI'][:5]
    print(f"   NDVI (first 5 values): {sample_ndvi}")
    
    print("\n🌧️ Sample CHIRPS data (Tashkent district):")
    sample_precip = chirps['data']['Tashkent']['precipitation'][:5]
    print(f"   Precipitation (first 5 values): {sample_precip}")