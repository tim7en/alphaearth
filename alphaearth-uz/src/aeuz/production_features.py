"""
Production-Ready Features for Drought & Vegetation Analysis

This module adds production-ready features to the drought analysis including:
- Memory optimization for large datasets
- API rate limiting simulation (for future GEE integration)
- Data validation and quality checks
- Chunked processing for memory efficiency
- Caching mechanisms
- Progress tracking and logging
"""

import time
import logging
from functools import wraps
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
import psutil
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory optimization utilities for large-scale analysis"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_memory_threshold(threshold_mb: float = 1000) -> bool:
        """Check if memory usage exceeds threshold"""
        current = MemoryOptimizer.get_memory_usage()
        return current > threshold_mb
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        # Convert float64 to float32 where possible
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        # Convert int64 to smaller int types where possible
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() >= -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
        
        return df

def memory_monitor(threshold_mb: float = 1000):
    """Decorator to monitor memory usage"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_memory = MemoryOptimizer.get_memory_usage()
            logger.info(f"Starting {func.__name__} - Memory: {start_memory:.1f} MB")
            
            if start_memory > threshold_mb:
                logger.warning(f"High memory usage detected: {start_memory:.1f} MB")
            
            result = func(*args, **kwargs)
            
            end_memory = MemoryOptimizer.get_memory_usage()
            memory_diff = end_memory - start_memory
            logger.info(f"Completed {func.__name__} - Memory: {end_memory:.1f} MB (Δ{memory_diff:+.1f} MB)")
            
            return result
        return wrapper
    return decorator

class APIRateLimiter:
    """Simulate API rate limiting for future GEE integration"""
    
    def __init__(self, calls_per_second: float = 5.0):
        """Initialize rate limiter
        
        Args:
            calls_per_second: Maximum API calls per second (GEE limit simulation)
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        self.call_count = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.call_count += 1
        
        if self.call_count % 100 == 0:
            logger.info(f"Processed {self.call_count} API calls")

def rate_limited(calls_per_second: float = 5.0):
    """Decorator for rate limiting API calls"""
    limiter = APIRateLimiter(calls_per_second)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class DataValidator:
    """Validate data quality and completeness"""
    
    @staticmethod
    def validate_time_series(data: np.ndarray, name: str = "data") -> Dict[str, Any]:
        """Validate time series data quality"""
        validation_results = {
            'name': name,
            'length': len(data),
            'missing_values': np.isnan(data).sum(),
            'missing_percentage': (np.isnan(data).sum() / len(data)) * 100,
            'min_value': np.nanmin(data),
            'max_value': np.nanmax(data),
            'mean_value': np.nanmean(data),
            'std_value': np.nanstd(data),
            'outliers': 0,
            'quality_score': 0.0
        }
        
        # Check for outliers (values beyond 3 standard deviations)
        if validation_results['std_value'] > 0:
            z_scores = np.abs((data - validation_results['mean_value']) / validation_results['std_value'])
            validation_results['outliers'] = np.sum(z_scores > 3)
        
        # Calculate quality score (0-100)
        quality_score = 100.0
        quality_score -= validation_results['missing_percentage']  # Penalize missing data
        quality_score -= min(validation_results['outliers'] / len(data) * 100 * 5, 20)  # Penalize outliers
        validation_results['quality_score'] = max(0, quality_score)
        
        return validation_results
    
    @staticmethod
    def validate_district_data(district_data: Dict, district_name: str) -> Dict[str, Any]:
        """Validate all data for a district"""
        validation_summary = {
            'district': district_name,
            'variables': {},
            'overall_quality': 0.0
        }
        
        quality_scores = []
        for variable_name, variable_data in district_data.items():
            if isinstance(variable_data, (list, np.ndarray)):
                validation = DataValidator.validate_time_series(
                    np.array(variable_data), f"{district_name}_{variable_name}"
                )
                validation_summary['variables'][variable_name] = validation
                quality_scores.append(validation['quality_score'])
        
        validation_summary['overall_quality'] = np.mean(quality_scores) if quality_scores else 0.0
        
        return validation_summary

class ChunkedProcessor:
    """Process large datasets in chunks to manage memory"""
    
    def __init__(self, chunk_size: int = 1000):
        """Initialize chunked processor
        
        Args:
            chunk_size: Number of records to process at once
        """
        self.chunk_size = chunk_size
    
    @memory_monitor(threshold_mb=500)
    def process_time_series_chunks(self, data: np.ndarray, 
                                 processing_func: Callable,
                                 **kwargs) -> np.ndarray:
        """Process time series data in chunks"""
        
        if len(data) <= self.chunk_size:
            return processing_func(data, **kwargs)
        
        logger.info(f"Processing {len(data)} records in chunks of {self.chunk_size}")
        
        results = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunk_result = processing_func(chunk, **kwargs)
            results.append(chunk_result)
            
            # Log progress
            progress = min(i + self.chunk_size, len(data))
            logger.info(f"Processed {progress}/{len(data)} records ({progress/len(data)*100:.1f}%)")
        
        return np.concatenate(results)

class CacheManager:
    """Simple caching mechanism for expensive computations"""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Cache a result"""
        self.cache[key] = value
        
        # Log cache size
        if len(self.cache) % 10 == 0:
            logger.debug(f"Cache size: {len(self.cache)} items")
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        logger.info("Cache cleared")

def cached_computation(cache_manager: CacheManager):
    """Decorator for caching expensive computations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached result for {func.__name__}")
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """Initialize progress tracker
        
        Args:
            total_steps: Total number of steps in the operation
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, step: int = 1, message: str = ""):
        """Update progress"""
        self.current_step += step
        
        # Calculate progress
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed_time = time.time() - self.start_time
        
        # Estimate remaining time
        if self.current_step > 0:
            time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * time_per_step
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: unknown"
        
        # Log progress
        if message:
            logger.info(f"{self.description}: {progress_pct:.1f}% ({self.current_step}/{self.total_steps}) - {message} - {eta_str}")
        else:
            logger.info(f"{self.description}: {progress_pct:.1f}% ({self.current_step}/{self.total_steps}) - {eta_str}")
    
    def complete(self):
        """Mark operation as complete"""
        total_time = time.time() - self.start_time
        logger.info(f"{self.description}: 100% complete in {total_time:.1f}s")

# Production-ready configuration
PRODUCTION_CONFIG = {
    'memory_threshold_mb': 1000,  # Alert if memory usage exceeds 1GB
    'api_rate_limit': 5.0,        # 5 calls per second (conservative GEE limit)
    'chunk_size': 1000,           # Process 1000 records at a time
    'enable_caching': True,       # Enable result caching
    'log_level': 'INFO',          # Logging level
    'max_retries': 3,             # Maximum retries for failed operations
    'timeout_seconds': 300,       # Timeout for long operations
}

def apply_production_optimizations(func: Callable) -> Callable:
    """Apply all production optimizations to a function"""
    
    # Create cache manager
    cache_manager = CacheManager()
    
    # Apply decorators
    optimized_func = memory_monitor(PRODUCTION_CONFIG['memory_threshold_mb'])(func)
    optimized_func = rate_limited(PRODUCTION_CONFIG['api_rate_limit'])(optimized_func)
    optimized_func = cached_computation(cache_manager)(optimized_func)
    
    return optimized_func

def log_system_status():
    """Log current system status"""
    memory_mb = MemoryOptimizer.get_memory_usage()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    logger.info(f"System Status - Memory: {memory_mb:.1f} MB, CPU: {cpu_percent:.1f}%")
    
    if memory_mb > PRODUCTION_CONFIG['memory_threshold_mb']:
        logger.warning(f"High memory usage: {memory_mb:.1f} MB")
    
    if cpu_percent > 80:
        logger.warning(f"High CPU usage: {cpu_percent:.1f}%")

if __name__ == "__main__":
    # Test production features
    print("🏭 Testing Production Features")
    print("=" * 40)
    
    # Test memory monitoring
    @memory_monitor()
    def test_memory_function():
        # Create some data to use memory
        data = np.random.random((10000, 100))
        return np.mean(data)
    
    result = test_memory_function()
    print(f"Test result: {result}")
    
    # Test rate limiting
    @rate_limited(calls_per_second=10.0)
    def test_api_call():
        return "API call completed"
    
    print("\nTesting rate limiting...")
    for i in range(5):
        result = test_api_call()
        print(f"Call {i+1}: {result}")
    
    # Test progress tracking
    print("\nTesting progress tracking...")
    tracker = ProgressTracker(10, "Test Operation")
    for i in range(10):
        time.sleep(0.1)  # Simulate work
        tracker.update(1, f"Step {i+1}")
    tracker.complete()
    
    # Log system status
    log_system_status()
    
    print("\n✅ Production features test completed")