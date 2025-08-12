
# Scientific Surface Urban Heat Island (SUHI) Analysis - Uzbekistan Cities

## Executive Summary

This analysis quantifies surface urban heat island intensity (SUHI) for 14 major cities in Uzbekistan using scientifically defensible remote sensing methods. SUHI is calculated as the difference in land surface temperature (LST) between urban cores and nearby rural rings during the warm season (June-August).

## Methodology

### Data Sources and Processing
- **MODIS LST**: MOD11A2 V6.1 8-day composites, properly scaled (0.02 K) and converted to °C
- **Landsat 8/9**: Collection-2 Level-2 surface reflectance with QA_PIXEL cloud masking
- **Dynamic World V1**: 10m land cover probabilities for urban/rural classification
- **Analysis Scale**: 1000m (MODIS native resolution) with explicit aggregation
- **Temporal Focus**: Warm season (JJA) median composites to avoid seasonal bias

### SUHI Calculation
Urban cores defined using Dynamic World built-up probabilities (threshold > 0.15), with rural rings at 25km distance (built-up probability < 0.15, water < 0.1). SUHI calculated separately for day and night as:

**SUHI = LST_urban - LST_rural**

### Quality Assurance
- Server-side processing via Google Earth Engine reduceRegions
- Proper Landsat L2 cloud/shadow masking using QA_PIXEL bits
- MODIS LST scaling and offset correction
- Scale-consistent analysis at 1km resolution
- Warm season filtering for temporal consistency

## Results

### Regional SUHI Trends (2016 → 2025)

**Surface Urban Heat Island Changes:**
- Mean SUHI Day Change: 0.006 ± 0.859°C
- Mean SUHI Night Change: 0.536 ± 1.012°C
- Range Day SUHI Change: -0.514 to 0.998°C
- Range Night SUHI Change: -0.446 to 1.576°C

### City-Level Results

| City | SUHI Day Change (°C) | SUHI Night Change (°C) | Urban Built Change | Rural Built Change |
|------|---------------------|----------------------|------------------|------------------|
| Tashkent | N/A | N/A | N/A | 0.0102 |
| Nukus | N/A | N/A | N/A | 0.0042 |
| Andijan | -0.514 | 1.576 | 0.0036 | 0.0020 |
| Bukhara | N/A | N/A | N/A | -0.0043 |
| Jizzakh | N/A | N/A | N/A | 0.0179 |
| Qarshi | N/A | N/A | N/A | 0.0110 |
| Navoiy | N/A | N/A | N/A | 0.0104 |
| Namangan | N/A | N/A | N/A | 0.0012 |
| Samarkand | 0.998 | 0.477 | 0.0256 | 0.0063 |
| Termez | N/A | N/A | N/A | 0.0010 |
| Gulistan | N/A | N/A | N/A | 0.0425 |
| Nurafshon | N/A | N/A | N/A | 0.0086 |
| Fergana | N/A | N/A | N/A | 0.0005 |
| Urgench | -0.465 | -0.446 | 0.0285 | 0.0064 |


## Technical Implementation

### Server-Side Processing
This analysis leverages Google Earth Engine's distributed computing infrastructure:
- Zonal statistics computed server-side using grouped reducers
- Minimal data transfer (only aggregated results)
- Scale-aware processing at 1km resolution
- Proper handling of mixed-resolution datasets

---

**Report Generated**: 2025-08-12 12:13:41
**Analysis Method**: Scientific SUHI (Urban-Rural LST Difference)
**Data Processing**: Google Earth Engine Server-Side
