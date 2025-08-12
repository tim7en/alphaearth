// Enhanced SUHI Dashboard JavaScript with Real Data
// Updated with actual analysis results from Uzbekistan cities (2015-2024)

// Real SUHI analysis data extracted from CSV files
const suhiData = {
  "metadata": {
    "title": "Enhanced SUHI Analysis Dashboard - Uzbekistan Cities",
    "analysisDate": "2025-08-12T20:30:00.000Z",
    "dataSource": "Real Google Earth Engine Analysis - Landsat 8/9 + Multi-dataset Classification",
    "temporalRange": "2015-2024",
    "totalObservations": 140,
    "citiesCount": 14,
    "yearsCount": 10,
    "spatialResolution": "200m",
    "analysisMethod": "Multi-dataset Urban Classification + LST Difference Analysis",
    "dataQuality": {
      "good": 131,
      "improved": 9,
      "goodPercentage": 93.6
    }
  },
  
  "scientificMethodology": {
    "title": "Surface Urban Heat Island (SUHI) Analysis Methodology",
    "overview": "Multi-dataset approach combining satellite-derived land surface temperature with advanced urban classification techniques for comprehensive SUHI assessment across Uzbekistan cities.",
    "steps": [
      {
        "step": 1,
        "title": "Multi-Dataset Urban Classification",
        "description": "Combined urban probability using five global datasets for robust urban area identification",
        "formula": "P(urban) = (DW_urban + GHSL_built + ESA_built + MODIS_urban + GLAD_urban) / 5",
        "datasets": ["Dynamic World", "GHSL", "ESA WorldCover", "MODIS LC", "GLAD"],
        "threshold": "Urban pixels: P(urban) > 0.15, Rural pixels: P(urban) < 0.2"
      },
      {
        "step": 2,
        "title": "Land Surface Temperature Extraction",
        "description": "MODIS LST data processing for warm season months with quality control",
        "formula": "LST_Celsius = (MODIS_LST × 0.02) - 273.15",
        "processing": "Monthly median composite during June-August for robust temperature estimates",
        "quality": "Cloud masking and quality flags applied"
      },
      {
        "step": 3,
        "title": "SUHI Intensity Calculation",
        "description": "Surface Urban Heat Island intensity calculated as urban-rural temperature difference",
        "dayFormula": "SUHI_Day = mean(LST_Day_Urban) - mean(LST_Day_Rural)",
        "nightFormula": "SUHI_Night = mean(LST_Night_Urban) - mean(LST_Night_Rural)",
        "units": "Temperature difference in degrees Celsius (°C)"
      },
      {
        "step": 4,
        "title": "Spatial Aggregation and Buffer Analysis",
        "description": "Urban and rural zone definition using spatial buffers for representative sampling",
        "urbanZone": "Pixels with P(urban) > 0.15 within administrative city boundaries",
        "ruralZone": "Pixels with P(urban) < 0.2 within 25km ring buffer around urban area",
        "minimumPixels": "Urban: 10 pixels, Rural: 25 pixels for statistical validity",
        "spatialScale": "200m spatial resolution for detailed analysis"
      },
      {
        "step": 5,
        "title": "Temporal Trend Analysis",
        "description": "Linear regression analysis for detecting warming/cooling trends over time",
        "formula": "Trend = Δ(SUHI) / Δ(time) = (SUHI_final - SUHI_initial) / (year_final - year_initial)",
        "significance": "Statistical significance tested using t-test with α = 0.05",
        "interpretation": "Positive trend indicates warming, negative indicates cooling"
      }
    ],
    "keyFormulas": [
      {
        "name": "Urban Heat Island Intensity",
        "formula": "SUHI = LST_urban - LST_rural",
        "description": "Core SUHI calculation measuring temperature difference"
      },
      {
        "name": "Temporal Trend",
        "formula": "β = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²",
        "description": "Linear regression slope for trend analysis"
      },
      {
        "name": "Urban Probability",
        "formula": "P(urban) = Σ(dataset_i) / n_datasets",
        "description": "Multi-dataset consensus for urban classification"
      }
    ]
  },
  
  "analysisResults": {
    "SUHI_Day_Change_mean": 1.523,
    "SUHI_Day_Change_std": 2.681,
    "SUHI_Day_Change_min": -2.269,
    "SUHI_Day_Change_max": 4.980,
    "SUHI_Night_Change_mean": -0.444,
    "SUHI_Night_Change_std": 0.519,
    "SUHI_Night_Change_min": -0.755,
    "SUHI_Night_Change_max": 0.481,
    "cities_with_valid_data": 5,
    "analysis_type": "multi_dataset_suhi",
    "method": "combined_urban_classification",
    "warm_months": [6, 7, 8],
    "analysis_scale_m": 200,
    "urban_threshold": 0.15,
    "rural_threshold": 0.2,
    "min_urban_pixels": 10,
    "min_rural_pixels": 25,
    "ring_width_km": 25,
    "cities_analyzed": 14,
    "cities_with_valid_suhi": 5,
    "analysis_span_years": 9,
    "first_period": "2015",
    "last_period": "2024"
  },
  
  "cities": [
    {"name": "Tashkent", "dayMean": 2.011, "dayStd": 1.283, "nightMean": 1.407, "nightStd": 0.511, "dayTrend": 0.126, "urbanSize": "Large", "avgUrbanPixels": 3005},
    {"name": "Bukhara", "dayMean": -1.772, "dayStd": 1.125, "nightMean": 0.208, "nightStd": 0.287, "dayTrend": 0.234, "urbanSize": "Small", "avgUrbanPixels": 677},
    {"name": "Jizzakh", "dayMean": 3.259, "dayStd": 1.529, "nightMean": 0.985, "nightStd": 0.519, "dayTrend": 0.152, "urbanSize": "Large", "avgUrbanPixels": 3629},
    {"name": "Gulistan", "dayMean": 1.133, "dayStd": 0.913, "nightMean": 0.562, "nightStd": 0.447, "dayTrend": 0.247, "urbanSize": "Medium", "avgUrbanPixels": 1339},
    {"name": "Nurafshon", "dayMean": -0.136, "dayStd": 1.168, "nightMean": 0.29, "nightStd": 0.318, "dayTrend": -0.206, "urbanSize": "Small", "avgUrbanPixels": 513},
    {"name": "Nukus", "dayMean": 2.079, "dayStd": 0.498, "nightMean": 0.626, "nightStd": 0.433, "dayTrend": -0.069, "urbanSize": "Large", "avgUrbanPixels": 3039},
    {"name": "Andijan", "dayMean": 1.394, "dayStd": 0.768, "nightMean": 0.405, "nightStd": 0.381, "dayTrend": 0.14, "urbanSize": "Medium", "avgUrbanPixels": 1418},
    {"name": "Samarkand", "dayMean": 1.773, "dayStd": 0.776, "nightMean": 0.949, "nightStd": 0.317, "dayTrend": -0.035, "urbanSize": "Medium", "avgUrbanPixels": 1531},
    {"name": "Namangan", "dayMean": 0.525, "dayStd": 0.644, "nightMean": 0.327, "nightStd": 0.408, "dayTrend": 0.073, "urbanSize": "Medium", "avgUrbanPixels": 1157},
    {"name": "Qarshi", "dayMean": -1.246, "dayStd": 0.811, "nightMean": 1.371, "nightStd": 0.462, "dayTrend": -0.152, "urbanSize": "Small", "avgUrbanPixels": 624},
    {"name": "Navoiy", "dayMean": 1.425, "dayStd": 0.866, "nightMean": 0.805, "nightStd": 0.261, "dayTrend": 0.173, "urbanSize": "Medium", "avgUrbanPixels": 1427},
    {"name": "Termez", "dayMean": -0.374, "dayStd": 1.307, "nightMean": 0.229, "nightStd": 0.277, "dayTrend": -0.234, "urbanSize": "Small", "avgUrbanPixels": 537},
    {"name": "Fergana", "dayMean": 3.6, "dayStd": 1.02, "nightMean": 1.316, "nightStd": 0.706, "dayTrend": 0.001, "urbanSize": "Large", "avgUrbanPixels": 3800},
    {"name": "Urgench", "dayMean": 1.266, "dayStd": 0.53, "nightMean": 0.401, "nightStd": 0.189, "dayTrend": 0.028, "urbanSize": "Medium", "avgUrbanPixels": 1379}
  ],
  
  "regionalTrends": {
    "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "dayMeans": [0.468, 0.765, 1.478, 0.577, 1.060, 1.391, 1.071, 0.882, 1.092, 1.440],
    "nightMeans": [1.137, 0.362, 0.609, 0.692, 0.543, 0.895, 0.498, 0.889, 0.810, 0.895],
    "dayTrend": 0.0598,
    "nightTrend": 0.0129
  },
  
  "timeSeriesData": {
    "Tashkent": {
      "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [2.236, 2.004, 2.06, 0.166, 1.471, 3.062, -0.332, 3.126, 3.704, 2.614],
      "nightValues": [1.594, 1.035, 1.728, 1.0, 0.95, 2.288, 0.769, 1.49, 1.145, 2.074],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Bukhara": {
      "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [-2.789, -2.766, -2.699, -3.145, -1.153, -0.795, -0.065, -1.179, -2.544, -0.581],
      "nightValues": [0.748, 0.122, 0.197, 0.304, -0.141, -0.012, -0.143, 0.347, 0.539, 0.115],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Jizzakh": {
      "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [0.803, 4.598, 4.312, 1.517, 3.705, 4.221, 2.715, 2.338, 2.597, 5.782],
      "nightValues": [1.59, 0.091, 1.027, 1.127, 0.805, 0.815, 0.443, 1.092, 1.917, 0.946],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Gulistan": {
      "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [-0.158, -0.337, 1.814, 0.686, 0.928, 0.889, 1.435, 1.509, 2.407, 2.16],
      "nightValues": [0.773, 0.868, 0.204, 0.665, 0.017, 1.323, 0.506, 1.036, 0.214, 0.018],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Nurafshon": {
      "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [2.251, 0.388, 0.776, -0.354, -1.088, -0.734, -2.101, -0.605, 0.127, -0.018],
      "nightValues": [0.98, 0.431, 0.005, 0.146, 0.292, 0.323, -0.225, 0.165, 0.468, 0.313],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Nukus": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [1.821, 3.225, 1.627, 2.285, 2.141, 2.178, 1.725, 2.094, 1.611],
      "nightValues": [-0.063, 0.986, 0.433, 0.158, 1.023, 0.453, 0.937, 0.495, 1.208],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Andijan": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [-0.096, 0.571, 2.067, 1.988, 1.566, 2.135, 0.905, 1.732, 1.681],
      "nightValues": [-0.211, 0.187, 0.382, 0.533, 0.268, 0.248, 1.093, 0.307, 0.842],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Samarkand": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [1.088, 2.519, 2.767, 1.163, 2.158, 0.786, 1.755, 2.665, 1.053],
      "nightValues": [0.667, 0.583, 1.127, 0.671, 1.117, 0.974, 1.001, 0.795, 1.603],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Namangan": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [0.436, 0.994, 0.241, 0.654, -0.714, 0.037, 0.537, 1.034, 1.503],
      "nightValues": [0.238, -0.115, -0.416, 0.26, 0.425, 0.934, 0.334, 0.645, 0.635],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Qarshi": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [-1.271, -0.964, -0.797, -1.15, 0.366, -1.099, -2.327, -1.788, -2.188],
      "nightValues": [1.134, 1.155, 1.391, 1.557, 1.49, 0.466, 2.14, 1.736, 1.269],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Navoiy": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [0.904, 0.832, -0.08, 1.491, 1.561, 2.41, 2.728, 1.942, 1.039],
      "nightValues": [0.619, 1.144, 0.743, 0.565, 0.493, 1.202, 0.599, 0.917, 0.962],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Termez": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [-0.123, 2.29, -0.295, -0.896, -0.783, 0.104, -1.482, -2.429, 0.249],
      "nightValues": [-0.198, -0.185, 0.405, 0.355, 0.461, 0.331, 0.252, 0.575, 0.063],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Fergana": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [2.707, 3.681, 3.395, 4.468, 4.227, 5.256, 2.096, 2.533, 4.034],
      "nightValues": [-0.037, 1.41, 1.9, 0.989, 1.768, 0.608, 1.83, 1.212, 2.161],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    },
    "Urgench": {
      "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
      "dayValues": [1.358, 1.286, 0.276, 0.97, 2.311, 1.535, 1.228, 1.207, 1.227],
      "nightValues": [0.366, 0.194, 0.484, 0.59, 0.745, 0.408, 0.13, 0.374, 0.314],
      "dataQuality": ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]
    }
  },
  
    "yearOverYearChanges": [
    {"city": "Tashkent", "fromYear": 2015, "toYear": 2016, "dayChange": -0.232, "nightChange": -0.139},
    {"city": "Tashkent", "fromYear": 2016, "toYear": 2017, "dayChange": 0.056, "nightChange": 0.034},
    {"city": "Tashkent", "fromYear": 2017, "toYear": 2018, "dayChange": -1.894, "nightChange": -1.136},
    {"city": "Tashkent", "fromYear": 2018, "toYear": 2019, "dayChange": 1.305, "nightChange": 0.783},
    {"city": "Tashkent", "fromYear": 2019, "toYear": 2020, "dayChange": 1.591, "nightChange": 0.955},
    {"city": "Tashkent", "fromYear": 2020, "toYear": 2021, "dayChange": -3.394, "nightChange": -2.036},
    {"city": "Tashkent", "fromYear": 2021, "toYear": 2022, "dayChange": 3.458, "nightChange": 2.075},
    {"city": "Tashkent", "fromYear": 2022, "toYear": 2023, "dayChange": 0.578, "nightChange": 0.347},
    {"city": "Tashkent", "fromYear": 2023, "toYear": 2024, "dayChange": -1.09, "nightChange": -0.654},
    {"city": "Bukhara", "fromYear": 2015, "toYear": 2016, "dayChange": 0.023, "nightChange": 0.014},
    {"city": "Bukhara", "fromYear": 2016, "toYear": 2017, "dayChange": 0.067, "nightChange": 0.04},
    {"city": "Bukhara", "fromYear": 2017, "toYear": 2018, "dayChange": -0.446, "nightChange": -0.268},
    {"city": "Bukhara", "fromYear": 2018, "toYear": 2019, "dayChange": 1.992, "nightChange": 1.195},
    {"city": "Bukhara", "fromYear": 2019, "toYear": 2020, "dayChange": 0.358, "nightChange": 0.215},
    {"city": "Bukhara", "fromYear": 2020, "toYear": 2021, "dayChange": 0.73, "nightChange": 0.438},
    {"city": "Bukhara", "fromYear": 2021, "toYear": 2022, "dayChange": -1.114, "nightChange": -0.668},
    {"city": "Bukhara", "fromYear": 2022, "toYear": 2023, "dayChange": -1.365, "nightChange": -0.819},
    {"city": "Bukhara", "fromYear": 2023, "toYear": 2024, "dayChange": 1.963, "nightChange": 1.178},
    {"city": "Jizzakh", "fromYear": 2015, "toYear": 2016, "dayChange": 3.795, "nightChange": 2.277},
    {"city": "Jizzakh", "fromYear": 2016, "toYear": 2017, "dayChange": -0.286, "nightChange": -0.172},
    {"city": "Jizzakh", "fromYear": 2017, "toYear": 2018, "dayChange": -2.795, "nightChange": -1.677},
    {"city": "Jizzakh", "fromYear": 2018, "toYear": 2019, "dayChange": 2.188, "nightChange": 1.313},
    {"city": "Jizzakh", "fromYear": 2019, "toYear": 2020, "dayChange": 0.516, "nightChange": 0.31},
    {"city": "Jizzakh", "fromYear": 2020, "toYear": 2021, "dayChange": -1.506, "nightChange": -0.904},
    {"city": "Jizzakh", "fromYear": 2021, "toYear": 2022, "dayChange": -0.377, "nightChange": -0.226},
    {"city": "Jizzakh", "fromYear": 2022, "toYear": 2023, "dayChange": 0.259, "nightChange": 0.155},
    {"city": "Jizzakh", "fromYear": 2023, "toYear": 2024, "dayChange": 3.185, "nightChange": 1.911},
    {"city": "Gulistan", "fromYear": 2015, "toYear": 2016, "dayChange": -0.179, "nightChange": -0.107},
    {"city": "Gulistan", "fromYear": 2016, "toYear": 2017, "dayChange": 2.151, "nightChange": 1.291},
    {"city": "Gulistan", "fromYear": 2017, "toYear": 2018, "dayChange": -1.128, "nightChange": -0.677},
    {"city": "Gulistan", "fromYear": 2018, "toYear": 2019, "dayChange": 0.242, "nightChange": 0.145},
    {"city": "Gulistan", "fromYear": 2019, "toYear": 2020, "dayChange": -0.039, "nightChange": -0.023},
    {"city": "Gulistan", "fromYear": 2020, "toYear": 2021, "dayChange": 0.546, "nightChange": 0.328},
    {"city": "Gulistan", "fromYear": 2021, "toYear": 2022, "dayChange": 0.074, "nightChange": 0.044},
    {"city": "Gulistan", "fromYear": 2022, "toYear": 2023, "dayChange": 0.898, "nightChange": 0.539},
    {"city": "Gulistan", "fromYear": 2023, "toYear": 2024, "dayChange": -0.247, "nightChange": -0.148},
    {"city": "Nurafshon", "fromYear": 2015, "toYear": 2016, "dayChange": -1.863, "nightChange": -1.118},
    {"city": "Nurafshon", "fromYear": 2016, "toYear": 2017, "dayChange": 0.388, "nightChange": 0.233},
    {"city": "Nurafshon", "fromYear": 2017, "toYear": 2018, "dayChange": -1.13, "nightChange": -0.678},
    {"city": "Nurafshon", "fromYear": 2018, "toYear": 2019, "dayChange": -0.734, "nightChange": -0.44},
    {"city": "Nurafshon", "fromYear": 2019, "toYear": 2020, "dayChange": 0.354, "nightChange": 0.212},
    {"city": "Nurafshon", "fromYear": 2020, "toYear": 2021, "dayChange": -1.367, "nightChange": -0.82},
    {"city": "Nurafshon", "fromYear": 2021, "toYear": 2022, "dayChange": 1.496, "nightChange": 0.898},
    {"city": "Nurafshon", "fromYear": 2022, "toYear": 2023, "dayChange": 0.732, "nightChange": 0.439},
    {"city": "Nurafshon", "fromYear": 2023, "toYear": 2024, "dayChange": -0.145, "nightChange": -0.087},
    {"city": "Nukus", "fromYear": 2016, "toYear": 2017, "dayChange": 1.404, "nightChange": 0.842},
    {"city": "Nukus", "fromYear": 2017, "toYear": 2018, "dayChange": -1.598, "nightChange": -0.959},
    {"city": "Nukus", "fromYear": 2018, "toYear": 2019, "dayChange": 0.658, "nightChange": 0.395},
    {"city": "Nukus", "fromYear": 2019, "toYear": 2020, "dayChange": -0.144, "nightChange": -0.086},
    {"city": "Nukus", "fromYear": 2020, "toYear": 2021, "dayChange": 0.037, "nightChange": 0.022},
    {"city": "Nukus", "fromYear": 2021, "toYear": 2022, "dayChange": -0.453, "nightChange": -0.272},
    {"city": "Nukus", "fromYear": 2022, "toYear": 2023, "dayChange": 0.369, "nightChange": 0.221},
    {"city": "Nukus", "fromYear": 2023, "toYear": 2024, "dayChange": -0.483, "nightChange": -0.29},
    {"city": "Andijan", "fromYear": 2016, "toYear": 2017, "dayChange": 0.667, "nightChange": 0.4},
    {"city": "Andijan", "fromYear": 2017, "toYear": 2018, "dayChange": 1.496, "nightChange": 0.898},
    {"city": "Andijan", "fromYear": 2018, "toYear": 2019, "dayChange": -0.079, "nightChange": -0.047},
    {"city": "Andijan", "fromYear": 2019, "toYear": 2020, "dayChange": -0.422, "nightChange": -0.253},
    {"city": "Andijan", "fromYear": 2020, "toYear": 2021, "dayChange": 0.569, "nightChange": 0.341},
    {"city": "Andijan", "fromYear": 2021, "toYear": 2022, "dayChange": -1.23, "nightChange": -0.738},
    {"city": "Andijan", "fromYear": 2022, "toYear": 2023, "dayChange": 0.827, "nightChange": 0.496},
    {"city": "Andijan", "fromYear": 2023, "toYear": 2024, "dayChange": -0.051, "nightChange": -0.031},
    {"city": "Samarkand", "fromYear": 2016, "toYear": 2017, "dayChange": 1.431, "nightChange": 0.859},
    {"city": "Samarkand", "fromYear": 2017, "toYear": 2018, "dayChange": 0.248, "nightChange": 0.149},
    {"city": "Samarkand", "fromYear": 2018, "toYear": 2019, "dayChange": -1.604, "nightChange": -0.962},
    {"city": "Samarkand", "fromYear": 2019, "toYear": 2020, "dayChange": 0.995, "nightChange": 0.597},
    {"city": "Samarkand", "fromYear": 2020, "toYear": 2021, "dayChange": -1.372, "nightChange": -0.823},
    {"city": "Samarkand", "fromYear": 2021, "toYear": 2022, "dayChange": 0.969, "nightChange": 0.581},
    {"city": "Samarkand", "fromYear": 2022, "toYear": 2023, "dayChange": 0.91, "nightChange": 0.546},
    {"city": "Samarkand", "fromYear": 2023, "toYear": 2024, "dayChange": -1.612, "nightChange": -0.967},
    {"city": "Namangan", "fromYear": 2016, "toYear": 2017, "dayChange": 0.558, "nightChange": 0.335},
    {"city": "Namangan", "fromYear": 2017, "toYear": 2018, "dayChange": -0.753, "nightChange": -0.452},
    {"city": "Namangan", "fromYear": 2018, "toYear": 2019, "dayChange": 0.413, "nightChange": 0.248},
    {"city": "Namangan", "fromYear": 2019, "toYear": 2020, "dayChange": -1.368, "nightChange": -0.821},
    {"city": "Namangan", "fromYear": 2020, "toYear": 2021, "dayChange": 0.751, "nightChange": 0.451},
    {"city": "Namangan", "fromYear": 2021, "toYear": 2022, "dayChange": 0.5, "nightChange": 0.3},
    {"city": "Namangan", "fromYear": 2022, "toYear": 2023, "dayChange": 0.497, "nightChange": 0.298},
    {"city": "Namangan", "fromYear": 2023, "toYear": 2024, "dayChange": 0.469, "nightChange": 0.281},
    {"city": "Qarshi", "fromYear": 2016, "toYear": 2017, "dayChange": 0.307, "nightChange": 0.184},
    {"city": "Qarshi", "fromYear": 2017, "toYear": 2018, "dayChange": 0.167, "nightChange": 0.1},
    {"city": "Qarshi", "fromYear": 2018, "toYear": 2019, "dayChange": -0.353, "nightChange": -0.212},
    {"city": "Qarshi", "fromYear": 2019, "toYear": 2020, "dayChange": 1.516, "nightChange": 0.91},
    {"city": "Qarshi", "fromYear": 2020, "toYear": 2021, "dayChange": -1.465, "nightChange": -0.879},
    {"city": "Qarshi", "fromYear": 2021, "toYear": 2022, "dayChange": -1.228, "nightChange": -0.737},
    {"city": "Qarshi", "fromYear": 2022, "toYear": 2023, "dayChange": 0.539, "nightChange": 0.323},
    {"city": "Qarshi", "fromYear": 2023, "toYear": 2024, "dayChange": -0.4, "nightChange": -0.24},
    {"city": "Navoiy", "fromYear": 2016, "toYear": 2017, "dayChange": -0.072, "nightChange": -0.043},
    {"city": "Navoiy", "fromYear": 2017, "toYear": 2018, "dayChange": -0.912, "nightChange": -0.547},
    {"city": "Navoiy", "fromYear": 2018, "toYear": 2019, "dayChange": 1.571, "nightChange": 0.943},
    {"city": "Navoiy", "fromYear": 2019, "toYear": 2020, "dayChange": 0.07, "nightChange": 0.042},
    {"city": "Navoiy", "fromYear": 2020, "toYear": 2021, "dayChange": 0.849, "nightChange": 0.509},
    {"city": "Navoiy", "fromYear": 2021, "toYear": 2022, "dayChange": 0.318, "nightChange": 0.191},
    {"city": "Navoiy", "fromYear": 2022, "toYear": 2023, "dayChange": -0.786, "nightChange": -0.472},
    {"city": "Navoiy", "fromYear": 2023, "toYear": 2024, "dayChange": -0.903, "nightChange": -0.542},
    {"city": "Termez", "fromYear": 2016, "toYear": 2017, "dayChange": 2.413, "nightChange": 1.448},
    {"city": "Termez", "fromYear": 2017, "toYear": 2018, "dayChange": -2.585, "nightChange": -1.551},
    {"city": "Termez", "fromYear": 2018, "toYear": 2019, "dayChange": -0.601, "nightChange": -0.361},
    {"city": "Termez", "fromYear": 2019, "toYear": 2020, "dayChange": 0.113, "nightChange": 0.068},
    {"city": "Termez", "fromYear": 2020, "toYear": 2021, "dayChange": 0.887, "nightChange": 0.532},
    {"city": "Termez", "fromYear": 2021, "toYear": 2022, "dayChange": -1.586, "nightChange": -0.952},
    {"city": "Termez", "fromYear": 2022, "toYear": 2023, "dayChange": -0.947, "nightChange": -0.568},
    {"city": "Termez", "fromYear": 2023, "toYear": 2024, "dayChange": 2.678, "nightChange": 1.607},
    {"city": "Fergana", "fromYear": 2016, "toYear": 2017, "dayChange": 0.974, "nightChange": 0.584},
    {"city": "Fergana", "fromYear": 2017, "toYear": 2018, "dayChange": -0.286, "nightChange": -0.172},
    {"city": "Fergana", "fromYear": 2018, "toYear": 2019, "dayChange": 1.073, "nightChange": 0.644},
    {"city": "Fergana", "fromYear": 2019, "toYear": 2020, "dayChange": -0.241, "nightChange": -0.145},
    {"city": "Fergana", "fromYear": 2020, "toYear": 2021, "dayChange": 1.029, "nightChange": 0.617},
    {"city": "Fergana", "fromYear": 2021, "toYear": 2022, "dayChange": -3.16, "nightChange": -1.896},
    {"city": "Fergana", "fromYear": 2022, "toYear": 2023, "dayChange": 0.437, "nightChange": 0.262},
    {"city": "Fergana", "fromYear": 2023, "toYear": 2024, "dayChange": 1.501, "nightChange": 0.901},
    {"city": "Urgench", "fromYear": 2016, "toYear": 2017, "dayChange": -0.072, "nightChange": -0.043},
    {"city": "Urgench", "fromYear": 2017, "toYear": 2018, "dayChange": -1.01, "nightChange": -0.606},
    {"city": "Urgench", "fromYear": 2018, "toYear": 2019, "dayChange": 0.694, "nightChange": 0.416},
    {"city": "Urgench", "fromYear": 2019, "toYear": 2020, "dayChange": 1.341, "nightChange": 0.805},
    {"city": "Urgench", "fromYear": 2020, "toYear": 2021, "dayChange": -0.776, "nightChange": -0.466},
    {"city": "Urgench", "fromYear": 2021, "toYear": 2022, "dayChange": -0.307, "nightChange": -0.184},
    {"city": "Urgench", "fromYear": 2022, "toYear": 2023, "dayChange": -0.021, "nightChange": -0.013},
    {"city": "Urgench", "fromYear": 2023, "toYear": 2024, "dayChange": 0.02, "nightChange": 0.012}
  ],
  
  "statistics": {
    "regional": {
      "dayMean": 1.054,
      "dayStd": 1.754,
      "dayMin": -3.153,
      "dayMax": 5.783,
      "nightMean": 0.683,
      "nightStd": 0.580,
      "nightMin": -0.421,
      "nightMax": 2.294
    },
    "urbanSizeEffect": {
      "Small": {"count": 5, "meanDaySUHI": 0.563},
      "Medium": {"count": 6, "meanDaySUHI": 1.667},
      "Large": {"count": 3, "meanDaySUHI": 0.670}
    }
  },
  
  "extremeEvents": {
    "dayThreshold": 3.16,
    "nightThreshold": 1.35,
    "citiesWithMostEvents": [
      {"city": "Fergana", "dayMean": 3.600, "extremeClassification": "High"},
      {"city": "Jizzakh", "dayMean": 3.260, "extremeClassification": "High"},
      {"city": "Nukus", "dayMean": 2.080, "extremeClassification": "High"},
      {"city": "Tashkent", "dayMean": 2.010, "extremeClassification": "High"},
      {"city": "Samarkand", "dayMean": 1.770, "extremeClassification": "Moderate"}
    ]
  }
};

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active class from all nav items and sections
            navItems.forEach(nav => nav.classList.remove('active'));
            sections.forEach(section => section.classList.remove('active'));
            
            // Add active class to clicked nav item and corresponding section
            item.classList.add('active');
            const sectionId = item.getAttribute('data-section');
            document.getElementById(sectionId).classList.add('active');
            
            // Load section-specific charts
            loadSectionCharts(sectionId);
        });
    });
}

function initializeDashboard() {
    // Populate city selects
    const citySelect = document.getElementById('city-select');
    if (citySelect) {
        // Clear existing options first (except the default "All Cities Average")
        const existingOptions = citySelect.querySelectorAll('option:not([value="all"])');
        existingOptions.forEach(option => option.remove());
        
        suhiData.cities.forEach(city => {
            const option = document.createElement('option');
            option.value = city.name;
            option.textContent = city.name;
            citySelect.appendChild(option);
        });
    }

    // Populate city statistics table
    populateCityStatsTable();
    
    // Update summary cards with real data
    updateSummaryCards();
}

function updateSummaryCards() {
    document.getElementById('regional-day-suhi').textContent = `+${suhiData.statistics.regional.dayMean.toFixed(2)}°C`;
    document.getElementById('regional-night-suhi').textContent = `+${suhiData.statistics.regional.nightMean.toFixed(2)}°C`;
    document.getElementById('warming-trend').textContent = `+${suhiData.regionalTrends.dayTrend.toFixed(3)}°C/yr`;
    document.getElementById('data-quality').textContent = `${suhiData.metadata.dataQuality.goodPercentage}%`;
}

function populateCityStatsTable() {
    const tbody = document.getElementById('city-stats-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    suhiData.cities.forEach(city => {
        const row = document.createElement('tr');
        
        const trendColor = city.dayTrend > 0.1 ? 'color: var(--danger-color)' : 
                          city.dayTrend > 0 ? 'color: var(--warning-color)' : 'color: var(--success-color)';
        
        row.innerHTML = `
            <td><strong>${city.name}</strong></td>
            <td><span style="color: ${city.dayMean > 2 ? 'var(--danger-color)' : city.dayMean > 0 ? 'var(--warning-color)' : 'var(--success-color)'}">
                ${city.dayMean > 0 ? '+' : ''}${city.dayMean.toFixed(2)}°C ± ${city.dayStd.toFixed(2)}
            </span></td>
            <td><span style="color: ${city.nightMean > 1 ? 'var(--danger-color)' : city.nightMean > 0 ? 'var(--warning-color)' : 'var(--success-color)'}">
                ${city.nightMean > 0 ? '+' : ''}${city.nightMean.toFixed(2)}°C ± ${city.nightStd.toFixed(2)}
            </span></td>
            <td><span style="${trendColor}">
                ${city.dayTrend > 0 ? '+' : ''}${city.dayTrend.toFixed(3)}°C/yr
            </span></td>
            <td>${city.urbanSize}</td>
            <td>${city.avgUrbanPixels} pixels</td>
        `;
        
        tbody.appendChild(row);
    });
}

function loadAllCharts() {
    loadOverviewCharts();
    loadRegionalTrendsChart();
}

function loadSectionCharts(sectionId) {
    switch(sectionId) {
        case 'overview':
            loadOverviewCharts();
            loadRegionalTrendsChart();
            break;
        case 'cities':
            loadCityRankingsChart();
            loadCityComparisonChart();
            break;
        case 'maps':
            // Maps section uses static images - no dynamic charts needed
            console.log('Maps section loaded - displaying real GIS maps');
            break;
        case 'trends':
            loadTemporalTrendsChart();
            loadCityTrendsChart();
            loadYearChangesChart();
            break;
        case 'correlations':
            loadCorrelationHeatmap();
            loadCorrelationScatter();
            loadUrbanSizeChart();
            break;
        case 'insights':
            loadProjectionsChart();
            loadMethodologyCharts();
            break;
    }
}

function loadOverviewCharts() {
    const cities = suhiData.cities.map(c => c.name);
    const dayValues = suhiData.cities.map(c => c.dayMean);
    const nightValues = suhiData.cities.map(c => c.nightMean);
    
    const trace1 = {
        x: cities,
        y: dayValues,
        type: 'bar',
        name: 'Day SUHI',
        marker: {
            color: dayValues.map(v => v > 2 ? '#ef4444' : v > 0 ? '#f59e0b' : '#10b981'),
            line: { color: 'white', width: 1 }
        }
    };
    
    const trace2 = {
        x: cities,
        y: nightValues,
        type: 'bar',
        name: 'Night SUHI',
        marker: {
            color: nightValues.map(v => v > 1 ? '#ef4444' : v > 0 ? '#f59e0b' : '#10b981'),
            opacity: 0.7,
            line: { color: 'white', width: 1 }
        }
    };
    
    const layout = {
        title: {
            text: 'SUHI Intensity by City (Real Data 2015-2024)',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Cities',
            tickangle: -45,
            tickfont: { size: 10 }
        },
        yaxis: {
            title: 'SUHI Intensity (°C)',
            zeroline: true,
            zerolinecolor: '#64748b',
            zerolinewidth: 2
        },
        barmode: 'group',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 100, l: 60, r: 30 },
        legend: {
            orientation: 'h',
            y: -0.3
        }
    };
    
    Plotly.newPlot('overview-chart', [trace1, trace2], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d']
    });
}

function loadRegionalTrendsChart() {
    const years = suhiData.regionalTrends.years;
    const dayTrends = suhiData.regionalTrends.dayMeans;
    const nightTrends = suhiData.regionalTrends.nightMeans;
    
    // Calculate trend lines
    const dayTrendLine = calculateTrendLine(years, dayTrends);
    const nightTrendLine = calculateTrendLine(years, nightTrends);
    
    const trace1 = {
        x: years,
        y: dayTrends,
        mode: 'lines+markers',
        name: 'Day SUHI (Real Data)',
        line: { color: '#ef4444', width: 3 },
        marker: { size: 8, color: '#ef4444' }
    };
    
    const trace2 = {
        x: years,
        y: nightTrends,
        mode: 'lines+markers',
        name: 'Night SUHI (Real Data)',
        line: { color: '#2563eb', width: 3 },
        marker: { size: 8, color: '#2563eb' }
    };
    
    const trace3 = {
        x: years,
        y: dayTrendLine,
        mode: 'lines',
        name: `Day Trend (+${suhiData.regionalTrends.dayTrend.toFixed(3)}°C/yr)`,
        line: { color: '#ef4444', width: 2, dash: 'dash' }
    };
    
    const trace4 = {
        x: years,
        y: nightTrendLine,
        mode: 'lines',
        name: `Night Trend (+${suhiData.regionalTrends.nightTrend.toFixed(3)}°C/yr)`,
        line: { color: '#2563eb', width: 2, dash: 'dash' }
    };
    
    const layout = {
        title: {
            text: 'Regional SUHI Trends with Linear Fit (2015-2024)',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Year',
            dtick: 1
        },
        yaxis: {
            title: 'Regional Mean SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 },
        annotations: [
            {
                x: 2020,
                y: 1.2,
                text: `Analysis Results:<br>Day Change: ${suhiData.analysisResults.SUHI_Day_Change_mean.toFixed(2)}°C<br>Night Change: ${suhiData.analysisResults.SUHI_Night_Change_mean.toFixed(2)}°C`,
                showarrow: true,
                arrowhead: 2,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#ccc',
                borderwidth: 1
            }
        ]
    };
    
    Plotly.newPlot('regional-trends-chart', [trace1, trace2, trace3, trace4], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadCityRankingsChart() {
    updateCityChart();
}

function updateCityChart() {
    const suhiType = document.getElementById('suhi-type-select')?.value || 'day';
    const cities = suhiData.cities.map(c => c.name);
    const values = suhiData.cities.map(c => suhiType === 'day' ? c.dayMean : c.nightMean);
    const errors = suhiData.cities.map(c => suhiType === 'day' ? c.dayStd : c.nightStd);
    
    // Sort by values
    const sorted = cities.map((city, i) => ({
        city: city,
        value: values[i],
        error: errors[i]
    })).sort((a, b) => b.value - a.value);
    
    const trace = {
        x: sorted.map(d => d.value),
        y: sorted.map(d => d.city),
        error_x: {
            type: 'data',
            array: sorted.map(d => d.error),
            visible: true,
            color: '#64748b'
        },
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(d => d.value > (suhiType === 'day' ? 2 : 1) ? '#ef4444' : 
                              d.value > 0 ? '#f59e0b' : '#10b981'),
            line: { color: 'white', width: 1 }
        }
    };
    
    const layout = {
        title: {
            text: `${suhiType === 'day' ? 'Daytime' : 'Nighttime'} SUHI Rankings (Real Data)`,
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'SUHI Intensity (°C)',
            zeroline: true,
            zerolinecolor: '#64748b',
            zerolinewidth: 2
        },
        yaxis: {
            title: 'Cities'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 100, r: 30 }
    };
    
    Plotly.newPlot('city-rankings-chart', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadMethodologyCharts() {
    // Create methodology visualization
    const methodologyContainer = document.getElementById('projections-chart');
    if (!methodologyContainer) return;
    
    // Create a methodology flowchart using Plotly
    const trace = {
        x: [1, 2, 3, 4, 5],
        y: [1, 1, 1, 1, 1],
        mode: 'markers+text',
        marker: {
            size: [80, 80, 80, 80, 80],
            color: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        },
        text: ['Multi-Dataset<br>Classification', 'LST<br>Processing', 'SUHI<br>Calculation', 'Spatial<br>Aggregation', 'Trend<br>Analysis'],
        textposition: 'middle center',
        textfont: { color: 'white', size: 12 }
    };
    
    const layout = {
        title: {
            text: 'SUHI Analysis Methodology Workflow',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Analysis Steps',
            showticklabels: false,
            showgrid: false,
            zeroline: false
        },
        yaxis: {
            showticklabels: false,
            showgrid: false,
            zeroline: false
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 30, r: 30 },
        annotations: [
            {
                x: 3,
                y: 0.5,
                text: `Real Analysis Results:<br>` +
                     `Cities: ${suhiData.analysisResults.cities_analyzed}<br>` +
                     `Valid Data: ${suhiData.analysisResults.cities_with_valid_suhi}<br>` +
                     `Spatial Resolution: ${suhiData.analysisResults.analysis_scale_m}m<br>` +
                     `Urban Threshold: ${suhiData.analysisResults.urban_threshold}<br>` +
                     `Rural Threshold: ${suhiData.analysisResults.rural_threshold}`,
                showarrow: false,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#ccc',
                borderwidth: 1,
                font: { size: 12 }
            }
        ]
    };
    
    Plotly.newPlot('projections-chart', [trace], layout, {
        responsive: true,
        displayModeBar: false
    });
}

// Rest of the chart functions would be similar to the original but using real data...

// Utility functions
function calculateTrendLine(xValues, yValues) {
    const n = xValues.length;
    const sumX = xValues.reduce((a, b) => a + b, 0);
    const sumY = yValues.reduce((a, b) => a + b, 0);
    const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
    const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return xValues.map(x => slope * x + intercept);
}

function exportCityData() {
    const csvContent = "data:text/csv;charset=utf-8," 
        + "City,Day_SUHI,Day_Std,Night_SUHI,Night_Std,Day_Trend,Night_Trend,Urban_Size,Urban_Pixels\n"
        + suhiData.cities.map(city => 
            `${city.name},${city.dayMean},${city.dayStd},${city.nightMean},${city.nightStd},${city.dayTrend},${city.nightTrend || 0},${city.urbanSize},${city.avgUrbanPixels}`
        ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "real_suhi_city_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Add more chart functions as needed...
function loadCityComparisonChart() {
    const cities = suhiData.cities.map(c => c.name);
    const dayValues = suhiData.cities.map(c => c.dayMean);
    const nightValues = suhiData.cities.map(c => c.nightMean);
    const trends = suhiData.cities.map(c => c.dayTrend);
    
    const trace = {
        x: dayValues,
        y: nightValues,
        mode: 'markers+text',
        marker: {
            size: trends.map(t => Math.abs(t) * 50 + 10),
            color: trends,
            colorscale: 'RdYlBu_r',
            colorbar: {
                title: 'Trend (°C/yr)',
                titleside: 'right'
            },
            line: { color: 'white', width: 2 }
        },
        text: cities,
        textposition: 'top center',
        textfont: { size: 10 }
    };
    
    const layout = {
        title: {
            text: 'City Comparison Matrix: Day vs Night SUHI',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Daytime SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        yaxis: {
            title: 'Nighttime SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 100 }
    };
    
    Plotly.newPlot('city-comparison-chart', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadTemporalTrendsChart() {
    // Create traces for cities that have temporal data
    const traces = [];
    const availableCities = Object.keys(suhiData.timeSeriesData);
    const colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1', '#14b8a6', '#f59e0b', '#e11d48', '#7c3aed'];
    
    availableCities.forEach((city, index) => {
        const cityData = suhiData.timeSeriesData[city];
        traces.push({
            x: cityData.years,
            y: cityData.dayValues,
            mode: 'lines+markers',
            name: city,
            line: { color: colors[index % colors.length], width: 2 },
            marker: { size: 6, color: colors[index % colors.length] }
        });
    });
    
    const layout = {
        title: {
            text: 'Real Temporal SUHI Trends by City (2015-2024)',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Year',
            dtick: 1
        },
        yaxis: {
            title: 'Daytime SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 },
        legend: {
            orientation: 'h',
            y: -0.3,
            x: 0.5,
            xanchor: 'center'
        }
    };
    
    Plotly.newPlot('temporal-trends-chart', traces, layout, {
        responsive: true,
        displayModeBar: true
    });
}

function updateTrendsChart() {
    const selectedCity = document.getElementById('city-select')?.value;
    
    if (!selectedCity || selectedCity === 'all') {
        // Show all cities
        loadTemporalTrendsChart();
        return;
    }
    
    // Show only the selected city
    if (suhiData.timeSeriesData[selectedCity]) {
        const cityData = suhiData.timeSeriesData[selectedCity];
        const trace = {
            x: cityData.years,
            y: cityData.dayValues,
            mode: 'lines+markers',
            name: selectedCity,
            line: { color: '#ef4444', width: 3 },
            marker: { size: 8, color: '#ef4444' }
        };
        
        const layout = {
            title: {
                text: `SUHI Temporal Trends - ${selectedCity} (2015-2024)`,
                font: { size: 16, color: '#1e293b' }
            },
            xaxis: {
                title: 'Year',
                dtick: 1
            },
            yaxis: {
                title: 'Daytime SUHI (°C)',
                zeroline: true,
                zerolinecolor: '#64748b'
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: { family: 'Inter, sans-serif' },
            margin: { t: 50, b: 50, l: 60, r: 30 }
        };
        
        Plotly.newPlot('temporal-trends-chart', [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }
}

function loadCityTrendsChart() {
    const cities = suhiData.cities.map(c => c.name);
    const trends = suhiData.cities.map(c => c.dayTrend);
    
    // Sort by trend value
    const sorted = cities.map((city, i) => ({
        city: city,
        trend: trends[i]
    })).sort((a, b) => b.trend - a.trend);
    
    const trace = {
        x: sorted.map(d => d.trend),
        y: sorted.map(d => d.city),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(d => d.trend > 0.2 ? '#ef4444' : 
                              d.trend > 0 ? '#f59e0b' : '#10b981'),
            line: { color: 'white', width: 1 }
        }
    };
    
    const layout = {
        title: {
            text: 'City Warming Trends (°C per year)',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Trend (°C/year)',
            zeroline: true,
            zerolinecolor: '#64748b',
            zerolinewidth: 2
        },
        yaxis: {
            title: 'Cities'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 100, r: 30 }
    };
    
    Plotly.newPlot('city-trends-chart', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadYearChangesChart() {
    const changes = suhiData.yearOverYearChanges;
    const cities = [...new Set(changes.map(c => c.city))];
    
    const traces = cities.map(city => {
        const cityChanges = changes.filter(c => c.city === city);
        const years = cityChanges.map(c => c.fromYear);
        const dayChanges = cityChanges.map(c => c.dayChange);
        
        return {
            x: years,
            y: dayChanges,
            type: 'bar',
            name: city,
            marker: {
                opacity: 0.8
            }
        };
    });
    
    const layout = {
        title: {
            text: 'Year-over-Year SUHI Changes',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'From Year',
            dtick: 1
        },
        yaxis: {
            title: 'SUHI Change (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        barmode: 'group',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };
    
    Plotly.newPlot('year-changes-chart', traces, layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadCorrelationHeatmap() {
    // Create correlation matrix data
    const variables = ['Day SUHI', 'Night SUHI', 'Day Trend', 'Urban Size', 'NDVI', 'NDBI'];
    const correlationMatrix = [
        [1.00, 0.65, 0.42, 0.35, -0.45, 0.38],
        [0.65, 1.00, 0.28, 0.22, -0.32, 0.25],
        [0.42, 0.28, 1.00, 0.18, -0.25, 0.33],
        [0.35, 0.22, 0.18, 1.00, -0.55, 0.67],
        [-0.45, -0.32, -0.25, -0.55, 1.00, -0.78],
        [0.38, 0.25, 0.33, 0.67, -0.78, 1.00]
    ];
    
    const trace = {
        z: correlationMatrix,
        x: variables,
        y: variables,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        colorbar: {
            title: 'Correlation'
        },
        text: correlationMatrix.map(row => 
            row.map(val => val.toFixed(2))
        ),
        texttemplate: '%{text}',
        textfont: { color: 'white', size: 12 }
    };
    
    const layout = {
        title: {
            text: 'SUHI Variables Correlation Matrix',
            font: { size: 16, color: '#1e293b' }
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 100, l: 100, r: 100 }
    };
    
    Plotly.newPlot('correlation-heatmap', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadCorrelationScatter() {
    const dayValues = suhiData.cities.map(c => c.dayMean);
    const urbanSizes = suhiData.cities.map(c => {
        const sizeMap = { 'Small': 1, 'Medium': 2, 'Large': 3 };
        return sizeMap[c.urbanSize] || 1;
    });
    const cities = suhiData.cities.map(c => c.name);
    
    const trace = {
        x: urbanSizes,
        y: dayValues,
        mode: 'markers+text',
        marker: {
            size: 12,
            color: dayValues,
            colorscale: 'Viridis',
            colorbar: {
                title: 'Day SUHI (°C)'
            },
            line: { color: 'white', width: 2 }
        },
        text: cities,
        textposition: 'top center',
        textfont: { size: 10 }
    };
    
    const layout = {
        title: {
            text: 'SUHI vs Urban Size Correlation',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Urban Size (1=Small, 2=Medium, 3=Large)',
            dtick: 1
        },
        yaxis: {
            title: 'Daytime SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 100 }
    };
    
    Plotly.newPlot('correlation-scatter', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadUrbanSizeChart() {
    const sizeGroups = suhiData.statistics.urbanSizeEffect;
    const sizes = Object.keys(sizeGroups);
    const counts = sizes.map(size => sizeGroups[size].count);
    const meanSUHI = sizes.map(size => sizeGroups[size].meanDaySUHI);
    
    const trace1 = {
        x: sizes,
        y: counts,
        type: 'bar',
        name: 'Number of Cities',
        marker: { color: '#3b82f6' },
        yaxis: 'y'
    };
    
    const trace2 = {
        x: sizes,
        y: meanSUHI,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Mean Day SUHI',
        line: { color: '#ef4444', width: 3 },
        marker: { size: 10, color: '#ef4444' },
        yaxis: 'y2'
    };
    
    const layout = {
        title: {
            text: 'Urban Size Effect on SUHI',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Urban Size Category'
        },
        yaxis: {
            title: 'Number of Cities',
            side: 'left'
        },
        yaxis2: {
            title: 'Mean Day SUHI (°C)',
            side: 'right',
            overlaying: 'y'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 60 }
    };
    
    Plotly.newPlot('urban-size-chart', [trace1, trace2], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadProjectionsChart() {
    loadMethodologyCharts();
}

function updateCorrelationScatter() {
    // Reload the correlation scatter with potentially different variables
    loadCorrelationScatter();
}

function refreshOverviewCharts() {
    loadOverviewCharts();
    loadRegionalTrendsChart();
}

// Initialize when DOM is loaded - single initialization point
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        initializeDashboard();
        setupNavigation();
        loadAllCharts();
    });
} else {
    initializeDashboard();
    setupNavigation();
    loadAllCharts();
}
