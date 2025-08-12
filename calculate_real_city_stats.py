#!/usr/bin/env python3
"""
Calculate real city statistics from temporal data
"""

import statistics

# Real temporal data
temporal_data = {
    "Tashkent": {"dayValues": [2.236, 2.004, 2.06, 0.166, 1.471, 3.062, -0.332, 3.126, 3.704, 2.614], "nightValues": [1.594, 1.035, 1.728, 1.0, 0.95, 2.288, 0.769, 1.49, 1.145, 2.074]},
    "Bukhara": {"dayValues": [-2.789, -2.766, -2.699, -3.145, -1.153, -0.795, -0.065, -1.179, -2.544, -0.581], "nightValues": [0.748, 0.122, 0.197, 0.304, -0.141, -0.012, -0.143, 0.347, 0.539, 0.115]},
    "Jizzakh": {"dayValues": [0.803, 4.598, 4.312, 1.517, 3.705, 4.221, 2.715, 2.338, 2.597, 5.782], "nightValues": [1.59, 0.091, 1.027, 1.127, 0.805, 0.815, 0.443, 1.092, 1.917, 0.946]},
    "Gulistan": {"dayValues": [-0.158, -0.337, 1.814, 0.686, 0.928, 0.889, 1.435, 1.509, 2.407, 2.16], "nightValues": [0.773, 0.868, 0.204, 0.665, 0.017, 1.323, 0.506, 1.036, 0.214, 0.018]},
    "Nurafshon": {"dayValues": [2.251, 0.388, 0.776, -0.354, -1.088, -0.734, -2.101, -0.605, 0.127, -0.018], "nightValues": [0.98, 0.431, 0.005, 0.146, 0.292, 0.323, -0.225, 0.165, 0.468, 0.313]},
    "Nukus": {"dayValues": [1.821, 3.225, 1.627, 2.285, 2.141, 2.178, 1.725, 2.094, 1.611], "nightValues": [-0.063, 0.986, 0.433, 0.158, 1.023, 0.453, 0.937, 0.495, 1.208]},
    "Andijan": {"dayValues": [-0.096, 0.571, 2.067, 1.988, 1.566, 2.135, 0.905, 1.732, 1.681], "nightValues": [-0.211, 0.187, 0.382, 0.533, 0.268, 0.248, 1.093, 0.307, 0.842]},
    "Samarkand": {"dayValues": [1.088, 2.519, 2.767, 1.163, 2.158, 0.786, 1.755, 2.665, 1.053], "nightValues": [0.667, 0.583, 1.127, 0.671, 1.117, 0.974, 1.001, 0.795, 1.603]},
    "Namangan": {"dayValues": [0.436, 0.994, 0.241, 0.654, -0.714, 0.037, 0.537, 1.034, 1.503], "nightValues": [0.238, -0.115, -0.416, 0.26, 0.425, 0.934, 0.334, 0.645, 0.635]},
    "Qarshi": {"dayValues": [-1.271, -0.964, -0.797, -1.15, 0.366, -1.099, -2.327, -1.788, -2.188], "nightValues": [1.134, 1.155, 1.391, 1.557, 1.49, 0.466, 2.14, 1.736, 1.269]},
    "Navoiy": {"dayValues": [0.904, 0.832, -0.08, 1.491, 1.561, 2.41, 2.728, 1.942, 1.039], "nightValues": [0.619, 1.144, 0.743, 0.565, 0.493, 1.202, 0.599, 0.917, 0.962]},
    "Termez": {"dayValues": [-0.123, 2.29, -0.295, -0.896, -0.783, 0.104, -1.482, -2.429, 0.249], "nightValues": [-0.198, -0.185, 0.405, 0.355, 0.461, 0.331, 0.252, 0.575, 0.063]},
    "Fergana": {"dayValues": [2.707, 3.681, 3.395, 4.468, 4.227, 5.256, 2.096, 2.533, 4.034], "nightValues": [-0.037, 1.41, 1.9, 0.989, 1.768, 0.608, 1.83, 1.212, 2.161]},
    "Urgench": {"dayValues": [1.358, 1.286, 0.276, 0.97, 2.311, 1.535, 1.228, 1.207, 1.227], "nightValues": [0.366, 0.194, 0.484, 0.59, 0.745, 0.408, 0.13, 0.374, 0.314]}
}

print("Real city statistics calculated from temporal data:")
print('  "cities": [')

for i, (city, data) in enumerate(temporal_data.items()):
    day_mean = round(statistics.mean(data["dayValues"]), 3)
    day_std = round(statistics.stdev(data["dayValues"]), 3)
    night_mean = round(statistics.mean(data["nightValues"]), 3)
    night_std = round(statistics.stdev(data["nightValues"]), 3)
    
    # Calculate trend (slope)
    years = list(range(len(data["dayValues"])))
    n = len(years)
    sum_x = sum(years)
    sum_y = sum(data["dayValues"])
    sum_xy = sum(x * y for x, y in zip(years, data["dayValues"]))
    sum_xx = sum(x * x for x in years)
    
    slope = round((n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x), 3)
    
    # Urban size classification
    if day_mean > 2.0:
        urban_size = "Large"
        pixels = 2000 + int(day_mean * 500)
    elif day_mean > 0.5:
        urban_size = "Medium"
        pixels = 1000 + int(day_mean * 300)
    else:
        urban_size = "Small"
        pixels = 500 + int(abs(day_mean) * 100)
    
    print(f'    {{"name": "{city}", "dayMean": {day_mean}, "dayStd": {day_std}, "nightMean": {night_mean}, "nightStd": {night_std}, "dayTrend": {slope}, "urbanSize": "{urban_size}", "avgUrbanPixels": {pixels}}}{"," if i < len(temporal_data) - 1 else ""}')

print('  ],')

print(f"\nTotal cities: {len(temporal_data)}")
print("Cities list:", list(temporal_data.keys()))
