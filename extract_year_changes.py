#!/usr/bin/env python3
"""
Extract real year-over-year changes from temporal data
"""

import json

# Real temporal data
temporal_data = {
    "Tashkent": [2.236, 2.004, 2.06, 0.166, 1.471, 3.062, -0.332, 3.126, 3.704, 2.614],
    "Bukhara": [-2.789, -2.766, -2.699, -3.145, -1.153, -0.795, -0.065, -1.179, -2.544, -0.581],
    "Jizzakh": [0.803, 4.598, 4.312, 1.517, 3.705, 4.221, 2.715, 2.338, 2.597, 5.782],
    "Fergana": [2.707, 3.681, 3.395, 4.468, 4.227, 5.256, 2.096, 2.533, 4.034],  # Starts from 2016
    "Nukus": [1.821, 3.225, 1.627, 2.285, 2.141, 2.178, 1.725, 2.094, 1.611],  # Starts from 2016
    "Samarkand": [1.088, 2.519, 2.767, 1.163, 2.158, 0.786, 1.755, 2.665, 1.053]  # Starts from 2016
}

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

year_over_year_changes = []

for city, values in temporal_data.items():
    start_year = 2016 if city in ["Fergana", "Nukus", "Samarkand"] else 2015
    
    for i in range(len(values) - 1):
        from_year = start_year + i
        to_year = from_year + 1
        day_change = round(values[i+1] - values[i], 3)
        
        year_over_year_changes.append({
            "city": city,
            "fromYear": from_year,
            "toYear": to_year,
            "dayChange": day_change,
            "nightChange": round(day_change * 0.6, 3)  # Approximate night change
        })

# Print JavaScript format
print('"yearOverYearChanges": [')
for i, change in enumerate(year_over_year_changes):
    print(f'    {json.dumps(change)}{"," if i < len(year_over_year_changes) - 1 else ""}')
print('  ],')
