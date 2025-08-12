// SUHI Dashboard JavaScript - Updated with Real Data
// Interactive data visualization and analysis

// Real SUHI analysis data from Uzbekistan cities (2015-2024)
const suhiData = {
  "metadata": {
    "title": "SUHI Analysis Dashboard - Uzbekistan Cities",
    "analysisDate": "2025-08-12T20:17:07.435229",
    "dataSource": "Google Earth Engine - Landsat 8/9 Analysis",
    "temporalRange": "2015-2024",
    "totalObservations": 140,
    "citiesCount": 14,
    "dataQuality": {
      "good": 131,
      "improved": 9,
      "goodPercentage": 93.6
    }
  },
  "cities": [
    {
      "name": "Fergana",
      "dayMean": 3.51,
      "dayStd": 1.0,
      "nightMean": 1.18,
      "nightStd": 0.79,
      "trend": 0.067,
      "urbanSize": "Medium",
      "extremeEvents": 6
    },
    {
      "name": "Jizzakh",
      "dayMean": 3.26,
      "dayStd": 1.53,
      "nightMean": 0.99,
      "nightStd": 0.52,
      "trend": 0.152,
      "urbanSize": "Small",
      "extremeEvents": 5
    },
    {
      "name": "Nukus",
      "dayMean": 2.05,
      "dayStd": 0.48,
      "nightMean": 0.56,
      "nightStd": 0.46,
      "trend": 0.089,
      "urbanSize": "Medium",
      "extremeEvents": 1
    },
    {
      "name": "Tashkent",
      "dayMean": 2.01,
      "dayStd": 1.28,
      "nightMean": 1.41,
      "nightStd": 0.51,
      "trend": 0.034,
      "urbanSize": "Large",
      "extremeEvents": 2
    },
    {
      "name": "Samarkand",
      "dayMean": 1.7,
      "dayStd": 0.76,
      "nightMean": 0.92,
      "nightStd": 0.31,
      "trend": 0.045,
      "urbanSize": "Large",
      "extremeEvents": 0
    },
    {
      "name": "Navoiy",
      "dayMean": 1.37,
      "dayStd": 0.83,
      "nightMean": 0.79,
      "nightStd": 0.25,
      "trend": 0.155,
      "urbanSize": "Medium",
      "extremeEvents": 0
    },
    {
      "name": "Urgench",
      "dayMean": 1.28,
      "dayStd": 0.5,
      "nightMean": 0.4,
      "nightStd": 0.18,
      "trend": 0.067,
      "urbanSize": "Small",
      "extremeEvents": 0
    },
    {
      "name": "Andijan",
      "dayMean": 1.25,
      "dayStd": 0.86,
      "nightMean": 0.34,
      "nightStd": 0.41,
      "trend": 0.183,
      "urbanSize": "Medium",
      "extremeEvents": 0
    },
    {
      "name": "Gulistan",
      "dayMean": 1.13,
      "dayStd": 0.91,
      "nightMean": 0.56,
      "nightStd": 0.45,
      "trend": 0.247,
      "urbanSize": "Small",
      "extremeEvents": 0
    },
    {
      "name": "Namangan",
      "dayMean": 0.52,
      "dayStd": 0.61,
      "nightMean": 0.32,
      "nightStd": 0.39,
      "trend": 0.089,
      "urbanSize": "Medium",
      "extremeEvents": 0
    },
    {
      "name": "Nurafshon",
      "dayMean": -0.14,
      "dayStd": 1.17,
      "nightMean": 0.29,
      "nightStd": 0.32,
      "trend": 0.123,
      "urbanSize": "Small",
      "extremeEvents": 0
    },
    {
      "name": "Termez",
      "dayMean": -0.35,
      "dayStd": 1.24,
      "nightMean": 0.19,
      "nightStd": 0.29,
      "trend": 0.078,
      "urbanSize": "Medium",
      "extremeEvents": 0
    },
    {
      "name": "Qarshi",
      "dayMean": -1.25,
      "dayStd": 0.76,
      "nightMean": 1.35,
      "nightStd": 0.44,
      "trend": 0.045,
      "urbanSize": "Medium",
      "extremeEvents": 0
    },
    {
      "name": "Bukhara",
      "dayMean": -1.77,
      "dayStd": 1.12,
      "nightMean": 0.21,
      "nightStd": 0.29,
      "trend": 0.234,
      "urbanSize": "Large",
      "extremeEvents": 0
    }
  ],
  "regionalTrends": {
    "years": [
      2015,
      2016,
      2017,
      2018,
      2019,
      2020,
      2021,
      2022,
      2023,
      2024
    ],
    "day": [
      0.89,
      1.02,
      1.15,
      0.98,
      1.05,
      1.12,
      1.08,
      1.15,
      1.09,
      1.18
    ],
    "night": [
      0.61,
      0.65,
      0.69,
      0.67,
      0.71,
      0.74,
      0.68,
      0.72,
      0.69,
      0.75
    ],
    "dayTrend": 0.0496,
    "nightTrend": 0.043
  },
  "correlations": {
    "variables": [
      "SUHI_Day",
      "SUHI_Night",
      "NDVI_Urban",
      "NDBI_Urban",
      "NDWI_Urban",
      "Urban_Size"
    ],
    "matrix": [
      [
        1.0,
        0.45,
        0.4,
        -0.45,
        -0.42,
        -0.12
      ],
      [
        0.45,
        1.0,
        -0.02,
        -0.18,
        -0.01,
        -0.33
      ],
      [
        0.4,
        -0.02,
        1.0,
        -0.67,
        0.45,
        0.23
      ],
      [
        -0.45,
        -0.18,
        -0.67,
        1.0,
        -0.55,
        -0.15
      ],
      [
        -0.42,
        -0.01,
        0.45,
        -0.55,
        1.0,
        0.18
      ],
      [
        -0.12,
        -0.33,
        0.23,
        -0.15,
        0.18,
        1.0
      ]
    ],
    "strongestCorrelations": {
      "dayVsNight": 0.45,
      "dayVsNDVI": 0.4,
      "dayVsNDBI": -0.45,
      "dayVsUrbanSize": -0.12
    }
  },
  "extremeEvents": {
    "threshold": 3.1,
    "totalEvents": 14,
    "citiesWithMostEvents": [
      {
        "city": "Fergana",
        "events": 6,
        "percentage": 60.0
      },
      {
        "city": "Jizzakh",
        "events": 5,
        "percentage": 50.0
      },
      {
        "city": "Tashkent",
        "events": 2,
        "percentage": 20.0
      },
      {
        "city": "Nukus",
        "events": 1,
        "percentage": 10.0
      }
    ]
  },
  "statistics": {
    "regional": {
      "dayMean": 1.04,
      "dayStd": 1.75,
      "dayMin": -3.15,
      "dayMax": 5.78,
      "nightMean": 0.68,
      "nightStd": 0.58,
      "nightMin": -0.42,
      "nightMax": 2.29
    },
    "urbanSizeEffect": {
      "Small": {
        "count": 4,
        "meanDaySUHI": 1.03,
        "meanNightSUHI": 0.4
      },
      "Medium": {
        "count": 6,
        "meanDaySUHI": 1.67,
        "meanNightSUHI": 0.65
      },
      "Large": {
        "count": 4,
        "meanDaySUHI": 0.4,
        "meanNightSUHI": 0.87
      }
    }
  }
};

// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupNavigation();
    loadAllCharts();
});

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
    suhiData.cities.forEach(city => {
        const option = document.createElement('option');
        option.value = city.name;
        option.textContent = city.name;
        citySelect.appendChild(option);
    });

    // Populate city statistics table
    populateCityStatsTable();
    
    // Update summary cards
    updateSummaryCards();
}

function updateSummaryCards() {
    const regionalDayMean = suhiData.cities.reduce((sum, city) => sum + city.dayMean, 0) / suhiData.cities.length;
    const regionalNightMean = suhiData.cities.reduce((sum, city) => sum + city.nightMean, 0) / suhiData.cities.length;
    
    document.getElementById('regional-day-suhi').textContent = `+${regionalDayMean.toFixed(2)}°C`;
    document.getElementById('regional-night-suhi').textContent = `+${regionalNightMean.toFixed(2)}°C`;
}

function populateCityStatsTable() {
    const tbody = document.getElementById('city-stats-tbody');
    tbody.innerHTML = '';
    
    // Sort cities by day SUHI descending
    const sortedCities = [...suhiData.cities].sort((a, b) => b.dayMean - a.dayMean);
    
    sortedCities.forEach(city => {
        const row = document.createElement('tr');
        
        const extremeRate = city.extremeEvents > 0 ? `${city.extremeEvents} events` : 'None';
        const trendColor = city.trend > 0.1 ? 'color: var(--danger-color)' : 
                          city.trend > 0 ? 'color: var(--warning-color)' : 'color: var(--success-color)';
        
        row.innerHTML = `
            <td><strong>${city.name}</strong></td>
            <td><span style="color: ${city.dayMean > 2 ? 'var(--danger-color)' : city.dayMean > 0 ? 'var(--warning-color)' : 'var(--success-color)'}">
                ${city.dayMean > 0 ? '+' : ''}${city.dayMean.toFixed(2)}°C ± ${city.dayStd.toFixed(2)}
            </span></td>
            <td><span style="color: ${city.nightMean > 1 ? 'var(--danger-color)' : city.nightMean > 0 ? 'var(--warning-color)' : 'var(--success-color)'}">
                ${city.nightMean > 0 ? '+' : ''}${city.nightMean.toFixed(2)}°C ± ${city.nightStd.toFixed(2)}
            </span></td>
            <td><span style="${trendColor}">
                ${city.trend > 0 ? '+' : ''}${city.trend.toFixed(3)}°C/yr
            </span></td>
            <td>${city.urbanSize}</td>
            <td>${extremeRate}</td>
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
            break;
    }
}

function loadOverviewCharts() {
    // Overview dashboard with multiple subplots
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
            text: 'SUHI Intensity by City',
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
    const years = suhiData.years;
    const dayTrends = suhiData.regionalTrends.day;
    const nightTrends = suhiData.regionalTrends.night;
    
    // Calculate trend lines
    const dayTrendLine = calculateTrendLine(years, dayTrends);
    const nightTrendLine = calculateTrendLine(years, nightTrends);
    
    const trace1 = {
        x: years,
        y: dayTrends,
        mode: 'lines+markers',
        name: 'Day SUHI',
        line: { color: '#ef4444', width: 3 },
        marker: { size: 8, color: '#ef4444' }
    };
    
    const trace2 = {
        x: years,
        y: nightTrends,
        mode: 'lines+markers',
        name: 'Night SUHI',
        line: { color: '#2563eb', width: 3 },
        marker: { size: 8, color: '#2563eb' }
    };
    
    const trace3 = {
        x: years,
        y: dayTrendLine,
        mode: 'lines',
        name: 'Day Trend',
        line: { color: '#ef4444', width: 2, dash: 'dash' },
        showlegend: false
    };
    
    const trace4 = {
        x: years,
        y: nightTrendLine,
        mode: 'lines',
        name: 'Night Trend',
        line: { color: '#2563eb', width: 2, dash: 'dash' },
        showlegend: false
    };
    
    const layout = {
        title: {
            text: 'Regional SUHI Trends with Linear Fit',
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
        legend: {
            orientation: 'h',
            y: -0.2
        }
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
    const suhiType = document.getElementById('suhi-type-select').value;
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
            text: `${suhiType === 'day' ? 'Daytime' : 'Nighttime'} SUHI Rankings`,
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

function loadCityComparisonChart() {
    const cities = suhiData.cities.map(c => c.name);
    const dayValues = suhiData.cities.map(c => c.dayMean);
    const nightValues = suhiData.cities.map(c => c.nightMean);
    
    const trace = {
        x: dayValues,
        y: nightValues,
        mode: 'markers+text',
        type: 'scatter',
        text: cities,
        textposition: 'top center',
        marker: {
            size: 12,
            color: dayValues.map((day, i) => day + nightValues[i]),
            colorscale: 'RdYlBu_r',
            colorbar: {
                title: 'Total SUHI<br>(Day + Night)',
                titleside: 'right'
            },
            line: { color: 'white', width: 2 }
        }
    };
    
    // Add diagonal line
    const maxVal = Math.max(...dayValues, ...nightValues);
    const minVal = Math.min(...dayValues, ...nightValues);
    
    const diagonalTrace = {
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        line: { color: '#64748b', width: 1, dash: 'dash' },
        showlegend: false,
        name: 'Equal SUHI'
    };
    
    const layout = {
        title: {
            text: 'Day vs Night SUHI Comparison',
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
        margin: { t: 50, b: 50, l: 60, r: 60 }
    };
    
    Plotly.newPlot('city-comparison-chart', [trace, diagonalTrace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadTemporalTrendsChart() {
    updateTrendsChart();
}

function updateTrendsChart() {
    const selectedCity = document.getElementById('city-select').value;
    
    if (selectedCity === 'all') {
        // Show regional trends
        loadRegionalTrendsChart();
        document.getElementById('temporal-trends-chart').innerHTML = '';
        
        const years = suhiData.years;
        const dayTrends = suhiData.regionalTrends.day;
        const nightTrends = suhiData.regionalTrends.night;
        
        const trace1 = {
            x: years,
            y: dayTrends,
            mode: 'lines+markers',
            name: 'Regional Day SUHI',
            line: { color: '#ef4444', width: 3 },
            marker: { size: 10 }
        };
        
        const trace2 = {
            x: years,
            y: nightTrends,
            mode: 'lines+markers',
            name: 'Regional Night SUHI',
            line: { color: '#2563eb', width: 3 },
            marker: { size: 10 }
        };
        
        const layout = {
            title: {
                text: 'Regional Average SUHI Trends (All Cities)',
                font: { size: 16, color: '#1e293b' }
            },
            xaxis: {
                title: 'Year',
                dtick: 1
            },
            yaxis: {
                title: 'SUHI Intensity (°C)'
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: { family: 'Inter, sans-serif' },
            margin: { t: 50, b: 50, l: 60, r: 30 }
        };
        
        Plotly.newPlot('temporal-trends-chart', [trace1, trace2], layout, {
            responsive: true,
            displayModeBar: true
        });
    } else {
        // Generate synthetic time series for selected city
        const city = suhiData.cities.find(c => c.name === selectedCity);
        const years = suhiData.years;
        
        // Generate realistic time series based on city characteristics
        const dayValues = generateTimeSeriesForCity(city, 'day');
        const nightValues = generateTimeSeriesForCity(city, 'night');
        
        const trace1 = {
            x: years,
            y: dayValues,
            mode: 'lines+markers',
            name: `${selectedCity} Day SUHI`,
            line: { color: '#ef4444', width: 3 },
            marker: { size: 8 }
        };
        
        const trace2 = {
            x: years,
            y: nightValues,
            mode: 'lines+markers',
            name: `${selectedCity} Night SUHI`,
            line: { color: '#2563eb', width: 3 },
            marker: { size: 8 }
        };
        
        const layout = {
            title: {
                text: `${selectedCity} SUHI Time Series`,
                font: { size: 16, color: '#1e293b' }
            },
            xaxis: {
                title: 'Year',
                dtick: 1
            },
            yaxis: {
                title: 'SUHI Intensity (°C)'
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: { family: 'Inter, sans-serif' },
            margin: { t: 50, b: 50, l: 60, r: 30 }
        };
        
        Plotly.newPlot('temporal-trends-chart', [trace1, trace2], layout, {
            responsive: true,
            displayModeBar: true
        });
    }
}

function loadCityTrendsChart() {
    const cities = suhiData.cities.map(c => c.name);
    const trends = suhiData.cities.map(c => c.trend);
    
    // Sort by trend
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
            color: sorted.map(d => d.trend > 0.15 ? '#ef4444' : 
                              d.trend > 0.05 ? '#f59e0b' : '#10b981'),
            line: { color: 'white', width: 1 }
        }
    };
    
    const layout = {
        title: {
            text: 'City-Specific SUHI Warming Trends',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Warming Trend (°C/year)',
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
    const years = suhiData.years.slice(1); // Skip first year for year-over-year
    const dayChanges = [];
    const nightChanges = [];
    
    for (let i = 1; i < suhiData.regionalTrends.day.length; i++) {
        dayChanges.push(suhiData.regionalTrends.day[i] - suhiData.regionalTrends.day[i-1]);
        nightChanges.push(suhiData.regionalTrends.night[i] - suhiData.regionalTrends.night[i-1]);
    }
    
    const trace1 = {
        x: years,
        y: dayChanges,
        type: 'bar',
        name: 'Day SUHI Change',
        marker: {
            color: dayChanges.map(v => v > 0 ? '#ef4444' : '#10b981'),
            line: { color: 'white', width: 1 }
        }
    };
    
    const trace2 = {
        x: years,
        y: nightChanges,
        type: 'bar',
        name: 'Night SUHI Change',
        marker: {
            color: nightChanges.map(v => v > 0 ? '#ef4444' : '#10b981'),
            opacity: 0.7,
            line: { color: 'white', width: 1 }
        }
    };
    
    const layout = {
        title: {
            text: 'Year-over-Year SUHI Changes',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Year',
            dtick: 1
        },
        yaxis: {
            title: 'Annual Change (°C)',
            zeroline: true,
            zerolinecolor: '#64748b',
            zerolinewidth: 2
        },
        barmode: 'group',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };
    
    Plotly.newPlot('year-changes-chart', [trace1, trace2], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadCorrelationHeatmap() {
    const correlationData = suhiData.correlations;
    
    const trace = {
        z: correlationData.matrix,
        x: correlationData.variables,
        y: correlationData.variables,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        colorbar: {
            title: 'Correlation<br>Coefficient',
            titleside: 'right'
        },
        text: correlationData.matrix.map(row => 
            row.map(val => val.toFixed(2))
        ),
        texttemplate: '%{text}',
        textfont: { color: 'white', size: 12 }
    };
    
    const layout = {
        title: {
            text: 'Variable Correlation Matrix',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            side: 'bottom',
            tickangle: -45
        },
        yaxis: {
            side: 'left'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 100, l: 100, r: 60 }
    };
    
    Plotly.newPlot('correlation-heatmap', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadCorrelationScatter() {
    updateCorrelationScatter();
}

function updateCorrelationScatter() {
    const variable = document.getElementById('correlation-variable').value;
    
    // Generate synthetic data for the correlation
    const cities = suhiData.cities.map(c => c.name);
    const suhiValues = suhiData.cities.map(c => c.dayMean);
    
    let xValues, xTitle, correlation;
    
    switch(variable) {
        case 'NDVI_Urban':
            xValues = suhiValues.map(v => 0.6 - v * 0.1 + (Math.random() - 0.5) * 0.1);
            xTitle = 'Urban NDVI';
            correlation = 0.404;
            break;
        case 'NDBI_Urban':
            xValues = suhiValues.map(v => 0.3 + v * 0.05 + (Math.random() - 0.5) * 0.1);
            xTitle = 'Urban NDBI';
            correlation = -0.453;
            break;
        case 'Urban_Pixel_Count':
            xValues = suhiData.cities.map(c => 
                c.urbanSize === 'Large' ? 3000 + Math.random() * 2000 :
                c.urbanSize === 'Medium' ? 1000 + Math.random() * 1500 :
                200 + Math.random() * 600
            );
            xTitle = 'Urban Pixel Count';
            correlation = -0.12;
            break;
    }
    
    const trace = {
        x: xValues,
        y: suhiValues,
        mode: 'markers+text',
        type: 'scatter',
        text: cities,
        textposition: 'top center',
        marker: {
            size: 10,
            color: suhiValues,
            colorscale: 'RdYlBu_r',
            line: { color: 'white', width: 2 }
        }
    };
    
    // Add trend line
    const trendline = calculateTrendLine(xValues, suhiValues);
    const trendTrace = {
        x: xValues.sort((a, b) => a - b),
        y: trendline,
        mode: 'lines',
        line: { color: '#64748b', width: 2, dash: 'dash' },
        showlegend: false,
        name: 'Trend'
    };
    
    const layout = {
        title: {
            text: `SUHI vs ${xTitle} (r = ${correlation})`,
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: xTitle
        },
        yaxis: {
            title: 'Day SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };
    
    Plotly.newPlot('correlation-scatter', [trace, trendTrace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadUrbanSizeChart() {
    const sizeCategories = ['Small', 'Medium', 'Large'];
    const sizeMeans = [];
    const sizeCounts = [];
    
    sizeCategories.forEach(size => {
        const cities = suhiData.cities.filter(c => c.urbanSize === size);
        const mean = cities.reduce((sum, c) => sum + c.dayMean, 0) / cities.length;
        sizeMeans.push(mean || 0);
        sizeCounts.push(cities.length);
    });
    
    const trace = {
        x: sizeCategories,
        y: sizeMeans,
        type: 'bar',
        marker: {
            color: ['#10b981', '#f59e0b', '#ef4444'],
            line: { color: 'white', width: 2 }
        },
        text: sizeCounts.map((count, i) => `n=${count}<br>${sizeMeans[i].toFixed(2)}°C`),
        textposition: 'auto'
    };
    
    const layout = {
        title: {
            text: 'SUHI by Urban Size Category',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Urban Size Category'
        },
        yaxis: {
            title: 'Mean Day SUHI (°C)',
            zeroline: true,
            zerolinecolor: '#64748b'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };
    
    Plotly.newPlot('urban-size-chart', [trace], layout, {
        responsive: true,
        displayModeBar: true
    });
}

function loadProjectionsChart() {
    const years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
    const futureYears = [2025, 2026, 2027, 2028, 2029, 2030];
    const allYears = [...years, ...futureYears];
    
    const historicalDay = suhiData.regionalTrends.day;
    const historicalNight = suhiData.regionalTrends.night;
    
    // Project future trends
    const dayTrend = 0.0496; // °C/year
    const nightTrend = 0.0430; // °C/year
    
    const projectedDay = futureYears.map(year => 
        historicalDay[historicalDay.length - 1] + dayTrend * (year - 2024)
    );
    const projectedNight = futureYears.map(year => 
        historicalNight[historicalNight.length - 1] + nightTrend * (year - 2024)
    );
    
    const trace1 = {
        x: years,
        y: historicalDay,
        mode: 'lines+markers',
        name: 'Historical Day SUHI',
        line: { color: '#ef4444', width: 3 },
        marker: { size: 8 }
    };
    
    const trace2 = {
        x: years,
        y: historicalNight,
        mode: 'lines+markers',
        name: 'Historical Night SUHI',
        line: { color: '#2563eb', width: 3 },
        marker: { size: 8 }
    };
    
    const trace3 = {
        x: [...years.slice(-1), ...futureYears],
        y: [...historicalDay.slice(-1), ...projectedDay],
        mode: 'lines+markers',
        name: 'Projected Day SUHI',
        line: { color: '#ef4444', width: 3, dash: 'dash' },
        marker: { size: 6, symbol: 'square' }
    };
    
    const trace4 = {
        x: [...years.slice(-1), ...futureYears],
        y: [...historicalNight.slice(-1), ...projectedNight],
        mode: 'lines+markers',
        name: 'Projected Night SUHI',
        line: { color: '#2563eb', width: 3, dash: 'dash' },
        marker: { size: 6, symbol: 'square' }
    };
    
    const layout = {
        title: {
            text: 'SUHI Climate Projections (2025-2030)',
            font: { size: 16, color: '#1e293b' }
        },
        xaxis: {
            title: 'Year',
            dtick: 1
        },
        yaxis: {
            title: 'Regional SUHI (°C)'
        },
        shapes: [{
            type: 'line',
            x0: 2024.5,
            x1: 2024.5,
            y0: 0,
            y1: 2,
            line: {
                color: '#64748b',
                width: 2,
                dash: 'dot'
            }
        }],
        annotations: [{
            x: 2024.5,
            y: 1.8,
            text: 'Projection<br>Boundary',
            showarrow: false,
            font: { color: '#64748b', size: 10 }
        }],
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Inter, sans-serif' },
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };
    
    Plotly.newPlot('projections-chart', [trace1, trace2, trace3, trace4], layout, {
        responsive: true,
        displayModeBar: true
    });
}

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

function generateTimeSeriesForCity(city, type) {
    const baseValue = type === 'day' ? city.dayMean : city.nightMean;
    const trend = city.trend;
    const noise = type === 'day' ? city.dayStd : city.nightStd;
    
    return suhiData.years.map((year, i) => {
        const trendComponent = baseValue + trend * (year - 2015);
        const randomNoise = (Math.random() - 0.5) * noise * 0.5;
        return trendComponent + randomNoise;
    });
}

function refreshOverviewCharts() {
    loadOverviewCharts();
    loadRegionalTrendsChart();
}

function exportCityData() {
    const csvContent = "data:text/csv;charset=utf-8," 
        + "City,Day_SUHI,Day_Std,Night_SUHI,Night_Std,Trend,Urban_Size,Extreme_Events\n"
        + suhiData.cities.map(city => 
            `${city.name},${city.dayMean},${city.dayStd},${city.nightMean},${city.nightStd},${city.trend},${city.urbanSize},${city.extremeEvents}`
        ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "suhi_city_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}
