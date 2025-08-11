// AlphaEarth Uzbekistan - SPA JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('AlphaEarth Uzbekistan Dashboard Initializing...');
    
    // Initialize the application
    initializeNavigation();
    initializeMap();
    initializeCharts();
    
    // Load initial data
    loadDashboardData();
});

// Navigation System
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const contentPanels = document.querySelectorAll('.content-panel');
    
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all nav items and panels
            navItems.forEach(nav => nav.classList.remove('active'));
            contentPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked item
            this.classList.add('active');
            
            // Show corresponding panel
            const module = this.getAttribute('data-module');
            const panel = document.getElementById(`${module}-panel`);
            if (panel) {
                panel.classList.add('active');
                
                // Load module-specific data
                loadModuleData(module);
            }
        });
    });
}

// Interactive Map with Leaflet
function initializeMap() {
    // Uzbekistan coordinates (approximate center)
    const uzbekistanCenter = [41.377491, 64.585262];
    
    // Initialize the map
    const map = L.map('main-map').setView(uzbekistanCenter, 6);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Define region boundaries (approximate)
    const regions = {
        'Karakalpakstan': {
            center: [43.8, 59.4],
            data: { waterStress: 0.85, afforestation: 0.65, urbanHeat: 22.1, biodiversity: 0.42 }
        },
        'Tashkent': {
            center: [41.3, 69.2],
            data: { waterStress: 0.72, afforestation: 0.78, urbanHeat: 26.4, biodiversity: 0.55 }
        },
        'Samarkand': {
            center: [39.7, 66.9],
            data: { waterStress: 0.68, afforestation: 0.74, urbanHeat: 25.8, biodiversity: 0.48 }
        },
        'Bukhara': {
            center: [39.8, 64.4],
            data: { waterStress: 0.79, afforestation: 0.69, urbanHeat: 25.2, biodiversity: 0.43 }
        },
        'Namangan': {
            center: [41.0, 71.7],
            data: { waterStress: 0.76, afforestation: 0.71, urbanHeat: 24.6, biodiversity: 0.46 }
        }
    };
    
    // Add region markers
    Object.entries(regions).forEach(([name, region]) => {
        const marker = L.marker(region.center).addTo(map);
        marker.bindPopup(`
            <div style="font-family: 'Segoe UI', sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: #2E8B57;">${name}</h4>
                <p><strong>Water Stress:</strong> ${(region.data.waterStress * 100).toFixed(1)}%</p>
                <p><strong>Afforestation Suitability:</strong> ${(region.data.afforestation * 100).toFixed(1)}%</p>
                <p><strong>Urban Heat (LST):</strong> ${region.data.urbanHeat}°C</p>
                <p><strong>Biodiversity Index:</strong> ${region.data.biodiversity.toFixed(2)}</p>
            </div>
        `);
    });
    
    // Handle layer selection
    const layerSelect = document.getElementById('layer-select');
    if (layerSelect) {
        layerSelect.addEventListener('change', function() {
            updateMapLayer(map, regions, this.value);
        });
    }
    
    // Store map reference globally
    window.alphaEarthMap = map;
    window.alphaEarthRegions = regions;
}

// Update map visualization based on selected layer
function updateMapLayer(map, regions, layerType) {
    // Remove existing circles
    map.eachLayer(function(layer) {
        if (layer instanceof L.Circle) {
            map.removeLayer(layer);
        }
    });
    
    // Color schemes for different layers
    const colorSchemes = {
        'water-stress': { 
            field: 'waterStress', 
            colors: ['#28A745', '#FFC107', '#DC3545'],
            name: 'Water Stress Level'
        },
        'afforestation': { 
            field: 'afforestation', 
            colors: ['#DC3545', '#FFC107', '#28A745'],
            name: 'Afforestation Suitability'
        },
        'urban-heat': { 
            field: 'urbanHeat', 
            colors: ['#17A2B8', '#FFC107', '#DC3545'],
            name: 'Urban Heat Intensity'
        },
        'biodiversity': { 
            field: 'biodiversity', 
            colors: ['#DC3545', '#FFC107', '#28A745'],
            name: 'Biodiversity Index'
        }
    };
    
    const scheme = colorSchemes[layerType];
    if (!scheme) return;
    
    // Add circles for each region
    Object.entries(regions).forEach(([name, region]) => {
        let value = region.data[scheme.field];
        let color;
        
        // Normalize value and assign color
        if (layerType === 'urban-heat') {
            // For temperature, use actual temperature value
            if (value < 24) color = scheme.colors[0];
            else if (value < 26) color = scheme.colors[1];
            else color = scheme.colors[2];
        } else {
            // For other metrics, use percentage-based coloring
            if (value < 0.5) color = scheme.colors[0];
            else if (value < 0.75) color = scheme.colors[1];
            else color = scheme.colors[2];
        }
        
        const circle = L.circle(region.center, {
            color: color,
            fillColor: color,
            fillOpacity: 0.3,
            radius: 50000 // 50km radius
        }).addTo(map);
        
        circle.bindPopup(`
            <div style="font-family: 'Segoe UI', sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: #2E8B57;">${name}</h4>
                <p><strong>${scheme.name}:</strong> 
                ${layerType === 'urban-heat' ? value + '°C' : (value * 100).toFixed(1) + '%'}</p>
            </div>
        `);
    });
}

// Initialize Charts with Plotly
function initializeCharts() {
    // Chart configurations
    const chartConfigs = {
        'soil-moisture-chart': createSoilMoistureChart,
        'soil-moisture-trends': createSoilMoistureTrends,
        'afforestation-chart': createAfforestationChart,
        'species-chart': createSpeciesChart
    };
    
    // Create charts if elements exist
    Object.entries(chartConfigs).forEach(([elementId, createFunction]) => {
        const element = document.getElementById(elementId);
        if (element) {
            createFunction(elementId);
        }
    });
}

// Soil Moisture Analysis Chart
function createSoilMoistureChart(elementId) {
    const data = [
        {
            x: ['Karakalpakstan', 'Tashkent', 'Samarkand', 'Bukhara', 'Namangan'],
            y: [32.4, 31.4, 32.0, 31.9, 30.6],
            type: 'bar',
            name: 'Soil Moisture (%)',
            marker: {
                color: ['#DC3545', '#FFC107', '#28A745', '#FFC107', '#DC3545'],
                line: { color: '#2E8B57', width: 2 }
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'Regional Soil Moisture Levels',
            font: { size: 16, color: '#212529' }
        },
        xaxis: {
            title: 'Region',
            titlefont: { color: '#6C757D' }
        },
        yaxis: {
            title: 'Soil Moisture (%)',
            titlefont: { color: '#6C757D' },
            range: [25, 35]
        },
        plot_bgcolor: '#F8F9FA',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 50, r: 30, b: 80, l: 60 }
    };
    
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, data, layout, config);
}

// Soil Moisture Trends Chart
function createSoilMoistureTrends(elementId) {
    const years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
    
    const data = [
        {
            x: years,
            y: [34.2, 33.8, 33.1, 32.5, 32.1, 31.8, 31.5, 31.2, 31.7],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'National Average',
            line: { color: '#2E8B57', width: 3 },
            marker: { size: 8, color: '#2E8B57' }
        },
        {
            x: years,
            y: [35.1, 34.5, 33.8, 33.2, 32.8, 32.4, 32.1, 31.8, 32.4],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Karakalpakstan',
            line: { color: '#DC3545', width: 2, dash: 'dash' },
            marker: { size: 6 }
        }
    ];
    
    const layout = {
        title: {
            text: 'Soil Moisture Temporal Trends (2017-2025)',
            font: { size: 16, color: '#212529' }
        },
        xaxis: {
            title: 'Year',
            titlefont: { color: '#6C757D' }
        },
        yaxis: {
            title: 'Soil Moisture (%)',
            titlefont: { color: '#6C757D' }
        },
        plot_bgcolor: '#F8F9FA',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 50, r: 30, b: 60, l: 60 },
        legend: { x: 0.7, y: 1 }
    };
    
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, data, layout, config);
}

// Afforestation Suitability Chart
function createAfforestationChart(elementId) {
    const data = [
        {
            x: ['Karakalpakstan', 'Tashkent', 'Samarkand', 'Bukhara', 'Namangan'],
            y: [719, 717, 716, 717, 706],
            type: 'bar',
            name: 'Suitability Score',
            marker: {
                color: '#28A745',
                line: { color: '#2E8B57', width: 2 }
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'Regional Afforestation Suitability',
            font: { size: 16, color: '#212529' }
        },
        xaxis: {
            title: 'Region',
            titlefont: { color: '#6C757D' }
        },
        yaxis: {
            title: 'Suitability Score (0-1000)',
            titlefont: { color: '#6C757D' },
            range: [700, 730]
        },
        plot_bgcolor: '#F8F9FA',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 50, r: 30, b: 80, l: 60 }
    };
    
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, data, layout, config);
}

// Species Suitability Chart
function createSpeciesChart(elementId) {
    const species = [
        'Elaeagnus angustifolia',
        'Populus alba',
        'Ulmus pumila',
        'Tamarix species',
        'Pinus sylvestris'
    ];
    
    const regions = ['Karakalpakstan', 'Tashkent', 'Samarkand', 'Bukhara', 'Namangan'];
    
    const data = [
        {
            z: [
                [0.52, 0.52, 0.52, 0.53, 0.52], // Elaeagnus
                [0.26, 0.25, 0.27, 0.27, 0.25], // Populus
                [0.39, 0.40, 0.40, 0.41, 0.40], // Ulmus
                [0.21, 0.23, 0.21, 0.21, 0.23], // Tamarix
                [0.16, 0.14, 0.15, 0.15, 0.14]  // Pinus
            ],
            x: regions,
            y: species,
            type: 'heatmap',
            colorscale: [
                [0, '#DC3545'],
                [0.5, '#FFC107'],
                [1, '#28A745']
            ],
            showscale: true,
            colorbar: {
                title: 'Suitability Score',
                titlefont: { color: '#6C757D' }
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'Species Suitability by Region',
            font: { size: 16, color: '#212529' }
        },
        xaxis: {
            title: 'Region',
            titlefont: { color: '#6C757D' }
        },
        yaxis: {
            title: 'Species',
            titlefont: { color: '#6C757D' }
        },
        plot_bgcolor: '#F8F9FA',
        paper_bgcolor: '#FFFFFF',
        margin: { t: 50, r: 100, b: 100, l: 200 }
    };
    
    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(elementId, data, layout, config);
}

// Load Dashboard Data
function loadDashboardData() {
    console.log('Loading dashboard data...');
    
    // Simulate data loading with promises
    Promise.all([
        fetchSoilMoistureData(),
        fetchAfforestationData(),
        fetchUrbanHeatData(),
        fetchBiodiversityData()
    ]).then(([soilData, afforestationData, urbanHeatData, biodiversityData]) => {
        console.log('All data loaded successfully');
        updateDashboardMetrics({
            soilData,
            afforestationData,
            urbanHeatData,
            biodiversityData
        });
    }).catch(error => {
        console.error('Error loading data:', error);
        showDataError();
    });
}

// Load Module-Specific Data
function loadModuleData(module) {
    console.log(`Loading data for module: ${module}`);
    
    switch (module) {
        case 'soil-moisture':
            loadSoilMoistureModule();
            break;
        case 'afforestation':
            loadAfforestationModule();
            break;
        case 'urban-heat':
            loadUrbanHeatModule();
            break;
        case 'biodiversity':
            loadBiodiversityModule();
            break;
        default:
            console.log('Module data loading not implemented for:', module);
    }
}

// Module-specific loading functions
function loadSoilMoistureModule() {
    // Update soil moisture charts with real-time data
    if (document.getElementById('soil-moisture-chart')) {
        createSoilMoistureChart('soil-moisture-chart');
    }
    if (document.getElementById('soil-moisture-trends')) {
        createSoilMoistureTrends('soil-moisture-trends');
    }
}

function loadAfforestationModule() {
    // Update afforestation charts
    if (document.getElementById('afforestation-chart')) {
        createAfforestationChart('afforestation-chart');
    }
    if (document.getElementById('species-chart')) {
        createSpeciesChart('species-chart');
    }
}

function loadUrbanHeatModule() {
    // Placeholder for urban heat module
    console.log('Urban heat module data loaded');
}

function loadBiodiversityModule() {
    // Placeholder for biodiversity module
    console.log('Biodiversity module data loaded');
}

// Data fetching functions (simulated)
function fetchSoilMoistureData() {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                nationalAverage: 31.7,
                severeStressAreas: 156,
                highStressAreas: 1685,
                trends: 'declining'
            });
        }, 500);
    });
}

function fetchAfforestationData() {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                suitableSites: 3681,
                averageSuitability: 71.7,
                modelAccuracy: 97.1,
                recommendedSpecies: 'Elaeagnus angustifolia'
            });
        }, 300);
    });
}

function fetchUrbanHeatData() {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                averageLST: 26.2,
                maxUHIIntensity: 19.8,
                highRiskAreas: 54,
                coolingPotential: 1872.3
            });
        }, 400);
    });
}

function fetchBiodiversityData() {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({
                ecosystemTypes: 6,
                habitatQuality: 0.403,
                fragmentationIndex: 0.25,
                conservationPriority: 'High'
            });
        }, 600);
    });
}

// Update Dashboard Metrics
function updateDashboardMetrics(data) {
    // Update metric cards with real data
    const metrics = document.querySelectorAll('.metric-value');
    if (metrics.length > 0) {
        // Update water stress metric
        if (metrics[0]) {
            metrics[0].textContent = '67.4%';
        }
        // Update afforestation metric
        if (metrics[1]) {
            metrics[1].textContent = data.afforestationData.suitableSites.toLocaleString();
        }
        // Update urban heat metric
        if (metrics[2]) {
            metrics[2].textContent = data.urbanHeatData.averageLST + '°C';
        }
        // Update biodiversity metric
        if (metrics[3]) {
            metrics[3].textContent = data.biodiversityData.ecosystemTypes;
        }
    }
}

// Error Handling
function showDataError() {
    const errorMessage = document.createElement('div');
    errorMessage.className = 'alert alert-warning';
    errorMessage.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <strong>Data Loading Error:</strong> Some data could not be loaded. 
        Please check your connection and try again.
    `;
    
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.insertBefore(errorMessage, mainContent.firstChild);
    }
}

// Utility Functions
function formatNumber(num) {
    return num.toLocaleString();
}

function formatPercentage(num) {
    return (num * 100).toFixed(1) + '%';
}

function formatCurrency(num) {
    return '$' + num.toLocaleString();
}

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeNavigation,
        initializeMap,
        initializeCharts,
        loadDashboardData
    };
}

console.log('AlphaEarth Uzbekistan Dashboard JavaScript loaded successfully');