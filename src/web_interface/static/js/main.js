/**
 * Main JavaScript for Titanic Survival Predictor Web Interface
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic Survival Predictor - JavaScript initialized');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize interactive elements
    initializeInteractiveElements();
    
    // Setup any chart placeholders with loading indicators
    setupChartPlaceholders();
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize interactive UI elements
 */
function initializeInteractiveElements() {
    // Add event listeners for any interactive elements that need them
    
    // Example: Add click event for collapsible cards
    const collapsibleCardHeaders = document.querySelectorAll('.card-header[data-bs-toggle="collapse"]');
    collapsibleCardHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const targetId = this.getAttribute('data-bs-target');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const isCollapsed = targetElement.classList.contains('show');
                // Toggle the collapse arrow
                const arrow = this.querySelector('.collapse-arrow');
                if (arrow) {
                    arrow.style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)';
                }
            }
        });
    });
    
    // Example: Setup range input value display
    const rangeInputs = document.querySelectorAll('input[type="range"][data-value-display]');
    rangeInputs.forEach(input => {
        const displayId = input.getAttribute('data-value-display');
        const displayElement = document.getElementById(displayId);
        
        if (displayElement) {
            // Update initially
            displayElement.textContent = input.value;
            
            // Update on change
            input.addEventListener('input', function() {
                displayElement.textContent = this.value;
            });
        }
    });
}

/**
 * Setup chart placeholders with loading indicators
 */
function setupChartPlaceholders() {
    // Find all chart containers
    const chartContainers = document.querySelectorAll('[id$="-chart"]');
    
    // Add loading indicators to empty charts
    chartContainers.forEach(container => {
        if (!container.hasChildNodes() || container.children.length === 0) {
            container.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2 text-muted">Chart loading...</p></div>';
        }
    });
}

/**
 * Format a number as a percentage
 * @param {number} value - The value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted percentage string
 */
function formatPercent(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Create a bar chart using Plotly.js
 * @param {string} elementId - The ID of the container element
 * @param {Array} labels - Chart labels
 * @param {Array} values - Chart values
 * @param {string} title - Chart title
 */
function createBarChart(elementId, labels, values, title) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const data = [{
        x: labels,
        y: values,
        type: 'bar',
        marker: {
            color: 'rgba(0, 123, 255, 0.7)'
        }
    }];
    
    const layout = {
        title: title,
        font: {
            family: 'Roboto, sans-serif'
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 50
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot(elementId, data, layout, config);
}

/**
 * Create a line chart using Plotly.js
 * @param {string} elementId - The ID of the container element
 * @param {Array} x - X-axis values
 * @param {Array} y - Y-axis values
 * @param {string} title - Chart title
 */
function createLineChart(elementId, x, y, title) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const data = [{
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines+markers',
        marker: {
            color: 'rgba(0, 123, 255, 0.7)'
        },
        line: {
            color: 'rgba(0, 123, 255, 0.7)',
            width: 2
        }
    }];
    
    const layout = {
        title: title,
        font: {
            family: 'Roboto, sans-serif'
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 50
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot(elementId, data, layout, config);
}

/**
 * Format a confusion matrix for display
 * @param {Array} matrix - 2x2 confusion matrix
 * @param {string} elementId - The ID of the container element
 */
function displayConfusionMatrix(matrix, elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const data = [{
        type: 'heatmap',
        z: matrix,
        x: ['Predicted No', 'Predicted Yes'],
        y: ['Actual No', 'Actual Yes'],
        colorscale: [
            [0, 'rgb(242,240,247)'],
            [1, 'rgb(0,123,255)']
        ],
        showscale: false,
        text: [
            ['True Negatives', 'False Positives'],
            ['False Negatives', 'True Positives']
        ],
        hoverinfo: 'text+z'
    }];
    
    const layout = {
        title: 'Confusion Matrix',
        annotations: [],
        xaxis: {
            title: 'Predicted',
            side: 'bottom'
        },
        yaxis: {
            title: 'Actual',
            autorange: 'reversed'
        },
        margin: {
            l: 80,
            r: 20,
            t: 50,
            b: 80
        },
        font: {
            family: 'Roboto, sans-serif'
        }
    };
    
    // Add annotations for each cell
    for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
            layout.annotations.push({
                x: j,
                y: i,
                text: matrix[i][j],
                showarrow: false,
                font: {
                    color: 'white'
                }
            });
        }
    }
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot(elementId, data, layout, config);
}

/**
 * Display ROC curve
 * @param {string} elementId - The ID of the container element
 * @param {Array} fpr - False positive rates
 * @param {Array} tpr - True positive rates
 * @param {number} auc - Area under curve value
 */
function displayRocCurve(elementId, fpr, tpr, auc) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const data = [
        // Diagonal line (random classifier)
        {
            x: [0, 1],
            y: [0, 1],
            type: 'scatter',
            mode: 'lines',
            line: {
                dash: 'dash',
                color: 'gray',
                width: 1
            },
            name: 'Random'
        },
        // ROC curve
        {
            x: fpr,
            y: tpr,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: 'rgba(0, 123, 255, 0.7)',
                width: 2
            },
            name: `ROC Curve (AUC = ${auc.toFixed(3)})`
        }
    ];
    
    const layout = {
        title: 'ROC Curve',
        xaxis: {
            title: 'False Positive Rate',
            showgrid: true,
            zeroline: true,
            range: [0, 1]
        },
        yaxis: {
            title: 'True Positive Rate',
            showgrid: true,
            zeroline: true,
            range: [0, 1]
        },
        font: {
            family: 'Roboto, sans-serif'
        },
        margin: {
            l: 60,
            r: 20,
            t: 50,
            b: 60
        },
        showlegend: true,
        legend: {
            x: 0.6,
            y: 0.1
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot(elementId, data, layout, config);
}
