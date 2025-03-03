/**
 * JavaScript for the Results page
 * 
 * This file contains the JavaScript code for the Results page of the Titanic Survival Predictor.
 * It handles the visualization of model metrics and comparison of models.
 */

// Global variable to track if Plotly is available
let plotlyAvailable = false;

// Check if Plotly is available when the script loads
if (typeof Plotly !== 'undefined') {
    plotlyAvailable = true;
    console.log('Plotly loaded successfully', Plotly.version);
} else {
    console.error('Plotly not available on initial load');
    // Will attempt to use CDN version in the error handler in HTML
}

document.addEventListener('DOMContentLoaded', function() {
    // Double check if Plotly has loaded by this point (might have loaded asynchronously)
    if (typeof Plotly !== 'undefined') {
        plotlyAvailable = true;
        console.log('Plotly available on DOMContentLoaded', Plotly.version);
    }
    
    // Initialize visualization components
    initializePerformanceChart();
    initializeRocCurvesChart();
    initializeFeatureImportanceChart();

    // Handle view details button clicks
    const viewDetailsButtons = document.querySelectorAll('.view-details-btn');
    viewDetailsButtons.forEach(button => {
        button.addEventListener('click', function() {
            const modelName = this.getAttribute('data-model');
            showModelDetails(modelName);
        });
    });

    // Handle metric selection for performance chart
    const metricButtons = document.querySelectorAll('[data-metric]');
    metricButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            metricButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            
            const metric = this.getAttribute('data-metric');
            updatePerformanceChart(metric);
        });
    });

    // Handle model selection for ROC curves
    const modelCheckboxes = document.querySelectorAll('[data-model]');
    modelCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateRocCurvesChart();
        });
    });

    // Handle model selection for feature importance
    const featureImportanceButtons = document.querySelectorAll('[data-model]');
    featureImportanceButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons in the group
            this.parentElement.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            
            const model = this.getAttribute('data-model');
            updateFeatureImportanceChart(model);
        });
    });

    // Function to show model details in modal
    function showModelDetails(modelName) {
        // Show loading spinners in the visualization containers
        document.getElementById('modal-confusion-matrix').innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`;
        
        document.getElementById('modal-roc-curve').innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`;
        
        document.getElementById('modal-feature-importance').innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`;
        
        // Set modal title immediately
        document.getElementById('modelDetailsModalLabel').textContent = 
            `${formatModelName(modelName)} Details`;
        
        // Show the modal immediately to show loading state
        const modalElement = document.getElementById('modelDetailsModal');
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
        
        // Hide error message (will show again if needed)
        const errorElement = document.getElementById('visualization-errors');
        if (errorElement) {
            errorElement.classList.add('d-none');
        }
        
        // Fetch model details from the server
        fetch(`/api/model-details?model=${modelName}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Model details data:', data);
                
                // Fill in model parameters
                const paramsHTML = Object.entries(data.params || {})
                    .map(([key, value]) => `<tr><th>${key}</th><td>${value}</td></tr>`)
                    .join('');
                document.getElementById('model-params').innerHTML = 
                    paramsHTML || '<tr><td colspan="2">No parameters available</td></tr>';
                
                // Fill in performance metrics
                const metricsHTML = Object.entries(data.metrics || {})
                    .filter(([key]) => !Array.isArray(data.metrics[key]) && typeof data.metrics[key] !== 'object')
                    .map(([key, value]) => {
                        const formattedValue = typeof value === 'number' ? 
                            (value < 1 ? (value * 100).toFixed(2) + '%' : value.toFixed(2)) : value;
                        return `<tr><th>${formatMetricName(key)}</th><td>${formattedValue}</td></tr>`;
                    })
                    .join('');
                document.getElementById('model-metrics').innerHTML = 
                    metricsHTML || '<tr><td colspan="2">No metrics available</td></tr>';
                
                // Render visualizations (with error handling for each)
                try {
                    renderConfusionMatrix(data.confusion_matrix, 'modal-confusion-matrix');
                } catch (error) {
                    console.error('Error rendering confusion matrix:', error);
                    document.getElementById('modal-confusion-matrix').innerHTML = `
                        <div class="alert alert-warning">
                            Error rendering confusion matrix. See console for details.
                        </div>`;
                    showVisualizationError('Confusion Matrix');
                }
                
                try {
                    renderRocCurve(data.roc_curve, 'modal-roc-curve');
                } catch (error) {
                    console.error('Error rendering ROC curve:', error);
                    document.getElementById('modal-roc-curve').innerHTML = `
                        <div class="alert alert-warning">
                            Error rendering ROC curve. See console for details.
                        </div>`;
                    showVisualizationError('ROC Curve');
                }
                
                try {
                    renderFeatureImportance(data.feature_importance, 'modal-feature-importance');
                } catch (error) {
                    console.error('Error rendering feature importance:', error);
                    document.getElementById('modal-feature-importance').innerHTML = `
                        <div class="alert alert-warning">
                            Error rendering feature importance. See console for details.
                        </div>`;
                    showVisualizationError('Feature Importance');
                }
            })
            .catch(error => {
                console.error('Error fetching model details:', error);
                
                // Show fallback message in each visualization area
                document.getElementById('modal-confusion-matrix').innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Failed to load data.</strong><br>
                        ${error.message}
                    </div>`;
                
                document.getElementById('modal-roc-curve').innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Failed to load data.</strong><br>
                        ${error.message}
                    </div>`;
                
                document.getElementById('modal-feature-importance').innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Failed to load data.</strong><br>
                        ${error.message}
                    </div>`;
                
                showVisualizationError('API Request');
            });
    }

    // Function to show visualization error
    function showVisualizationError(component) {
        // Show error message area
        const errorElement = document.getElementById('visualization-errors');
        const errorDetailsElement = document.getElementById('visualization-error-details');
        
        if (errorElement && errorDetailsElement) {
            errorElement.classList.remove('d-none');
            
            // Add specific error
            const errorItem = document.createElement('p');
            errorItem.innerHTML = `<strong>${component}</strong>: Failed to render visualization. Check browser console for details.`;
            errorDetailsElement.appendChild(errorItem);
            
            // Add general troubleshooting tips if this is the first error
            if (errorDetailsElement.children.length === 1) {
                const tipsList = document.createElement('ul');
                tipsList.innerHTML = `
                    <li>Make sure your browser is up to date</li>
                    <li>Check if you have JavaScript enabled</li>
                    <li>Try refreshing the page</li>
                    <li>Check if you have any browser extensions that might be blocking scripts</li>
                `;
                errorDetailsElement.appendChild(tipsList);
            }
        }
    }

    // Function to render confusion matrix
    function renderConfusionMatrix(confusionMatrix, elementId) {
        const element = document.getElementById(elementId);
        
        if (!confusionMatrix || !Array.isArray(confusionMatrix) || confusionMatrix.length !== 2) {
            element.innerHTML = 
                '<div class="alert alert-warning">Confusion matrix data not available</div>';
            return;
        }
        
        // Check if Plotly is available
        if (!plotlyAvailable) {
            // Fallback to table representation
            renderConfusionMatrixAsTable(confusionMatrix, elementId);
            return;
        }
        
        // Create confusion matrix visualization using Plotly
        try {
            const data = [{
                z: confusionMatrix,
                x: ['Predicted Died', 'Predicted Survived'],
                y: ['Actual Died', 'Actual Survived'],
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: false,
                text: [
                    [`TN: ${confusionMatrix[0][0]}`, `FP: ${confusionMatrix[0][1]}`],
                    [`FN: ${confusionMatrix[1][0]}`, `TP: ${confusionMatrix[1][1]}`]
                ],
                hoverinfo: 'text'
            }];
            
            const layout = {
                title: 'Confusion Matrix',
                xaxis: { title: 'Predicted' },
                yaxis: { title: 'Actual' },
                annotations: []
            };
            
            // Add text annotations to each cell
            for (let i = 0; i < 2; i++) {
                for (let j = 0; j < 2; j++) {
                    const annotationText = confusionMatrix[i][j].toString();
                    layout.annotations.push({
                        x: j,
                        y: i,
                        text: annotationText,
                        font: { color: 'white' },
                        showarrow: false
                    });
                }
            }
            
            Plotly.newPlot(elementId, data, layout, { responsive: true });
        } catch (error) {
            console.error('Error creating confusion matrix plot:', error);
            renderConfusionMatrixAsTable(confusionMatrix, elementId);
        }
    }
    
    // Function to render confusion matrix as HTML table (fallback)
    function renderConfusionMatrixAsTable(confusionMatrix, elementId) {
        const element = document.getElementById(elementId);
        
        // Create HTML table
        const tableHTML = `
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th></th>
                        <th>Predicted Died</th>
                        <th>Predicted Survived</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Actual Died</th>
                        <td class="bg-info text-white">${confusionMatrix[0][0]}</td>
                        <td>${confusionMatrix[0][1]}</td>
                    </tr>
                    <tr>
                        <th>Actual Survived</th>
                        <td>${confusionMatrix[1][0]}</td>
                        <td class="bg-info text-white">${confusionMatrix[1][1]}</td>
                    </tr>
                </tbody>
            </table>
            <p class="text-muted small">Fallback table mode: Plotly visualization not available</p>
        `;
        
        element.innerHTML = tableHTML;
    }

    // Function to render ROC curve
    function renderRocCurve(rocData, elementId) {
        const element = document.getElementById(elementId);
        
        if (!rocData || !rocData.fpr || !rocData.tpr) {
            element.innerHTML = 
                '<div class="alert alert-warning">ROC curve data not available</div>';
            return;
        }
        
        // Check if Plotly is available
        if (!plotlyAvailable) {
            // Fallback to simplified display
            renderRocCurveAsSummary(rocData, elementId);
            return;
        }
        
        // Create ROC curve visualization using Plotly
        try {
            const data = [
                {
                    x: rocData.fpr,
                    y: rocData.tpr,
                    type: 'scatter',
                    mode: 'lines',
                    name: `ROC Curve (AUC = ${rocData.auc.toFixed(3)})`,
                    line: { color: 'blue', width: 2 }
                },
                {
                    x: [0, 1],
                    y: [0, 1],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Random',
                    line: { color: 'grey', width: 2, dash: 'dash' }
                }
            ];
            
            const layout = {
                title: 'ROC Curve',
                xaxis: { title: 'False Positive Rate' },
                yaxis: { title: 'True Positive Rate' },
                legend: { x: 0.6, y: 0.2 },
                showlegend: true
            };
            
            Plotly.newPlot(elementId, data, layout, { responsive: true });
        } catch (error) {
            console.error('Error creating ROC curve plot:', error);
            renderRocCurveAsSummary(rocData, elementId);
        }
    }
    
    // Function to render ROC curve as simplified summary (fallback)
    function renderRocCurveAsSummary(rocData, elementId) {
        const element = document.getElementById(elementId);
        
        // Create a simple card with AUC value
        const auc = rocData.auc.toFixed(3);
        const cardHTML = `
            <div class="card">
                <div class="card-body text-center">
                    <h5 class="card-title">ROC Curve Summary</h5>
                    <p class="card-text">Area Under Curve (AUC): <strong>${auc}</strong></p>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" 
                            style="width: ${auc * 100}%" 
                            aria-valuenow="${auc * 100}" aria-valuemin="0" aria-valuemax="100">
                            ${(auc * 100).toFixed(1)}%
                        </div>
                    </div>
                    <p class="text-muted small">Fallback mode: Plotly visualization not available</p>
                </div>
            </div>
        `;
        
        element.innerHTML = cardHTML;
    }

    // Function to render feature importance
    function renderFeatureImportance(featureImportance, elementId) {
        const element = document.getElementById(elementId);
        
        if (!featureImportance || !Array.isArray(featureImportance) || featureImportance.length === 0) {
            element.innerHTML = 
                '<div class="alert alert-warning">Feature importance data not available</div>';
            return;
        }
        
        // Extract features and importance values
        const features = featureImportance.map(item => item[0]);
        const importanceValues = featureImportance.map(item => item[1]);
        
        // Sort in descending order of importance
        const indices = Array.from(Array(features.length).keys())
            .sort((a, b) => importanceValues[b] - importanceValues[a]);
        
        const sortedFeatures = indices.map(i => features[i]);
        const sortedImportance = indices.map(i => importanceValues[i]);
        
        // Only show top 10 features
        const topFeatures = sortedFeatures.slice(0, 10);
        const topImportance = sortedImportance.slice(0, 10);
        
        // Check if Plotly is available
        if (!plotlyAvailable) {
            // Fallback to horizontal bar representation with HTML/CSS
            renderFeatureImportanceAsTable(topFeatures, topImportance, elementId);
            return;
        }
        
        // Create feature importance visualization using Plotly
        try {
            const data = [{
                y: topFeatures,
                x: topImportance,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: 'rgba(58, 123, 213, 0.8)',
                    line: {
                        color: 'rgba(58, 123, 213, 1.0)',
                        width: 1
                    }
                }
            }];
            
            const layout = {
                title: 'Top 10 Feature Importance',
                xaxis: { title: 'Importance' },
                yaxis: { title: 'Feature', automargin: true },
                margin: { l: 150 } // Add more space for feature names
            };
            
            Plotly.newPlot(elementId, data, layout, { responsive: true });
        } catch (error) {
            console.error('Error creating feature importance plot:', error);
            renderFeatureImportanceAsTable(topFeatures, topImportance, elementId);
        }
    }
    
    // Function to render feature importance as HTML table with bars (fallback)
    function renderFeatureImportanceAsTable(features, importance, elementId) {
        const element = document.getElementById(elementId);
        
        // Create HTML table with CSS progress bars
        let tableHTML = `
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Add rows
        features.forEach((feature, index) => {
            const value = importance[index];
            const percentage = (value * 100).toFixed(1);
            
            tableHTML += `
                <tr>
                    <td>${feature}</td>
                    <td>
                        <div class="d-flex align-items-center">
                            <div class="progress flex-grow-1 me-2" style="height: 10px;">
                                <div class="progress-bar bg-primary" role="progressbar" 
                                    style="width: ${percentage}%" 
                                    aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                                </div>
                            </div>
                            <span>${value.toFixed(3)}</span>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        tableHTML += `
                </tbody>
            </table>
            <p class="text-muted small">Fallback table mode: Plotly visualization not available</p>
        `;
        
        element.innerHTML = tableHTML;
    }

    // Function to initialize main performance chart
    function initializePerformanceChart() {
        // Fetch model comparison data
        fetch('/api/model-comparison')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Create performance chart
                const ctx = document.getElementById('performance-chart');
                if (!ctx) {
                    console.error('Performance chart element not found');
                    return;
                }
                
                ctx.innerHTML = ''; // Clear any placeholder content
                
                // Format model names for display
                const formattedModelNames = data.models.map(formatModelName);
                
                // Create chart data
                const chartData = {
                    labels: formattedModelNames,
                    datasets: [{
                        label: 'Accuracy',
                        data: data.accuracy,
                        backgroundColor: 'rgba(75, 192, 192, 0.8)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                };
                
                // Store data for other metrics
                ctx.dataset.models = JSON.stringify(formattedModelNames);
                ctx.dataset.accuracy = JSON.stringify(data.accuracy);
                ctx.dataset.precision = JSON.stringify(data.precision);
                ctx.dataset.recall = JSON.stringify(data.recall);
                ctx.dataset.f1 = JSON.stringify(data.f1);
                ctx.dataset.auc = JSON.stringify(data.auc);
                
                // Check if Plotly is available
                if (!plotlyAvailable) {
                    // Fallback to table
                    renderPerformanceAsTable(data, 'performance-chart');
                    return;
                }
                
                // Create bar chart with Plotly
                try {
                    Plotly.newPlot('performance-chart', [{
                        x: formattedModelNames,
                        y: data.accuracy,
                        type: 'bar',
                        name: 'Accuracy',
                        marker: {
                            color: 'rgba(75, 192, 192, 0.8)'
                        }
                    }], {
                        title: 'Model Performance (Accuracy)',
                        xaxis: { title: 'Model' },
                        yaxis: { title: 'Accuracy (%)', range: [0, 100] }
                    }, { responsive: true });
                } catch (error) {
                    console.error('Error creating performance chart:', error);
                    renderPerformanceAsTable(data, 'performance-chart');
                }
            })
            .catch(error => {
                console.error('Error fetching model comparison data:', error);
                document.getElementById('performance-chart').innerHTML = 
                    `<div class="alert alert-danger">Error loading performance data: ${error.message}</div>`;
            });
    }
    
    // Function to render performance as table (fallback)
    function renderPerformanceAsTable(data, elementId) {
        const element = document.getElementById(elementId);
        
        // Create HTML table
        let tableHTML = `
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>AUC</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Add rows
        data.models.forEach((model, index) => {
            tableHTML += `
                <tr>
                    <td>${formatModelName(model)}</td>
                    <td>${data.accuracy[index].toFixed(1)}%</td>
                    <td>${data.precision[index].toFixed(1)}%</td>
                    <td>${data.recall[index].toFixed(1)}%</td>
                    <td>${data.f1[index].toFixed(1)}%</td>
                    <td>${data.auc[index].toFixed(3)}</td>
                </tr>
            `;
        });
        
        tableHTML += `
                </tbody>
            </table>
            <p class="text-muted small">Fallback table mode: Plotly visualization not available</p>
        `;
        
        element.innerHTML = tableHTML;
    }

    // Function to update performance chart based on selected metric
    function updatePerformanceChart(metric) {
        const chartElement = document.getElementById('performance-chart');
        if (!chartElement) {
            console.error('Performance chart element not found');
            return;
        }
        
        // Get data from the dataset
        let models, metricData;
        try {
            models = JSON.parse(chartElement.dataset.models);
            metricData = JSON.parse(chartElement.dataset[metric]);
        } catch (error) {
            console.error('Error parsing chart data:', error);
            return;
        }
        
        // Check if Plotly is available
        if (!plotlyAvailable) {
            // Fallback already handled in initializePerformanceChart
            return;
        }
        
        // Update chart
        try {
            Plotly.react('performance-chart', [{
                x: models,
                y: metricData,
                type: 'bar',
                name: formatMetricName(metric),
                marker: {
                    color: getMetricColor(metric)
                }
            }], {
                title: `Model Performance (${formatMetricName(metric)})`,
                xaxis: { title: 'Model' },
                yaxis: { title: `${formatMetricName(metric)} (${metric === 'auc' ? '' : '%'})`, range: [0, metric === 'auc' ? 1 : 100] }
            }, { responsive: true });
        } catch (error) {
            console.error('Error updating performance chart:', error);
        }
    }

    // Function to initialize ROC curves chart
    function initializeRocCurvesChart() {
        document.getElementById('roc-curves-chart').innerHTML = 
            '<div class="alert alert-info">Select models to display ROC curves</div>';
    }

    // Function to update ROC curves chart based on selected models
    function updateRocCurvesChart() {
        const selectedModels = [];
        document.querySelectorAll('[data-model]:checked').forEach(checkbox => {
            selectedModels.push(checkbox.getAttribute('data-model'));
        });
        
        if (selectedModels.length === 0) {
            document.getElementById('roc-curves-chart').innerHTML = 
                '<div class="alert alert-info">Select at least one model to display ROC curve</div>';
            return;
        }
        
        // Fetch ROC curve data for selected models
        fetch(`/api/roc-curves?models=${selectedModels.join(',')}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const chartElement = document.getElementById('roc-curves-chart');
                chartElement.innerHTML = ''; // Clear any placeholder content
                
                // Check if Plotly is available
                if (!plotlyAvailable) {
                    // Fallback to table of AUC values
                    renderRocCurvesAsTable(data, chartElement);
                    return;
                }
                
                // Create chart traces for each model
                const traces = [];
                
                // Add random baseline
                traces.push({
                    x: [0, 1],
                    y: [0, 1],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Random',
                    line: {
                        color: 'grey',
                        width: 2,
                        dash: 'dash'
                    }
                });
                
                // Add model curves
                Object.entries(data).forEach(([modelName, rocData], index) => {
                    if (rocData && rocData.fpr && rocData.tpr) {
                        traces.push({
                            x: rocData.fpr,
                            y: rocData.tpr,
                            type: 'scatter',
                            mode: 'lines',
                            name: `${formatModelName(modelName)} (AUC = ${rocData.auc.toFixed(3)})`,
                            line: {
                                color: getModelColor(index),
                                width: 2
                            }
                        });
                    }
                });
                
                // Create plot
                try {
                    Plotly.newPlot('roc-curves-chart', traces, {
                        title: 'ROC Curves Comparison',
                        xaxis: { title: 'False Positive Rate' },
                        yaxis: { title: 'True Positive Rate' },
                        legend: { x: 0.6, y: 0.2 }
                    }, { responsive: true });
                } catch (error) {
                    console.error('Error creating ROC curves comparison:', error);
                    renderRocCurvesAsTable(data, chartElement);
                }
            })
            .catch(error => {
                console.error('Error fetching ROC curves data:', error);
                document.getElementById('roc-curves-chart').innerHTML = 
                    `<div class="alert alert-danger">Error loading ROC curves: ${error.message}</div>`;
            });
    }
    
    // Function to render ROC curves as table (fallback)
    function renderRocCurvesAsTable(data, element) {
        // Create HTML table of AUC values
        let tableHTML = `
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>AUC</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Add rows
        Object.entries(data).forEach(([modelName, rocData]) => {
            if (rocData && rocData.auc) {
                const auc = rocData.auc.toFixed(3);
                const percentage = (rocData.auc * 100).toFixed(1);
                
                tableHTML += `
                    <tr>
                        <td>${formatModelName(modelName)}</td>
                        <td>${auc}</td>
                        <td>
                            <div class="progress" style="height: 10px;">
                                <div class="progress-bar" role="progressbar" 
                                    style="width: ${percentage}%" 
                                    aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                                </div>
                            </div>
                        </td>
                    </tr>
                `;
            }
        });
        
        tableHTML += `
                </tbody>
            </table>
            <p class="text-muted small">Fallback table mode: Plotly visualization not available</p>
        `;
        
        element.innerHTML = tableHTML;
    }

    // Function to initialize feature importance chart
    function initializeFeatureImportanceChart() {
        // Default to the first model (gradient_boosting)
        updateFeatureImportanceChart('gradient_boosting');
    }

    // Function to update feature importance chart based on selected model
    function updateFeatureImportanceChart(modelName) {
        // Fetch feature importance data for the selected model
        fetch(`/api/feature-importance?model=${modelName}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const chartElement = document.getElementById('feature-importance-chart');
                chartElement.innerHTML = ''; // Clear any placeholder content
                
                if (data.error) {
                    chartElement.innerHTML = 
                        `<div class="alert alert-warning">${data.error}</div>`;
                    return;
                }
                
                // Extract features and importance values
                let features, importance;
                
                if (Array.isArray(data)) {
                    // If the data is already an array of [feature, importance] pairs
                    features = data.map(item => item[0]);
                    importance = data.map(item => item[1]);
                } else if (data.features && data.importance) {
                    // If the data is in the { features: [], importance: [] } format
                    features = data.features;
                    importance = data.importance;
                } else {
                    chartElement.innerHTML = 
                        '<div class="alert alert-warning">Invalid feature importance data format</div>';
                    return;
                }
                
                // Sort by importance
                const indices = Array.from(Array(features.length).keys())
                    .sort((a, b) => importance[b] - importance[a]);
                
                const sortedFeatures = indices.map(i => features[i]);
                const sortedImportance = indices.map(i => importance[i]);
                
                // Take top 10 features
                const topFeatures = sortedFeatures.slice(0, 10);
                const topImportance = sortedImportance.slice(0, 10);
                
                // Check if Plotly is available
                if (!plotlyAvailable) {
                    // Fallback to table
                    renderFeatureImportanceAsTable(topFeatures, topImportance, 'feature-importance-chart');
                    return;
                }
                
                // Create chart
                try {
                    Plotly.newPlot('feature-importance-chart', [{
                        y: topFeatures,
                        x: topImportance,
                        type: 'bar',
                        orientation: 'h',
                        marker: {
                            color: 'rgba(58, 123, 213, 0.8)',
                            line: {
                                color: 'rgba(58, 123, 213, 1.0)',
                                width: 1
                            }
                        }
                    }], {
                        title: `Feature Importance for ${formatModelName(modelName)}`,
                        xaxis: { title: 'Importance' },
                        yaxis: { title: 'Feature', automargin: true },
                        margin: { l: 150 } // Add more space for feature names
                    }, { responsive: true });
                } catch (error) {
                    console.error('Error creating feature importance chart:', error);
                    renderFeatureImportanceAsTable(topFeatures, topImportance, 'feature-importance-chart');
                }
            })
            .catch(error => {
                console.error('Error fetching feature importance data:', error);
                document.getElementById('feature-importance-chart').innerHTML = 
                    `<div class="alert alert-danger">Error loading feature importance: ${error.message}</div>`;
            });
    }

    // Helper function to format model name for display
    function formatModelName(modelName) {
        if (!modelName) return '';
        
        // Convert snake_case to Title Case
        return modelName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Helper function to format metric name for display
    function formatMetricName(metricName) {
        if (!metricName) return '';
        
        const metricMap = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'roc_auc': 'ROC AUC',
            'auc': 'AUC-ROC',
            'average_precision': 'Average Precision'
        };
        
        return metricMap[metricName] || metricName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Helper function to get color for a metric
    function getMetricColor(metric) {
        const colorMap = {
            'accuracy': 'rgba(75, 192, 192, 0.8)',
            'precision': 'rgba(54, 162, 235, 0.8)',
            'recall': 'rgba(255, 159, 64, 0.8)',
            'f1': 'rgba(153, 102, 255, 0.8)',
            'auc': 'rgba(255, 99, 132, 0.8)'
        };
        
        return colorMap[metric] || 'rgba(75, 192, 192, 0.8)';
    }

    // Helper function to get color for a model
    function getModelColor(index) {
        const colors = [
            'rgb(31, 119, 180)',  // blue
            'rgb(255, 127, 14)',  // orange
            'rgb(44, 160, 44)',   // green
            'rgb(214, 39, 40)',   // red
            'rgb(148, 103, 189)', // purple
            'rgb(140, 86, 75)'    // brown
        ];
        
        return colors[index % colors.length];
    }
});
