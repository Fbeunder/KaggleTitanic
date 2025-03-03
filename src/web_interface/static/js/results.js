/**
 * JavaScript for the Results page
 * 
 * This file contains the JavaScript code for the Results page of the Titanic Survival Predictor.
 * It handles the visualization of model metrics and comparison of models.
 */

document.addEventListener('DOMContentLoaded', function() {
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
        // Fetch model details from the server
        fetch(`/api/model-details?model=${modelName}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Set modal title
                document.getElementById('modelDetailsModalLabel').textContent = 
                    `${formatModelName(modelName)} Details`;
                
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
                
                // Render visualizations
                renderConfusionMatrix(data.confusion_matrix, 'modal-confusion-matrix');
                renderRocCurve(data.roc_curve, 'modal-roc-curve');
                renderFeatureImportance(data.feature_importance, 'modal-feature-importance');
                
                // Show the modal
                const modalElement = document.getElementById('modelDetailsModal');
                const modal = new bootstrap.Modal(modalElement);
                modal.show();
            })
            .catch(error => {
                console.error('Error fetching model details:', error);
                alert(`Error loading model details: ${error.message}`);
            });
    }

    // Function to render confusion matrix
    function renderConfusionMatrix(confusionMatrix, elementId) {
        const element = document.getElementById(elementId);
        
        if (!confusionMatrix || !Array.isArray(confusionMatrix) || confusionMatrix.length !== 2) {
            element.innerHTML = 
                '<div class="alert alert-warning">Confusion matrix data not available</div>';
            return;
        }
        
        // Create confusion matrix visualization using Plotly
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
    }

    // Function to render ROC curve
    function renderRocCurve(rocData, elementId) {
        const element = document.getElementById(elementId);
        
        if (!rocData || !rocData.fpr || !rocData.tpr) {
            element.innerHTML = 
                '<div class="alert alert-warning">ROC curve data not available</div>';
            return;
        }
        
        // Create ROC curve visualization using Plotly
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
        
        // Create feature importance visualization using Plotly
        const data = [{
            y: sortedFeatures.slice(0, 10), // Show top 10 features
            x: sortedImportance.slice(0, 10),
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
                
                // Create bar chart
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
            })
            .catch(error => {
                console.error('Error fetching model comparison data:', error);
                document.getElementById('performance-chart').innerHTML = 
                    `<div class="alert alert-danger">Error loading performance data: ${error.message}</div>`;
            });
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
        
        // Update chart
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
                Plotly.newPlot('roc-curves-chart', traces, {
                    title: 'ROC Curves Comparison',
                    xaxis: { title: 'False Positive Rate' },
                    yaxis: { title: 'True Positive Rate' },
                    legend: { x: 0.6, y: 0.2 }
                }, { responsive: true });
            })
            .catch(error => {
                console.error('Error fetching ROC curves data:', error);
                document.getElementById('roc-curves-chart').innerHTML = 
                    `<div class="alert alert-danger">Error loading ROC curves: ${error.message}</div>`;
            });
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
                
                // Create chart
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
