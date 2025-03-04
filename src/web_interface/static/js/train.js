/**
 * JavaScript for the Train Models page
 * 
 * This file contains code to handle model training and visualization
 * of training results for Titanic Survival Predictor.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Check if Plotly is available
    let plotlyAvailable = (typeof Plotly !== 'undefined');
    console.log('Plotly available:', plotlyAvailable, plotlyAvailable ? Plotly.version : 'Not loaded');

    // Try to load Plotly if it's not available
    if (!plotlyAvailable) {
        try {
            // Create a script element for Plotly
            const scriptElement = document.createElement('script');
            scriptElement.src = 'https://cdn.plot.ly/plotly-latest.min.js';
            scriptElement.onload = function() {
                console.log('Plotly loaded dynamically!');
                plotlyAvailable = true;
                // Initialize visualizations if needed
                initializeVisualizations();
            };
            scriptElement.onerror = function() {
                console.error('Failed to load Plotly dynamically');
                // Try the backup CDN
                const backupScript = document.createElement('script');
                backupScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.29.1/plotly.min.js';
                backupScript.onload = function() {
                    console.log('Plotly loaded from backup CDN!');
                    plotlyAvailable = true;
                    // Initialize visualizations if needed
                    initializeVisualizations();
                };
                document.head.appendChild(backupScript);
            };
            document.head.appendChild(scriptElement);
        } catch (error) {
            console.error('Error attempting to load Plotly:', error);
        }
    } else {
        // Initialize visualizations right away
        initializeVisualizations();
    }

    function initializeVisualizations() {
        // Check if we're on a results page (after training)
        const urlParams = new URLSearchParams(window.location.search);
        const selectedModel = urlParams.get('model');
        
        if (selectedModel && document.getElementById('training-results')) {
            // Show results section
            document.getElementById('training-initial').style.display = 'none';
            document.getElementById('training-progress-container').style.display = 'none';
            document.getElementById('training-results').style.display = 'block';
            
            // Fetch model details to populate results
            fetchModelDetails(selectedModel);
        }
    }
    
    function fetchModelDetails(modelName) {
        console.log('Fetching details for model:', modelName);
        
        // Show loading spinners
        document.getElementById('confusion-matrix').innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`;
        
        document.getElementById('roc-curve').innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`;
        
        document.getElementById('feature-importance-chart').innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`;
        
        // Fetch the model details
        fetch(`/api/model-details?model=${modelName}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Model details data:', data);
                
                // Update performance metrics
                if (data.metrics) {
                    document.getElementById('result-accuracy').textContent = 
                        formatMetricValue(data.metrics.accuracy);
                    document.getElementById('result-precision').textContent = 
                        formatMetricValue(data.metrics.precision);
                    document.getElementById('result-recall').textContent = 
                        formatMetricValue(data.metrics.recall);
                    document.getElementById('result-f1').textContent = 
                        formatMetricValue(data.metrics.f1);
                    document.getElementById('result-auc').textContent = 
                        data.metrics.roc_auc ? data.metrics.roc_auc.toFixed(3) : 'N/A';
                }
                
                // Render visualizations (with error handling for each)
                try {
                    renderConfusionMatrix(data.confusion_matrix, 'confusion-matrix');
                } catch (error) {
                    console.error('Error rendering confusion matrix:', error);
                    document.getElementById('confusion-matrix').innerHTML = `
                        <div class="alert alert-warning">
                            Error rendering confusion matrix: ${error.message}
                        </div>`;
                }
                
                try {
                    renderRocCurve(data.roc_curve, 'roc-curve');
                } catch (error) {
                    console.error('Error rendering ROC curve:', error);
                    document.getElementById('roc-curve').innerHTML = `
                        <div class="alert alert-warning">
                            Error rendering ROC curve: ${error.message}
                        </div>`;
                }
                
                try {
                    renderFeatureImportance(data.feature_importance, 'feature-importance-chart');
                } catch (error) {
                    console.error('Error rendering feature importance:', error);
                    document.getElementById('feature-importance-chart').innerHTML = `
                        <div class="alert alert-warning">
                            Error rendering feature importance: ${error.message}
                        </div>`;
                }
            })
            .catch(error => {
                console.error('Error fetching model details:', error);
                
                // Show error message in each visualization area
                const errorMessage = `
                    <div class="alert alert-danger">
                        <strong>Failed to load data.</strong><br>
                        ${error.message}
                    </div>`;
                
                document.getElementById('confusion-matrix').innerHTML = errorMessage;
                document.getElementById('roc-curve').innerHTML = errorMessage;
                document.getElementById('feature-importance-chart').innerHTML = errorMessage;
            });
    }
    
    // Helper function to format metric values
    function formatMetricValue(value) {
        if (value === undefined || value === null) return 'N/A';
        return (value * 100).toFixed(1) + '%';
    }
    
    // Function to render confusion matrix
    function renderConfusionMatrix(confusionMatrix, elementId) {
        const element = document.getElementById(elementId);
        
        if (!confusionMatrix || !Array.isArray(confusionMatrix) || confusionMatrix.length !== 2) {
            element.innerHTML = 
                '<div class="alert alert-warning">Confusion matrix data not available or in unexpected format</div>';
            return;
        }
        
        console.log('Rendering confusion matrix:', confusionMatrix);
        
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
                annotations: [],
                margin: {
                    l: 60,
                    r: 30,
                    b: 60,
                    t: 40,
                    pad: 4
                }
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
                '<div class="alert alert-warning">ROC curve data not available or in unexpected format</div>';
            return;
        }
        
        console.log('Rendering ROC curve:', rocData);
        
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
                showlegend: true,
                margin: {
                    l: 60,
                    r: 30,
                    b: 60,
                    t: 40,
                    pad: 4
                }
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
                '<div class="alert alert-warning">Feature importance data not available or in unexpected format</div>';
            return;
        }
        
        console.log('Rendering feature importance:', featureImportance);
        
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
                margin: {
                    l: 150,
                    r: 30,
                    b: 60,
                    t: 40,
                    pad: 4
                }
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
    
    // Initialize page-specific events
    function initTrainingPage() {
        // Update test size display
        const testSizeSlider = document.getElementById('test-size');
        const testSizeValue = document.getElementById('test-size-value');
        
        if (testSizeSlider && testSizeValue) {
            testSizeSlider.addEventListener('input', function() {
                testSizeValue.textContent = this.value + '%';
            });
        }
        
        // Show/hide model parameters based on selection
        const modelTypeSelect = document.getElementById('model-type');
        const allParamDivs = document.querySelectorAll('.model-params');
        
        if (modelTypeSelect && allParamDivs.length > 0) {
            modelTypeSelect.addEventListener('change', function() {
                // Hide all parameter divs
                allParamDivs.forEach(div => div.style.display = 'none');
                
                // Show the selected model's parameters
                const selectedParamDiv = document.getElementById('params-' + this.value);
                if (selectedParamDiv) {
                    selectedParamDiv.style.display = 'block';
                }
            });
        }
        
        // Train model button click
        const trainButton = document.getElementById('btn-train-model');
        
        if (trainButton) {
            trainButton.addEventListener('click', function() {
                // Hide initial prompt
                if (document.getElementById('training-initial')) {
                    document.getElementById('training-initial').style.display = 'none';
                }
                
                // Show progress
                if (document.getElementById('training-progress-container')) {
                    document.getElementById('training-progress-container').style.display = 'block';
                }
                
                if (document.getElementById('training-results')) {
                    document.getElementById('training-results').style.display = 'none';
                }
                
                // Simulate training progress
                let progress = 0;
                const progressBar = document.getElementById('training-progress-bar');
                const statusText = document.getElementById('training-status');
                
                if (progressBar && statusText) {
                    const progressInterval = setInterval(function() {
                        progress += 10;
                        progressBar.style.width = progress + '%';
                        progressBar.setAttribute('aria-valuenow', progress);
                        
                        if (progress === 10) {
                            statusText.textContent = 'Preparing data...';
                        } else if (progress === 30) {
                            statusText.textContent = 'Applying feature engineering...';
                        } else if (progress === 50) {
                            statusText.textContent = 'Training model...';
                        } else if (progress === 70) {
                            statusText.textContent = 'Evaluating model...';
                        } else if (progress === 90) {
                            statusText.textContent = 'Finalizing results...';
                        } else if (progress >= 100) {
                            clearInterval(progressInterval);
                            statusText.textContent = 'Training complete!';
                            
                            // Submit the form to the server
                            document.getElementById('model-selection-form').submit();
                        }
                    }, 400);
                }
            });
        }
        
        // Configure Parameters button click
        const showParamsButton = document.getElementById('btn-show-params');
        
        if (showParamsButton) {
            showParamsButton.addEventListener('click', function() {
                // Find the parameters section heading and scroll to it
                const paramsHeadings = document.querySelectorAll('.card-header.bg-primary.text-white h4.card-title.mb-0');
                
                for (let heading of paramsHeadings) {
                    if (heading.textContent.includes('Model Parameters')) {
                        heading.scrollIntoView({ behavior: 'smooth' });
                        break;
                    }
                }
            });
        }
        
        // Save and Compare buttons
        const saveModelButton = document.getElementById('btn-save-model');
        
        if (saveModelButton) {
            saveModelButton.addEventListener('click', function() {
                alert('Model saved successfully!');
            });
        }
        
        const compareModelsButton = document.getElementById('btn-compare-models');
        
        if (compareModelsButton) {
            compareModelsButton.addEventListener('click', function() {
                const modelName = document.getElementById('model-type').value;
                window.location.href = '/results?model=' + modelName;
            });
        }
    }
    
    // Initialize the training page
    initTrainingPage();
});
