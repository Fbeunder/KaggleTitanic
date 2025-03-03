/**
 * Debugging script for visualization issues
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Debugging visualization script loaded');
    
    // Check if Plotly is available
    if (typeof Plotly !== 'undefined') {
        console.log('Plotly is loaded correctly', Plotly.version);
    } else {
        console.error('Plotly is not loaded!');
    }

    // Test API endpoints
    testModelDetailsEndpoint();
    testFeatureImportanceEndpoint();
    testRocCurvesEndpoint();
    testModelComparisonEndpoint();

    // Add test visualization button
    const debugButton = document.createElement('button');
    debugButton.textContent = 'Debug Visualizations';
    debugButton.className = 'btn btn-warning mt-3';
    debugButton.addEventListener('click', function() {
        testVisualizations();
    });
    
    // Add the button to the page if on results page
    if (document.getElementById('performance-chart')) {
        document.getElementById('performance-chart').parentNode.appendChild(debugButton);
    }
});

// Test the model-details API endpoint
function testModelDetailsEndpoint() {
    const modelName = 'gradient_boosting';
    fetch(`/api/model-details?model=${modelName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API /api/model-details response:', data);
            // Check for required visualization data
            if (!data.confusion_matrix) {
                console.error('Missing confusion_matrix data in API response');
            }
            if (!data.roc_curve) {
                console.error('Missing roc_curve data in API response');
            }
            if (!data.feature_importance) {
                console.error('Missing feature_importance data in API response');
            }
        })
        .catch(error => {
            console.error('Error testing model-details API:', error);
        });
}

// Test the feature-importance API endpoint
function testFeatureImportanceEndpoint() {
    const modelName = 'gradient_boosting';
    fetch(`/api/feature-importance?model=${modelName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API /api/feature-importance response:', data);
        })
        .catch(error => {
            console.error('Error testing feature-importance API:', error);
        });
}

// Test the roc-curves API endpoint
function testRocCurvesEndpoint() {
    const models = 'gradient_boosting,random_forest';
    fetch(`/api/roc-curves?models=${models}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API /api/roc-curves response:', data);
        })
        .catch(error => {
            console.error('Error testing roc-curves API:', error);
        });
}

// Test the model-comparison API endpoint
function testModelComparisonEndpoint() {
    fetch('/api/model-comparison')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API /api/model-comparison response:', data);
        })
        .catch(error => {
            console.error('Error testing model-comparison API:', error);
        });
}

// Test creating visualizations directly
function testVisualizations() {
    console.log('Testing direct visualization creation');
    
    // Test area for visualizations
    const testDiv = document.createElement('div');
    testDiv.id = 'test-visualization';
    testDiv.style.height = '300px';
    testDiv.style.width = '100%';
    testDiv.style.marginTop = '20px';
    testDiv.style.border = '1px solid #ccc';
    testDiv.style.padding = '10px';
    
    // Add it to the page
    document.body.appendChild(testDiv);
    
    // Create a simple test plot
    const data = [{
        x: [1, 2, 3, 4, 5],
        y: [1, 2, 4, 8, 16],
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Test Data'
    }];
    
    const layout = {
        title: 'Test Visualization',
        xaxis: { title: 'X Axis' },
        yaxis: { title: 'Y Axis' }
    };
    
    // Create the plot
    try {
        Plotly.newPlot('test-visualization', data, layout);
        console.log('Test visualization created successfully');
    } catch (error) {
        console.error('Error creating test visualization:', error);
    }
}
