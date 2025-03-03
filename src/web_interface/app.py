# -*- coding: utf-8 -*-
"""
Main Web Application Module

This module is the entry point for the web application interface of the Titanic survival prediction project.
"""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

from src.web_interface.model_interface import ModelInterface
from src.web_interface.dashboard import create_dashboard_layout


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For deployment

# Initialize model interface
model_interface = ModelInterface()

# Define app layout
app.layout = html.Div([
    html.H1("Titanic Survival Prediction"),
    
    # Tabs for different sections of the app
    dcc.Tabs(id='tabs', value='tab-data-exploration', children=[
        dcc.Tab(label='Data Exploration', value='tab-data-exploration'),
        dcc.Tab(label='Model Training', value='tab-model-training'),
        dcc.Tab(label='Model Evaluation', value='tab-model-evaluation'),
        dcc.Tab(label='Prediction', value='tab-prediction'),
    ]),
    
    # Content of the tabs will be rendered here
    html.Div(id='tabs-content')
])


@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_tabs_content(tab):
    """
    Render content based on the selected tab.
    
    Args:
        tab (str): The tab value.
        
    Returns:
        dash_html_components.Div: The content for the selected tab.
    """
    if tab == 'tab-data-exploration':
        return create_dashboard_layout()
    elif tab == 'tab-model-training':
        return html.Div([html.H3("Model Training")]) # Placeholder
    elif tab == 'tab-model-evaluation':
        return html.Div([html.H3("Model Evaluation")]) # Placeholder
    elif tab == 'tab-prediction':
        return html.Div([html.H3("Prediction")]) # Placeholder


# Main entry point
if __name__ == '__main__':
    app.run_server(debug=True)
