# -*- coding: utf-8 -*-
"""
Dashboard Module

This module is responsible for creating the dashboard visualizations for the Titanic dataset.
"""

from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_processing.data_loader import DataLoader


def create_dashboard_layout():
    """
    Create the layout for the data exploration dashboard.
    
    Returns:
        dash_html_components.Div: The dashboard layout.
    """
    # Try to load data
    loader = DataLoader()
    df = loader.load_train_data()
    
    if df is None:
        return html.Div([
            html.H3("Data Exploration"),
            html.P("Error: Could not load data. Please make sure the training data is available.")
        ])
    
    # Create dashboard layout
    layout = html.Div([
        html.H3("Data Exploration"),
        
        # Basic dataset information
        html.Div([
            html.H4("Dataset Information"),
            html.P(f"Number of passengers: {len(df)}"),
            html.P(f"Number of features: {len(df.columns) - 1}"),  # Excluding target variable
            html.P(f"Survival rate: {df['Survived'].mean():.2%}"),
        ]),
        
        # Placeholder for visualizations
        html.Div([
            html.H4("Visualizations"),
            html.P("Visualizations will be shown here when implemented.")
        ])
    ])
    
    return layout


def create_survival_by_feature_plot(df, feature_name):
    """
    Create a plot showing survival rates by a specific feature.
    
    Args:
        df (pandas.DataFrame): The dataset.
        feature_name (str): The feature to analyze.
        
    Returns:
        plotly.graph_objects.Figure: The plot figure.
    """
    # Placeholder function for creating survival by feature plots
    # Actual implementation would depend on the feature type (categorical/numeric)
    
    # Example for categorical feature
    if df[feature_name].dtype == 'object' or len(df[feature_name].unique()) < 10:
        survival_by_feature = df.groupby(feature_name)['Survived'].mean().reset_index()
        fig = px.bar(
            survival_by_feature, 
            x=feature_name, 
            y='Survived',
            title=f'Survival Rate by {feature_name}',
            labels={'Survived': 'Survival Rate'}
        )
    # Example for numeric feature
    else:
        fig = px.histogram(
            df, 
            x=feature_name, 
            color='Survived',
            barmode='group',
            title=f'Distribution of {feature_name} by Survival',
            labels={'Survived': 'Survived'}
        )
    
    return fig
