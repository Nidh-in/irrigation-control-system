import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_correlation_matrix(df):
    """
    Create a correlation matrix plot for the dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe containing the data
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure object containing the plot
    """
    # Create a copy of the dataframe with numerical columns only
    numeric_df = df.copy()
    
    # Convert pump to numeric (already 0 or 1)
    if 'pump_status' in numeric_df.columns:
        numeric_df['pump_numeric'] = numeric_df['pump_status'].map({'on': 1, 'off': 0})
    else:
        numeric_df['pump_numeric'] = numeric_df['pump']
    
    # Select only numeric columns
    numeric_df = numeric_df.select_dtypes(include=['number'])
    
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Matrix')
    
    return fig

def plot_feature_importance(feature_importance, feature_names):
    """
    Create a bar plot of feature importance
    
    Parameters:
    -----------
    feature_importance : array-like
        Feature importance values
    feature_names : list
        Names of features
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure object containing the plot
    """
    # Create a dataframe for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    return fig

def plot_moisture_by_features(df):
    """
    Create scatter plots of soil moisture vs temperature
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe containing the data
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure object containing the plots
    """
    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for pump status
    colors = {1: 'green', 0: 'red'}
    
    # Scatter plot for temperature vs moisture
    scatter = ax.scatter(df['temp'], df['moisture'], c=df['pump'].map(colors), alpha=0.7)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Moisture Level')
    ax.set_title('Temperature vs Moisture Level')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Pump ON'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Pump OFF')
    ]
    ax.legend(handles=legend_elements)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
