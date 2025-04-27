import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the input dataframe for machine learning
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data containing crop, moisture, temperature, and pump status
        
    Returns:
    --------
    X : array-like
        Features for machine learning
    y : array-like
        Target variable
    feature_names : list
        Names of features
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert column names to lowercase if they aren't already
    df_copy.columns = [col.lower() for col in df_copy.columns]
    
    # Convert pump status to string labels for easier interpretation
    # 1 = "on" (open the pump), 0 = "off" (close the pump)
    df_copy['pump_status'] = df_copy['pump'].map({1: 'on', 0: 'off'})
    
    # Extract features and target
    X = df_copy[['moisture', 'temp']]
    y = df_copy['pump_status']
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy arrays
    X = X.values
    
    return X, y, feature_names

def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    X_train : array-like
    X_test : array-like
    y_train : array-like
    y_test : array-like
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
