import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X_train, y_train, model_type="Random Forest"):
    """
    Train a machine learning model on the given data
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target variable
    model_type : str
        Type of model to train (Random Forest, Decision Tree, or Logistic Regression)
        
    Returns:
    --------
    model : trained model object
    feature_importance : array of feature importance values
    """
    # Select model based on user choice
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Extract feature importance
    if model_type == "Logistic Regression":
        # For logistic regression, coefficients serve as feature importance
        feature_importance = np.abs(model.coef_[0])
    else:
        # For tree-based models
        feature_importance = model.feature_importances_
    
    return model, feature_importance

def predict(model, input_data):
    """
    Make a prediction using the trained model
    
    Parameters:
    -----------
    model : trained model object
    input_data : DataFrame
        Input data for prediction
        
    Returns:
    --------
    prediction : str
        Predicted class ('on' or 'off')
    probability : float
        Probability of the predicted class
    """
    # Make prediction
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]
    
    # Get probability of the predicted class
    if prediction == 'on':
        probability = prediction_proba[1]  # Probability of 'on' class
    else:
        probability = prediction_proba[0]  # Probability of 'off' class
    
    return prediction, probability

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    
    Parameters:
    -----------
    model : trained model object
    X_test : array-like
        Test features
    y_test : array-like
        Test target variable
        
    Returns:
    --------
    accuracy : float
    precision : float
    recall : float
    f1 : float
    """
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For binary classification metrics, specify the 'positive' class as 'on'
    # (since we're primarily interested in when to turn on the pump)
    precision = precision_score(y_test, y_pred, pos_label='on')
    recall = recall_score(y_test, y_pred, pos_label='on')
    f1 = f1_score(y_test, y_pred, pos_label='on')
    
    return accuracy, precision, recall, f1
