import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

from data_processing import preprocess_data, split_data
from ml_model import train_model, predict, evaluate_model
from data_visualization import plot_correlation_matrix, plot_feature_importance, plot_moisture_by_features

# Set page configuration
st.set_page_config(
    page_title="Irrigation Control System",
    page_icon="💧",
    layout="wide"
)

# Main title
st.title("🌱 Smart Irrigation Control System")
st.subheader("Using Machine Learning to Optimize Water Usage")

# Introduction
with st.expander("ℹ️ About This Application", expanded=True):
    st.markdown("""
    This application helps in predicting when to open a water pump based on environmental data:
    * **Crop Type**: Type of crop being monitored
    * **Temperature**: Ambient temperature (°C)
    * **Moisture Level**: Moisture level in soil
    * **Pump Status**: Whether the pump should be ON (1) or OFF (0)
    
    Upload your training data, train a machine learning model, and get predictions on when to open or close your irrigation pump.
    """)

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload & Training", "Real-time Prediction", "Model Performance"])

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'precision' not in st.session_state:
    st.session_state.precision = None
if 'recall' not in st.session_state:
    st.session_state.recall = None
if 'f1' not in st.session_state:
    st.session_state.f1 = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Function to load and process data
def load_data(uploaded_file=None, use_sample=False):
    try:
        if use_sample and os.path.exists('data/data.csv'):
            # Load sample data
            df = pd.read_csv('data/data.csv')
            return df
        elif uploaded_file is not None:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_columns = ['crop', 'moisture', 'temp', 'pump']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
                
            # Check if pump column has the right values
            pump_values = df['pump'].unique()
            valid_pump = all(pump in [0, 1] for pump in pump_values)
            
            if not valid_pump:
                st.error("Pump column should only contain 0 or 1 values.")
                st.stop()
                
            # Process data
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data Upload & Training Page
if page == "Data Upload & Training":
    st.header("Data Upload & Model Training")
    
    use_sample_data = st.checkbox("Use sample dataset", value=True)
    
    uploaded_file = None
    if not use_sample_data:
        uploaded_file = st.file_uploader("Upload your CSV training dataset", type=["csv"])
    
    sample_format = pd.DataFrame({
        'crop': ['cotton', 'cotton', 'cotton', 'cotton'],
        'moisture': [750, 650, 350, 200],
        'temp': [25, 30, 22, 28],
        'pump': [1, 1, 0, 0]
    })
    
    with st.expander("📝 View Sample Data Format"):
        st.write(sample_format)
        buffer = io.StringIO()
        sample_format.to_csv(buffer, index=False)
        st.download_button(
            label="Download Sample CSV",
            data=buffer.getvalue(),
            file_name="sample_irrigation_data.csv",
            mime="text/csv"
        )
    
    # Load data based on user choice
    df = None
    if use_sample_data:
        df = load_data(use_sample=True)
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
    
    if df is not None:
        st.session_state.df = df
        st.success("Data successfully loaded!")
        
        st.subheader("Preview of Dataset")
        st.write(df.head())
        
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        # Data Visualization
        st.subheader("Data Visualization")
        
        # Display correlation matrix
        st.write("Correlation Matrix:")
        fig_corr = plot_correlation_matrix(df)
        st.pyplot(fig_corr)
        
        # Display moisture vs temperature scatter plot
        st.write("Moisture vs Temperature:")
        fig_scatter = plot_moisture_by_features(df)
        st.pyplot(fig_scatter)
        
        # Train Model Section
        st.subheader("Train Model")
        
        # Model selection
        model_type = st.selectbox(
            "Select Machine Learning Algorithm:",
            ["Random Forest", "Decision Tree", "Logistic Regression"]
        )
        
        # Split ratio selector
        test_size = st.slider("Test Data Size (%)", 10, 50, 20)
        
        # Training button
        if st.button("Train Model"):
            with st.spinner("Training model... Please wait."):
                # Preprocess data
                X, y, feature_names = preprocess_data(df)
                
                # Split data
                X_train, X_test, y_train, y_test = split_data(X, y, test_size/100)
                
                # Save split data to session state
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                
                # Train model
                model, feature_importance = train_model(X_train, y_train, model_type)
                
                # Save model to session state
                st.session_state.model = model
                st.session_state.feature_importance = feature_importance
                
                # Evaluate model
                accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
                
                # Save metrics to session state
                st.session_state.accuracy = accuracy
                st.session_state.precision = precision
                st.session_state.recall = recall
                st.session_state.f1 = f1
                
            st.success("Model trained successfully!")
            
            # Display model performance
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{precision:.2f}")
            col3.metric("Recall", f"{recall:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")
            
            # Display feature importance
            st.write("Feature Importance:")
            fig_importance = plot_feature_importance(feature_importance, feature_names)
            st.pyplot(fig_importance)

# Real-time Prediction Page
elif page == "Real-time Prediction":
    st.header("Real-time Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model first!")
        st.sidebar.info("Go to 'Data Upload & Training' to train your model.")
    else:
        st.success("Model is ready for prediction!")
        
        st.subheader("Enter Environmental Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=25, step=1)
        
        with col2:
            moisture = st.number_input("Moisture Level", min_value=0, max_value=1200, value=600, step=10, 
                                       help="Higher values indicate more moisture, lower values indicate drier soil")
        
        if st.button("Predict Pump Status"):
            # Create input data
            input_data = pd.DataFrame({
                'moisture': [moisture],
                'temp': [temperature]
            })
            
            # Make prediction
            prediction, probability = predict(st.session_state.model, input_data)
            
            # Display prediction
            st.subheader("Prediction Result")
            
            # Create two columns for the prediction result
            prediction_col, gauge_col = st.columns([1, 2])
            
            with prediction_col:
                if prediction == 'on':
                    st.error("⚠️ Low moisture detected")
                    st.success("✅ Recommendation: OPEN the water pump")
                else:
                    st.success("💦 Adequate moisture detected")
                    st.info("❌ Recommendation: CLOSE the water pump")
                
                st.write(f"Confidence: {probability:.2f}")
            
            with gauge_col:
                # Create a simple gauge chart for the moisture
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Define the gauge colors
                colors = ['#FF9999', '#FFCC99', '#FFFF99', '#99FF99', '#99CCFF']
                
                # Define gauge thresholds
                thresholds = [0, 200, 400, 600, 800, 1000]
                
                # Draw gauge segments
                for i in range(len(thresholds) - 1):
                    ax.barh(0, thresholds[i+1] - thresholds[i], left=thresholds[i], height=0.5, color=colors[i])
                
                # Add moisture marker
                ax.plot(moisture, 0, 'ko', markersize=10)
                
                # Add threshold labels
                for t in thresholds:
                    ax.text(t, -0.2, str(t), ha='center', va='center')
                
                # Configure plot
                ax.set_xlim(0, 1000)
                ax.set_ylim(-0.5, 0.5)
                ax.set_title('Moisture Level Gauge')
                ax.set_xlabel('Moisture')
                ax.set_yticks([])
                
                # Display the chart
                st.pyplot(fig)
            
            # Provide context for the prediction
            if st.session_state.df is not None:
                st.subheader("Context for This Prediction")
                
                # Find similar records in training data
                df = st.session_state.df
                
                # Calculate Euclidean distance for each record
                df['distance'] = np.sqrt(
                    (df['temp'] - temperature) ** 2 + 
                    (df['moisture'] - moisture) ** 2
                )
                
                # Get the 5 most similar records
                similar_records = df.sort_values('distance').head(5).drop('distance', axis=1)
                
                st.write("Similar conditions from the training data:")
                st.write(similar_records)

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance & Analysis")
    
    if st.session_state.model is None:
        st.warning("Please train a model first!")
        st.sidebar.info("Go to 'Data Upload & Training' to train your model.")
    else:
        # Display model metrics
        st.subheader("Model Evaluation Metrics")
        
        # Create a metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Accuracy", f"{st.session_state.accuracy:.2f}")
        col2.metric("Precision", f"{st.session_state.precision:.2f}")
        col3.metric("Recall", f"{st.session_state.recall:.2f}")
        col4.metric("F1 Score", f"{st.session_state.f1:.2f}")
        
        # Add explanation for metrics
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            - **Accuracy**: The proportion of correct predictions among the total number of predictions.
            - **Precision**: The proportion of true positive predictions among all positive predictions (how many predicted 'pump on' cases were actually correct).
            - **Recall**: The proportion of true positive predictions among all actual positive cases (how many actual 'pump on' cases were correctly predicted).
            - **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
            """)
        
        # Display feature importance
        st.subheader("Feature Importance")
        fig_importance = plot_feature_importance(
            st.session_state.feature_importance, 
            st.session_state.feature_names
        )
        st.pyplot(fig_importance)
        
        # Explain feature importance
        with st.expander("Understanding Feature Importance"):
            st.markdown("""
            Feature importance indicates how much each factor contributes to the prediction:
            - Higher values mean the feature has more influence on the model's decision
            - This can help you understand which environmental factors most affect pump activation decisions
            """)
        
        # Optional: Add more advanced visualizations if data is available
        if st.session_state.df is not None:
            st.subheader("Data Distribution")
            
            # Display correlation matrix
            st.write("Correlation Matrix:")
            fig_corr = plot_correlation_matrix(st.session_state.df)
            st.pyplot(fig_corr)
            
            with st.expander("Understanding Correlation Matrix"):
                st.markdown("""
                The correlation matrix shows relationships between variables:
                - Values close to 1 indicate strong positive correlation
                - Values close to -1 indicate strong negative correlation
                - Values close to 0 indicate little to no correlation
                """)

# Footer
st.markdown("---")
st.markdown("💧 Smart Irrigation Control System - Using ML to Optimize Water Usage")
