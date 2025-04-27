# Smart Irrigation Control System

A machine learning application that predicts when to activate irrigation pumps based on environmental data.

## Overview

This Streamlit application helps farmers optimize water usage by predicting when to open or close water pumps based on soil moisture and temperature data. The system uses machine learning algorithms to learn the patterns from historical data and make intelligent recommendations.

## Features

- **Data Analysis**: Visualize and analyze your crop data with interactive charts
- **Model Training**: Train machine learning models using Random Forest, Decision Tree, or Logistic Regression
- **Real-time Prediction**: Get immediate recommendations on pump activation based on current conditions
- **Performance Analysis**: Evaluate model performance with key metrics (accuracy, precision, recall, F1 score)

## Sample Dataset Structure

The application expects data in the following format:

| crop    | moisture | temp | pump |
|---------|----------|------|------|
| cotton  | 750      | 25   | 1    |
| cotton  | 650      | 30   | 1    |
| cotton  | 350      | 22   | 0    |
| cotton  | 200      | 28   | 0    |

Where:
- **crop**: Type of crop (e.g., cotton)
- **moisture**: Soil moisture level
- **temp**: Temperature in °C
- **pump**: Target variable (1 = pump ON, 0 = pump OFF)

## Installation & Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install streamlit pandas numpy matplotlib scikit-learn seaborn
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## How to Use

1. Navigate to the "Data Upload & Training" tab
2. Upload your dataset or use the provided sample data
3. Explore data visualizations to understand patterns
4. Train a model using your preferred algorithm
5. Go to the "Real-time Prediction" tab to get recommendations based on current conditions
6. Check model performance in the "Model Performance" tab

## Technologies Used

- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

## License

This project is open-source and available under the MIT License.