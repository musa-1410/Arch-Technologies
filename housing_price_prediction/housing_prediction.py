import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel

# Load dataset
def load_data(file_path='C:/Users/dellp/OneDrive - Higher Education Commission/Desktop/Arch Tech/Month 1/housing_price_prediction/housing.csv'):
    """Load California housing dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please download it from Kaggle.")
        exit(1)

# Preprocess data
def preprocess_data(df):
    """Clean and preprocess data"""
    # Handle missing values
    if df.isnull().sum().any():
        imputer = SimpleImputer(strategy='median')
        df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])
    
    # Convert ocean_proximity to numerical values
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='op')
    
    # Separate features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    return X, y

# Feature selection
def select_features(X, y):
    """Select important features using RandomForest feature importance"""
    # Separate numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # Preprocess all features
    X_processed = preprocessor.fit_transform(X)
    
    # Train model to get feature importances
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_processed, y)
    
    # Select features
    selector = SelectFromModel(model, threshold='median', prefit=True)
    X_selected = selector.transform(X_processed)
    
    # Get feature names after preprocessing
    num_feature_names = num_cols.tolist()
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_feature_names + cat_feature_names
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [all_feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
    
    print("\nSelected Features:")
    print(selected_features)
    
    return X_selected, selected_features, preprocessor

# Main execution
if __name__ == "__main__":
    # Step 1: Load data
    print("Loading dataset...")
    df = load_data()
    
    # Step 2: Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Step 3: Feature selection
    print("Selecting important features...")
    X_selected, selected_features, preprocessor = select_features(X, y)
    
    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Step 5: Train model
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 6: Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Step 7: Show sample prediction
    sample_idx = np.random.randint(0, len(X_test))
    sample_pred = model.predict([X_test[sample_idx]])
    actual_value = y_test.iloc[sample_idx]
    
    print("\nSample Prediction:")
    print(f"Actual Price: ${actual_value:,.2f}")
    print(f"Predicted Price: ${sample_pred[0]:,.2f}")
    print(f"Difference: ${abs(actual_value - sample_pred[0]):,.2f}")
    
    print("\nTask completed successfully!")