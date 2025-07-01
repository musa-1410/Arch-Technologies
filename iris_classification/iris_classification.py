# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data(file_path='C:/Users/dellp/OneDrive - Higher Education Commission/Desktop/Arch Tech/Month 1/iris_classification/Iris.csv'):
    """Load Iris dataset from CSV file"""
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    """Clean data and separate features/target"""
    # Drop unnecessary column
    df = df.drop(columns=['Id'], errors='ignore')
    
    # Handle missing values (if any)
    if df.isnull().sum().any():
        df = df.fillna(df.mean(numeric_only=True))
    
    # Separate features (X) and target (y)
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Train model
def train_model(X_train, y_train):
    """Train Support Vector Machine classifier"""
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, report

# Main execution
if __name__ == "__main__":
    # Step 1: Load data
    print("Loading dataset...")
    df = load_data()
    
    # Step 2: Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Step 3: Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    print("Evaluating model...")
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print("\nTask completed successfully!")