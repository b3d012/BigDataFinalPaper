import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import os  # For checking and creating directories

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocessing: handle missing values, feature scaling, and encoding
def preprocess_data(df):
    # Print column names to debug and check for protocol column
    print("Columns in the dataset:", df.columns)

    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Handle missing values: impute numerical columns with median and categorical with mode
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical columns with the most frequent value

    # Exclude the timestamp column 'frame.time' from scaling (since it's not numeric)
    numeric_columns = [col for col in numeric_columns if col != 'frame.time']
    
    # Check if 'protocol' or similar columns exist
    if 'protocol' not in df.columns:
        print("No 'protocol' column found. Checking for alternative protocol-related columns.")
        # List other potential protocol-related columns
        alternative_protocol_columns = ['mqtt.protoname', 'tcp.srcport', 'udp.port']  # Example: Modify with your own column names
        
        # If protocol-related columns are found, use those for OneHotEncoding
        categorical_features = alternative_protocol_columns if any(col in df.columns for col in alternative_protocol_columns) else []
        print(f"Using these columns for categorical encoding: {categorical_features}")
    else:
        categorical_features = ['protocol']  # If 'protocol' exists, use it for encoding
    
    # Convert all non-numeric columns (e.g., 'mqtt.protoname', 'tcp.srcport', 'udp.port') to strings
    for col in categorical_features:
        df[col] = df[col].astype(str)  # Ensure all values are strings for one-hot encoding

    # Extract relevant IoT features (modify as needed for your protocol-specific features)
    features = df[['frame.time', 'ip.src_host', 'ip.dst_host', 'tcp.srcport', 'tcp.dstport', 'udp.port', 'tcp.flags']]
    
    # Apply OneHotEncoding only if categorical features exist
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),  # Now exclude 'frame.time' from scaling
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Apply preprocessing to the data
    df_processed = preprocessor.fit_transform(df)
    
    return df_processed, preprocessor

# Handle class imbalance using SMOTE
def handle_imbalance(X, y):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# Split the data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Main function to load data, process it, and save the result
def main():
    # Load the dataset
    file_path = 'data/ML-EdgeIIoT-dataset.csv'
    df = load_data(file_path)
    
    # Extract features and labels
    X = df.drop(columns=['Attack_label'])  # Features (exclude target label)
    y = df['Attack_label']  # Target label
    
    # Preprocess the data
    X_processed, preprocessor = preprocess_data(X)
    
    # Handle class imbalance using SMOTE
    X_resampled, y_resampled = handle_imbalance(X_processed, y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    
    # Ensure the 'models' directory exists before saving the preprocessor
    if not os.path.exists('models'):
        os.makedirs('models')  # Create the 'models' directory if it doesn't exist
    
    # Save the processed data
    pd.DataFrame(X_resampled).to_csv('data/processed_data.csv', index=False)
    
    # Save preprocessor for later use
    import joblib
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    print("Feature Engineering complete. Processed data saved to 'data/processed_data.csv'.")

if __name__ == '__main__':
    main()