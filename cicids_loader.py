"""
CIC-IDS Dataset Loader for Network Anomaly Detection
Supports CIC-IDS-2017 and CIC-IDS-2018 datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class CICIDSLoader:
    """
    Load and prepare CIC-IDS dataset for anomaly detection
    """
    
    def __init__(self):
        self.label_column = 'Label'  # Default label column name
        
    def load_from_csv(self, file_path):
        """
        Load CIC-IDS dataset from CSV file
        
        Args:
            file_path: Path to CSV file (or folder with multiple CSVs)
        
        Returns:
            DataFrame with features and labels
        """
        print(f"\n📊 Loading CIC-IDS Dataset from: {file_path}")
        
        # Check if it's a folder or file
        if os.path.isdir(file_path):
            # Load all CSV files in folder
            all_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
            print(f"   Found {len(all_files)} CSV files")
            
            dfs = []
            for file in all_files:
                full_path = os.path.join(file_path, file)
                print(f"   Loading: {file}...")
                df = pd.read_csv(full_path)
                dfs.append(df)
            
            data = pd.concat(dfs, ignore_index=True)
            print(f"   ✓ Combined data: {data.shape}")
            
        else:
            # Single file
            data = pd.read_csv(file_path)
            print(f"   ✓ Loaded: {data.shape}")
        
        return data
    
    def load_from_parquet(self, file_path):
        """
        Load CIC-IDS dataset from Parquet file (V2 format)
        
        Args:
            file_path: Path to parquet file
        
        Returns:
            DataFrame with features and labels
        """
        print(f"\n📊 Loading CIC-IDS Dataset from: {file_path}")
        
        data = pd.read_parquet(file_path)
        print(f"   ✓ Loaded: {data.shape}")
        
        return data
    
    def preprocess_cicids(self, data, sample_size=None):
        """
        Preprocess CIC-IDS data for anomaly detection
        
        Args:
            data: Raw DataFrame
            sample_size: Optional - sample N rows for faster processing
        
        Returns:
            X (features), y (labels), label_mapping
        """
        print("\n🔧 Preprocessing CIC-IDS data...")
        
        # Sample if requested
        if sample_size and len(data) > sample_size:
            print(f"   Sampling {sample_size} rows from {len(data)} total...")
            data = data.sample(n=sample_size, random_state=42)
        
        # Find the label column (different names in different versions)
        possible_label_cols = ['Label', ' Label', 'label', 'Attack', 'attack_cat']
        label_col = None
        
        for col in possible_label_cols:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Could not find label column. Available columns: " + str(data.columns.tolist()))
        
        print(f"   Using label column: '{label_col}'")
        
        # Display label distribution
        print(f"\n   Label distribution:")
        label_counts = data[label_col].value_counts()
        for label, count in label_counts.head(10).items():
            print(f"      {label}: {count}")
        
        # Create binary labels: BENIGN=0, any attack=1
        benign_labels = ['BENIGN', 'Benign', 'benign', 'Normal', 'normal']
        data['binary_label'] = data[label_col].apply(
            lambda x: 0 if x in benign_labels else 1
        )
        
        normal_count = (data['binary_label'] == 0).sum()
        attack_count = (data['binary_label'] == 1).sum()
        print(f"\n   Binary distribution:")
        print(f"      Normal: {normal_count} ({normal_count/len(data)*100:.1f}%)")
        print(f"      Attack: {attack_count} ({attack_count/len(data)*100:.1f}%)")
        
        # Separate features and labels
        y = data['binary_label']
        X = data.drop(columns=[label_col, 'binary_label'])
        
        # Handle special columns
        # Remove timestamp/ID columns if present
        drop_cols = []
        for col in X.columns:
            if 'Timestamp' in col or 'timestamp' in col or 'Flow ID' in col:
                drop_cols.append(col)
        
        if drop_cols:
            print(f"   Dropping timestamp/ID columns: {drop_cols}")
            X = X.drop(columns=drop_cols)
        
        # Handle infinity and missing values
        print(f"   Handling missing/infinite values...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with 0 (or could use median)
        X = X.fillna(0)
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"   Converting '{col}' to numeric...")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        print(f"\n   ✓ Final feature shape: {X.shape}")
        print(f"   ✓ Features: {X.shape[1]}")
        
        # Create label mapping for reference
        label_mapping = {
            0: 'BENIGN',
            1: 'ATTACK'
        }
        
        return X, y, label_mapping
    
    def create_train_test_split(self, X, y, test_size=0.3, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            random_state: Random seed
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Keep label distribution balanced
        )
        from sklearn.utils import resample
    
    # Separate normal and attack samples
        normal_idx = y_train[y_train == 0].index
        attack_idx = y_train[y_train == 1].index
    
    # Undersample normal to 2x the number of attacks
        target_normal = len(attack_idx) * 2
        normal_idx_balanced = resample(normal_idx, 
                                   n_samples=target_normal, 
                                   random_state=random_state)
    
    # Combine balanced normal + all attacks
        balanced_idx = list(normal_idx_balanced) + list(attack_idx)
    
        X_train_balanced = X_train.loc[balanced_idx]
        y_train_balanced = y_train.loc[balanced_idx]
    
        print(f"   ✓ Balanced training set: {X_train_balanced.shape}")
        print(f"   ✓ Normal: {(y_train_balanced == 0).sum()}")
        print(f"   ✓ Attack: {(y_train_balanced == 1).sum()}")
        print(f"   ✓ Test set: {X_test.shape}")
    
        return X_train_balanced, X_test, y_train_balanced, y_test
        
        '''
        print(f"   ✓ Training set: {X_train.shape}")
        print(f"   ✓ Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        '''


# Quick usage example
if __name__ == "__main__":
    print("="*70)
    print("📊 CIC-IDS DATASET LOADER")
    print("="*70)
    
    loader = CICIDSLoader()
    
    print("\nUsage Examples:")
    print("\n1. Load from CSV:")
    print("   data = loader.load_from_csv('path/to/dataset.csv')")
    print("   # or for folder with multiple CSVs:")
    print("   data = loader.load_from_csv('path/to/dataset_folder/')")
    
    print("\n2. Load from Parquet (V2):")
    print("   data = loader.load_from_parquet('path/to/dataset.parquet')")
    
    print("\n3. Preprocess:")
    print("   X, y, mapping = loader.preprocess_cicids(data, sample_size=50000)")
    
    print("\n4. Split:")
    print("   X_train, X_test, y_train, y_test = loader.create_train_test_split(X, y)")
    
    print("\n5. Use with detector:")
    print("   from network_anomaly_detector import NetworkAnomalyDetector")
    print("   detector = NetworkAnomalyDetector()")
    print("   detector.train_models(X_train, y_train)")
    print("   detector.evaluate_models(X_test, y_test)")
    
    print("\n" + "="*70)
