"""
Run Network Anomaly Detection with CIC-IDS Dataset
Handles all 8 parquet files from CIC-IDS-2017
"""

from cicids_loader import CICIDSLoader
from network_anomaly_detector import NetworkAnomalyDetector
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

def run_with_cicids_multiple_files(dataset_folder, sample_size=None):
    """
    Run anomaly detection on multiple CIC-IDS parquet files
    
    Args:
        dataset_folder: Path to folder containing parquet files
        sample_size: Number of samples to use per file (None = use all)
    """
    
    print("="*70)
    print("🛡️  NETWORK ANOMALY DETECTION - CIC-IDS DATASET")
    print("="*70)
    
    # Step 1: Find all parquet files
    print(f"\n📁 Scanning folder: {dataset_folder}")
    
    parquet_files = [
        'Benign-Monday-no-metadata.parquet',
        'Botnet-Friday-no-metadata.parquet',
        'Bruteforce-Tuesday-no-metadata.parquet',
        'DDoS-Friday-no-metadata.parquet',
        'DoS-Wednesday-no-metadata.parquet',
        'Infiltration-Thursday-no-metadata.parquet',
        'Portscan-Friday-no-metadata.parquet',
        'WebAttacks-Thursday-no-metadata.parquet'
    ]
    
    # Check which files exist
    available_files = []
    for file in parquet_files:
        full_path = os.path.join(dataset_folder, file)
        if os.path.exists(full_path):
            available_files.append(full_path)
            print(f"   ✓ Found: {file}")
        else:
            print(f"   ✗ Missing: {file}")
    
    if not available_files:
        print("\n❌ No parquet files found!")
        print(f"   Make sure the files are in: {dataset_folder}")
        return
    
    print(f"\n   Total files to process: {len(available_files)}")
    
    # Step 2: Load and combine all files
    print("\n📊 Loading all datasets...")
    loader = CICIDSLoader()
    
    all_dataframes = []
    
    for file_path in available_files:
        file_name = os.path.basename(file_path)
        print(f"\n   Loading: {file_name}...")
        
        try:
            df = pd.read_parquet(file_path)
            print(f"      Shape: {df.shape}")
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"      Sampled to: {df.shape}")
            
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"      ✗ Error loading {file_name}: {e}")
    
    # Combine all data
    print("\n🔗 Combining all datasets...")
    combined_data = pd.concat(all_dataframes, ignore_index=True)
    print(f"   ✓ Combined shape: {combined_data.shape}")
    print(f"   ✓ Total samples: {len(combined_data):,}")
    
    # Step 3: Preprocess
    X, y, label_mapping = loader.preprocess_cicids(combined_data)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = loader.create_train_test_split(X, y, test_size=0.3)
    
    # Step 5: Scale features
    print("\n⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features normalized")
    
    # Step 6: Initialize detector
    print("\n🤖 Initializing anomaly detector...")
    # Calculate contamination based on actual attack ratio
    contamination = y_train.sum() / len(y_train)
    print(f"   Setting contamination={contamination:.3f} based on data")
    
    detector = NetworkAnomalyDetector(
        contamination=contamination,
        n_estimators=100
    )
    
    # Step 7: Train models
    detector.train_models(X_train_scaled, y_train)
    
    # Step 8: Evaluate
    iforest_pred, lof_pred, iforest_scores, lof_scores = detector.evaluate_models(
        X_test_scaled, y_test
    )
    
    # Step 9: Visualize
    detector.visualize_results(y_test, iforest_pred, lof_pred, iforest_scores, lof_scores)
    
    # Step 10: Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Dataset Summary:")
    print(f"   Files processed: {len(available_files)}")
    print(f"   Total samples: {len(combined_data):,}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Features: {X.shape[1]}")
    
    print(f"\n🎯 Model Performance:")
    print(f"   Isolation Forest:")
    print(f"      Accuracy:  {detector.metrics['isolation_forest']['accuracy']:.4f}")
    print(f"      F1-Score:  {detector.metrics['isolation_forest']['f1_score']:.4f}")
    print(f"      ROC-AUC:   {detector.metrics['isolation_forest']['roc_auc']:.4f}")
    
    print(f"\n   Local Outlier Factor:")
    print(f"      Accuracy:  {detector.metrics['lof']['accuracy']:.4f}")
    print(f"      F1-Score:  {detector.metrics['lof']['f1_score']:.4f}")
    print(f"      ROC-AUC:   {detector.metrics['lof']['roc_auc']:.4f}")
    
    print(f"\n📁 Results saved to: anomaly_detection_results.png")
    print("="*70)
    
    return detector


def run_with_single_file(file_path, sample_size=None):
    """
    Run anomaly detection on a single CIC-IDS file
    
    Args:
        file_path: Path to single parquet file
        sample_size: Number of samples to use (None = use all)
    """
    
    print("="*70)
    print("🛡️  NETWORK ANOMALY DETECTION - SINGLE FILE")
    print("="*70)
    
    loader = CICIDSLoader()
    
    # Load data
    data = loader.load_from_parquet(file_path)
    
    # Sample if requested
    if sample_size and len(data) > sample_size:
        print(f"\n   Sampling {sample_size} rows from {len(data)} total...")
        data = data.sample(n=sample_size, random_state=42)
    
    # Preprocess
    X, y, label_mapping = loader.preprocess_cicids(data)
    
    # Split
    X_train, X_test, y_train, y_test = loader.create_train_test_split(X, y, test_size=0.3)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize detector
    contamination = y_train.sum() / len(y_train)
    detector = NetworkAnomalyDetector(contamination=contamination, n_estimators=100)
    
    # Train
    detector.train_models(X_train_scaled, y_train)
    
    # Evaluate
    iforest_pred, lof_pred, iforest_scores, lof_scores = detector.evaluate_models(
        X_test_scaled, y_test
    )
    
    # Visualize
    detector.visualize_results(y_test, iforest_pred, lof_pred, iforest_scores, lof_scores)
    
    print("\n✅ Complete! Results saved.")
    
    return detector


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("📊 CIC-IDS NETWORK ANOMALY DETECTION")
    print("="*70)
    
    print("\nUsage:")
    print("  Option 1 - Process ALL parquet files in a folder:")
    print("    python run_cicids.py /path/to/dataset/folder")
    print()
    print("  Option 2 - Process a SINGLE file:")
    print("    python run_cicids.py /path/to/file.parquet")
    print()
    print("  Option 3 - Process with SAMPLING (faster):")
    print("    python run_cicids.py /path/to/folder 50000")
    print("    (uses 50,000 samples per file)")
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("💡 INTERACTIVE MODE")
        print("="*70)
        
        dataset_path = input("\nEnter path to dataset folder or file: ").strip()
        
        sample_input = input("Sample size per file (press Enter for all data): ").strip()
        sample_size = int(sample_input) if sample_input else None
        
    else:
        dataset_path = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Check if it's a folder or file
    if os.path.isdir(dataset_path):
        print(f"\n📁 Folder mode: Processing all parquet files in {dataset_path}")
        detector = run_with_cicids_multiple_files(dataset_path, sample_size)
    elif os.path.isfile(dataset_path):
        print(f"\n📄 Single file mode: Processing {dataset_path}")
        detector = run_with_single_file(dataset_path, sample_size)
    else:
        print(f"\n❌ Error: {dataset_path} not found!")
        print("   Make sure the path is correct")
