"""
Real-Time Network Anomaly Detection System
Using PyOD Isolation Forest on NSL-KDD Dataset

Project: BCSE497J - Network Traffic Anomaly Detection
Team: Ved Kulkarni, Tanya Bhardwaj, Kashish Hussain
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# ML and Anomaly Detection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    roc_curve,
    f1_score,
    accuracy_score
)

# Scikit-learn Anomaly Detection (replacing PyOD due to network constraints)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For Redis storage (optional - can be enabled later)
# import redis
import json
from datetime import datetime

class NetworkAnomalyDetector:
    """
    Lightweight network anomaly detection using PyOD Isolation Forest
    """
    
    def __init__(self, contamination=0.1, n_estimators=100):
        """
        Initialize the detector
        
        Args:
            contamination: Expected proportion of anomalies in dataset
            n_estimators: Number of trees in Isolation Forest
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        # Models
        self.iforest_model = None
        self.lof_model = None
        self.scaler = StandardScaler()
        
        # Label encoders for categorical features
        self.label_encoders = {}
        
        # Feature names
        self.feature_names = None
        self.categorical_features = []
        
        # Performance metrics
        self.metrics = {}
        
        print(f"🔧 Network Anomaly Detector initialized")
        print(f"   - Contamination rate: {contamination}")
        print(f"   - Isolation Forest trees: {n_estimators}")
    
    def load_nsl_kdd_data(self, train_path=None, test_path=None):
        """
        Load and prepare NSL-KDD dataset
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
        """
        print("\n📊 Loading NSL-KDD Dataset...")
        
        # NSL-KDD column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        
        # First, check for local files in 'data' folder
        if train_path is None:
            local_train = 'data/KDDTrain+.txt'
            if os.path.exists(local_train):
                train_path = local_train
                print(f"   ℹ️  Found local training data: {train_path}")
        
        if test_path is None:
            local_test = 'data/KDDTest+.txt'
            if os.path.exists(local_test):
                test_path = local_test
                print(f"   ℹ️  Found local test data: {test_path}")
        
        # If still None, try online sources (will likely fail)
        if train_path is None:
            train_path = 'http://205.174.165.80/CICDataset/NSL-KDD/Dataset/KDDTrain+.txt'
        if test_path is None:
            test_path = 'http://205.174.165.80/CICDataset/NSL-KDD/Dataset/KDDTest+.txt'
        
        try:
            # Load data
            train_data = pd.read_csv(train_path, names=columns)
            test_data = pd.read_csv(test_path, names=columns)
            
            print(f"   ✓ Training data loaded: {train_data.shape}")
            print(f"   ✓ Test data loaded: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            print(f"   ✗ Error loading data: {e}")
            print("   ℹ️  Run 'python download_dataset.py' to download a dataset")
            print("   ℹ️  Or generating sample synthetic data for demonstration...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample data if NSL-KDD download fails"""
        np.random.seed(42)
        n_samples = 10000
        
        # Create synthetic features
        data = {
            'duration': np.random.exponential(10, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'smtp', 'ftp', 'ssh', 'other'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], n_samples),
            'src_bytes': np.random.exponential(1000, n_samples),
            'dst_bytes': np.random.exponential(1000, n_samples),
            'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'wrong_fragment': np.random.poisson(0.1, n_samples),
            'urgent': np.random.poisson(0.05, n_samples),
            'hot': np.random.poisson(0.5, n_samples),
            'num_failed_logins': np.random.poisson(0.1, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'count': np.random.poisson(50, n_samples),
            'srv_count': np.random.poisson(30, n_samples),
            'serror_rate': np.random.uniform(0, 1, n_samples),
            'srv_serror_rate': np.random.uniform(0, 1, n_samples),
            'rerror_rate': np.random.uniform(0, 1, n_samples),
            'srv_rerror_rate': np.random.uniform(0, 1, n_samples),
            'same_srv_rate': np.random.uniform(0, 1, n_samples),
            'diff_srv_rate': np.random.uniform(0, 1, n_samples),
            'attack_type': np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r'], 
                                           n_samples, p=[0.7, 0.15, 0.1, 0.03, 0.02])
        }
        
        df = pd.DataFrame(data)
        
        # Split into train and test
        train_data = df.iloc[:7000].copy()
        test_data = df.iloc[7000:].copy()
        
        print(f"   ✓ Generated synthetic training data: {train_data.shape}")
        print(f"   ✓ Generated synthetic test data: {test_data.shape}")
        
        return train_data, test_data
    
    def preprocess_data(self, train_data, test_data):
        """
        Preprocess the network traffic data
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n🔧 Preprocessing data...")
        
        # Identify categorical features
        self.categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()
        if 'attack_type' in self.categorical_features:
            self.categorical_features.remove('attack_type')
        if 'difficulty' in self.categorical_features:
            self.categorical_features.remove('difficulty')
        
        # Create binary labels: normal=0, attack=1
        train_data['label'] = (train_data['attack_type'] != 'normal').astype(int)
        test_data['label'] = (test_data['attack_type'] != 'normal').astype(int)
        
        print(f"   - Attack distribution (train): {train_data['label'].value_counts().to_dict()}")
        print(f"   - Attack distribution (test): {test_data['label'].value_counts().to_dict()}")
        
        # Drop non-feature columns
        drop_cols = ['attack_type']
        if 'difficulty' in train_data.columns:
            drop_cols.append('difficulty')
        
        X_train = train_data.drop(columns=drop_cols + ['label'])
        y_train = train_data['label']
        
        X_test = test_data.drop(columns=drop_cols + ['label'])
        y_test = test_data['label']
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in X_train.columns:
                self.label_encoders[col] = LabelEncoder()
                X_train[col] = self.label_encoders[col].fit_transform(X_train[col].astype(str))
                # Handle unseen categories in test set
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
                )
                if 'unknown' not in self.label_encoders[col].classes_:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'unknown'
                    )
                X_test[col] = self.label_encoders[col].transform(X_test[col])
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   ✓ Preprocessed features: {X_train_scaled.shape[1]}")
        print(f"   ✓ Categorical features encoded: {len(self.categorical_features)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train anomaly detection models
        
        Args:
            X_train: Training features
            y_train: Training labels (for evaluation only)
        """
        print("\n🤖 Training anomaly detection models...")
        
        # Isolation Forest (primary model)
        print("   - Training Isolation Forest...")
        self.iforest_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.iforest_model.fit(X_train)
        print("   ✓ Isolation Forest trained")
        
        # LOF for comparison
        print("   - Training Local Outlier Factor...")
        self.lof_model = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=20,
            n_jobs=-1,
            novelty=True  # Required for sklearn LOF to predict on new data
        )
        self.lof_model.fit(X_train)
        print("   ✓ LOF trained")
        
        print("   ✓ All models trained successfully")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n📈 Evaluating models...")
        
        # Predictions (sklearn returns -1 for anomalies, 1 for normal; convert to 0/1)
        iforest_pred_raw = self.iforest_model.predict(X_test)
        lof_pred_raw = self.lof_model.predict(X_test)
        
        iforest_pred = np.where(iforest_pred_raw == -1, 1, 0)  # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        lof_pred = np.where(lof_pred_raw == -1, 1, 0)
        
        # Get anomaly scores (negative_outlier_factor for sklearn)
        iforest_scores = -self.iforest_model.score_samples(X_test)  # Higher score = more anomalous
        lof_scores = -self.lof_model.score_samples(X_test)
        
        # Calculate metrics for Isolation Forest
        print("\n   🌲 Isolation Forest Results:")
        iforest_metrics = self._calculate_metrics(y_test, iforest_pred, iforest_scores, "Isolation Forest")
        
        # Calculate metrics for LOF
        print("\n   📍 Local Outlier Factor Results:")
        lof_metrics = self._calculate_metrics(y_test, lof_pred, lof_scores, "LOF")
        
        # Store metrics
        self.metrics = {
            'isolation_forest': iforest_metrics,
            'lof': lof_metrics
        }
        
        return iforest_pred, lof_pred, iforest_scores, lof_scores
    
    def _calculate_metrics(self, y_true, y_pred, scores, model_name):
        """Calculate and display metrics"""
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, scores)
        except:
            auc = 0.0
        
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
        print(f"      ROC-AUC:   {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"      Confusion Matrix:")
        print(f"      TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"      FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # False Positive Rate
        fpr = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
        print(f"      False Positive Rate: {fpr:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'fpr': fpr
        }
    
    def generate_alert(self, flow_data, anomaly_score, threshold=0.5):
        """
        Generate alert for anomalous flow
        
        Args:
            flow_data: Dictionary containing flow information
            anomaly_score: Anomaly score from model
            threshold: Score threshold for alerting
            
        Returns:
            Alert dictionary
        """
        if anomaly_score < threshold:
            return None
        
        # Determine severity based on score
        if anomaly_score >= 0.8:
            severity = "CRITICAL"
        elif anomaly_score >= 0.6:
            severity = "HIGH"
        elif anomaly_score >= 0.4:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'anomaly_score': float(anomaly_score),
            'flow_info': flow_data,
            'message': f"{severity} anomaly detected with score {anomaly_score:.3f}"
        }
        
        return alert
    
    def visualize_results(self, y_test, iforest_pred, lof_pred, iforest_scores, lof_scores):
        """
        Create visualization of results
        """
        print("\n📊 Generating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix - Isolation Forest
        cm_iforest = confusion_matrix(y_test, iforest_pred)
        sns.heatmap(cm_iforest, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Isolation Forest - Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Confusion Matrix - LOF
        cm_lof = confusion_matrix(y_test, lof_pred)
        sns.heatmap(cm_lof, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
        axes[0, 1].set_title('LOF - Confusion Matrix')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        # 3. Anomaly Score Distribution - Isolation Forest
        axes[1, 0].hist(iforest_scores[y_test==0], bins=50, alpha=0.5, label='Normal', color='blue')
        axes[1, 0].hist(iforest_scores[y_test==1], bins=50, alpha=0.5, label='Anomaly', color='red')
        axes[1, 0].set_title('Isolation Forest - Anomaly Score Distribution')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Model Comparison
        models = ['Isolation Forest', 'LOF']
        accuracies = [
            self.metrics['isolation_forest']['accuracy'],
            self.metrics['lof']['accuracy']
        ]
        f1_scores = [
            self.metrics['isolation_forest']['f1_score'],
            self.metrics['lof']['f1_score']
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        axes[1, 1].bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1.0])
        
        plt.tight_layout()
        #plt.savefig('/mnt/user-data/outputs/anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        print("   ✓ Visualization saved")
        
        return fig
    
    def predict_single_flow(self, flow_features):
        """
        Predict anomaly for a single flow
        
        Args:
            flow_features: Dictionary or array of flow features
            
        Returns:
            prediction, anomaly_score, alert
        """
        # Convert to DataFrame if dictionary
        if isinstance(flow_features, dict):
            flow_df = pd.DataFrame([flow_features])
        else:
            flow_df = pd.DataFrame([flow_features], columns=self.feature_names)
        
        # Preprocess
        for col in self.categorical_features:
            if col in flow_df.columns and col in self.label_encoders:
                flow_df[col] = flow_df[col].astype(str).map(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
                )
                flow_df[col] = self.label_encoders[col].transform(flow_df[col])
        
        flow_scaled = self.scaler.transform(flow_df)
        
        # Predict (sklearn returns -1 for anomaly, 1 for normal)
        prediction_raw = self.iforest_model.predict(flow_scaled)[0]
        prediction = 1 if prediction_raw == -1 else 0  # Convert to binary
        score = -self.iforest_model.score_samples(flow_scaled)[0]  # Higher = more anomalous
        
        # Generate alert if anomaly
        alert = None
        if prediction == 1:
            alert = self.generate_alert(flow_features, score)
        
        return prediction, score, alert


def main():
    """Main execution function"""
    
    print("="*70)
    print("🛡️  NETWORK ANOMALY DETECTION SYSTEM")
    print("    Real-Time Anomaly Detection using PyOD Isolation Forest")
    print("    Dataset: NSL-KDD")
    print("="*70)
    
    # Initialize detector
    detector = NetworkAnomalyDetector(contamination=0.1, n_estimators=100)
    
    # Load data
    train_data, test_data = detector.load_nsl_kdd_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test = detector.preprocess_data(train_data, test_data)
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Evaluate
    iforest_pred, lof_pred, iforest_scores, lof_scores = detector.evaluate_models(X_test, y_test)
    
    # Visualize
    detector.visualize_results(y_test, iforest_pred, lof_pred, iforest_scores, lof_scores)
    
    # Demo: Predict on sample flows
    print("\n" + "="*70)
    print("🔍 SAMPLE PREDICTIONS")
    print("="*70)
    
    # Get a few test samples
    sample_indices = [0, 100, 500]
    
    for idx in sample_indices:
        if idx < len(X_test):
            flow_features = dict(zip(detector.feature_names, X_test[idx]))
            prediction, score, alert = detector.predict_single_flow(flow_features)
            
            print(f"\n   Sample {idx}:")
            print(f"      Prediction: {'ANOMALY' if prediction == 1 else 'NORMAL'}")
            print(f"      Score: {score:.4f}")
            print(f"      Actual: {'ANOMALY' if y_test.iloc[idx] == 1 else 'NORMAL'}")
            if alert:
                print(f"      Alert: {alert['severity']} - {alert['message']}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📁 Results saved to: /mnt/user-data/outputs/anomaly_detection_results.png")
    print(f"\n💡 Key Findings:")
    print(f"   - Isolation Forest Accuracy: {detector.metrics['isolation_forest']['accuracy']:.4f}")
    print(f"   - Isolation Forest F1-Score: {detector.metrics['isolation_forest']['f1_score']:.4f}")
    print(f"   - LOF Accuracy: {detector.metrics['lof']['accuracy']:.4f}")
    print(f"   - LOF F1-Score: {detector.metrics['lof']['f1_score']:.4f}")
    
    return detector


if __name__ == "__main__":
    detector = main()
