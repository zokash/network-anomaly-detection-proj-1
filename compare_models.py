"""
Compare Multiple Models for Anomaly Detection
Tests: Isolation Forest, LOF, One-Class SVM, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from cicids_loader import CICIDSLoader
from sklearn.preprocessing import StandardScaler
import time

class ModelComparison:
    """Compare multiple anomaly detection models"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.models = {}
        self.results = {}
        
    def train_unsupervised_models(self, X_train):
        """Train unsupervised models"""
        print("\n🤖 Training Unsupervised Models...")
        
        # 1. Isolation Forest
        print("   1/3 Isolation Forest...")
        start = time.time()
        self.models['IsolationForest'] = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.models['IsolationForest'].fit(X_train)
        print(f"      ✓ Trained in {time.time()-start:.2f}s")
        
        # 2. LOF
        print("   2/3 Local Outlier Factor...")
        start = time.time()
        self.models['LOF'] = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=20,
            novelty=True,
            n_jobs=-1
        )
        self.models['LOF'].fit(X_train)
        print(f"      ✓ Trained in {time.time()-start:.2f}s")
        
        # 3. One-Class SVM
        print("   3/3 One-Class SVM...")
        start = time.time()
        self.models['OneClassSVM'] = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='auto'
        )
        self.models['OneClassSVM'].fit(X_train)
        print(f"      ✓ Trained in {time.time()-start:.2f}s")
        
    def train_supervised_models(self, X_train, y_train):
        """Train supervised models (if labels available)"""
        print("\n🤖 Training Supervised Models...")
        
        # 4. Random Forest
        print("   1/2 Random Forest...")
        start = time.time()
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.models['RandomForest'].fit(X_train, y_train)
        print(f"      ✓ Trained in {time.time()-start:.2f}s")
        
        # 5. XGBoost
        print("   2/2 XGBoost...")
        start = time.time()
        self.models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss'
        )
        self.models['XGBoost'].fit(X_train, y_train)
        print(f"      ✓ Trained in {time.time()-start:.2f}s")
        
    def evaluate_all(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n📈 Evaluating All Models...")
        
        for name, model in self.models.items():
            print(f"\n   📊 {name}:")
            
            # Predict
            if name in ['IsolationForest', 'LOF', 'OneClassSVM']:
                # Unsupervised models
                pred_raw = model.predict(X_test)
                predictions = np.where(pred_raw == -1, 1, 0)
                scores = -model.score_samples(X_test) if hasattr(model, 'score_samples') else None
            else:
                # Supervised models
                predictions = model.predict(X_test)
                scores = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            try:
                auc = roc_auc_score(y_test, scores) if scores is not None else 0
            except:
                auc = 0
            
            # Store results
            self.results[name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc,
                'predictions': predictions,
                'scores': scores
            }
            
            # Print
            print(f"      Accuracy:  {acc:.4f}")
            print(f"      F1-Score:  {f1:.4f}")
            print(f"      ROC-AUC:   {auc:.4f}")
    
    def plot_comparison(self):
        """Visualize model comparison"""
        print("\n📊 Creating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy Comparison
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_ylim([0, 1.0])
        ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score Comparison
        f1_scores = [self.results[m]['f1_score'] for m in models]
        
        ax2 = axes[0, 1]
        bars = ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_ylim([0, 1.0])
        ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 3. ROC-AUC Comparison
        aucs = [self.results[m]['roc_auc'] for m in models]
        
        ax3 = axes[1, 0]
        bars = ax3.bar(models, aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax3.set_ylim([0, 1.0])
        ax3.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('ROC-AUC')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 4. All Metrics Together
        ax4 = axes[1, 1]
        x = np.arange(len(models))
        width = 0.25
        
        ax4.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
        ax4.bar(x, f1_scores, width, label='F1-Score', color='lightcoral')
        ax4.bar(x + width, aucs, width, label='ROC-AUC', color='lightgreen')
        
        ax4.set_ylabel('Score')
        ax4.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()
        ax4.set_ylim([0, 1.0])
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved as 'model_comparison_results.png'")
        
    def print_summary(self):
        """Print summary table"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-"*70)
        
        for model, results in self.results.items():
            print(f"{model:<20} {results['accuracy']:<12.4f} {results['f1_score']:<12.4f} {results['roc_auc']:<12.4f}")
        
        print("="*70)
        
        # Best model
        best_acc = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_auc = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        
        print("\n🏆 Best Models:")
        print(f"   Accuracy:  {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
        print(f"   F1-Score:  {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        print(f"   ROC-AUC:   {best_auc[0]} ({best_auc[1]['roc_auc']:.4f})")


def main():
    """Main comparison function"""
    
    print("="*70)
    print("🔬 MODEL COMPARISON - NETWORK ANOMALY DETECTION")
    print("="*70)
    
    # Get dataset path
    dataset_path = input("\nEnter path to dataset: ").strip()
    sample_input = input("Sample size (press Enter for 50000): ").strip()
    sample_size = int(sample_input) if sample_input else 50000
    
    # Load data
    loader = CICIDSLoader()
    print("\n📊 Loading dataset...")
    
    import os
    all_files = [f for f in os.listdir(dataset_path) if f.endswith('.parquet')]  # Use 3 files
    dfs = []
    for file in all_files:
        df = pd.read_parquet(os.path.join(dataset_path, file))
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    X, y, _ = loader.preprocess_cicids(data)
    X_train, X_test, y_train, y_test = loader.create_train_test_split(X, y, test_size=0.3)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate contamination
    contamination = y_train.sum() / len(y_train)
    
    # Initialize comparison
    comparison = ModelComparison(contamination=contamination)
    
    # Train models
    comparison.train_unsupervised_models(X_train_scaled)
    comparison.train_supervised_models(X_train_scaled, y_train)
    
    # Evaluate
    comparison.evaluate_all(X_test_scaled, y_test)
    
    # Visualize
    comparison.plot_comparison()
    
    # Summary
    comparison.print_summary()
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()