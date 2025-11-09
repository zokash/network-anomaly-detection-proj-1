import os
import sys
import subprocess

print("="*70)
print("🚀 CICIDS ANOMALY DETECTION - AUTO SETUP")
print("="*70)

# Step 1: Check if kaggle.json exists
kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
if os.name == 'nt':  # Windows
    kaggle_path = os.path.expanduser("~\\.kaggle\\kaggle.json")

if not os.path.exists(kaggle_path):
    print("\n❌ Kaggle API key not found!")
    print("\n📝 DO THIS ONCE:")
    print("1. Go to: https://www.kaggle.com/")
    print("2. Sign in")
    print("3. Click your profile pic → Settings")
    print("4. Scroll to 'API' section")
    print("5. Click 'Create New API Token'")
    print("6. A file 'kaggle.json' will download")
    print("\n7. Move it here:")
    
    # Create .kaggle folder
    kaggle_dir = os.path.dirname(kaggle_path)
    os.makedirs(kaggle_dir, exist_ok=True)
    print(f"   {kaggle_dir}")
    
    input("\nPress ENTER after you've moved kaggle.json to that folder...")
    
    if not os.path.exists(kaggle_path):
        print("❌ Still can't find kaggle.json. Exiting.")
        sys.exit(1)

print("✅ Kaggle key found!")

# Step 2: Install packages
print("\n📦 Installing required packages...")
packages = ["kagglehub", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "pyarrow"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✅ Packages installed!")

# Step 3: Download dataset
print("\n📥 Downloading CIC-IDS dataset (this may take a few minutes)...")
import kagglehub
dataset_path = kagglehub.dataset_download("dhoogla/cicids2017")
print(f"✅ Dataset downloaded to: {dataset_path}")

# Step 4: Run training
print("\n🤖 Starting model training...")
print("="*70)

# Import and run
from run_cicids import run_with_cicids_multiple_files

detector = run_with_cicids_multiple_files(dataset_path, sample_size=50000)

print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)
print(f"\nCheck 'anomaly_detection_results.png' for results!")