"""
Verification script to check if the project is set up correctly
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    exists = os.path.isdir(dirpath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    """Main verification function"""
    print("="*60)
    print("Competitor Pricing Optimizer - Setup Verification")
    print("="*60)
    print()
    
    all_checks = []
    
    # Check core files
    print("Core Files:")
    all_checks.append(check_file_exists("requirements.txt", "Requirements file"))
    all_checks.append(check_file_exists("config.py", "Configuration file"))
    all_checks.append(check_file_exists("README.md", "README"))
    all_checks.append(check_file_exists("train.py", "Training script"))
    all_checks.append(check_file_exists("app.py", "Streamlit app"))
    print()
    
    # Check source files
    print("Source Files:")
    all_checks.append(check_file_exists("src/scraper.py", "Scraper module"))
    all_checks.append(check_file_exists("src/preprocessing.py", "Preprocessing module"))
    all_checks.append(check_file_exists("src/models.py", "Models module"))
    all_checks.append(check_file_exists("src/utils.py", "Utils module"))
    print()
    
    # Check directories
    print("Directories:")
    all_checks.append(check_directory_exists("data/raw", "Raw data directory"))
    all_checks.append(check_directory_exists("data/processed", "Processed data directory"))
    all_checks.append(check_directory_exists("models", "Models directory"))
    all_checks.append(check_directory_exists("notebooks", "Notebooks directory"))
    all_checks.append(check_directory_exists("src", "Source directory"))
    print()
    
    # Check notebook
    print("Notebooks:")
    all_checks.append(check_file_exists("notebooks/eda.ipynb", "EDA notebook"))
    print()
    
    # Summary
    print("="*60)
    passed = sum(all_checks)
    total = len(all_checks)
    print(f"Verification: {passed}/{total} checks passed")
    
    if passed == total:
        print("[SUCCESS] All checks passed! Project is set up correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Generate sample data: python src/scraper.py --use-sample")
        print("3. Train models: python train.py")
        print("4. Run dashboard: streamlit run app.py")
    else:
        print("[WARNING] Some checks failed. Please review the output above.")
    
    print("="*60)

if __name__ == "__main__":
    main()

