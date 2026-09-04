"""
Phase 1: Environment & Setup Verification Script
AI-Powered Deepfake Detection System
"""

import sys

def check_environment():
    print("==================================================")
    print("   Deepfake Detection System - Environment Check   ")
    print("==================================================\n")
    
    print(f"Python Version: {sys.version.split()[0]}")
    
    libraries = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("OpenCV", "cv2"),
        ("Pillow (PIL)", "PIL"),
        ("Scikit-Learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
        ("Streamlit", "streamlit"),
        ("TensorFlow", "tensorflow"),
    ]
    
    all_passed = True
    print("\nChecking required libraries...")
    print("-" * 50)
    
    for name, module_name in libraries:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "Installed (no __version__ attribute)")
            print(f"[OK] {name:<15} : v{version}")
        except ImportError:
            print(f"[MISSING] {name:<15} : Not installed!")
            all_passed = False
            
    print("-" * 50)
    if all_passed:
        print("\nSUCCESS: All required libraries are installed and ready!")
    else:
        print("\nWARNING: Some libraries are missing. Please install them using requirements.txt")

if __name__ == "__main__":
    check_environment()
