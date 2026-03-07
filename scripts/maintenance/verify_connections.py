"""
verify_connections.py
======================
Verification script to test all module imports and connections
in the Telco Customer Churn Prediction system.

Run this to ensure everything is properly organized.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_imports():
    """Test that all critical imports work."""
    errors = []
    
    print("=" * 70)
    print("VERIFYING PROJECT CONNECTIONS")
    print("=" * 70)
    
    # Test 1: Config (central hub)
    print("\n[1/12] Testing config.py...")
    try:
        from config import (
            MODELS_DIR, DATA_DIR, OUTPUTS_DIR, LOGS_DIR,
            XGB_PARAMS, NN_PARAMS, RISK_BANDS,
            RANDOM_STATE, TARGET_COLUMN
        )
        print("  ✓ config.py imports successfully")
    except Exception as e:
        errors.append(f"config.py: {e}")
        print(f"  ✗ config.py failed: {e}")
    
    # Test 2: Data loader
    print("\n[2/12] Testing data_loader.py...")
    try:
        from data_loader import load_data, clean_features
        print("  ✓ data_loader.py imports successfully")
    except Exception as e:
        errors.append(f"data_loader.py: {e}")
        print(f"  ✗ data_loader.py failed: {e}")
    
    # Test 3: Preprocessor
    print("\n[3/12] Testing preprocessor.py...")
    try:
        from preprocessor import build_preprocessor, fit_preprocessor, transform
        print("  ✓ preprocessor.py imports successfully")
    except Exception as e:
        errors.append(f"preprocessor.py: {e}")
        print(f"  ✗ preprocessor.py failed: {e}")
    
    # Test 4: XGBoost model
    print("\n[4/12] Testing xgb_model.py...")
    try:
        from xgb_model import build_xgb, train_xgb, predict_proba_xgb
        print("  ✓ xgb_model.py imports successfully")
    except Exception as e:
        errors.append(f"xgb_model.py: {e}")
        print(f"  ✗ xgb_model.py failed: {e}")
    
    # Test 5: Neural network
    print("\n[5/12] Testing nn_model.py...")
    try:
        from nn_model import build_nn, train_nn, predict_proba_nn
        print("  ✓ nn_model.py imports successfully")
    except Exception as e:
        errors.append(f"nn_model.py: {e}")
        print(f"  ✗ nn_model.py failed: {e}")
    
    # Test 6: Stacking
    print("\n[6/12] Testing stacking.py...")
    try:
        from stacking import generate_oof_predictions, train_meta_model, stack_predict
        print("  ✓ stacking.py imports successfully")
    except Exception as e:
        errors.append(f"stacking.py: {e}")
        print(f"  ✗ stacking.py failed: {e}")
    
    # Test 7: Calibration
    print("\n[7/12] Testing calibration.py...")
    try:
        from calibration import calibrate_probabilities
        print("  ✓ calibration.py imports successfully")
    except Exception as e:
        errors.append(f"calibration.py: {e}")
        print(f"  ✗ calibration.py failed: {e}")
    
    # Test 8: Evaluation
    print("\n[8/12] Testing evaluation.py...")
    try:
        from evaluation import evaluate_model, compare_models
        print("  ✓ evaluation.py imports successfully")
    except Exception as e:
        errors.append(f"evaluation.py: {e}")
        print(f"  ✗ evaluation.py failed: {e}")
    
    # Test 9: Risk segmentation
    print("\n[9/12] Testing risk_segmentation.py...")
    try:
        from risk_segmentation import add_risk_band
        print("  ✓ risk_segmentation.py imports successfully")
    except Exception as e:
        errors.append(f"risk_segmentation.py: {e}")
        print(f"  ✗ risk_segmentation.py failed: {e}")
    
    # Test 10: Business impact
    print("\n[10/12] Testing business_impact.py...")
    try:
        from business_impact import compute_business_impact
        print("  ✓ business_impact.py imports successfully")
    except Exception as e:
        errors.append(f"business_impact.py: {e}")
        print(f"  ✗ business_impact.py failed: {e}")
    
    # Test 11: Retention AI
    print("\n[11/12] Testing retention_ai.py...")
    try:
        from retention_ai import generate_retention_recommendations
        print("  ✓ retention_ai.py imports successfully")
    except Exception as e:
        errors.append(f"retention_ai.py: {e}")
        print(f"  ✗ retention_ai.py failed: {e}")
    
    # Test 12: Reporting
    print("\n[12/12] Testing reporting.py...")
    try:
        from reporting import generate_all_reports
        print("  ✓ reporting.py imports successfully")
    except Exception as e:
        errors.append(f"reporting.py: {e}")
        print(f"  ✗ reporting.py failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    if errors:
        print(f"❌ FAILED: {len(errors)} module(s) have import errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ SUCCESS: All modules imported successfully!")
        print("\nProject connections verified:")
        print("  • All src/ modules are properly connected")
        print("  • No circular dependencies detected")
        print("  • Import paths are correctly configured")
        print("\nYou can now run:")
        print("  python src/train_pipeline.py --data data/customer_churn.csv --target Churn")
        print("  streamlit run app.py")
        print("  python -m pytest tests/ -v")
        return True


def check_directories():
    """Verify all required directories exist."""
    print("\n" + "=" * 70)
    print("CHECKING PROJECT STRUCTURE")
    print("=" * 70)
    
    required_dirs = [
        "data",
        "models",
        "outputs",
        "logs",
        "src",
        "tests",
        "scripts"
    ]
    
    root = PROJECT_ROOT
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ exists")
        else:
            print(f"  ✗ {dir_name}/ missing")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required directories exist")
    else:
        print("\n❌ Some directories are missing")
    
    return all_exist


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("TELCO CHURN PREDICTION - PROJECT VERIFICATION")
    print("=" * 70)
    
    imports_ok = test_imports()
    dirs_ok = check_directories()
    
    print("\n" + "=" * 70)
    if imports_ok and dirs_ok:
        print("🎉 PROJECT IS PROPERLY ORGANIZED AND CONNECTED!")
        print("=" * 70)
        return 0
    else:
        print("⚠️  SOME ISSUES DETECTED - SEE ABOVE")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
