# Quick Reference: Project Organization
**Telco Customer Churn Prediction System**

## ✅ What Was Organized

### 1. **File Structure Cleanup**
- Moved `test_ai_recommendations.py` → `scripts/demo_ai_recommendations.py`
- Created `scripts/` directory for utility and demo scripts
- Added `scripts/__init__.py` for proper package structure

### 2. **All Module Connections Verified**
- ✓ All 12 core modules in `src/` properly connected
- ✓ No circular dependencies
- ✓ All imports working correctly
- ✓ Entry points (`app.py`, `example_predict.py`) connected to src/
- ✓ Tests properly configured with `conftest.py`

### 3. **Project Structure**
```
📁 Root Directory
├── app.py                      ✓ Streamlit dashboard
├── example_predict.py          ✓ Simple prediction demo
├── verify_connections.py       ✓ NEW: Verification script
├── PROJECT_STRUCTURE.md        ✓ NEW: Complete connection map
│
├── 📁 src/                     ✓ All 14 modules connected
│   ├── config.py              ← Central hub (no dependencies)
│   ├── data_loader.py         ← Depends on: config
│   ├── preprocessor.py        ← Depends on: config
│   ├── xgb_model.py           ← Depends on: config
│   ├── nn_model.py            ← Depends on: config
│   ├── stacking.py            ← Depends on: config, xgb_model, nn_model
│   ├── calibration.py         ← Depends on: config
│   ├── evaluation.py          ← No internal dependencies
│   ├── risk_segmentation.py   ← Depends on: config
│   ├── business_impact.py     ← Depends on: config
│   ├── retention_ai.py        ← Depends on: config (+ Gemini API)
│   ├── reporting.py           ← Depends on: config
│   ├── train_pipeline.py      ← Orchestrator (uses all modules)
│   └── predict_stacked.py     ← CLI inference
│
├── 📁 tests/                   ✓ All 5 test files connected
│   ├── conftest.py            ← Sets up src/ imports
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_business_impact.py
│   └── test_risk_segmentation.py
│
├── 📁 scripts/                 ✓ NEW: Demo & utility scripts
│   ├── __init__.py
│   └── demo_ai_recommendations.py
│
├── 📁 data/                    ✓ Training data
├── 📁 models/                  ✓ Trained artifacts
├── 📁 outputs/                 ✓ Predictions & reports
└── 📁 logs/                    ✓ Run logs
```

---

## 🎯 How Everything Connects

### **Import Pattern Overview**

1. **config.py** = Central hub (imported by everyone)
2. **Base modules** = Import only from config
3. **Composite modules** = Import from config + other modules
4. **Orchestrators** = Import from all modules

```
           config.py (no dependencies)
                  ↓
        ┌─────────┼─────────┐
        ↓         ↓         ↓
   data_loader  xgb_model  nn_model
        ↓         ↓         ↓
        └─────────┼─────────┘
                  ↓
             stacking.py
                  ↓
          train_pipeline.py
```

---

## 🚀 Ready to Use Commands

### **1. Verify Everything Works**
```powershell
python verify_connections.py
```

### **2. Train the Model**
```powershell
python src/train_pipeline.py --data data/customer_churn.csv --target Churn
```

### **3. Run Dashboard**
```powershell
streamlit run app.py
```

### **4. Make Single Prediction**
```powershell
python example_predict.py
```

### **5. CLI Prediction**
```powershell
python src/predict_stacked.py --input tenure=12 MonthlyCharges=85
```

### **6. Test AI Recommendations**
```powershell
python scripts/demo_ai_recommendations.py
```

### **7. Run Tests**
```powershell
python -m pytest tests/ -v
```

---

## 📝 Key Files Created/Modified

### New Files:
1. `scripts/demo_ai_recommendations.py` - AI recommendation demo
2. `scripts/__init__.py` - Scripts package initialization
3. `PROJECT_STRUCTURE.md` - Complete connection documentation
4. `verify_connections.py` - Connection verification script
5. `QUICK_REFERENCE.md` - This file

### Modified Files:
1. `src/retention_ai.py` - Fixed Gemini API compatibility
   - Updated to use `genai.configure()` and `genai.GenerativeModel()`
   - Fixed JSON serialization for numpy types
   - Added proper error handling

---

## ✅ Verification Results

All modules tested and working:
- ✓ config.py
- ✓ data_loader.py
- ✓ preprocessor.py
- ✓ xgb_model.py
- ✓ nn_model.py
- ✓ stacking.py
- ✓ calibration.py
- ✓ evaluation.py
- ✓ risk_segmentation.py
- ✓ business_impact.py
- ✓ retention_ai.py
- ✓ reporting.py

All directories exist:
- ✓ data/
- ✓ models/
- ✓ outputs/
- ✓ logs/
- ✓ src/
- ✓ tests/
- ✓ scripts/

---

## 🔍 Import Path Patterns

### Pattern 1: Root-level entry points
```python
# app.py, example_predict.py
sys.path.insert(0, str(Path("src").resolve()))
```

### Pattern 2: Source modules (src/)
```python
# All modules in src/ directory
from config import MODELS_DIR  # Direct import, same directory
```

### Pattern 3: Tests (tests/)
```python
# tests/conftest.py and all test files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

### Pattern 4: Scripts (scripts/)
```python
# scripts/demo_ai_recommendations.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

---

## 🎉 Project Status

**FULLY ORGANIZED AND CONNECTED** ✅

All files are in their correct locations, all modules are properly connected, and the system is ready for use. The project follows Python best practices with:

- Clear separation of concerns
- No circular dependencies
  Central configuration management
- Proper package structure
- Comprehensive testing setup
- Easy-to-use entry points

---

**Need Help?** 
- See `PROJECT_STRUCTURE.md` for detailed connection diagrams
- See `PROJECT_DOCUMENTATION.md` for full technical specs
- See `STUDENT_GUIDE.md` for educational walkthrough
- See `README.md` for quick start guide
