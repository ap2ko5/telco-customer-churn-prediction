# Churn Intelligence Demo — One-Click Automation Pipeline

## Overview

This automation pipeline provides a **one-click demo** for the Churn Intelligence System:

```
Double-click run_system.bat
        ↓
Python pipeline executes (generates predictions)
        ↓
CSV updated with churn predictions (in ₹ rupees format)
        ↓
Power BI dashboard opens automatically
```

## Files Included

### 1. **run_system.bat** — Main Orchestrator (Windows Batch)
The entry point for everything. This batch file:
- Ensures Python environment is ready
- Runs the complete ML pipeline
- Waits for file I/O to complete
- Opens your Power BI dashboard

**To use:**
- Edit line 19 with your Power BI file path:
  ```batch
  SET "PBIX_PATH=C:\Users\ABEL ABRAHAM\Downloads\dash1.pbix"
  ```
- Double-click the file

---

### 2. **run_demo.py** — Demo Execution Script
Python script that orchestrates the pipeline using pre-trained models:

**What it does:**
1. ✓ Loads customer data from CSV
2. ✓ Uses pre-trained ML models (no retraining)
3. ✓ Generates churn predictions
4. ✓ Assigns risk bands (Low/Medium/High/Critical)
5. ✓ Calculates business impact in ₹ rupees
6. ✓ Creates visualizations
7. ✓ **Safely writes CSV** with atomic file replacement
8. ✓ Displays metrics with Indian rupee formatting

**Usage from command line:**
```bash
python run_demo.py
python run_demo.py --skip-ai        # Faster: skip AI recommendations
python run_demo.py --data path/to/file.csv
```

**Output:**
```
📊 KEY METRICS:
   Total Customers        : 7,043
   High-Risk Customers    : 1,238
   Critical Risk          : 287
   Avg Churn Probability  : 26.4%
   💰 Total Revenue at Risk : ₹1,40,68,495.32

✨ Ready for Power BI! The CSV file is stable and fully written.
```

---

### 3. **src/safe_csv_writer.py** — Atomic CSV Operations
Handles safe writing of CSV files using atomic operations:

```python
from safe_csv_writer import safe_write_csv, add_rupee_formatted_columns
import pandas as pd

# Create DataFrame
df = pd.DataFrame({...})

# Option 1: Simple safe write
safe_write_csv(df, "output.csv", verbose=True)

# Option 2: With column ordering
column_order = ["churn_probability", "churn_band", "expected_revenue_loss"]
safe_write_csv(df, "output.csv", columns_order=column_order, verbose=True)

# Option 3: Add formatted rupee columns for Power BI
df_formatted = add_rupee_formatted_columns(df)
# Now includes: expected_revenue_loss_rupees, monthly_charges_rupees, etc.
safe_write_csv(df_formatted, "output.csv")
```

**Key Features:**
- ✓ Writes to temporary file first
- ✓ Atomically replaces target file (no partial writes)
- ✓ Prevents Power BI file locking issues
- ✓ Safe cleanup on error
- ✓ Optional automatic backups

---

### 4. **src/indian_currency.py** — Rupee Formatting
Already in project; used throughout for ₹ currency display:

```python
from indian_currency import format_indian_currency, format_indian_currency_short

format_indian_currency(1440684.95)  # ₹14,40,684.95
format_indian_currency(100000)      # ₹1,00,000.00

# Short form with Lakhs/Crores
format_indian_currency_short(1440684.95)  # ₹14.41L
format_indian_currency_short(10000000)    # ₹1.00Cr
```

---

## Output CSV Structure

The generated **churn_predictions.csv** includes (in order):

| Column | Format | Example | Use in Power BI |
|--------|--------|---------|-----------------|
| `churn_probability` | Float (0-1) | 0.8234 | Risk gauge, charts |
| `churn_band` | Category | "Critical" | Color coding, filtering |
| `expected_revenue_loss` | Numeric (₹) | 15000.50 | Calculations |
| `expected_revenue_loss_rupees` | Text (₹) | ₹15,000.50 | Direct display |
| `retention_recommendation` | Text | "Urgent outreach..." | AI insights |
| `tenure` | Int | 12 | Demographics |
| `MonthlyCharges` | Float | 89.50 | Analysis |
| `monthly_charges_rupees` | Text | ₹89.50 | Display |
| ... | ... | ... | ... |

---

## Execution Flow

### Step-by-Step Process

```
┌─────────────────────────────────────────┐
│ 1. Double-click run_system.bat          │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 2. Batch file validates Python environ  │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 3. run_demo.py executes:                │
│    • Load data                          │
│    • Load pre-trained models            │
│    • Generate predictions               │
│    • Calculate business impact          │
│    • Create visualizations              │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 4. Format currency in ₹ (rupees)        │
│    Add rupee columns to DataFrame       │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 5. SAFE CSV WRITE:                      │
│    • Write to temp file                 │
│    • Atomic file replacement            │
│    • Prevent Power BI locking           │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 6. Wait 2 seconds for file stability    │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ 7. Open Power BI dashboard              │
│    Power BI reads updated CSV           │
└─────────────────────────────────────────┘
```

---

## Key Features

### ✓ Stable Outputs
- Uses **pre-trained models** — no retraining on each run
- Same raw predictions every time
- Only change outputs intentionally via training script

### ✓ Safe File Writing
- Atomic operations (all-or-nothing)
- Prevents partial/corrupt CSV files
- No Power BI file locking issues
- Easy cleanup on failure

### ✓ Rupee Currency Format
- All financial metrics in ₹ Indian format
- Examples:
  - ₹95,000.00 (ninety-five thousand)
  - ₹14,40,684.95 (fourteen lakhs forty thousand)
  - ₹1,00,00,000.00 (one crore)
- Both numeric and formatted text columns

### ✓ Demo-Ready
- Simple one-click execution
- Fast runtime (~1-3 minutes)
- Clear progress indicators
- Helpful error messages

---

## Configuration

### Power BI Integration

1. **Connect Power BI to CSV:**
   - Open Power BI Desktop
   - Data → Get Data → Text/CSV
   - Select `outputs/churn_predictions.csv`
   - Check "Use first row as headers"
   - Load

2. **Auto-refresh (Optional):**
   - Power BI will prompt to refresh after CSV is updated
   - Click "Refresh" to see latest predictions

3. **Display Rupee Columns:**
   - Use `expected_revenue_loss_rupees` in visual labels
   - Use numeric `expected_revenue_loss` for calculations

### Customize Pipeline

Edit `run_demo.py` to:
- Change input data path: `--data`
- Skip AI recommendations: `--skip-ai` (faster)
- Modify feature engineering in preprocessing
- Adjust risk band thresholds in config.py

---

## Troubleshooting

### Power BI file not found
**Error:** "Power BI file not found"
**Solution:** Update line 19 in `run_system.bat` with correct path to your `.pbix` file

### Python not found
**Error:** "Python executable not found"
**Solution:** Initialize virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### CSV not updating in Power BI
**Solution:**
1. Run script again
2. Click "Refresh" in Power BI
3. Check that `outputs/churn_predictions.csv` was updated (check file timestamp)

### Wait time issues
**If Power BI opens before CSV is ready:**
- Edit `run_system.bat` line 82-83
- Increase timeout from 2 to 5 seconds:
  ```batch
  timeout /t 5 /nobreak
  ```

---

## Performance

| Step | Time |
|------|------|
| Data loading | ~5 sec |
| Preprocessing | ~10 sec |
| Model prediction | ~30 sec |
| Risk segmentation | ~5 sec |
| Business impact | ~5 sec |
| Visualizations | ~15 sec |
| CSV write | <1 sec |
| **Total** | **~1:10 minutes** |

---

## Example Metrics Output

```
📊 KEY METRICS:
   Total Customers        : 7,043
   High-Risk Customers    : 1,238
   Critical Risk          : 287
   Avg Churn Probability  : 26.4%
   💰 Total Revenue at Risk : ₹1,40,68,495.32

📁 Files Created:
   📊 Predictions         : outputs/churn_predictions.csv
   📄 Summary report      : outputs/summary_report.txt
   📈 Visualizations      : outputs/ [prob_distribution.png, band_distribution.png, shap_importance.png]

⏱️  Total time: 71.3 seconds

✨ Ready for Power BI! The CSV file is stable and fully written.
```

---

## Quick Start Checklist

- [ ] Open Terminal in project root
- [ ] Run: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
- [ ] Edit `run_system.bat` line 19 — add Power BI file path
- [ ] Double-click `run_system.bat`
- [ ] Wait for Power BI to open
- [ ] Refresh data in Power BI if needed
- [ ] Done! ✨

---

## Support

For issues or questions:
1. Check error message in console
2. Review troubleshooting section above
3. Check that pre-trained models exist in `models/` directory
4. Verify data file format matches expected columns

---

**Created:** March 2026  
**Version:** 1.0  
**Last Updated:** March 25, 2026
