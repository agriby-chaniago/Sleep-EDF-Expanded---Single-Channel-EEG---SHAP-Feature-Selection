# 🚀 QUICK START GUIDE

## Ready to Run!

Your production notebook is **complete and ready for execution**. Follow these simple steps:

---

## Step 1: Verify Dataset (30 seconds)

Check that you have the Sleep-EDF data:

```bash
ls physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/ | head
```

You should see files like:

```
SC4001E0-PSG.edf
SC4001EC-Hypnogram.edf
SC4002E0-PSG.edf
...
```

✅ **If you see these files**, you're ready!  
❌ **If not**, download from: https://physionet.org/content/sleep-edfx/1.0.0/

---

## Step 2: Install Dependencies (2-3 minutes)

```bash
pip install -r requirements.txt
```

Or create a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

---

## Step 3: Open the Notebook

**In VS Code:**

1. Open `sleep_edf_production.ipynb`
2. Select Python kernel (3.8+)
3. Click "Run All"

**In Jupyter:**

```bash
jupyter notebook sleep_edf_production.ipynb
```

Then: Kernel → Restart & Run All

---

## Step 4: Monitor Progress (90-120 minutes)

The notebook will display:

- ✅ Setup progress (directories, memory governor)
- ✅ Feature extraction progress bar (20-30 min)
- ✅ Per-fold training updates (15-20 min each × 5 folds)
- ✅ Real-time memory monitoring
- ✅ Statistical results
- ✅ Visualization generation
- ✅ Final summary

**Coffee break recommended! ☕**

---

## Step 5: Review Results

After completion, check:

### Console Output

Look for the final summary:

```
================================================================================
FINAL EXPERIMENT SUMMARY
================================================================================

RESEARCH QUESTION
Does SHAP-based feature selection improve XGBoost performance?

ANSWER
✓ YES - SHAP significantly improves performance (p=0.XXXX***)
...
```

### Generated Files

```bash
# View tables
ls -lh results/tables/*.csv

# View figures
ls -lh results/figures/*/*.png

# Check logs
cat experiment_log.txt | tail -50
```

### Key Files to Open

1. **results/tables/statistical_tests.csv** - Primary results
2. **results/figures/main/model_comparison_boxplot.png** - Visual summary
3. **results/tables/feature_stability.csv** - Top features
4. **experiment_log.txt** - Detailed execution log

---

## Quick Commands

### Check current status:

```bash
# How many folds completed?
ls checkpoints/*.pkl | wc -l

# What's in the log?
tail -f experiment_log.txt
```

### Resume if interrupted:

Just run "Run All" again! The notebook automatically:

- ✅ Detects completed folds
- ✅ Loads checkpoints
- ✅ Skips to next incomplete fold

### Clean restart:

```bash
# Delete all results (start fresh)
rm -rf results/ checkpoints/ cache/ experiment_log.txt
```

---

## Expected Timeline

| Stage                | Time           | What's Happening                          |
| -------------------- | -------------- | ----------------------------------------- |
| Setup                | 1-2 min        | Imports, directories, config              |
| Feature Extraction   | 20-30 min      | Computing 120-180 features for all epochs |
| CV Verification      | 1-2 min        | Checking fold distributions               |
| **Fold 1**           | 15-20 min      | Training RF, XGB-Full, SHAP, XGB-SHAP     |
| **Fold 2**           | 15-20 min      | Same as Fold 1                            |
| **Fold 3**           | 15-20 min      | Same as Fold 1                            |
| **Fold 4**           | 15-20 min      | Same as Fold 1                            |
| **Fold 5**           | 15-20 min      | Same as Fold 1                            |
| Statistical Analysis | 2-3 min        | Wilcoxon, Cohen's d, Bayes Factor         |
| Interpretability     | 2-3 min        | Feature stability, biological meaning     |
| Visualizations       | 5-8 min        | Generating 14+ figures at 300 DPI         |
| Final Summary        | 1 min          | Comprehensive report                      |
| **TOTAL**            | **90-120 min** | ☕☕☕                                    |

_Times based on i5-12450HX with RTX 3050, 12GB RAM_

---

## Success Indicators

### During Execution

✅ Memory usage stays below 10 GB  
✅ Progress bars complete smoothly  
✅ No Python errors in console  
✅ Checkpoints appear in `checkpoints/`  
✅ Log file grows continuously

### After Completion

✅ See "EXPERIMENT COMPLETE!" message  
✅ 14+ PNG files in `results/figures/`  
✅ 18+ CSV files in `results/tables/`  
✅ 5 checkpoint files (fold_0 to fold_4)  
✅ Final summary with p-value printed

---

## Troubleshooting (If Issues Occur)

### "Out of Memory" Error

```python
# In cell 4, reduce batch size:
BATCH_SIZE = 5  # Instead of 10
```

### "GPU not found" (Not an error!)

```
# Expected message if no CUDA GPU:
"XGBoostFactory: GPU not available, using CPU (tree_method=hist)"
# Notebook continues normally, just slower
```

### "Feature extraction too slow"

```python
# In cell 4, increase workers (if you have cores):
N_JOBS = 4  # Instead of 3
```

### Cache issues

```bash
# Delete cache and rebuild:
rm -rf cache/
# Then re-run the notebook
```

### Resume from specific fold

```bash
# Keep only completed folds, delete others:
ls checkpoints/
rm checkpoints/fold_3_complete.pkl  # If fold 3 failed
# Re-run notebook - it will redo fold 3
```

---

## What to Do After Completion

### 1. Quick Check (5 minutes)

- Open `results/tables/statistical_tests.csv`
- Look at p-value for "XGB-Full vs XGB-SHAP"
- If p < 0.05: **Success!** SHAP significantly improved performance
- Check `results/figures/main/model_comparison_boxplot.png`

### 2. Deep Dive (30 minutes)

- Read through all tables in `results/tables/`
- View all figures in `results/figures/`
- Read final summary in notebook output
- Review top stable features

### 3. Paper Writing (hours to days)

- Copy figures to manuscript
- Reference tables in results section
- Discuss feature interpretability
- Compare with literature
- Acknowledge limitations

---

## Need Help?

### Check These First

1. **IMPLEMENTATION_COMPLETE.md** - Full documentation
2. **README.md** - Project overview
3. **experiment_log.txt** - Detailed execution trace
4. **requirements.txt** - Dependency versions

### Common Questions

**Q: Can I run this on CPU only?**  
A: Yes! GPU is optional. It will be ~2-3× slower (3-4 hours total).

**Q: What if I only have 8GB RAM?**  
A: Reduce `BATCH_SIZE = 5` and `N_JOBS = 2` in cell 4.

**Q: Can I use fewer subjects?**  
A: Yes, but statistical power decreases. Edit `get_all_cassette_subjects()` to return fewer IDs.

**Q: How do I cite this work?**  
A: See README.md for APA citation template.

**Q: Can I modify hyperparameters?**  
A: Yes! Edit `RF_PARAMS` and `XGB_PARAMS` in cell 4.

---

## Ready? Let's Go! 🎉

```bash
# Open the notebook
jupyter notebook sleep_edf_production.ipynb

# Or in VS Code:
code sleep_edf_production.ipynb
```

**Click "Run All" and grab a coffee! ☕**

Expected completion: **90-120 minutes**  
Expected outcome: **Publication-ready sleep staging analysis**

---

_Good luck with your research!_ 🚀

**Questions?** Check IMPLEMENTATION_COMPLETE.md for detailed documentation.
