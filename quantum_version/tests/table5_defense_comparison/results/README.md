# Table 5 Defense Comparison - Results Placeholder

Results will be generated here when you run:
```bash
python run_all_tests.py
```

## Generated Files

- `table5_results_YYYYMMDD_HHMMSS.json` - Full results with all metrics
- `table5_results_YYYYMMDD_HHMMSS.csv` - CSV format for spreadsheet
- `table5_latex_YYYYMMDD_HHMMSS.txt` - LaTeX table code for paper
- `table5_comparison_YYYYMMDD_HHMMSS.png` - Visual comparison plots

## Expected Results (Placeholder)

| Method | Detection Rate (%) | FPR (%) | F1-Score | Test Acc (%) |
|--------|-------------------|---------|----------|--------------|
| FedAvg (No Defense) | N/A | N/A | N/A | ~10-15 |
| Krum | ~65 | ~15 | ~0.65 | ~68 |
| Median | ~82 | ~8 | ~0.82 | ~76 |
| Trimmed-Mean | ~78 | ~10 | ~0.78 | ~72 |
| RobustAvg | ~73 | ~13 | ~0.73 | ~69 |
| **QuantumDefend** | **>95** | **<5** | **>0.95** | **~85** |

Run the tests to get actual results!
