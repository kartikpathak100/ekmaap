# Additional modules

Six small, self-contained modules added to the `ekmaap` package. Each has its own tests in `tests/test_feature_*.py`. None of them changes the existing pipeline.

| Module | What it does | Used for |
| --- | --- | --- |
| `ekmaap/sampling.py` | Pools several photos into a lot-level result with a 95 % interval (Wilson); estimates how many photos a lot needs | Reporting how sure a lot percentage is |
| `ekmaap/export.py` | Writes an analysis as CSV (one row per onion) and as a short text summary | Auditors, spreadsheets, messages |
| `ekmaap/quality_checks.py` | Warns about blurred, dark, bright, glared or too-small photos before analysis | Fewer bad photos reaching the grader |
| `ekmaap/rules_io.py` | Saves, loads, validates and diffs grading rule sets as JSON | Loading the official limits when published |
| `ekmaap/size_stats.py` | Mean, spread, percentiles, uniformity (CV) and size histogram of a lot | Uniformity information for buyers |
| `ekmaap/agreement.py` | Agreement and Cohen's kappa between officer labels and system labels | The grader-agreement study in the field-validation plan |

## Status
- The modules are covered by unit tests that run on synthetic data and fixed numbers.
- The photo-quality thresholds are starting values and have not been tuned on real photographs.
- `agreement.py` needs real paired labels; no field data has been collected yet.
- They are not yet connected to the API or the officer app.

## Run the tests
```bash
python -m pytest tests/test_feature_*.py -q
```
