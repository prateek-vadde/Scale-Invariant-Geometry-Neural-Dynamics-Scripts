# Analysis Scripts Repository

This repository contains all analysis scripts used to generate the statistics and results reported in:

**"Scale-Invariant Geometry of Neural Dynamics Across Biological Timescales"**

Prateek Vadde (2025)

## Contents

All scripts are documented with their corresponding statistics in `STATISTIC_TO_SCRIPT_MAP.md`.

### Main Analysis Scripts

1. **ANALYZE_TMS_EEG.py** - TMS theta burst stimulation analysis
   - Produces: `tms_results.json`
   - Statistics: n=10 subjects, Δα = +0.026 ± 0.013, permutation p = 0.0035

2. **TMS_ENHANCED_STATS.py** - Enhanced TMS statistical analysis
   - Produces: `tms_enhanced_stats.json`
   - Requires: `tms_results.json`

3. **ANALYZE_PROPOFOL_PHARMA_EEG.py** - Propofol pharmaco-EEG analysis
   - Produces: `propofol_results.json`
   - Statistics: n=11 subjects, 2,032 windows, β = +0.023, p = 0.001

4. **MANIPULATION1_MSIT_RESCUE.py** - MSIT cognitive conflict analysis
   - Produces: `MSIT_RESCUE_results.json`
   - Statistics: n=41, Δα = +0.086 ± 0.033, t(40) = 2.60, p = 0.013

5. **MANIPULATION1C_STATE_LEVERS.py** - Continuous conflict analysis
   - Produces: `STATE_LEVERS_results.json`
   - Statistics: β = +0.0095 ± 0.0029, p = 0.001

6. **BEHAVIORAL_RT_PREDICTION.py** - Trial-level RT correlation
   - Produces: `behavioral_rt_prediction_results.json`
   - Statistics: r = 0.132, p < 0.001, t(41) = 9.71

7. **mathematical_derivation/nencki_rt_correlation_FIXED.py** - Trial-level κ-RT correlation
   - Produces: `mathematical_derivation/nencki_rt_correlation_results_FIXED.json`
   - Statistics: r = 0.132, R² = 0.017 (1.7%)

8. **NEUROPIXELS_IMPROVED.py** - Cross-species mouse Neuropixels analysis
   - Produces: `neuropixels_improved_results.pkl`, `neuropixels_improved.png`
   - Statistics: α = -0.64 ± 0.08, R² = 0.935

9. **clinical_aszed_analysis.py** - Schizophrenia clinical analysis
   - Produces: `clinical_aszed_results.json`
   - Statistics: n=27 patients, n=56 controls

10. **complete_unrejectability_analysis_GPU.py** - Comprehensive clinical analysis
    - Produces: `comprehensive_unrejectability_results.json`
    - Statistics: t = -3.19, p = 0.002, BF₁₀ = 494

11. **create_calibration_plot.py** - Theory-data calibration
    - Produces: `calibration_results_improved.json`, `calibration_plot_improved.png`
    - Statistics: Spearman ρ = 0.929, p = 0.0025

## Usage

All scripts can be run independently. See `STATISTIC_TO_SCRIPT_MAP.md` for detailed execution instructions and output verification.

### Execution Order (if reproducing all analyses):

```bash
# 1. TMS analysis
python ANALYZE_TMS_EEG.py
python TMS_ENHANCED_STATS.py

# 2. Propofol analysis
python ANALYZE_PROPOFOL_PHARMA_EEG.py

# 3. MSIT conflict analysis
python MANIPULATION1_MSIT_RESCUE.py

# 4. State levers
python MANIPULATION1C_STATE_LEVERS.py

# 5. Behavioral RT correlation
python mathematical_derivation/nencki_rt_correlation_FIXED.py

# 6. Neuropixels
python NEUROPIXELS_IMPROVED.py

# 7. Clinical schizophrenia
python complete_unrejectability_analysis_GPU.py

# 8. Calibration plot
python create_calibration_plot.py
```

## Requirements

See main manuscript repository for data dependencies and package requirements.

## Citation

If you use these scripts, please cite:

```
Vadde, P. (2025). Scale-Invariant Geometry of Neural Dynamics
Across Biological Timescales. bioRxiv. [DOI]
```

## License

These analysis scripts are provided for reproducibility and transparency.

## Contact

Prateek Vadde - prateek.vadde@gmail.com
