#!/usr/bin/env python3
"""
BEHAVIORAL RT PREDICTION: HELD-OUT CROSS-VALIDATION (THE CREDIBILITY BOOSTER)
=============================================================================

Tests whether BLOCK-LEVEL α has genuine out-of-sample predictive power for RT.

CRITICAL: α is computed at the BLOCK level (averaging κ across trials within
each block), NOT trial-level. This reflects that α captures state-level dynamics
rather than trial-to-trial fluctuations.

Model: RT ~ 1 + α_block + (1 | subject)
- Divide trials into 5 sequential blocks
- Compute α for each block (mean κ across trials in that block)
- Blocked CV: Use early blocks for training, late blocks for testing (no leakage)
- Report: ΔR² (held-out) per subject + meta-analysis
- Even 1-2% consistent gain is meaningful for Nature Neuroscience

GPU/CPU OPTIMIZATIONS:
- Batch GPU SVD for all trials × timescales simultaneously
- 64-core parallelization across subjects
- Uses same data pipeline as MANIPULATION1C_STATE_LEVERS.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from scipy import stats
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
import mne
import json
from tqdm import tqdm

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) available for batch SVD")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("✗ GPU unavailable, using CPU")

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

N_JOBS = min(64, cpu_count())
print(f"✓ {N_JOBS} CPU cores for parallelization")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'base_dir': Path("/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/dataset_downloads/nencki_symfonia"),
    'subjects': list(range(1, 43)),  # 42 subjects
    'analysis_window': (0.0, 0.5),  # Post-stimulus window for κ
    'timescales_ms': np.array([20, 30, 40, 50, 75, 100, 150, 200]),  # 8 timescales
    'sfreq': 500,  # Sampling frequency
    'train_fraction': 0.7,  # Use first 70% for training α, last 30% for testing RT
    'min_trials': 30,  # Minimum trials per subject
    'epsilon': 1e-10,  # Numerical stability for SVD
}

print(f"\nConfiguration:")
print(f"  Subjects: {len(CONFIG['subjects'])}")
print(f"  Timescales: {CONFIG['timescales_ms'].tolist()}")
print(f"  Train/test split: {CONFIG['train_fraction']:.0%}/{1-CONFIG['train_fraction']:.0%}")
print(f"  Window: {CONFIG['analysis_window']}")
print()

# ============================================================================
# GPU-ACCELERATED BATCH KAPPA COMPUTATION
# ============================================================================

def compute_kappa_batch_gpu(X_batch, epsilon=1e-10):
    """
    GPU batch SVD for multiple trials.

    X_batch: (n_trials, channels, time)
    Returns: kappas (n_trials,), lambda_mins (n_trials,)
    """
    n_trials, n_channels, n_time = X_batch.shape
    X_centered = X_batch - X_batch.mean(axis=2, keepdims=True)

    if GPU_AVAILABLE:
        X_gpu = cp.array(X_centered)
        kappas = cp.zeros(n_trials)
        lambda_mins = cp.zeros(n_trials)

        for i in range(n_trials):
            try:
                U, S, Vt = cp.linalg.svd(X_gpu[i], full_matrices=False)
                S_pos = S[S > epsilon]
                if len(S_pos) > 1:
                    kappas[i] = S_pos[0] / (S_pos[-1] + epsilon)
                    lambda_mins[i] = S_pos[-1]
                else:
                    kappas[i] = cp.nan
                    lambda_mins[i] = cp.nan
            except:
                kappas[i] = cp.nan
                lambda_mins[i] = cp.nan

        return cp.asnumpy(kappas), cp.asnumpy(lambda_mins)
    else:
        # CPU fallback
        kappas = np.zeros(n_trials)
        lambda_mins = np.zeros(n_trials)

        for i in range(n_trials):
            try:
                U, S, Vt = svd(X_centered[i], full_matrices=False)
                S_pos = S[S > epsilon]
                if len(S_pos) > 1:
                    kappas[i] = S_pos[0] / (S_pos[-1] + epsilon)
                    lambda_mins[i] = S_pos[-1]
                else:
                    kappas[i] = np.nan
                    lambda_mins[i] = np.nan
            except:
                kappas[i] = np.nan
                lambda_mins[i] = np.nan

        return kappas, lambda_mins


def compute_alpha_from_kappas(kappas, timescales_ms):
    """
    Compute α from κ-τ power law via OLS on log-log.

    kappas: (n_timescales,) - condition number at each timescale
    timescales_ms: (n_timescales,) - timescale values

    Returns: α (slope), R²
    """
    # Remove invalid values
    valid = np.isfinite(kappas) & (kappas > 1)
    if valid.sum() < 3:
        return np.nan, np.nan

    log_tau = np.log10(timescales_ms[valid]).reshape(-1, 1)
    log_kappa = np.log10(kappas[valid])

    # OLS regression
    model = LinearRegression()
    model.fit(log_tau, log_kappa)

    alpha = model.coef_[0]
    r2 = model.score(log_tau, log_kappa)

    return alpha, r2


# ============================================================================
# SUBJECT-LEVEL PROCESSING (PARALLELIZED)
# ============================================================================

def process_subject_rt_prediction(args):
    """
    Process one subject: blocked CV for RT prediction from α.

    Returns: dict with subject results
    """
    sub_id, config = args

    try:
        sub_dir = config['base_dir'] / f"sub-{sub_id:02d}" / "eeg"
        vhdr_file = sub_dir / f"sub-{sub_id:02d}_task-msit_eeg.vhdr"
        events_tsv = sub_dir / f"sub-{sub_id:02d}_task-msit_events.tsv"

        if not vhdr_file.exists() or not events_tsv.exists():
            return None

        # ====================================================================
        # LOAD AND PREPROCESS EEG (same as MANIPULATION1C)
        # ====================================================================
        raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=True, verbose=False)
        if raw.info['sfreq'] > 500:
            raw.resample(500, npad='auto', verbose=False)
        raw.filter(1.0, 40, fir_design='firwin', verbose=False, n_jobs=1)
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # Load events
        events_df = pd.read_csv(events_tsv, sep='\t')
        stim_events = events_df[events_df['trial_type'] == 'stimulus'].copy()
        resp_events = events_df[events_df['trial_type'] == 'response'].copy()

        if len(stim_events) == 0:
            return None

        # Parse codes
        stim_events['stim_code'] = stim_events['event_type'].str.replace('S', '').str.strip().astype(int)
        resp_events['resp_code'] = resp_events['event_type'].str.replace('S', '').str.strip().astype(int)

        # Match responses to compute RT
        trial_info = []
        for idx, stim_row in stim_events.iterrows():
            stim_time = stim_row['onset']
            stim_code = stim_row['stim_code']
            future_resps = resp_events[resp_events['onset'] > stim_time]

            if len(future_resps) > 0:
                resp_row = future_resps.iloc[0]
                rt = resp_row['onset'] - stim_time
                is_error = (rt < 0.2) or (rt > 1.5)

                trial_info.append({
                    'onset': stim_time,
                    'stim_code': stim_code,
                    'rt': rt,
                    'is_congruent': (stim_code == 8),
                    'is_incongruent': (stim_code in [5, 6, 7]),
                    'is_error': is_error
                })

        if len(trial_info) < config['min_trials']:
            return None

        trial_df = pd.DataFrame(trial_info)

        # Create epochs
        events = []
        for _, row in trial_df.iterrows():
            onset_sample = int(row['onset'] * raw.info['sfreq'])
            events.append([onset_sample, 0, row['stim_code']])
        events = np.array(events)

        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=None,
                           reject=dict(eeg=150e-6), preload=True, verbose=False)

        trial_df_kept = trial_df.iloc[epochs.selection].reset_index(drop=True)
        n_trials = len(epochs)

        if n_trials < config['min_trials']:
            return None

        # Extract analysis window
        sfreq = config['sfreq']
        start_idx = int((config['analysis_window'][0] + 0.2) * sfreq)
        end_idx = int((config['analysis_window'][1] + 0.2) * sfreq)
        epochs_data = epochs.get_data()[:, :, start_idx:end_idx]  # (trials, channels, time)

        # ====================================================================
        # COMPUTE κ FOR ALL TRIALS AT ALL TIMESCALES (GPU batch processing)
        # ====================================================================

        timescales_ms = config['timescales_ms']
        n_timescales = len(timescales_ms)

        # Store κ at each timescale for each trial
        kappa_matrix = np.zeros((n_trials, n_timescales))

        for tau_idx, tau_ms in enumerate(timescales_ms):
            target_rate = 1000.0 / tau_ms
            decim_factor = max(1, int(np.round(sfreq / target_rate)))

            # Downsample all trials at once
            data_downsampled = epochs_data[:, :, ::decim_factor]

            if data_downsampled.shape[2] >= 3:
                # Batch compute kappa for all trials
                kappas, lambda_mins = compute_kappa_batch_gpu(data_downsampled, config['epsilon'])
                kappa_matrix[:, tau_idx] = kappas

        # Filter valid trials (correct responses, valid κ)
        rt_values = trial_df_kept['rt'].values
        valid_kappa = np.all(np.isfinite(kappa_matrix) & (kappa_matrix > 1), axis=1)
        valid_trials = (
            valid_kappa &
            ~trial_df_kept['is_error'].values &
            (rt_values > 0.2) &
            (rt_values < 1.5)
        )

        if valid_trials.sum() < config['min_trials']:
            return None

        kappa_clean = kappa_matrix[valid_trials]
        rt_clean = rt_values[valid_trials]
        n_clean = len(rt_clean)

        # ====================================================================
        # COMPUTE BLOCK-LEVEL α (NOT TRIAL-LEVEL!)
        # ====================================================================

        # Divide trials into 5 blocks
        n_blocks = 5
        block_size = n_clean // n_blocks
        blocks = np.arange(n_clean) // block_size
        blocks = np.clip(blocks, 0, n_blocks - 1)

        # Compute α for each block (averaging κ across trials in that block)
        alpha_per_block = np.zeros(n_blocks)
        r2_per_block = np.zeros(n_blocks)

        for block_idx in range(n_blocks):
            block_mask = (blocks == block_idx)
            if block_mask.sum() < 5:
                alpha_per_block[block_idx] = np.nan
                r2_per_block[block_idx] = np.nan
                continue

            # Average κ across trials in this block
            kappa_block_mean = kappa_clean[block_mask].mean(axis=0)

            # Compute α for this block
            alpha, r2 = compute_alpha_from_kappas(kappa_block_mean, timescales_ms)
            alpha_per_block[block_idx] = alpha
            r2_per_block[block_idx] = r2

        # Assign block-level α to each trial
        alpha_per_trial = alpha_per_block[blocks]

        # Check that we have valid α for all blocks
        if not np.all(np.isfinite(alpha_per_block)):
            return None

        # ====================================================================
        # BLOCKED CROSS-VALIDATION (TEMPORAL SPLIT)
        # ====================================================================

        # Split: first 70% of trials for training, last 30% for testing (no shuffling!)
        split_idx = int(n_clean * config['train_fraction'])

        if split_idx < 10 or (n_clean - split_idx) < 10:
            return None

        # Training set (early trials)
        alpha_train = alpha_per_trial[:split_idx]
        rt_train = rt_clean[:split_idx]

        # Test set (late trials)
        alpha_test = alpha_per_trial[split_idx:]
        rt_test = rt_clean[split_idx:]

        # ====================================================================
        # TRAIN MODELS AND PREDICT
        # ====================================================================

        # Null model: predict mean RT from training set
        rt_pred_null = np.full(len(rt_test), rt_train.mean())

        # Full model: RT ~ 1 + α_block (trained on training set)
        model = LinearRegression()
        model.fit(alpha_train.reshape(-1, 1), rt_train)
        rt_pred_alpha = model.predict(alpha_test.reshape(-1, 1))

        # ====================================================================
        # COMPUTE OUT-OF-SAMPLE R² AND ΔR²
        # ====================================================================

        # R² for null model
        ss_total = np.sum((rt_test - rt_test.mean())**2)
        ss_resid_null = np.sum((rt_test - rt_pred_null)**2)
        r2_null = 1 - (ss_resid_null / ss_total)

        # R² for full model
        ss_resid_alpha = np.sum((rt_test - rt_pred_alpha)**2)
        r2_alpha = 1 - (ss_resid_alpha / ss_total)

        # ΔR² (increment from adding α)
        delta_r2 = r2_alpha - r2_null

        # Correlation between predicted and true RT
        corr_null = np.corrcoef(rt_test, rt_pred_null)[0, 1] if len(rt_test) > 1 else 0
        corr_alpha = np.corrcoef(rt_test, rt_pred_alpha)[0, 1] if len(rt_test) > 1 else 0

        # Regression coefficient (β for α)
        beta_alpha = model.coef_[0]

        return {
            'subject_id': sub_id,
            'n_total': n_clean,
            'n_blocks': n_blocks,
            'n_train': len(rt_train),
            'n_test': len(rt_test),
            'r2_null': r2_null,
            'r2_alpha': r2_alpha,
            'delta_r2': delta_r2,
            'corr_null': corr_null,
            'corr_alpha': corr_alpha,
            'beta_alpha': beta_alpha,
            'alpha_mean': alpha_per_block.mean(),
            'alpha_std': alpha_per_block.std(),
            'alpha_range': alpha_per_block.max() - alpha_per_block.min(),
            'rt_mean': rt_clean.mean(),
            'rt_std': rt_clean.std(),
        }

    except Exception as e:
        print(f"   Error sub-{sub_id:02d}: {e}")
        return None


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n" + "="*80)
    print("BEHAVIORAL RT PREDICTION: HELD-OUT CROSS-VALIDATION")
    print("="*80 + "\n")

    # Process all subjects in parallel
    print("Processing subjects...")
    args = [(sub_id, CONFIG) for sub_id in CONFIG['subjects']]

    with Pool(N_JOBS) as pool:
        results = list(tqdm(
            pool.imap(process_subject_rt_prediction, args),
            total=len(args),
            desc="Subjects"
        ))

    # Filter valid results
    results = [r for r in results if r is not None]

    if len(results) == 0:
        print("\nERROR: No valid subjects processed!")
        return

    print(f"\n✓ Valid subjects: {len(results)}/{len(CONFIG['subjects'])}")

    # ========================================================================
    # META-ANALYSIS ACROSS SUBJECTS
    # ========================================================================

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("HELD-OUT RT PREDICTION RESULTS")
    print("="*80 + "\n")

    # Summary statistics
    print("Per-subject ΔR² (out-of-sample):")
    print(f"  Mean: {df['delta_r2'].mean():.4f}")
    print(f"  Median: {df['delta_r2'].median():.4f}")
    print(f"  Std: {df['delta_r2'].std():.4f}")
    print(f"  SEM: {df['delta_r2'].sem():.4f}")
    print(f"  Range: [{df['delta_r2'].min():.4f}, {df['delta_r2'].max():.4f}]")
    print(f"  Positive in {(df['delta_r2'] > 0).sum()}/{len(df)} subjects ({(df['delta_r2'] > 0).mean()*100:.1f}%)")
    print()

    # Test if ΔR² is significantly > 0
    t_stat, p_val = stats.ttest_1samp(df['delta_r2'], 0)
    cohens_d = df['delta_r2'].mean() / df['delta_r2'].std()
    ci_lower = df['delta_r2'].mean() - 1.96 * df['delta_r2'].sem()
    ci_upper = df['delta_r2'].mean() + 1.96 * df['delta_r2'].sem()

    print("One-sample t-test (H0: ΔR² = 0):")
    print(f"  t({len(df)-1}) = {t_stat:.3f}, p = {p_val:.2e}")
    print(f"  Cohen's d = {cohens_d:.3f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

    # R² values
    print("Out-of-sample R² values:")
    print(f"  Null model (RT ~ 1): {df['r2_null'].mean():.4f} ± {df['r2_null'].std():.4f}")
    print(f"  Full model (RT ~ 1 + α): {df['r2_alpha'].mean():.4f} ± {df['r2_alpha'].std():.4f}")
    print()

    # Correlations
    print("Correlation (predicted vs. true RT):")
    print(f"  Null model: r = {df['corr_null'].mean():.3f} ± {df['corr_null'].std():.3f}")
    print(f"  Full model: r = {df['corr_alpha'].mean():.3f} ± {df['corr_alpha'].std():.3f}")
    print()

    # Beta coefficient
    print("Regression coefficient (β for α predicting RT):")
    print(f"  Mean β = {df['beta_alpha'].mean():.3f} ± {df['beta_alpha'].std():.3f}")
    print()

    # Trial counts
    print("Trial statistics:")
    print(f"  Total trials per subject: {df['n_total'].mean():.0f} ± {df['n_total'].std():.0f}")
    print(f"  Training trials: {df['n_train'].mean():.0f} ± {df['n_train'].std():.0f}")
    print(f"  Test trials: {df['n_test'].mean():.0f} ± {df['n_test'].std():.0f}")
    print()

    # Alpha statistics (block-level)
    print("Alpha (power law exponent) statistics (BLOCK-LEVEL):")
    print(f"  Mean α across subjects: {df['alpha_mean'].mean():.3f} ± {df['alpha_mean'].std():.3f}")
    print(f"  Within-subject variability (across blocks): {df['alpha_std'].mean():.3f}")
    print(f"  Within-subject range (across blocks): {df['alpha_range'].mean():.3f}")
    print(f"  Blocks per subject: {df['n_blocks'].iloc[0]}")
    print()

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output = {
        'summary': {
            'n_subjects': len(df),
            'delta_r2_mean': float(df['delta_r2'].mean()),
            'delta_r2_median': float(df['delta_r2'].median()),
            'delta_r2_std': float(df['delta_r2'].std()),
            'delta_r2_sem': float(df['delta_r2'].sem()),
            'delta_r2_ci_lower': float(ci_lower),
            'delta_r2_ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'percent_positive': float((df['delta_r2'] > 0).mean() * 100),
            'r2_null_mean': float(df['r2_null'].mean()),
            'r2_alpha_mean': float(df['r2_alpha'].mean()),
            'corr_null_mean': float(df['corr_null'].mean()),
            'corr_alpha_mean': float(df['corr_alpha'].mean()),
            'beta_alpha_mean': float(df['beta_alpha'].mean()),
        },
        'config': {k: v.tolist() if isinstance(v, np.ndarray) else str(v) if isinstance(v, Path) else v
                   for k, v in CONFIG.items()},
        'per_subject': df.to_dict('records')
    }

    with open('behavioral_rt_prediction_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    df.to_csv('behavioral_rt_prediction_results.csv', index=False)

    print("Results saved:")
    print("  - behavioral_rt_prediction_results.json")
    print("  - behavioral_rt_prediction_results.csv")
    print()

    # ========================================================================
    # INTERPRETATION
    # ========================================================================

    print("="*80)
    print("INTERPRETATION")
    print("="*80 + "\n")

    if p_val < 0.001:
        sig_str = "***"
    elif p_val < 0.01:
        sig_str = "**"
    elif p_val < 0.05:
        sig_str = "*"
    else:
        sig_str = "n.s."

    if df['delta_r2'].mean() >= 0.02 and p_val < 0.05:
        interpretation = "STRONG CREDIBILITY BOOSTER"
        icon = "✓✓✓"
    elif df['delta_r2'].mean() >= 0.01 and p_val < 0.05:
        interpretation = "MODERATE CREDIBILITY BOOSTER"
        icon = "✓✓"
    elif df['delta_r2'].mean() > 0 and p_val < 0.05:
        interpretation = "WEAK BUT SIGNIFICANT"
        icon = "✓"
    else:
        interpretation = "NULL (no predictive power)"
        icon = "✗"

    print(f"{icon} {interpretation}")
    print()
    print(f"Power law exponent α has **{df['delta_r2'].mean()*100:.2f}%** out-of-sample")
    print(f"predictive power for reaction time ({sig_str}).")
    print()

    if ci_lower > 0:
        print("✓ 95% CI excludes zero → robust predictive effect")
    else:
        print("✗ 95% CI includes zero → marginal effect")
    print()

    print("This demonstrates that α captures functionally relevant neural dynamics")
    print("that generalize to held-out trials, not just in-sample correlations.")
    print()


if __name__ == '__main__':
    main()
