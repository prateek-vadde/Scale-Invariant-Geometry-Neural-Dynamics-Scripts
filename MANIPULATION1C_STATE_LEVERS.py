#!/usr/bin/env python3
"""
MANIPULATION 1C: MULTIPLE STATE LEVERS - HEAVILY GPU/CPU OPTIMIZED

Tests 4 orthogonal state manipulations:
1. POST-ERROR vs PRE-ERROR
2. TIME-ON-TASK / FATIGUE
3. VIGILANCE / LAPSES (RT)
4. CONFLICT STRENGTH

OPTIMIZATIONS:
- GPU batch processing (all trials × timescales at once per subject)
- 64-core parallelization for subject loading
- Vectorized state indicator computation
- Minimal Python loops
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count
from scipy import stats
from scipy.linalg import svd
import mne
from statsmodels.formula.api import mixedlm
import json
from functools import partial

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback
    print("⚠ Using CPU (CuPy not available)")

N_JOBS = min(64, cpu_count())
print(f"✓ {N_JOBS} CPU cores for parallelization")

CONFIG = {
    'SESOI': 0.05,
    'ROPE': 0.03,
    'timescales_ms': np.array([20, 30, 40, 50, 75, 100, 150, 200]),
    'analysis_window': (0.0, 0.5),
    'sfreq': 500,
    'base_dir': Path("/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/dataset_downloads/nencki_symfonia")
}

print("\n" + "="*80)
print("STATE LEVERS - GPU/CPU OPTIMIZED")
print("="*80)

def compute_kappa_batch_gpu(X_batch, epsilon=1e-10):
    """GPU batch SVD for multiple trials

    X_batch: (n_trials, channels, time)
    Returns: kappas (n_trials,), lambda_mins (n_trials,)
    """
    n_trials, n_channels, n_time = X_batch.shape

    # Center each trial
    X_centered = X_batch - X_batch.mean(axis=2, keepdims=True)

    if GPU_AVAILABLE:
        # Move to GPU
        X_gpu = cp.array(X_centered)

        # Batch SVD
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

        # Move back to CPU
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

def process_subject_full(args):
    """Load subject and compute ALL kappas for ALL timescales (GPU accelerated)"""
    sub_id, config = args

    try:
        sub_dir = config['base_dir'] / f"sub-{sub_id:02d}" / "eeg"
        vhdr_file = sub_dir / f"sub-{sub_id:02d}_task-msit_eeg.vhdr"
        events_tsv = sub_dir / f"sub-{sub_id:02d}_task-msit_events.tsv"

        if not vhdr_file.exists() or not events_tsv.exists():
            return None

        # Load EEG
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

        # Match responses
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

        if len(trial_info) == 0:
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

        # Extract analysis window
        sfreq = config['sfreq']
        start_idx = int((config['analysis_window'][0] + 0.2) * sfreq)
        end_idx = int((config['analysis_window'][1] + 0.2) * sfreq)
        epochs_data = epochs.get_data()[:, :, start_idx:end_idx]  # (trials, channels, time)

        # Compute state indicators (VECTORIZED)
        post_error = np.zeros(n_trials, dtype=bool)
        post_error[1:] = trial_df_kept['is_error'].values[:-1]

        trial_index_norm = np.arange(n_trials) / n_trials
        quartiles = pd.qcut(trial_index_norm, q=4, labels=False)
        is_early = (quartiles == 0)
        is_late = (quartiles == 3)

        rt_values = trial_df_kept['rt'].values
        rt_z = stats.zscore(rt_values)
        rt_quartiles = pd.qcut(rt_values, q=4, labels=False, duplicates='drop')
        is_fast = (rt_quartiles == 0)
        is_slow = (rt_quartiles == 3)

        conflict_strength = trial_df_kept['is_incongruent'].astype(float).values

        # GPU BATCH PROCESSING: compute kappa for all trials × all timescales
        rows = []

        for tau_ms in config['timescales_ms']:
            target_rate = 1000.0 / tau_ms
            decim_factor = max(1, int(np.round(sfreq / target_rate)))

            # Downsample all trials at once
            data_downsampled = epochs_data[:, :, ::decim_factor]  # (trials, channels, time_down)

            if data_downsampled.shape[2] >= 3:
                # Batch compute kappa for all trials
                kappas, lambda_mins = compute_kappa_batch_gpu(data_downsampled)

                # Filter valid trials
                valid = np.isfinite(kappas) & (kappas > 1) & (lambda_mins > 1e-10)

                # Build rows for this timescale
                for trial_idx in np.where(valid)[0]:
                    rows.append({
                        'subject_id': sub_id,
                        'trial_idx': int(trial_idx),
                        'timescale_ms': int(tau_ms),
                        'log_tau': np.log10(tau_ms),
                        'kappa': float(kappas[trial_idx]),
                        'log_kappa': np.log10(kappas[trial_idx]),
                        'post_error': int(post_error[trial_idx]),
                        'time_on_task': float(trial_index_norm[trial_idx]),
                        'is_early': int(is_early[trial_idx]),
                        'is_late': int(is_late[trial_idx]),
                        'rt_z': float(rt_z[trial_idx]),
                        'is_fast': int(is_fast[trial_idx]),
                        'is_slow': int(is_slow[trial_idx]),
                        'conflict_strength': float(conflict_strength[trial_idx]),
                        'lambda_min': float(lambda_mins[trial_idx])
                    })

        print(f"   Sub-{sub_id:02d}: {n_trials} trials → {len(rows)} valid rows")
        return rows

    except Exception as e:
        print(f"   Error sub-{sub_id:02d}: {e}")
        return None

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

print("\n[1/5] Parallel processing all subjects (GPU batch + 64-core)...")

subject_ids = list(range(1, 43))
args_list = [(sid, CONFIG) for sid in subject_ids]

with Pool(N_JOBS) as pool:
    results = pool.map(process_subject_full, args_list)

# Flatten results
all_rows = []
for result in results:
    if result is not None:
        all_rows.extend(result)

df = pd.DataFrame(all_rows)
print(f"\n✓ Created dataframe: {len(df):,} rows")
print(f"  Subjects: {df['subject_id'].nunique()}")
print(f"  Post-error: {df['post_error'].sum():,}")
print(f"  Early/Late: {df['is_early'].sum():,} / {df['is_late'].sum():,}")
print(f"  Fast/Slow: {df['is_fast'].sum():,} / {df['is_slow'].sum():,}")

# =============================================================================
# FIT MODELS
# =============================================================================

print("\n[2/5] Fitting LMMs for each state lever...")

results_dict = {}

# 1. POST-ERROR
print("\n[POST-ERROR]", end=" ")
df_pe = df[df['post_error'].isin([0, 1])]
model = mixedlm("log_kappa ~ log_tau + post_error + log_tau:post_error + time_on_task",
                df_pe, groups=df_pe["subject_id"], re_formula="~log_tau")
res = model.fit(method='lbfgs', maxiter=200)
beta = res.params['log_tau:post_error']
se = res.bse['log_tau:post_error']
p = res.pvalues['log_tau:post_error']
ci = res.conf_int(alpha=0.05).loc['log_tau:post_error']
results_dict['post_error'] = {
    'beta': float(beta), 'se': float(se), 'p': float(p),
    'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
    'exceeds_SESOI': bool(abs(beta) >= CONFIG['SESOI']),
    'within_ROPE': bool(abs(beta) < CONFIG['ROPE']),
    'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)
}
print(f"β={beta:.4f}, p={p:.4g}")

# 2. TIME-ON-TASK (continuous)
print("[TIME-CONT]", end=" ")
model = mixedlm("log_kappa ~ log_tau + time_on_task + log_tau:time_on_task",
                df, groups=df["subject_id"], re_formula="~log_tau")
res = model.fit(method='lbfgs', maxiter=200)
beta = res.params['log_tau:time_on_task']
se = res.bse['log_tau:time_on_task']
p = res.pvalues['log_tau:time_on_task']
ci = res.conf_int(alpha=0.05).loc['log_tau:time_on_task']
results_dict['time_continuous'] = {
    'beta': float(beta), 'se': float(se), 'p': float(p),
    'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
    'exceeds_SESOI': bool(abs(beta) >= CONFIG['SESOI']),
    'within_ROPE': bool(abs(beta) < CONFIG['ROPE']),
    'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)
}
print(f"β={beta:.4f}, p={p:.4g}")

# 3. TIME-ON-TASK (quartiles)
print("[TIME-QUART]", end=" ")
df_el = df[(df['is_early'] == 1) | (df['is_late'] == 1)]
model = mixedlm("log_kappa ~ log_tau + is_late + log_tau:is_late + time_on_task",
                df_el, groups=df_el["subject_id"], re_formula="~log_tau")
res = model.fit(method='lbfgs', maxiter=200)
beta = res.params['log_tau:is_late']
se = res.bse['log_tau:is_late']
p = res.pvalues['log_tau:is_late']
ci = res.conf_int(alpha=0.05).loc['log_tau:is_late']
results_dict['time_quartiles'] = {
    'beta': float(beta), 'se': float(se), 'p': float(p),
    'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
    'exceeds_SESOI': bool(abs(beta) >= CONFIG['SESOI']),
    'within_ROPE': bool(abs(beta) < CONFIG['ROPE']),
    'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)
}
print(f"β={beta:.4f}, p={p:.4g}")

# 4. VIGILANCE (RT)
print("[VIGILANCE]", end=" ")
df_fs = df[(df['is_fast'] == 1) | (df['is_slow'] == 1)]
model = mixedlm("log_kappa ~ log_tau + is_slow + log_tau:is_slow + time_on_task",
                df_fs, groups=df_fs["subject_id"], re_formula="~log_tau")
res = model.fit(method='lbfgs', maxiter=200)
beta = res.params['log_tau:is_slow']
se = res.bse['log_tau:is_slow']
p = res.pvalues['log_tau:is_slow']
ci = res.conf_int(alpha=0.05).loc['log_tau:is_slow']
results_dict['vigilance'] = {
    'beta': float(beta), 'se': float(se), 'p': float(p),
    'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
    'exceeds_SESOI': bool(abs(beta) >= CONFIG['SESOI']),
    'within_ROPE': bool(abs(beta) < CONFIG['ROPE']),
    'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)
}
print(f"β={beta:.4f}, p={p:.4g}")

# 5. CONFLICT
print("[CONFLICT]", end=" ")
model = mixedlm("log_kappa ~ log_tau + conflict_strength + log_tau:conflict_strength + time_on_task",
                df, groups=df["subject_id"], re_formula="~log_tau")
res = model.fit(method='lbfgs', maxiter=200)
beta = res.params['log_tau:conflict_strength']
se = res.bse['log_tau:conflict_strength']
p = res.pvalues['log_tau:conflict_strength']
ci = res.conf_int(alpha=0.05).loc['log_tau:conflict_strength']
results_dict['conflict'] = {
    'beta': float(beta), 'se': float(se), 'p': float(p),
    'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
    'exceeds_SESOI': bool(abs(beta) >= CONFIG['SESOI']),
    'within_ROPE': bool(abs(beta) < CONFIG['ROPE']),
    'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)
}
print(f"β={beta:.4f}, p={p:.4g}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n\n[3/5] " + "="*80)
print("SUMMARY: ALL STATE LEVERS")
print("="*80)

for name, res in results_dict.items():
    hit = "✓✓ HIT" if (res['exceeds_SESOI'] and res['ci_excludes_zero']) else ("✓ TREND" if res['ci_excludes_zero'] else "NULL")
    print(f"\n{name.upper():20s} {hit}")
    print(f"  β = {res['beta']:+.4f} ± {res['se']:.4f}, 95% CI [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]")
    print(f"  p = {res['p']:.4g}, SESOI: {res['exceeds_SESOI']}, ROPE: {res['within_ROPE']}")

# =============================================================================
# SAVE
# =============================================================================

print("\n[4/5] Saving results...")

output = {
    'config': {k: v.tolist() if isinstance(v, np.ndarray) else str(v) if isinstance(v, Path) else v
               for k, v in CONFIG.items()},
    'n_subjects': int(df['subject_id'].nunique()),
    'n_rows': int(len(df)),
    'results': results_dict
}

with open('STATE_LEVERS_results.json', 'w') as f:
    json.dump(output, f, indent=2)

df.to_csv('STATE_LEVERS_data.csv', index=False)

# =============================================================================
# INTERPRETATION
# =============================================================================

print("\n[5/5] INTERPRETATION:")

hits = [name for name, res in results_dict.items() if res['exceeds_SESOI'] and res['ci_excludes_zero']]
trends = [name for name, res in results_dict.items() if res['ci_excludes_zero'] and not res['exceeds_SESOI']]
nulls = [name for name, res in results_dict.items() if not res['ci_excludes_zero']]

if len(hits) > 0:
    print(f"\n✓✓ {len(hits)} STRONG HIT(S) (exceed SESOI + CI excludes zero):")
    for n in hits:
        print(f"   {n}: β = {results_dict[n]['beta']:.4f}, p = {results_dict[n]['p']:.4g}")

if len(trends) > 0:
    print(f"\n✓ {len(trends)} TREND(S) (CI excludes zero but below SESOI):")
    for n in trends:
        print(f"   {n}: β = {results_dict[n]['beta']:.4f}, p = {results_dict[n]['p']:.4g}")

if len(nulls) > 0:
    print(f"\n→ {len(nulls)} NULL(S) / NEGATIVE CONTROLS:")
    for n in nulls:
        print(f"   {n}: β = {results_dict[n]['beta']:.4f}, p = {results_dict[n]['p']:.4g}")

print("\n" + "="*80)
print("✓ COMPLETE - Results: STATE_LEVERS_results.json")
print("="*80)
