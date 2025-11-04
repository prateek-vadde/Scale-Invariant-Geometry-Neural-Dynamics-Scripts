"""
TMS-EEG Theta Burst Stimulation Analysis - EXTREME GPU ACCELERATION
====================================================================

OPTIMIZATIONS FOR GH200:
- Batch ALL windows across ALL files on GPU simultaneously
- CuPy for all array operations (SVD, PCA, filtering)
- Pre-load all data into GPU memory
- Vectorized operations across window batches
- Numba JIT for CPU-side coordination only
"""

import h5py
import numpy as np
import os
import json
from scipy import stats as sp_stats
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')
import time

# GPU ACCELERATION
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    USE_GPU = True
    print("🚀 GPU ACCELERATION ENABLED (CuPy)")
    try:
        print(f"   GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except:
        print(f"   GPU: Device 0")
    mempool = cp.get_default_memory_pool()
    print(f"   GPU Memory available")
except ImportError:
    cp = np
    USE_GPU = False
    print("⚠️  GPU not available")

from numba import jit

# =============================================================================
# PARAMETERS
# =============================================================================

DATA_DIR = '/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/tms_eeg_data'
OUTPUT_FILE = '/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/tms_results.json'

TIMESCALES_MS = np.array([50, 75, 100, 125, 150, 175, 200])
WINDOW_SEC = 4.0
OVERLAP_SEC = 3.5
R2_THRESHOLD = 0.50

LOWCUT_HZ = 1.0
HIGHCUT_HZ = 40.0

N_PCA_COMPONENTS = 10
EPSILON = 1e-12
ALPHA = 0.05

print("=" * 80)
print("TMS-EEG THETA BURST STIMULATION ANALYSIS")
print("=" * 80)

# =============================================================================
# BATCHED GPU FUNCTIONS
# =============================================================================

def load_all_fieldtrip_files(filepaths):
    """
    Load multiple FieldTrip files and return concatenated data.
    """
    all_data = []

    for filepath in filepaths:
        try:
            with h5py.File(filepath, 'r') as f:
                data_struct = f['data']
                fsample = data_struct['fsample'][()][0,0]

                trial_refs = data_struct['trial'][0, :]
                for ref in trial_refs:
                    trial_data = f[ref][()].T
                    all_data.append(trial_data)
        except Exception as e:
            print(f"    ⚠ Error loading {filepath}: {e}")
            continue

    if len(all_data) == 0:
        return None, None

    # Concatenate all trials
    data_concat = np.concatenate(all_data, axis=1)
    return data_concat, fsample

def bandpass_filter_gpu_batch(data_gpu, lowcut, highcut, fs, order=4):
    """
    GPU-accelerated bandpass filter using CuPy.
    """
    from scipy.signal import butter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')

    if USE_GPU:
        # Transfer SOS coefficients to GPU
        sos_gpu = cp.asarray(sos)
        # Use CuPy's sosfiltfilt
        filtered = cp_signal.sosfiltfilt(sos_gpu, data_gpu, axis=1)
    else:
        from scipy.signal import sosfiltfilt
        filtered = sosfiltfilt(sos, data_gpu, axis=1)

    return filtered

def extract_windows_gpu(data_gpu, window_samples, stride_samples):
    """
    Extract ALL windows at once on GPU using stride tricks.

    Returns:
        windows_gpu: (n_windows, n_channels, window_samples)
    """
    n_channels, n_timepoints = data_gpu.shape
    n_windows = (n_timepoints - window_samples) // stride_samples + 1

    if USE_GPU:
        # Pre-allocate on GPU
        windows = cp.zeros((n_windows, n_channels, window_samples), dtype=data_gpu.dtype)

        # Extract windows (vectorized)
        for i in range(n_windows):
            start = i * stride_samples
            end = start + window_samples
            windows[i] = data_gpu[:, start:end]
    else:
        windows = np.zeros((n_windows, n_channels, window_samples), dtype=data_gpu.dtype)
        for i in range(n_windows):
            start = i * stride_samples
            end = start + window_samples
            windows[i] = data_gpu[:, start:end]

    return windows

def downsample_batch_gpu(windows_gpu, downsample_factor):
    """
    Downsample all windows at once.

    Args:
        windows_gpu: (n_windows, n_channels, n_timepoints)

    Returns:
        downsampled: (n_windows, n_channels, n_timepoints // factor)
    """
    if downsample_factor <= 1:
        return windows_gpu

    return windows_gpu[:, :, ::downsample_factor]

def compute_kappa_batch_gpu(windows_gpu):
    """
    Compute condition number κ for ALL windows in parallel on GPU.

    Args:
        windows_gpu: (n_windows, n_channels, n_timepoints) on GPU

    Returns:
        kappas: (n_windows,) array of condition numbers
    """
    n_windows = windows_gpu.shape[0]

    if USE_GPU:
        kappas = cp.zeros(n_windows)

        # Process each window (could be further optimized with batched SVD)
        for i in range(n_windows):
            window = windows_gpu[i]  # (n_channels, n_timepoints)

            # Zero-mean
            window_centered = window - cp.mean(window, axis=1, keepdims=True)

            # PCA dimensionality reduction
            if window_centered.shape[1] > N_PCA_COMPONENTS:
                try:
                    U, S, Vt = cp.linalg.svd(window_centered, full_matrices=False)
                    window_reduced = U[:, :N_PCA_COMPONENTS] @ cp.diag(S[:N_PCA_COMPONENTS])
                except:
                    kappas[i] = cp.nan
                    continue
            else:
                window_reduced = window_centered

            # SVD
            try:
                U, S, Vt = cp.linalg.svd(window_reduced, full_matrices=False)
            except:
                kappas[i] = cp.nan
                continue

            # Condition number
            S_pos = S[S > EPSILON]
            if len(S_pos) >= 2:
                kappas[i] = S_pos[0] / (S_pos[-1] + EPSILON)
            else:
                kappas[i] = cp.nan

        return cp.asnumpy(kappas)  # Transfer back to CPU
    else:
        from sklearn.decomposition import PCA
        kappas = np.zeros(n_windows)

        for i in range(n_windows):
            window = windows_gpu[i]
            window_centered = window - np.mean(window, axis=1, keepdims=True)

            if window_centered.shape[1] > N_PCA_COMPONENTS:
                pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver='full')
                try:
                    window_reduced = pca.fit_transform(window_centered.T).T
                except:
                    kappas[i] = np.nan
                    continue
            else:
                window_reduced = window_centered

            try:
                U, S, Vt = np.linalg.svd(window_reduced, full_matrices=False)
            except:
                kappas[i] = np.nan
                continue

            S_pos = S[S > EPSILON]
            if len(S_pos) >= 2:
                kappas[i] = S_pos[0] / (S_pos[-1] + EPSILON)
            else:
                kappas[i] = np.nan

        return kappas

@jit(nopython=True, fastmath=True)
def fit_alpha_batch_jit(kappas_matrix, timescales_ms, r2_threshold):
    """
    Fit α for all windows in parallel (CPU JIT).

    Args:
        kappas_matrix: (n_windows, n_timescales)
        timescales_ms: (n_timescales,)

    Returns:
        alphas: (n_windows,) - includes NaN for bad fits
    """
    n_windows = kappas_matrix.shape[0]
    alphas = np.full(n_windows, np.nan)

    log_timescales = np.log(timescales_ms)

    for i in range(n_windows):
        kappas = kappas_matrix[i]

        # Remove invalid
        valid_mask = np.isfinite(kappas) & (kappas > 0)
        if np.sum(valid_mask) < 3:
            continue

        kappas_valid = kappas[valid_mask]
        log_tau_valid = log_timescales[valid_mask]
        log_kappa = np.log(kappas_valid)

        # Linear regression
        n = len(log_tau_valid)
        mean_x = np.mean(log_tau_valid)
        mean_y = np.mean(log_kappa)

        numerator = np.sum((log_tau_valid - mean_x) * (log_kappa - mean_y))
        denominator = np.sum((log_tau_valid - mean_x) ** 2)

        if denominator < 1e-12:
            continue

        slope = numerator / denominator
        intercept = mean_y - slope * mean_x

        # R²
        y_pred = slope * log_tau_valid + intercept
        ss_res = np.sum((log_kappa - y_pred) ** 2)
        ss_tot = np.sum((log_kappa - mean_y) ** 2)

        if ss_tot < 1e-12:
            r2 = 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        if r2 >= r2_threshold:
            alphas[i] = slope

    return alphas

def process_files_batch_gpu(filepaths, fsample):
    """
    Process ALL files at once on GPU with batched operations.

    Returns:
        alphas: list of valid alpha values
    """
    t0 = time.time()

    # Load all data
    print(f"    Loading {len(filepaths)} files...", end=' ', flush=True)
    data_cpu, fs = load_all_fieldtrip_files(filepaths)
    if data_cpu is None:
        return []
    print(f"Done ({data_cpu.shape[1] / fs:.1f}s of data)")

    # Transfer to GPU
    print(f"    Transferring to GPU...", end=' ', flush=True)
    if USE_GPU:
        data_gpu = cp.asarray(data_cpu)
    else:
        data_gpu = data_cpu
    print(f"Done")

    # Bandpass filter on GPU
    print(f"    Bandpass filtering on GPU...", end=' ', flush=True)
    data_filt_gpu = bandpass_filter_gpu_batch(data_gpu, LOWCUT_HZ, HIGHCUT_HZ, fs)
    print(f"Done")

    # Extract ALL windows at once
    window_samples = int(WINDOW_SEC * fs)
    overlap_samples = int(OVERLAP_SEC * fs)
    stride_samples = window_samples - overlap_samples

    print(f"    Extracting windows...", end=' ', flush=True)
    windows_gpu = extract_windows_gpu(data_filt_gpu, window_samples, stride_samples)
    n_windows = windows_gpu.shape[0]
    print(f"Done ({n_windows} windows)")

    # Compute κ at ALL timescales for ALL windows
    kappas_matrix = np.zeros((n_windows, len(TIMESCALES_MS)))

    print(f"    Computing κ at {len(TIMESCALES_MS)} timescales...")
    for tau_idx, tau_ms in enumerate(TIMESCALES_MS):
        downsample_factor = int(fs / (1000.0 / tau_ms))

        # Downsample all windows
        windows_downsampled = downsample_batch_gpu(windows_gpu, downsample_factor)

        # Compute κ for all windows at this timescale
        kappas = compute_kappa_batch_gpu(windows_downsampled)
        kappas_matrix[:, tau_idx] = kappas

        print(f"      τ={tau_ms}ms: κ range [{np.nanmin(kappas):.2e}, {np.nanmax(kappas):.2e}]")

    # Fit α for all windows (CPU JIT - very fast)
    print(f"    Fitting α for all windows...", end=' ', flush=True)
    alphas_all = fit_alpha_batch_jit(kappas_matrix, TIMESCALES_MS, R2_THRESHOLD)
    alphas_valid = alphas_all[np.isfinite(alphas_all)]
    print(f"Done ({len(alphas_valid)}/{n_windows} passed R²≥{R2_THRESHOLD})")

    t1 = time.time()
    print(f"    ⚡ Total processing time: {t1-t0:.1f}s")

    return alphas_valid.tolist()

def analyze_session_pre_post(subject_id, session_id):
    """
    Analyze pre vs post resting state EEG for one session.
    """
    subject_dir = os.path.join(DATA_DIR, subject_id)
    session_dir = os.path.join(subject_dir, session_id)

    if not os.path.exists(session_dir):
        return None

    print(f"\n  {subject_id} {session_id}")

    # Find files
    try:
        all_files = os.listdir(session_dir)
    except:
        return None

    pre_files = sorted([os.path.join(session_dir, f) for f in all_files if 'pre-rest' in f and f.endswith('.mat')])
    post_files = sorted([os.path.join(session_dir, f) for f in all_files if 'post-rest' in f and f.endswith('.mat')])

    if len(pre_files) == 0 or len(post_files) == 0:
        print(f"    ⚠ No pre/post files")
        return None

    # Process PRE (batched on GPU)
    print(f"  PRE condition:")
    alphas_pre = process_files_batch_gpu(pre_files, 2048.0)

    # Process POST (batched on GPU)
    print(f"  POST condition:")
    alphas_post = process_files_batch_gpu(post_files, 2048.0)

    # Trial-count parity
    n_min = min(len(alphas_pre), len(alphas_post))

    if n_min < 10:
        print(f"    ⚠ Insufficient windows: pre={len(alphas_pre)}, post={len(alphas_post)}")
        return None

    rng = np.random.RandomState(42)
    if len(alphas_pre) > n_min:
        idx = rng.choice(len(alphas_pre), n_min, replace=False)
        alphas_pre_matched = np.array(alphas_pre)[idx]
    else:
        alphas_pre_matched = np.array(alphas_pre)

    if len(alphas_post) > n_min:
        idx = rng.choice(len(alphas_post), n_min, replace=False)
        alphas_post_matched = np.array(alphas_post)[idx]
    else:
        alphas_post_matched = np.array(alphas_post)

    alpha_pre = np.mean(alphas_pre_matched)
    alpha_post = np.mean(alphas_post_matched)
    se_pre = np.std(alphas_pre_matched, ddof=1) / np.sqrt(len(alphas_pre_matched))
    se_post = np.std(alphas_post_matched, ddof=1) / np.sqrt(len(alphas_post_matched))
    delta_alpha = alpha_post - alpha_pre

    print(f"\n  RESULTS:")
    print(f"    Pre:  α = {alpha_pre:.3f} ± {se_pre:.3f}")
    print(f"    Post: α = {alpha_post:.3f} ± {se_post:.3f}")
    print(f"    Δα = {delta_alpha:+.3f}")

    return {
        'subject': subject_id,
        'session': session_id,
        'alpha_pre': float(alpha_pre),
        'alpha_post': float(alpha_post),
        'se_pre': float(se_pre),
        'se_post': float(se_post),
        'delta_alpha': float(delta_alpha),
        'n_windows': int(n_min),
        'alphas_pre_windows': alphas_pre_matched.tolist(),
        'alphas_post_windows': alphas_post_matched.tolist()
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

t_start = time.time()

subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"\nFound {len(subjects)} subjects: {subjects}")

all_results = []

for subject in subjects:
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subject_dir) if d.startswith('ses-') and os.path.isdir(os.path.join(subject_dir, d))])

    print(f"\n{'=' * 80}")
    print(f"{subject}: {len(sessions)} sessions")
    print(f"{'=' * 80}")

    for session in sessions:
        result = analyze_session_pre_post(subject, session)
        if result is not None:
            all_results.append(result)

t_end = time.time()

print(f"\n{'=' * 80}")
print(f"✅ ANALYSIS COMPLETE")
print(f"{'=' * 80}")
print(f"Sessions analyzed: {len(all_results)}")
print(f"Total time: {t_end - t_start:.1f}s")
print(f"{'=' * 80}")

if len(all_results) == 0:
    print("\n⚠️  No valid results - cannot perform statistics")
    exit(1)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("MIXED-EFFECTS HIERARCHICAL ANALYSIS")
print("=" * 80)

data_rows = []
for res in all_results:
    for alpha_val in res['alphas_pre_windows']:
        data_rows.append({
            'subject': res['subject'],
            'session': res['session'],
            'condition': 'pre',
            'alpha': alpha_val
        })
    for alpha_val in res['alphas_post_windows']:
        data_rows.append({
            'subject': res['subject'],
            'session': res['session'],
            'condition': 'post',
            'alpha': alpha_val
        })

df = pd.DataFrame(data_rows)
df['condition_coded'] = (df['condition'] == 'post').astype(int)

print(f"\nTotal observations: {len(df)}")
print(f"  Pre windows: {np.sum(df['condition'] == 'pre')}")
print(f"  Post windows: {np.sum(df['condition'] == 'post')}")
print(f"  Unique sessions: {len(df.groupby(['subject', 'session']))}")

df['subject_session'] = df['subject'] + '_' + df['session']

model = MixedLM.from_formula('alpha ~ condition_coded', data=df, groups=df['subject_session'])
result = model.fit(method='lbfgs', reml=True)

beta_condition = result.params['condition_coded']
se_condition = result.bse['condition_coded']
t_mixed = result.tvalues['condition_coded']

n_sessions = len(df['subject_session'].unique())
df_conservative = n_sessions - 1

p_mixed = 2 * (1 - sp_stats.t.cdf(abs(t_mixed), df_conservative))

ci_low = beta_condition - sp_stats.t.ppf(1 - ALPHA/2, df_conservative) * se_condition
ci_high = beta_condition + sp_stats.t.ppf(1 - ALPHA/2, df_conservative) * se_condition

print(f"\n{'=' * 80}")
print("MIXED-EFFECTS MODEL RESULTS")
print(f"{'=' * 80}")
print(f"Δα = {beta_condition:+.4f} ± {se_condition:.4f}")
print(f"t({df_conservative}) = {t_mixed:.2f}")
print(f"p = {p_mixed:.4f} {'***' if p_mixed < 0.001 else '**' if p_mixed < 0.01 else '*' if p_mixed < 0.05 else 'ns'}")
print(f"95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")

alpha_pre_all = df[df['condition'] == 'pre']['alpha'].mean()
alpha_post_all = df[df['condition'] == 'post']['alpha'].mean()
se_pre_all = df[df['condition'] == 'pre']['alpha'].sem()
se_post_all = df[df['condition'] == 'post']['alpha'].sem()

print(f"\n{'=' * 80}")
print("GRAND AVERAGES")
print(f"{'=' * 80}")
print(f"Pre-TMS:  α = {alpha_pre_all:.3f} ± {se_pre_all:.3f}")
print(f"Post-TMS: α = {alpha_post_all:.3f} ± {se_post_all:.3f}")
print(f"Δα = {alpha_post_all - alpha_pre_all:+.3f}")

# Save results
output = {
    'dataset': 'Nikolin_2022_TMS_EEG',
    'n_subjects': len(subjects),
    'n_sessions': len(all_results),
    'n_observations': len(df),
    'alpha_pre_mean': float(alpha_pre_all),
    'alpha_post_mean': float(alpha_post_all),
    'alpha_pre_se': float(se_pre_all),
    'alpha_post_se': float(se_post_all),
    'delta_alpha': float(beta_condition),
    'delta_alpha_se': float(se_condition),
    't_statistic': float(t_mixed),
    'df': int(df_conservative),
    'p_value': float(p_mixed),
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'significant': bool(p_mixed < ALPHA),
    'per_session_results': all_results
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 80}")
print(f"✅ Results saved to: {OUTPUT_FILE}")
print(f"{'=' * 80}\n")
