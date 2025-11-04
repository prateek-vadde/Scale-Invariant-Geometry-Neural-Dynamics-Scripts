#!/usr/bin/env python3
"""
Trial-by-Trial Curvature vs Reaction Time Correlation (FIXED)
==============================================================

CORRECTED: Computes κ for EACH trial's temporal trajectory independently.

Previous bug: Was computing κ across trials (sliding window), not within trials.
Fix: Each trial gets its own κ from its (n_times, n_channels) trajectory.

Dataset: Nencki MSIT (42 subjects, cognitive control task)

Author: Prateek
Date: 2025-10-27
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
import json
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from functools import partial
import warnings

# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# GPU setup
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Parameters
PARAMS = {
    'base_dir': '/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/dataset_downloads/nencki_symfonia',
    'task': 'msit',

    # Time windows
    'baseline_window': (-0.5, 0.0),
    'event_window': (0.05, 0.2),  # Post-stimulus window

    # Epoch parameters
    'epoch_tmin': -1.5,
    'epoch_tmax': 1.0,

    # Kappa computation - NO sliding window needed!
    'use_gpu': GPU_AVAILABLE,

    # RT filtering
    'rt_min': 0.2,  # Minimum plausible RT (200 ms)
    'rt_max': 2.0,  # Maximum plausible RT (2000 ms)

    # Analysis
    'n_subjects': 42,

    # Parallel processing
    'n_cores': 42,  # Run all subjects in parallel!
}

def compute_kappa_per_trial_gpu(trial_data):
    """
    Compute curvature for ONE trial's temporal trajectory.

    Parameters:
    -----------
    trial_data : array (n_channels, n_times)
        Neural data for one trial

    Returns:
    --------
    kappa : float
        Curvature of the temporal trajectory
    """
    # Transpose to (n_times, n_channels) - trajectory over time
    trajectory = trial_data.T

    if GPU_AVAILABLE:
        trajectory_gpu = cp.asarray(trajectory)
        _, s, _ = cp.linalg.svd(trajectory_gpu, full_matrices=False)
        s = cp.asnumpy(s)
    else:
        _, s, _ = np.linalg.svd(trajectory, full_matrices=False)

    if len(s) < 2:
        return np.nan

    kappa = s[0] / (s[-1] + 1e-10)
    return kappa

def parse_events_with_rt(events_file):
    """Parse events.tsv and compute RTs for each trial"""
    # Load events
    events_df = pd.read_csv(events_file, sep='\t')

    # Stimulus codes (S 5, S 6, S 7, S 8)
    stimulus_codes = ['S  5', 'S  6', 'S  7', 'S  8']

    # Response codes (S 1, S 2, S 3)
    response_codes = ['S  1', 'S  2', 'S  3']

    trials = []

    i = 0
    while i < len(events_df) - 1:
        row = events_df.iloc[i]

        # Check if this is a stimulus event
        if row['event_type'] in stimulus_codes:
            stim_onset = row['onset']
            stim_code = row['event_type']

            # Look for the next response
            for j in range(i + 1, min(i + 10, len(events_df))):  # Search next 10 events
                next_row = events_df.iloc[j]

                if next_row['event_type'] in response_codes:
                    resp_onset = next_row['onset']
                    resp_code = next_row['event_type']
                    rt = resp_onset - stim_onset

                    trials.append({
                        'stim_onset': stim_onset,
                        'stim_code': stim_code.strip(),
                        'resp_code': resp_code.strip(),
                        'rt': rt,
                    })

                    i = j  # Skip to response
                    break

        i += 1

    return pd.DataFrame(trials)

def analyze_subject(subject_num, params):
    """Analyze one subject: compute κ for each trial and correlate with RT"""

    subject_id = f"sub-{subject_num:02d}"
    base_dir = Path(params['base_dir'])

    # Load EEG data
    eeg_dir = base_dir / subject_id / 'eeg'
    eeg_file = eeg_dir / f"{subject_id}_task-{params['task']}_eeg.vhdr"
    events_file = eeg_dir / f"{subject_id}_task-{params['task']}_events.tsv"

    if not eeg_file.exists() or not events_file.exists():
        return None

    print(f"\n{subject_id}:")

    # Parse events and get RTs
    trials_df = parse_events_with_rt(events_file)

    # Filter by RT range
    valid_trials = trials_df[
        (trials_df['rt'] >= params['rt_min']) &
        (trials_df['rt'] <= params['rt_max'])
    ]

    n_trials_total = len(trials_df)
    n_trials_valid = len(valid_trials)

    print(f"  Trials: {n_trials_valid}/{n_trials_total} valid (RT: {params['rt_min']}-{params['rt_max']}s)")

    if n_trials_valid < 10:  # Need enough trials
        print(f"  ✗ Too few valid trials")
        return None

    # Load raw data
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(l_freq=1.0, h_freq=100.0, verbose=False)

    # Create epochs for valid trials
    events_mne = []
    for idx, trial in valid_trials.iterrows():
        sample = int(trial['stim_onset'] * raw.info['sfreq'])
        events_mne.append([sample, 0, 1])  # All coded as event_id=1

    events_mne = np.array(events_mne)

    epochs = mne.Epochs(
        raw, events_mne, event_id={'stimulus': 1},
        tmin=params['epoch_tmin'], tmax=params['epoch_tmax'],
        baseline=params['baseline_window'],
        preload=True, verbose=False
    )

    # Extract event window data
    times = epochs.times
    event_mask = (times >= params['event_window'][0]) & (times <= params['event_window'][1])

    data = epochs.get_data()  # (n_trials, n_channels, n_times)
    data_event = data[:, :, event_mask]  # (n_trials, n_channels, n_event_times)

    n_trials, n_channels, n_times = data_event.shape

    print(f"  Data shape: ({n_trials} trials, {n_channels} channels, {n_times} time points)")

    # Compute κ for EACH trial independently
    kappas = []
    for trial_idx in range(n_trials):
        trial_data = data_event[trial_idx]  # (n_channels, n_times)
        kappa = compute_kappa_per_trial_gpu(trial_data)

        if not np.isnan(kappa):
            kappas.append(kappa)
        else:
            kappas.append(np.nan)

    kappas = np.array(kappas)

    # Remove NaN values
    valid_mask = ~np.isnan(kappas)
    kappas_valid = kappas[valid_mask]
    rts_valid = valid_trials['rt'].values[valid_mask]

    if len(kappas_valid) < 10:
        print(f"  ✗ Too few valid κ values: {len(kappas_valid)}")
        return None

    print(f"  κ range: [{np.min(kappas_valid):.2f}, {np.max(kappas_valid):.2f}]")
    print(f"  RT range: [{np.min(rts_valid):.3f}, {np.max(rts_valid):.3f}] s")

    # Compute correlations
    r_pearson, p_pearson = pearsonr(kappas_valid, rts_valid)
    r_spearman, p_spearman = spearmanr(kappas_valid, rts_valid)

    print(f"  Pearson:  r = {r_pearson:+.3f}, p = {p_pearson:.4f}")
    print(f"  Spearman: ρ = {r_spearman:+.3f}, p = {p_spearman:.4f}")

    return {
        'subject': subject_num,
        'n_trials': len(kappas_valid),
        'kappas': kappas_valid,
        'rts': rts_valid,
        'mean_kappa': np.mean(kappas_valid),
        'std_kappa': np.std(kappas_valid),
        'mean_rt': np.mean(rts_valid),
        'std_rt': np.std(rts_valid),
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
    }

def main():
    import time
    start_time = time.time()

    print("="*80)
    print("Trial-by-Trial Curvature vs Reaction Time Analysis (FIXED)")
    print("="*80)
    print(f"\nFIX: Now computing κ for each trial's temporal trajectory independently")
    print(f"     (not across trials with sliding window)")
    print(f"\nParameters:")
    print(f"  Dataset: Nencki MSIT")
    print(f"  Subjects: {PARAMS['n_subjects']}")
    print(f"  Event window: {PARAMS['event_window']} s")
    print(f"  Valid RT range: {PARAMS['rt_min']}-{PARAMS['rt_max']} s")
    print(f"  GPU: {'✓ Enabled' if GPU_AVAILABLE else '✗ Disabled (CPU only)'}")
    print(f"  Parallel workers: {PARAMS['n_cores']}")

    # Analyze all subjects in parallel
    print(f"\n🚀 Launching {PARAMS['n_cores']} parallel workers...")
    print("(Output will be mixed from multiple processes)\n")

    subject_nums = list(range(1, PARAMS['n_subjects'] + 1))
    analyze_func = partial(analyze_subject, params=PARAMS)

    with mp.Pool(processes=PARAMS['n_cores']) as pool:
        results_raw = pool.map(analyze_func, subject_nums)

    # Filter out None results
    results = [r for r in results_raw if r is not None]

    print(f"\n{'='*80}")
    print(f"COMPLETED: {len(results)}/{PARAMS['n_subjects']} subjects analyzed")
    print(f"{'='*80}")

    if len(results) == 0:
        print("\n✗ No valid results")
        return

    # Aggregate statistics
    r_pearsons = [r['r_pearson'] for r in results]
    p_pearsons = [r['p_pearson'] for r in results]
    r_spearmans = [r['r_spearman'] for r in results]

    print(f"\nWithin-Subject Correlations:")
    print(f"  Pearson r:  {np.mean(r_pearsons):+.3f} ± {np.std(r_pearsons):.3f}")
    print(f"  Significant: {np.sum(np.array(p_pearsons) < 0.05)}/{len(results)} subjects (p < 0.05)")
    print(f"  Spearman ρ: {np.mean(r_spearmans):+.3f} ± {np.std(r_spearmans):.3f}")

    # Test if mean correlation is significantly different from zero
    t_stat, t_p = stats.ttest_1samp(r_pearsons, 0)
    print(f"\nGroup-Level Test (one-sample t-test):")
    print(f"  H0: mean correlation = 0")
    print(f"  t({len(results)-1}) = {t_stat:.3f}, p = {t_p:.4f}")

    if t_p < 0.05:
        direction = "positive" if np.mean(r_pearsons) > 0 else "negative"
        print(f"  ✓ Significant {direction} correlation across subjects!")
    else:
        print(f"  ✗ No significant correlation at group level")

    # Across-subject correlation (mean κ vs mean RT)
    mean_kappas = [r['mean_kappa'] for r in results]
    mean_rts = [r['mean_rt'] for r in results]

    r_across, p_across = pearsonr(mean_kappas, mean_rts)

    print(f"\nAcross-Subject Correlation:")
    print(f"  Does subject's average κ predict average RT?")
    print(f"  Pearson r = {r_across:+.3f}, p = {p_across:.4f}")

    # Save results
    output_dir = Path('/home/ubuntu/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/mathematical_derivation')

    # Save detailed results (without raw arrays for JSON compatibility)
    results_summary = []
    for r in results:
        results_summary.append({
            'subject': r['subject'],
            'n_trials': r['n_trials'],
            'mean_kappa': float(r['mean_kappa']),
            'std_kappa': float(r['std_kappa']),
            'mean_rt': float(r['mean_rt']),
            'std_rt': float(r['std_rt']),
            'r_pearson': float(r['r_pearson']),
            'p_pearson': float(r['p_pearson']),
            'r_spearman': float(r['r_spearman']),
            'p_spearman': float(r['p_spearman']),
        })

    output_data = {
        'parameters': {k: v for k, v in PARAMS.items() if k != 'base_dir'},
        'n_subjects_analyzed': len(results),
        'within_subject': {
            'mean_r_pearson': float(np.mean(r_pearsons)),
            'std_r_pearson': float(np.std(r_pearsons)),
            'n_significant': int(np.sum(np.array(p_pearsons) < 0.05)),
            't_statistic': float(t_stat),
            'p_value': float(t_p),
        },
        'across_subject': {
            'r_pearson': float(r_across),
            'p_value': float(p_across),
        },
        'subjects': results_summary,
    }

    output_file = output_dir / 'nencki_rt_correlation_results_FIXED.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved: {output_file}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Within-subject correlation distribution
    ax = axes[0, 0]
    ax.hist(r_pearsons, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='r=0')
    ax.axvline(np.mean(r_pearsons), color='blue', linestyle='-', linewidth=2,
               label=f'Mean r={np.mean(r_pearsons):.3f}')
    ax.set_xlabel('Within-Subject Correlation (Pearson r)', fontsize=12)
    ax.set_ylabel('Number of Subjects', fontsize=12)
    ax.set_title(f'Distribution of κ-RT Correlations\n(n={len(results)} subjects)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Across-subject correlation
    ax = axes[0, 1]
    ax.scatter(mean_kappas, mean_rts, s=100, alpha=0.6, edgecolors='black')

    # Add regression line
    z = np.polyfit(mean_kappas, mean_rts, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(mean_kappas), max(mean_kappas), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r={r_across:.3f}, p={p_across:.4f}')

    ax.set_xlabel('Mean Neural Curvature (κ)', fontsize=12)
    ax.set_ylabel('Mean Reaction Time (s)', fontsize=12)
    ax.set_title('Across-Subject Correlation\n(Subject Averages)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Example subject (pick one with strongest correlation)
    ax = axes[1, 0]
    best_idx = np.argmax(np.abs(r_pearsons))
    best_subject = results[best_idx]

    ax.scatter(best_subject['kappas'], best_subject['rts'], s=50, alpha=0.6, edgecolors='black')

    # Regression line
    z = np.polyfit(best_subject['kappas'], best_subject['rts'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(best_subject['kappas']), max(best_subject['kappas']), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2,
            label=f"r={best_subject['r_pearson']:.3f}, p={best_subject['p_pearson']:.4f}")

    ax.set_xlabel('Neural Curvature (κ)', fontsize=12)
    ax.set_ylabel('Reaction Time (s)', fontsize=12)
    ax.set_title(f"Example Subject (sub-{best_subject['subject']:02d})\nStrongest Correlation", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    FUNCTIONAL SIGNIFICANCE ANALYSIS (FIXED)
    ========================================

    Within-Subject Correlations:
      • Mean r = {np.mean(r_pearsons):+.3f} ± {np.std(r_pearsons):.3f}
      • {np.sum(np.array(p_pearsons) < 0.05)}/{len(results)} subjects significant (p<0.05)
      • Group t-test: t={t_stat:.2f}, p={t_p:.4f}

    Across-Subject Correlation:
      • r = {r_across:+.3f}, p = {p_across:.4f}

    Interpretation:
    """

    if t_p < 0.05:
        if np.mean(r_pearsons) > 0:
            summary_text += "  ✓ Higher κ → Slower RT (consistent)\n"
            summary_text += "  → More complex trajectories slow processing"
        else:
            summary_text += "  ✓ Higher κ → Faster RT (consistent)\n"
            summary_text += "  → More complex trajectories speed processing"
    else:
        summary_text += "  ✗ No consistent κ-RT relationship\n"
        summary_text += "  → Curvature may not predict RT"

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    fig_file = output_dir / 'nencki_rt_correlation_figure_FIXED.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {fig_file}")

    plt.close()

    elapsed_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"   Time per subject: {elapsed_time/len(results):.1f} seconds")

if __name__ == '__main__':
    main()
