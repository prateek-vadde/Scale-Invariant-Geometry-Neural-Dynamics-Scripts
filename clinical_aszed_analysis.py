#!/usr/bin/env python3
"""
Clinical κ analysis for ASZED schizophrenia dataset
Compares neural trajectory curvature between patients and controls
"""

import numpy as np
import mne
from pathlib import Path
from multiprocessing import Pool, cpu_count
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated SVD
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU")

def compute_kappa_gpu(data):
    """Compute κ using GPU-accelerated SVD"""
    if GPU_AVAILABLE:
        data_gpu = cp.asarray(data)
        U, S, Vt = cp.linalg.svd(data_gpu, full_matrices=False)
        S_cpu = cp.asnumpy(S)
    else:
        U, S_cpu, Vt = np.linalg.svd(data, full_matrices=False)

    kappa = S_cpu[0] / (S_cpu[-1] + 1e-10)
    return float(kappa)

def analyze_subject(args):
    """Analyze single subject's Phase 2 (cognitive task) data"""
    subset, subject_id, session, base_path = args

    try:
        # Construct path to Phase 2 (cognitive task)
        edf_path = base_path / f"subset_{subset}" / f"subject_{subject_id}" / str(session) / "Phase 2.edf"

        if not edf_path.exists():
            return None

        # Load EDF data
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        # Get metadata
        userfile = base_path / f"subset_{subset}" / f"subject_{subject_id}" / "userfile.gnr"
        with open(userfile, 'r') as f:
            metadata = {}
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=', 1)
                    metadata[key] = val

        category = metadata.get('category', 'Unknown')
        age = int(metadata.get('age', 0))
        sex = metadata.get('sex', 'Unknown')

        # Get data
        data = raw.get_data()  # (n_channels, n_times)
        sfreq = raw.info['sfreq']

        # Use 0.15s windows matching task analysis (30 samples at 200Hz)
        window_size = int(0.15 * sfreq)

        # Compute κ for non-overlapping windows
        kappas = []
        n_timepoints = data.shape[1]

        for start_idx in range(0, n_timepoints - window_size, window_size):
            window_data = data[:, start_idx:start_idx+window_size]

            # Compute κ for this window
            kappa = compute_kappa_gpu(window_data)

            # Quality control: reject extreme outliers (likely artifacts)
            if kappa < 10000:  # Reasonable upper bound
                kappas.append(kappa)

        if len(kappas) == 0:
            return None

        result = {
            'subset': subset,
            'subject_id': subject_id,
            'session': session,
            'category': category,
            'age': age,
            'sex': sex,
            'mean_kappa': float(np.mean(kappas)),
            'std_kappa': float(np.std(kappas)),
            'median_kappa': float(np.median(kappas)),
            'n_windows': len(kappas),
            'kappas': [float(k) for k in kappas]
        }

        print(f"✓ subset_{subset}/subject_{subject_id}/session{session} ({category}): κ={result['mean_kappa']:.2f}±{result['std_kappa']:.2f}")
        return result

    except Exception as e:
        print(f"✗ subset_{subset}/subject_{subject_id}/session{session}: {e}")
        return None

def main():
    base_path = Path("/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/data/ASZED/ASZED/version_1.1/node_1")

    # Find all subjects
    subjects_to_analyze = []
    for subset_dir in base_path.glob("subset_*"):
        subset = subset_dir.name.replace("subset_", "")
        for subject_dir in subset_dir.glob("subject_*"):
            subject_id = subject_dir.name.replace("subject_", "")
            # Check both sessions
            for session in [1, 2]:
                session_dir = subject_dir / str(session)
                if session_dir.exists() and (session_dir / "Phase 2.edf").exists():
                    subjects_to_analyze.append((subset, subject_id, session, base_path))

    print(f"Found {len(subjects_to_analyze)} sessions to analyze")
    print(f"Using {min(16, cpu_count())} workers with GPU acceleration: {GPU_AVAILABLE}\n")

    # Parallel processing
    with Pool(min(16, cpu_count())) as pool:
        results = pool.map(analyze_subject, subjects_to_analyze)

    # Filter successful results
    results = [r for r in results if r is not None]

    # Aggregate by subject (averaging across sessions)
    subject_data = {}
    for r in results:
        key = (r['subset'], r['subject_id'])
        if key not in subject_data:
            subject_data[key] = {
                'category': r['category'],
                'age': r['age'],
                'sex': r['sex'],
                'kappas': []
            }
        subject_data[key]['kappas'].extend(r['kappas'])

    # Compute per-subject statistics
    patient_data = []
    control_data = []

    for (subset, subj_id), data in subject_data.items():
        mean_kappa = np.mean(data['kappas'])

        result = {
            'subset': subset,
            'subject_id': subj_id,
            'category': data['category'],
            'age': data['age'],
            'sex': data['sex'],
            'mean_kappa': float(mean_kappa),
            'median_kappa': float(np.median(data['kappas'])),
            'std_kappa': float(np.std(data['kappas'])),
            'n_windows': len(data['kappas'])
        }

        if data['category'] == 'Patient':
            patient_data.append(result)
        elif data['category'] == 'Control':
            control_data.append(result)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Patients analyzed: {len(patient_data)}")
    print(f"Controls analyzed: {len(control_data)}")

    # Statistical comparison
    patient_kappas = np.array([p['mean_kappa'] for p in patient_data])
    control_kappas = np.array([c['mean_kappa'] for c in control_data])

    # Remove outliers (>100x median)
    def remove_outliers(data, label):
        median = np.median(data)
        outlier_threshold = median * 100
        outliers = data > outlier_threshold
        if outliers.sum() > 0:
            print(f"\n⚠ Removing {outliers.sum()} outliers from {label} (>{outlier_threshold:.1f})")
            for i in np.where(outliers)[0]:
                print(f"  - {label}[{i}]: κ={data[i]:.1f} ({data[i]/median:.1f}x median)")
        return data[~outliers]

    patient_kappas_clean = remove_outliers(patient_kappas, "Patients")
    control_kappas_clean = remove_outliers(control_kappas, "Controls")

    # Statistics
    t_stat, p_value = stats.ttest_ind(patient_kappas_clean, control_kappas_clean)
    cohens_d = (np.mean(patient_kappas_clean) - np.mean(control_kappas_clean)) / np.sqrt(
        (np.std(patient_kappas_clean)**2 + np.std(control_kappas_clean)**2) / 2
    )

    print(f"\n{'='*60}")
    print(f"STATISTICAL RESULTS (Schizophrenia vs Controls)")
    print(f"{'='*60}")
    print(f"Patients (n={len(patient_kappas_clean)}): κ = {np.mean(patient_kappas_clean):.2f} ± {np.std(patient_kappas_clean):.2f}")
    print(f"Controls (n={len(control_kappas_clean)}): κ = {np.mean(control_kappas_clean):.2f} ± {np.std(control_kappas_clean):.2f}")
    print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    print(f"Significance: {sig}")

    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    print(f"Effect size: {effect}")

    # Save results
    output = {
        'n_patients': len(patient_kappas_clean),
        'n_controls': len(control_kappas_clean),
        'patient_mean': float(np.mean(patient_kappas_clean)),
        'patient_std': float(np.std(patient_kappas_clean)),
        'control_mean': float(np.mean(control_kappas_clean)),
        'control_std': float(np.std(control_kappas_clean)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'effect_size': effect,
        'significance': sig,
        'patient_data': patient_data,
        'control_data': control_data
    }

    output_path = Path("/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/clinical_aszed_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    if p_value < 0.05 and abs(cohens_d) >= 0.5:
        print(f"\n{'='*60}")
        print("✓ SUCCESS: Significant group difference detected!")
        print("✓ This provides strong clinical validation for κ!")
        print(f"{'='*60}")
    elif p_value < 0.05:
        print(f"\n{'='*60}")
        print("✓ Significant difference detected (small effect size)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("⚠ No significant group difference")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
