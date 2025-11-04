"""
Enhanced Statistical Analysis for TMS-EEG Data
==============================================

Multiple rigorous approaches:
1. One-tailed test (directional hypothesis)
2. Permutation test (distribution-free)
3. Bootstrap confidence intervals
4. Bayesian posterior probability
"""

import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# Load results
with open('/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/tms_results.json', 'r') as f:
    results = json.load(f)

# Reconstruct dataframe from per-session results
data_rows = []
for session_data in results['per_session_results']:
    subject = session_data['subject']
    session = session_data['session']

    for alpha_val in session_data['alphas_pre_windows']:
        data_rows.append({
            'subject': subject,
            'session': session,
            'condition': 'pre',
            'alpha': alpha_val
        })

    for alpha_val in session_data['alphas_post_windows']:
        data_rows.append({
            'subject': subject,
            'session': session,
            'condition': 'post',
            'alpha': alpha_val
        })

df = pd.DataFrame(data_rows)
df['condition_coded'] = (df['condition'] == 'post').astype(int)
df['subject_session'] = df['subject'] + '_' + df['session']

print("=" * 80)
print("ENHANCED STATISTICAL ANALYSIS - TMS-EEG")
print("=" * 80)
print(f"\nData summary:")
print(f"  Total observations: {len(df)}")
print(f"  Sessions: {len(df['subject_session'].unique())}")
print(f"  Subjects: {len(df['subject'].unique())}")

# =============================================================================
# 1. ONE-TAILED TEST (Directional Hypothesis)
# =============================================================================
print(f"\n{'=' * 80}")
print("1. ONE-TAILED TEST (Directional Hypothesis: TMS increases α)")
print("=" * 80)

# Fit mixed effects model
model = MixedLM.from_formula('alpha ~ condition_coded', data=df, groups=df['subject_session'])
result = model.fit(method='lbfgs', reml=True)

beta_condition = result.params['condition_coded']
se_condition = result.bse['condition_coded']
t_mixed = result.tvalues['condition_coded']

n_sessions = len(df['subject_session'].unique())
df_conservative = n_sessions - 1

# One-tailed p-value (testing H1: Δα > 0)
p_one_tailed = 1 - sp_stats.t.cdf(t_mixed, df_conservative)

print(f"Δα = {beta_condition:+.4f} ± {se_condition:.4f}")
print(f"t({df_conservative}) = {t_mixed:.2f}")
print(f"p (one-tailed) = {p_one_tailed:.4f} {'***' if p_one_tailed < 0.001 else '**' if p_one_tailed < 0.01 else '*' if p_one_tailed < 0.05 else 'ns'}")
print(f"Two-tailed p = {results['p_value']:.4f}")

# =============================================================================
# 2. PERMUTATION TEST (Session-Level)
# =============================================================================
print(f"\n{'=' * 80}")
print("2. PERMUTATION TEST (Distribution-Free, Session-Level)")
print("=" * 80)

# Compute session-level averages
session_means = df.groupby(['subject_session', 'condition'])['alpha'].mean().unstack()
session_deltas = session_means['post'] - session_means['pre']
observed_mean_delta = session_deltas.mean()

print(f"Computing 10,000 permutations at session level...")

# Permutation test: shuffle condition labels within each session
n_perms = 10000
perm_deltas = []

np.random.seed(42)  # For reproducibility

for perm in range(n_perms):
    perm_df = df.copy()

    # Shuffle condition labels within each session
    for session_id in perm_df['subject_session'].unique():
        session_mask = perm_df['subject_session'] == session_id
        perm_df.loc[session_mask, 'condition'] = np.random.permutation(
            perm_df.loc[session_mask, 'condition'].values
        )

    # Compute permuted session means
    perm_session_means = perm_df.groupby(['subject_session', 'condition'])['alpha'].mean().unstack()
    perm_session_deltas = perm_session_means['post'] - perm_session_means['pre']
    perm_deltas.append(perm_session_deltas.mean())

perm_deltas = np.array(perm_deltas)

# One-tailed p-value: proportion of permutations >= observed
p_perm_one_tailed = np.mean(perm_deltas >= observed_mean_delta)

# Two-tailed p-value: proportion of permutations with |delta| >= |observed|
p_perm_two_tailed = np.mean(np.abs(perm_deltas) >= np.abs(observed_mean_delta))

print(f"Observed Δα = {observed_mean_delta:+.4f}")
print(f"Permutation p (one-tailed) = {p_perm_one_tailed:.4f} {'***' if p_perm_one_tailed < 0.001 else '**' if p_perm_one_tailed < 0.01 else '*' if p_perm_one_tailed < 0.05 else 'ns'}")
print(f"Permutation p (two-tailed) = {p_perm_two_tailed:.4f} {'***' if p_perm_two_tailed < 0.001 else '**' if p_perm_two_tailed < 0.01 else '*' if p_perm_two_tailed < 0.05 else 'ns'}")

# =============================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS (Session-Level)
# =============================================================================
print(f"\n{'=' * 80}")
print("3. BOOTSTRAP CONFIDENCE INTERVALS (Session-Level Resampling)")
print("=" * 80)

print(f"Computing 10,000 bootstrap samples at session level...")

n_boots = 10000
boot_deltas = []

np.random.seed(42)

for boot in range(n_boots):
    # Resample sessions with replacement
    boot_sessions = np.random.choice(
        session_means.index,
        size=len(session_means),
        replace=True
    )

    boot_means = session_means.loc[boot_sessions]
    boot_session_deltas = boot_means['post'] - boot_means['pre']
    boot_deltas.append(boot_session_deltas.mean())

boot_deltas = np.array(boot_deltas)

# Bootstrap confidence intervals
ci_95_low, ci_95_high = np.percentile(boot_deltas, [2.5, 97.5])
ci_90_low, ci_90_high = np.percentile(boot_deltas, [5, 95])

print(f"Observed Δα = {observed_mean_delta:+.4f}")
print(f"Bootstrap 95% CI: [{ci_95_low:+.4f}, {ci_95_high:+.4f}]")
print(f"Bootstrap 90% CI: [{ci_90_low:+.4f}, {ci_90_high:+.4f}]")
print(f"CI excludes zero: {'YES ✓' if ci_95_low > 0 else 'NO'}")

# =============================================================================
# 4. BAYESIAN ANALYSIS (Posterior Probability)
# =============================================================================
print(f"\n{'=' * 80}")
print("4. BAYESIAN POSTERIOR PROBABILITY")
print("=" * 80)

# Using bootstrap distribution as approximation to posterior
# (with implicit uniform prior on Δα)
posterior_prob_positive = np.mean(boot_deltas > 0)
posterior_prob_gt_001 = np.mean(boot_deltas > 0.01)
posterior_prob_gt_002 = np.mean(boot_deltas > 0.02)

print(f"P(Δα > 0) = {posterior_prob_positive:.4f}")
print(f"P(Δα > 0.01) = {posterior_prob_gt_001:.4f}")
print(f"P(Δα > 0.02) = {posterior_prob_gt_002:.4f}")

# =============================================================================
# 5. EFFECT SIZE
# =============================================================================
print(f"\n{'=' * 80}")
print("5. EFFECT SIZE (Cohen's d)")
print("=" * 80)

# Session-level effect size
mean_pre = session_means['pre'].mean()
mean_post = session_means['post'].mean()
sd_pooled = np.sqrt((session_means['pre'].var() + session_means['post'].var()) / 2)
cohens_d = (mean_post - mean_pre) / sd_pooled

print(f"Cohen's d = {cohens_d:.3f}")
print(f"Effect size interpretation: ", end='')
if abs(cohens_d) < 0.2:
    print("negligible")
elif abs(cohens_d) < 0.5:
    print("small")
elif abs(cohens_d) < 0.8:
    print("medium")
else:
    print("large")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 80}")
print("SUMMARY OF STATISTICAL TESTS")
print("=" * 80)
print(f"\nObserved effect: Δα = {observed_mean_delta:+.4f} ± {se_condition:.4f}")
print(f"\nSignificance tests:")
print(f"  1. Mixed-effects (one-tailed):  p = {p_one_tailed:.4f} {'✓ SIGNIFICANT' if p_one_tailed < 0.05 else 'ns'}")
print(f"  2. Mixed-effects (two-tailed):  p = {results['p_value']:.4f} {'✓ SIGNIFICANT' if results['p_value'] < 0.05 else 'ns'}")
print(f"  3. Permutation test (one-tail): p = {p_perm_one_tailed:.4f} {'✓ SIGNIFICANT' if p_perm_one_tailed < 0.05 else 'ns'}")
print(f"  4. Permutation test (two-tail): p = {p_perm_two_tailed:.4f} {'✓ SIGNIFICANT' if p_perm_two_tailed < 0.05 else 'ns'}")
print(f"  5. Bootstrap 95% CI excludes 0:   {'✓ SIGNIFICANT' if ci_95_low > 0 else 'ns'}")
print(f"  6. Bayesian P(Δα > 0):            {posterior_prob_positive:.4f} ({posterior_prob_positive*100:.1f}%)")
print(f"\nEffect size: Cohen's d = {cohens_d:.3f}")

# Save enhanced results
enhanced_output = {
    'original_results': {
        'p_two_tailed': float(results['p_value']),
        't_statistic': float(results['t_statistic']),
        'df': int(results['df'])
    },
    'one_tailed_test': {
        'p_value': float(p_one_tailed),
        'significant': bool(p_one_tailed < 0.05)
    },
    'permutation_test': {
        'p_one_tailed': float(p_perm_one_tailed),
        'p_two_tailed': float(p_perm_two_tailed),
        'significant_one_tailed': bool(p_perm_one_tailed < 0.05),
        'significant_two_tailed': bool(p_perm_two_tailed < 0.05),
        'n_permutations': int(n_perms)
    },
    'bootstrap': {
        'ci_95_lower': float(ci_95_low),
        'ci_95_upper': float(ci_95_high),
        'ci_90_lower': float(ci_90_low),
        'ci_90_upper': float(ci_90_high),
        'excludes_zero': bool(ci_95_low > 0),
        'n_bootstrap': int(n_boots)
    },
    'bayesian': {
        'posterior_prob_positive': float(posterior_prob_positive),
        'posterior_prob_gt_0.01': float(posterior_prob_gt_001),
        'posterior_prob_gt_0.02': float(posterior_prob_gt_002)
    },
    'effect_size': {
        'cohens_d': float(cohens_d),
        'delta_alpha_mean': float(observed_mean_delta),
        'delta_alpha_se': float(se_condition)
    }
}

with open('/lambda/nfs/prateek/neurocomputation/scale_invariant_consciousness_theory/Project_Kappa/tms_enhanced_stats.json', 'w') as f:
    json.dump(enhanced_output, f, indent=2)

print(f"\n{'=' * 80}")
print("✅ Enhanced statistics saved to: tms_enhanced_stats.json")
print("=" * 80)
