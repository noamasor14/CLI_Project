# %% [markdown]
# # CLI v6 — Relative Feature Analysis
# 
# **Project:** Predicting and Preventing Burnout Using Cognitive Load Analysis
# **Notebook:** 06 — Relative (Baseline-Deviation) Features
# 
# ---
# 
# ## Motivation
# 
# Notebooks 01–05 classify cognitive load from absolute physiological feature values.
# A persistent challenge is that absolute physiology differs strongly between people:
# what counts as high EDA or low HRV for one participant may be the other person's resting state.
# The cross-person GBT model must implicitly learn to handle this inter-individual variability.
# 
# This notebook tests a direct approach: instead of only giving the model absolute values,
# also give it features that express **deviation from each participant's own resting baseline**.
# 
# ### Two derived feature types
# 
# For every original feature X, two new features are computed:
# 
# - **Delta** — absolute deviation from baseline:
#   `X_delta = X - baseline_mean`
# 
# - **Ratio** — relative deviation from baseline:
#   `X_ratio = X / (baseline_mean + epsilon)`
# 
# These features are zero (or one) at baseline and grow in magnitude as the participant
# deviates from their own resting state — regardless of their absolute physiological level.
# 
# ### Baseline calibration scenario
# 
# For each participant, baseline statistics are estimated from their
# **`relaxation_video`** windows only.
# If that segment is unavailable, all Low-label windows serve as fallback.
# 
# This is a **realistic calibration scenario**, not a leakage-free research protocol:
# - The test participant's baseline is computed from their own Low windows
# - The test participant is still fully unseen to the model during training
# - Only their pre-task resting windows are used to anchor the feature scale
# 
# The model is never retrained on the test participant. Only the coordinate system
# of their features changes — the same GBT model trained on 23 other participants
# is applied unchanged.
# 
# ### What this is NOT
# - Not personalized model training
# - Not adding new physiological signals
# - Not filtering the test data
# - Not changing the labels
# 
# ### Ablation design
# 
# Three feature sets are compared using the same LOPO protocol and GBT model:
# 
# | Ablation | Features | Purpose |
# |---|---|---|
# | A1 — Original | Absolute features only | Baseline (equivalent to v2) |
# | A2 — Orig + Delta | Absolute + deviation | Test if deviations help |
# | A3 — Orig + Delta + Ratio | All three groups | Test if relative scaling adds further value |
# 
# This ablation isolates the contribution of each feature group and guards against
# the risk that ratio features add noise without adding signal.

# %%
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, recall_score, roc_curve, auc as sklearn_auc,
)

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_ROOT     = Path('.')
SESSIONS      = ['Lab1', 'Lab2']
FEATURE_FILES = ['EDA_features', 'HRV_features', 'TEMP_features']

LOW_EXACT   = {'relaxation_video', 'video_baseline'}
LOW_SUFFIX  = '_easy'
HIGH_SUFFIX = '_hard'
RELAX_SEGS  = {'relaxation_video', 'video_baseline'}

EPSILON    = 1e-6   # denominator guard in ratio computation
RATIO_CLIP = 50     # clip ratio features to [-50, 50] before normalization

BASE_MODEL = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)

print('Imports OK.')
print(f'Model  : {BASE_MODEL.__class__.__name__}  (n_estimators=100, same as v2)')
print(f'Epsilon: {EPSILON}   Ratio clip: +/-{RATIO_CLIP}')

# %% [markdown]
# ## Step 1 — Load Feature Dataset
# 
# Same loader as all previous notebooks.
# Labels assigned by design: easy/relaxation = Low (0), hard = High (1).
# Lab1 + Lab2 sessions only. EDA + HRV + TEMP features. No filtering.

# %%
def segment_to_label(segment_name):
    name = segment_name.lower()
    if name in LOW_EXACT or name.endswith(LOW_SUFFIX):
        return 0, 'Low'
    if name.endswith(HIGH_SUFFIX):
        return 1, 'High'
    return None, None


def load_segment(seg_dir, pid, session):
    seg_name = seg_dir.name
    label, load_level = segment_to_label(seg_name)
    if label is None:
        return None
    dfs = []
    for feat_name in FEATURE_FILES:
        fpath = seg_dir / f'{feat_name}.pickle'
        if not fpath.exists():
            return None
        with open(fpath, 'rb') as f:
            feat_df = pickle.load(f)
        if not isinstance(feat_df, pd.DataFrame):
            feat_df = pd.DataFrame(feat_df)
        dfs.append(feat_df.reset_index(drop=True))
    combined = pd.concat(dfs, axis=1)
    combined['label']          = label
    combined['load_level']     = load_level
    combined['participant_id'] = pid
    combined['session']        = session
    combined['segment']        = seg_name
    combined['window_idx']     = range(len(combined))
    return combined


def load_all_features():
    all_segs = []
    for pdir in sorted(DATA_ROOT.glob('UN_*')):
        if not pdir.is_dir():
            continue
        for session in SESSIONS:
            fp = pdir / session / 'Features'
            if not fp.exists():
                continue
            for sd in sorted(d for d in fp.iterdir() if d.is_dir()):
                df = load_segment(sd, pdir.name, session)
                if df is not None:
                    all_segs.append(df)
    return pd.concat(all_segs, ignore_index=True)


dataset = load_all_features()

META_COLS    = ['label', 'load_level', 'participant_id', 'session', 'segment', 'window_idx']
FEATURE_COLS = [c for c in dataset.columns if c not in META_COLS]

nan_count = dataset[FEATURE_COLS].isnull().sum().sum()
dataset[FEATURE_COLS] = dataset[FEATURE_COLS].fillna(dataset[FEATURE_COLS].median())

print(f'Dataset: {dataset.shape[0]:,} windows, {len(FEATURE_COLS)} features')
print(f'Participants: {dataset["participant_id"].nunique()}')
print(f'NaN cells filled: {nan_count}')
print()
print('Class balance:')
print(dataset['load_level'].value_counts().to_frame('windows'))

# %% [markdown]
# ## Step 2 — Relative Feature Construction
# 
# ### Why relative features may help
# 
# When a model trains on 23 participants and tests on the 24th, it must cope with the fact
# that absolute physiological values differ between people. A GBT tree splits on thresholds
# like "EDA_mean > 3.2" — but 3.2 μS might be elevated stress for person A and normal
# resting state for person B. The tree's threshold generalizes poorly across different
# physiological baselines.
# 
# Relative features sidestep this by expressing each window in terms of *how much it
# deviates from that person's own resting state*. A threshold of "EDA_mean_delta > 1.0"
# means the same thing for everyone: one unit above their personal resting level.
# 
# ### Absolute vs relative physiology
# 
# | Absolute (current) | Relative (this notebook) |
# |---|---|
# | EDA = 4.2 μS | EDA_delta = +1.8 above resting |
# | HRV_RMSSD = 28 ms | HRV_delta = -5 below resting |
# | TEMP_mean = 32.1 C | TEMP_ratio = 0.97× resting level |
# 
# The relative features don't replace absolute features — they supplement them.
# The model receives all three groups and can learn from whichever representation
# is most discriminative for each feature.
# 
# ### Why this is not personalization
# 
# - The GBT model is **not retrained** on the test participant
# - No labeled task data from the test participant is ever used
# - Only the test participant's Low/relaxation windows are used — for calibration only
# - This is equivalent to asking a new employee to complete a 5-minute rest before
#   their first monitoring session (the same scenario as notebook 05)
# 
# ### Baseline computation
# 
# For each participant, baseline statistics (mean) are computed from:
# 1. **`relaxation_video` windows** — pure rest baseline (preferred)
# 2. **All Low-label windows** — fallback if relaxation_video is unavailable
# 
# The ratio denominator uses `baseline_mean + epsilon` to guard against near-zero values.
# Ratio features are clipped to `[-50, 50]` before normalization to suppress outliers
# from participants whose baseline_mean is very close to zero for certain features.

# %%
def compute_participant_baselines(dataset, feature_cols):
    # Returns dict: pid -> {mean: Series, n_calib_windows: int, calib_type: str}
    records = {}
    for pid, grp in dataset.groupby('participant_id'):
        relax_mask = grp['segment'].str.lower().isin(RELAX_SEGS)
        if relax_mask.sum() > 0:
            calib_data = grp.loc[relax_mask, feature_cols]
            calib_type = 'relaxation_video'
        else:
            low_mask   = grp['label'] == 0
            calib_data = grp.loc[low_mask, feature_cols]
            calib_type = 'all_low_fallback'
        records[pid] = {
            'mean':            calib_data.mean(),
            'n_calib_windows': len(calib_data),
            'calib_type':      calib_type,
        }
    return records


def build_augmented_dataset(dataset, feature_cols, baselines,
                            epsilon=EPSILON, clip=RATIO_CLIP):
    # Adds delta and ratio columns for every original feature
    out = dataset.copy()
    delta_cols = [f'{c}_delta' for c in feature_cols]
    ratio_cols = [f'{c}_ratio' for c in feature_cols]

    for col in delta_cols + ratio_cols:
        out[col] = np.nan

    for pid, grp in dataset.groupby('participant_id'):
        bm = baselines[pid]['mean'][feature_cols].values  # (n_features,)
        X  = grp[feature_cols].values                     # (n_windows, n_features)

        delta = X - bm
        ratio = np.clip(X / (bm + epsilon), -clip, clip)

        out.loc[grp.index, delta_cols] = delta
        out.loc[grp.index, ratio_cols] = ratio

    return out, delta_cols, ratio_cols


def normalize_per_participant(df, feature_cols):
    # Per-column z-score per participant (same as v2)
    out = df.copy()
    for pid, grp in out.groupby('participant_id'):
        mu  = grp[feature_cols].mean()
        sig = grp[feature_cols].std().replace(0, 1)
        out.loc[grp.index, feature_cols] = (grp[feature_cols] - mu) / sig
    return out


print('Feature engineering functions defined.')

# %%
# Compute per-participant baselines
baselines = compute_participant_baselines(dataset, FEATURE_COLS)

baseline_log = pd.DataFrame([
    {'participant':      pid,
     'calib_type':       v['calib_type'],
     'n_calib_windows':  v['n_calib_windows']}
    for pid, v in sorted(baselines.items())
])
print('Calibration baseline summary:')
display(baseline_log)
print()
print('Calibration source:')
print(baseline_log['calib_type'].value_counts().to_string())
print()

# Build the augmented dataset (adds _delta and _ratio columns)
dataset_aug, DELTA_COLS, RATIO_COLS = build_augmented_dataset(
    dataset, FEATURE_COLS, baselines
)
ALL_COLS  = FEATURE_COLS + DELTA_COLS + RATIO_COLS
FEAT_A1   = FEATURE_COLS              # Ablation 1: original only
FEAT_A2   = FEATURE_COLS + DELTA_COLS # Ablation 2: original + delta
FEAT_A3   = ALL_COLS                  # Ablation 3: original + delta + ratio

print(f'Feature set sizes:')
print(f'  A1 Original only         : {len(FEAT_A1)}')
print(f'  A2 Original + Delta      : {len(FEAT_A2)}')
print(f'  A3 Original + Delta+Ratio: {len(FEAT_A3)}')
print()

# Sanity check
n_inf = np.isinf(dataset_aug[RATIO_COLS].values).sum()
n_nan = dataset_aug[RATIO_COLS].isnull().sum().sum()
print(f'Ratio feature quality — NaN: {n_nan}  Inf: {n_inf}')
print()

# Normalize all augmented columns once.
# Since normalization is per-column and per-participant, normalizing all columns
# together gives identical results for each column as normalizing them separately.
dataset_aug_norm = normalize_per_participant(dataset_aug, ALL_COLS)
print(f'Normalization complete. Augmented dataset shape: {dataset_aug_norm.shape}')

# %% [markdown]
# ## Step 3 — LOPO Cross-Validation
# 
# A single `run_lopo` function handles all three ablations.
# It accepts any normalized dataset and a list of feature columns to use,
# so the same 24-fold LOPO logic applies to A1, A2, and A3 without code duplication.
# 
# Feature importances are collected from each fold and averaged across folds.
# This provides a stable estimate of which features the GBT relied on most.

# %%
def _fold_metrics(y_te, y_pred, y_proba, test_pid):
    try:
        auc_val = roc_auc_score(y_te, y_proba)
    except ValueError:
        auc_val = np.nan
    return {
        'participant':  test_pid,
        'accuracy':     round(accuracy_score(y_te, y_pred), 3),
        'f1_macro':     round(f1_score(y_te, y_pred, average='macro',
                                       zero_division=0), 3),
        'roc_auc':      round(auc_val, 3) if not np.isnan(auc_val) else np.nan,
        'balanced_acc': round(balanced_accuracy_score(y_te, y_pred), 3),
        'recall_high':  round(recall_score(y_te, y_pred, pos_label=1,
                                           zero_division=0), 3),
    }


def run_lopo(df_norm, feature_cols, label=''):
    participants = sorted(df_norm['participant_id'].unique())
    fold_records, window_records, importances = [], [], []

    if label:
        print(f'LOPO ({label}): {len(participants)} folds, {len(feature_cols)} features ...')

    for k, test_pid in enumerate(participants):
        train_mask = df_norm['participant_id'] != test_pid
        test_mask  = df_norm['participant_id'] == test_pid

        X_tr = df_norm.loc[train_mask, feature_cols].values
        y_tr = df_norm.loc[train_mask, 'label'].values
        X_te = df_norm.loc[test_mask,  feature_cols].values
        y_te = df_norm.loc[test_mask,  'label'].values

        model = clone(BASE_MODEL)
        model.fit(X_tr, y_tr)
        importances.append(model.feature_importances_)

        y_proba = model.predict_proba(X_te)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        fold_records.append(_fold_metrics(y_te, y_pred, y_proba, test_pid))
        meta = df_norm.loc[test_mask,
                           ['participant_id','session','segment','label']].copy()
        meta['proba_high'] = y_proba
        window_records.append(meta)

        if label and (k + 1) % 6 == 0:
            print(f'  fold {k+1}/{len(participants)} done')

    fold_df   = pd.DataFrame(fold_records)
    window_df = pd.concat(window_records, ignore_index=True)
    imp_mean  = np.mean(importances, axis=0)
    return fold_df, window_df['label'].values, window_df['proba_high'].values, window_df, imp_mean


def segment_level_metrics(window_df, threshold=0.5):
    seg = (
        window_df
        .groupby(['participant_id','session','segment'], sort=False)
        .agg(proba_mean=('proba_high','mean'), true_label=('label','first'))
        .reset_index()
    )
    y_true  = seg['true_label'].values
    y_proba = seg['proba_mean'].values
    y_pred  = (y_proba >= threshold).astype(int)
    try:
        auc_val = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc_val = np.nan
    return {
        'n_segments':   len(seg),
        'accuracy':     round(accuracy_score(y_true, y_pred), 3),
        'balanced_acc': round(balanced_accuracy_score(y_true, y_pred), 3),
        'f1_macro':     round(f1_score(y_true, y_pred, average='macro',
                                       zero_division=0), 3),
        'roc_auc':      round(auc_val, 3) if not np.isnan(auc_val) else np.nan,
        'recall_high':  round(recall_score(y_true, y_pred, pos_label=1,
                                           zero_division=0), 3),
    }, seg


print('LOPO functions defined.')

# %% [markdown]
# ## Step 4 — Run Ablation Study
# 
# Three LOPO runs: 24 folds each, same GBT model, same evaluation.
# Total: 72 model fits. Expected runtime: 4–8 minutes.

# %%
print('=' * 60)
print('Ablation 1 — Original features only (v2 equivalent)')
print('=' * 60)
fold_a1, y_true_a1, y_proba_a1, win_a1, imp_a1 = run_lopo(
    dataset_aug_norm, FEAT_A1, label='A1-original'
)
print()

print('=' * 60)
print('Ablation 2 — Original + Delta features')
print('=' * 60)
fold_a2, y_true_a2, y_proba_a2, win_a2, imp_a2 = run_lopo(
    dataset_aug_norm, FEAT_A2, label='A2-orig+delta'
)
print()

print('=' * 60)
print('Ablation 3 — Original + Delta + Ratio features')
print('=' * 60)
fold_a3, y_true_a3, y_proba_a3, win_a3, imp_a3 = run_lopo(
    dataset_aug_norm, FEAT_A3, label='A3-orig+delta+ratio'
)
print()

seg_a1, seg_df_a1 = segment_level_metrics(win_a1)
seg_a2, seg_df_a2 = segment_level_metrics(win_a2)
seg_a3, seg_df_a3 = segment_level_metrics(win_a3)

def fold_summary(fold_df, label):
    print(f'--- {label} ---')
    for col in ['accuracy', 'balanced_acc', 'f1_macro', 'roc_auc', 'recall_high']:
        vals = fold_df[col].dropna()
        print(f'  {col:<15}: {vals.mean():.3f} +/- {vals.std():.3f}')
    print()

fold_summary(fold_a1, 'A1 Original only')
fold_summary(fold_a2, 'A2 Original + Delta')
fold_summary(fold_a3, 'A3 Original + Delta + Ratio')

print(f'Segment AUC:')
print(f'  A1 Original only       : {seg_a1["roc_auc"]:.3f}')
print(f'  A2 Original + Delta    : {seg_a2["roc_auc"]:.3f}')
print(f'  A3 Original+Delta+Ratio: {seg_a3["roc_auc"]:.3f}')

# %%
def make_row(label, n_feats, fold_df, seg_metrics):
    fd = fold_df.dropna(subset=['roc_auc'])
    return {
        'Variant':         label,
        'N Features':      n_feats,
        'Accuracy':        round(fd['accuracy'].mean(),     3),
        'Balanced Acc':    round(fd['balanced_acc'].mean(), 3),
        'F1 (macro)':      round(fd['f1_macro'].mean(),     3),
        'LOPO AUC':        round(fd['roc_auc'].mean(),      3),
        'Recall High':     round(fd['recall_high'].mean(),  3),
        'Segment AUC':     seg_metrics['roc_auc'],
        'Seg Recall High': seg_metrics['recall_high'],
    }

rows = [
    make_row('A1 — Original only',         len(FEAT_A1), fold_a1, seg_a1),
    make_row('A2 — Orig + Delta',          len(FEAT_A2), fold_a2, seg_a2),
    make_row('A3 — Orig + Delta + Ratio',  len(FEAT_A3), fold_a3, seg_a3),
]
comp_df = pd.DataFrame(rows).set_index('Variant')

print('=' * 75)
print('ABLATION COMPARISON TABLE')
print('=' * 75)
display(comp_df)
print()

numeric_cols = ['Accuracy', 'Balanced Acc', 'F1 (macro)', 'LOPO AUC',
                'Recall High', 'Segment AUC', 'Seg Recall High']
print('Delta vs A1 (original only):')
v_a1 = {col: float(comp_df.loc['A1 — Original only', col]) for col in numeric_cols}
for row_label in ['A2 — Orig + Delta', 'A3 — Orig + Delta + Ratio']:
    print(f'  {row_label}:')
    for col in numeric_cols:
        v   = float(comp_df.loc[row_label, col])
        d   = v - v_a1[col]
        sgn = '+' if d >= 0 else ''
        print(f'    {col:<18}: {v:.3f}  ({sgn}{d:.3f})')

# %%
participants_sorted = sorted(fold_a1['participant'].values)
x     = np.arange(len(participants_sorted))
width = 0.26

fd_a1 = fold_a1.set_index('participant')
fd_a2 = fold_a2.set_index('participant')
fd_a3 = fold_a3.set_index('participant')

auc_a1 = [fd_a1.loc[p, 'roc_auc'] if p in fd_a1.index else np.nan for p in participants_sorted]
auc_a2 = [fd_a2.loc[p, 'roc_auc'] if p in fd_a2.index else np.nan for p in participants_sorted]
auc_a3 = [fd_a3.loc[p, 'roc_auc'] if p in fd_a3.index else np.nan for p in participants_sorted]

fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(x - width, auc_a1, width=width, color='steelblue',  alpha=0.78,
       label='A1: Original', edgecolor='white')
ax.bar(x,         auc_a2, width=width, color='darkorange', alpha=0.78,
       label='A2: +Delta', edgecolor='white')
ax.bar(x + width, auc_a3, width=width, color='seagreen',   alpha=0.78,
       label='A3: +Delta+Ratio', edgecolor='white')
ax.axhline(0.5, color='black', linestyle='--', lw=1.2, label='Chance (0.5)')
ax.set_xticks(x)
ax.set_xticklabels([p.replace('UN_', 'P') for p in participants_sorted],
                   rotation=45, ha='right', fontsize=9)
ax.set_ylabel('LOPO ROC-AUC')
ax.set_title('Per-Participant LOPO ROC-AUC: Ablation A1 vs A2 vs A3')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('nb06_participant_auc.png', dpi=150, bbox_inches='tight')
plt.show()

# Participants where A3 > A1
gains = [(p, auc_a1[i], auc_a3[i], round(auc_a3[i]-auc_a1[i], 3))
         for i, p in enumerate(participants_sorted)
         if not np.isnan(auc_a1[i]) and not np.isnan(auc_a3[i])]
improved = [g for g in gains if g[3] > 0]
declined = [g for g in gains if g[3] < 0]
print(f'A3 vs A1 — improved: {len(improved)}  declined: {len(declined)}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

palette = [('A1: Original',       y_true_a1, y_proba_a1, 'steelblue'),
           ('A2: +Delta',         y_true_a2, y_proba_a2, 'darkorange'),
           ('A3: +Delta+Ratio',   y_true_a3, y_proba_a3, 'seagreen')]

for label, y_true, y_proba, color in palette:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    a = sklearn_auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{label}  (AUC={a:.3f})')

axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Chance')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Window-Level ROC (LOPO aggregated)')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

seg_palette = [('A1: Original',     seg_df_a1, 'steelblue'),
               ('A2: +Delta',       seg_df_a2, 'darkorange'),
               ('A3: +Delta+Ratio', seg_df_a3, 'seagreen')]

for label, seg_df_plot, color in seg_palette:
    fpr_s, tpr_s, _ = roc_curve(seg_df_plot['true_label'], seg_df_plot['proba_mean'])
    a_s = sklearn_auc(fpr_s, tpr_s)
    axes[1].plot(fpr_s, tpr_s, color=color, lw=2, label=f'{label}  (AUC={a_s:.3f})')

axes[1].plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Chance')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Segment-Level ROC (LOPO aggregated)')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.suptitle('Ablation ROC Curves: A1 vs A2 vs A3', fontsize=11)
plt.tight_layout()
plt.savefig('nb06_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Feature importance for A3 (full relative model), averaged over 24 folds
imp_df = pd.DataFrame({'feature': list(FEAT_A3), 'importance': imp_a3})
imp_df['group'] = 'original'
imp_df.loc[imp_df['feature'].str.endswith('_delta'), 'group'] = 'delta'
imp_df.loc[imp_df['feature'].str.endswith('_ratio'), 'group'] = 'ratio'

top_n = 25
top   = imp_df.nlargest(top_n, 'importance').reset_index(drop=True)

color_map = {'original': 'steelblue', 'delta': 'darkorange', 'ratio': 'seagreen'}
bar_colors = top['group'].map(color_map).values

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top)), top['importance'].values, color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(top)))
labels_short = (top['feature']
                .str.replace('_delta', '_d', regex=False)
                .str.replace('_ratio', '_r', regex=False))
ax.set_yticklabels(labels_short.values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mean Feature Importance (averaged over 24 folds)')
ax.set_title(f'Top {top_n} Features by Importance — A3 (Orig + Delta + Ratio)')
patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
ax.legend(handles=patches, loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('nb06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print('Feature importance by group (A3):')
group_summary = imp_df.groupby('group')['importance'].agg(['mean', 'sum']).round(4)
print(group_summary.to_string())
print()

# Identify top-2 original features (used in distribution plot below)
top2_orig = (imp_df[imp_df['group'] == 'original']
             .nlargest(2, 'importance')['feature']
             .tolist())
print(f'Top-2 original features for distribution plot: {top2_orig}')

# %%
# Correlation analysis: original vs delta vs ratio for a sample of 5 features
# Use the top-5 original features by importance in A1
imp_df_a1 = pd.DataFrame({'feature': list(FEAT_A1), 'importance': imp_a1})
top5_orig  = imp_df_a1.nlargest(5, 'importance')['feature'].tolist()

corr_cols = []
for f in top5_orig:
    corr_cols += [f, f'{f}_delta', f'{f}_ratio']

corr_data   = dataset_aug_norm[corr_cols].dropna()
corr_matrix = corr_data.corr()

short = [c.replace('_delta', '_d').replace('_ratio', '_r') for c in corr_cols]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.75, label='Pearson r')
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(short, rotation=90, fontsize=7)
ax.set_yticklabels(short, fontsize=7)
ax.set_title('Correlation Matrix: Original / Delta / Ratio\n(top-5 features by A1 importance)')
plt.tight_layout()
plt.savefig('nb06_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print('Pairwise correlations (original <-> relative):')
for f in top5_orig:
    r_od = corr_matrix.loc[f, f'{f}_delta']
    r_or = corr_matrix.loc[f, f'{f}_ratio']
    r_dr = corr_matrix.loc[f'{f}_delta', f'{f}_ratio']
    print(f'  {f[:30]:<30}: orig-delta={r_od:+.3f}  orig-ratio={r_or:+.3f}  delta-ratio={r_dr:+.3f}')

# %%
# Distribution of top-2 original features: Low vs High, for each representation
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
class_colors = {0: 'steelblue', 1: 'tomato'}
class_labels = {0: 'Low', 1: 'High'}

for row, feat in enumerate(top2_orig[:2]):
    feat_versions = [feat, f'{feat}_delta', f'{feat}_ratio']
    col_titles    = ['Original (absolute)', 'Delta (deviation)', 'Ratio (relative scale)']

    for col, (fv, title) in enumerate(zip(feat_versions, col_titles)):
        ax = axes[row, col]
        for lv in [0, 1]:
            vals = dataset_aug_norm.loc[dataset_aug_norm['label'] == lv, fv].dropna()
            ax.hist(vals, bins=40, alpha=0.55,
                    color=class_colors[lv], label=class_labels[lv], density=True)
        ax.set_title(f'{title}\n{fv[:35]}', fontsize=8)
        ax.set_xlabel('Normalized value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

plt.suptitle('Feature Distributions by Class: Original vs Delta vs Ratio\n(Top-2 features by A3 importance)', fontsize=10)
plt.tight_layout()
plt.savefig('nb06_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Conclusion — Interpreting the Results
# 
# ---
# 
# ### What was tested
# 
# This notebook added **delta** (absolute deviation from personal baseline) and **ratio**
# (current-to-baseline ratio) features to the existing absolute physiological features,
# and tested whether they improve cross-person cognitive load prediction in LOPO evaluation.
# 
# A three-way ablation was used:
# - **A1** — original features only (v2 equivalent)
# - **A2** — original + delta
# - **A3** — original + delta + ratio
# 
# The same GBT model, LOPO protocol, labels, and normalization were used throughout.
# 
# ---
# 
# ### If A2 or A3 improve performance vs A1
# 
# 1. **Inter-person feature scale mismatch was a limiting factor.** The cross-person model
#    struggled because absolute physiological thresholds do not generalize well.
#    Expressing features relative to each participant's resting baseline gave the model
#    information that the absolute features could not provide.
# 
# 2. **The ablation points to which feature group drove the improvement.** If A2 improves
#    but A3 does not add further, delta features are sufficient and ratio features
#    add noise. If A3 further improves on A2, ratio features capture a complementary
#    signal (multiplicative scaling) that delta alone misses.
# 
# 3. **Practical implication.** A brief pre-task baseline session improves the model's
#    ability to work with a new, unseen participant. This motivates including a
#    short onboarding calibration in any deployed wearable CLI system.
# 
# ---
# 
# ### If A2 and A3 do not improve vs A1
# 
# 1. **The per-participant z-score normalization already accounts for baseline differences.**
#    The z-scoring in v2 normalizes each participant's features to zero mean and unit
#    variance, effectively centering each person's distribution. After z-scoring, the
#    delta features are highly correlated with the original features (as the correlation
#    analysis in c15 reveals) and therefore add little new information.
# 
# 2. **The bottleneck is not feature representation.** If even personalized relative
#    features do not help, the signal-to-noise ratio in the physiological features
#    themselves may be the limiting factor — not how we scale them. This would point
#    toward richer feature engineering, longer signal windows, or additional modalities.
# 
# 3. **GBT with 100 trees may underfit the wider feature space.** Tripling feature
#    count without increasing model capacity is a known risk. A degradation in A3
#    compared to A2 (but not A1) would indicate the ratio features add noise that
#    100 shallow trees cannot filter. This is an honest limitation of fixing
#    n_estimators=100 across all ablations.
# 
# ---
# 
# ### What this analysis is — and is not
# 
# | This IS | This is NOT |
# |---|---|
# | A controlled ablation of relative feature engineering | Full personalization |
# | A test using the same model, labels, and protocol | Hyperparameter optimization |
# | An honest comparison of absolute vs relative features | A claim of production-ready accuracy |
# | A calibration-based scenario (Low windows only) | Use of any test-participant task labels |
# 
# ---
# 
# ### Connection to the project narrative
# 
# This notebook completes the progressive improvement story:
# 
# | Notebook | Contribution |
# |---|---|
# | 02 | Established the v2 baseline with LOPO and normalization |
# | 03 | Identified label noise via NASA-TLX questionnaire |
# | 04 | Tested noisy-label filtering (v3) as a training data improvement |
# | 05 | Tested baseline calibration as a normalization improvement |
# | 06 | Tested relative feature engineering as a feature representation improvement |
# 
# Each notebook addresses one dimension of the cross-person generalization problem.
# Together they form a systematic analysis of the factors limiting CLI performance
# for new, unseen participants.


