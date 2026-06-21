#!/usr/bin/env python3
"""Build 5-fold REMAP SitToStand splits grouped by patient.

The script reads the label workbook, removes rows with missing
`MDS-UPDRS_score_3.9 _arising_from_chair`, assigns each patient to one of five
folds with stratification on the label distribution, and writes per-fold
train/test CSVs for the ProtoGCN REMAP config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COL = 'MDS-UPDRS_score_3.9 _arising_from_chair'
MAX_LABEL = 3


def _resolve_workbook(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    fallback = path.parent / 'SitToStand_human_labels.xls'
    if fallback.exists():
        return fallback
    raise FileNotFoundError(path)


def _sequence_name(participant_id: int, cohort: str, transition_id: int) -> str:
    return f'Pt{participant_id}_{cohort}_n_{transition_id}'


def load_metadata(label_file: Path, skeleton_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(_resolve_workbook(label_file))
    if LABEL_COL not in df.columns:
        raise KeyError(f'Label column not found: {LABEL_COL}')

    df = df.dropna(subset=[LABEL_COL]).copy()
    df['Participant ID number'] = df['Participant ID number'].astype(int)
    df['Transition ID'] = df['Transition ID'].astype(int)
    df['label'] = df[LABEL_COL].astype(int)
    df = df[df['label'] <= MAX_LABEL].copy()
    df['patient_id'] = df['Participant ID number'].astype(str)
    df['cohort'] = df['PD_or_C'].astype(str).str.strip()
    df['frame_dir'] = [
        _sequence_name(pid, cohort, tid)
        for pid, cohort, tid in zip(df['Participant ID number'], df['cohort'], df['Transition ID'])
    ]
    df['file'] = df['frame_dir'] + '.csv'
    df['csv_path'] = df['file'].map(lambda x: skeleton_dir / x)
    df['total_frames'] = df['csv_path'].map(lambda p: pd.read_csv(p, comment='#', header=None).shape[0] if p.exists() else np.nan)
    df = df.dropna(subset=['total_frames']).copy()
    df['total_frames'] = df['total_frames'].astype(int)
    return df


def _label_entropy(label_counts: np.ndarray) -> float:
    total = label_counts.sum()
    if total == 0:
        return 0.0
    p = label_counts.astype(np.float32) / float(total)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _aggregate_patient_stats(df: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    rows = []
    for patient_id, sub in df.groupby('patient_id'):
        label_counts = np.bincount(sub['label'].to_numpy(dtype=np.int64), minlength=n_classes)
        rows.append(
            dict(
                patient_id=patient_id,
                sequence_count=int(len(sub)),
                label_counts=label_counts,
                label_entropy=_label_entropy(label_counts),
            )
        )
    return pd.DataFrame(rows)


def _assignment_score(fold_counts: np.ndarray, target_counts: np.ndarray, fold_sizes: np.ndarray, target_size: float) -> float:
    label_cost = ((fold_counts - target_counts) ** 2 / (target_counts + 1e-6)).sum()
    size_cost = (((fold_sizes - target_size) ** 2) / (target_size + 1e-6)).sum()
    return float(label_cost + 0.25 * size_cost)


def _choose_fold_for_patient(patient_counts: np.ndarray,
                             fold_counts: np.ndarray,
                             target_counts: np.ndarray,
                             fold_sizes: np.ndarray,
                             target_size: float) -> int:
    scores = []
    for fold_idx in range(fold_counts.shape[0]):
        new_counts = fold_counts.copy()
        new_counts[fold_idx] += patient_counts
        new_sizes = fold_sizes.copy()
        new_sizes[fold_idx] += patient_counts.sum()
        scores.append((_assignment_score(new_counts, target_counts, new_sizes, target_size), fold_idx))
    scores.sort(key=lambda x: (x[0], x[1]))
    return scores[0][1]


def _refine_assignment(patient_df: pd.DataFrame,
                       assignment: dict,
                       target_counts: np.ndarray,
                       target_size: float,
                       n_splits: int) -> dict:
    fold_counts = np.zeros((n_splits, len(target_counts)), dtype=np.float32)
    fold_sizes = np.zeros(n_splits, dtype=np.float32)
    for _, row in patient_df.iterrows():
        fold_idx = assignment[row['patient_id']]
        fold_counts[fold_idx] += row['label_counts']
        fold_sizes[fold_idx] += row['sequence_count']

    improved = True
    while improved:
        improved = False
        current_score = _assignment_score(fold_counts, target_counts, fold_sizes, target_size)
        for _, row in patient_df.iterrows():
            pid = row['patient_id']
            src = assignment[pid]
            patient_counts = row['label_counts']
            patient_size = row['sequence_count']

            best_fold = src
            best_score = current_score
            for dst in range(n_splits):
                if dst == src:
                    continue
                new_fold_counts = fold_counts.copy()
                new_fold_sizes = fold_sizes.copy()
                new_fold_counts[src] -= patient_counts
                new_fold_counts[dst] += patient_counts
                new_fold_sizes[src] -= patient_size
                new_fold_sizes[dst] += patient_size
                score = _assignment_score(new_fold_counts, target_counts, new_fold_sizes, target_size)
                if score + 1e-9 < best_score:
                    best_score = score
                    best_fold = dst
                    best_counts = new_fold_counts
                    best_sizes = new_fold_sizes

            if best_fold != src:
                assignment[pid] = best_fold
                fold_counts = best_counts
                fold_sizes = best_sizes
                current_score = best_score
                improved = True

    return assignment


def assign_folds(df: pd.DataFrame, random_state: int = 42, n_splits: int = 5, n_restarts: int = 128) -> pd.DataFrame:
    df = df.copy()
    n_classes = int(df['label'].max()) + 1
    patient_df = _aggregate_patient_stats(df, n_classes)
    target_counts = np.bincount(df['label'].to_numpy(dtype=np.int64), minlength=n_classes).astype(np.float32) / float(n_splits)
    target_size = len(df) / float(n_splits)

    priority = patient_df.copy()
    priority['rarity_score'] = priority['label_counts'].apply(
        lambda counts: float(np.dot(counts, 1.0 / (target_counts + 1e-6))) + 0.1 * _label_entropy(counts)
    )
    priority = priority.sort_values(['rarity_score', 'sequence_count'], ascending=[False, False]).reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    best_assignment = None
    best_score = None

    for restart in range(n_restarts):
        order = priority.copy()
        order['_tie_break'] = rng.rand(len(order)) if restart > 0 else np.arange(len(order), dtype=np.float32)
        order = order.sort_values(
            ['rarity_score', 'sequence_count', '_tie_break'],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        fold_counts = np.zeros((n_splits, n_classes), dtype=np.float32)
        fold_sizes = np.zeros(n_splits, dtype=np.float32)
        assignment = {}

        for _, row in order.iterrows():
            fold_idx = _choose_fold_for_patient(
                row['label_counts'], fold_counts, target_counts, fold_sizes, target_size)
            fold_counts[fold_idx] += row['label_counts']
            fold_sizes[fold_idx] += row['sequence_count']
            assignment[row['patient_id']] = fold_idx

        assignment = _refine_assignment(priority, assignment, target_counts, target_size, n_splits)
        fold_counts = np.zeros((n_splits, n_classes), dtype=np.float32)
        fold_sizes = np.zeros(n_splits, dtype=np.float32)
        for _, row in priority.iterrows():
            fold_idx = assignment[row['patient_id']]
            fold_counts[fold_idx] += row['label_counts']
            fold_sizes[fold_idx] += row['sequence_count']

        score = _assignment_score(fold_counts, target_counts, fold_sizes, target_size)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError('Failed to assign folds.')

    df['fold'] = df['patient_id'].map(best_assignment).astype(int)
    return df


def write_splits(df: pd.DataFrame, output_dir: Path, n_splits: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_fold = (
        df[['patient_id', 'fold']]
        .drop_duplicates()
        .sort_values(['fold', 'patient_id'])
    )
    patient_fold.to_csv(output_dir / 'patient_folds.csv', index=False)

    for fold in range(n_splits):
        test_fold = fold
        train_mask = df['fold'] != test_fold
        test_mask = df['fold'] == test_fold

        common_cols = [
            'file', 'frame_dir', 'patient_id', 'Participant ID number', 'cohort',
            'PD_or_C', 'Transition ID', 'label', LABEL_COL, 'total_frames', 'fold'
        ]
        train_df = df.loc[train_mask, common_cols].copy()
        test_df = df.loc[test_mask, common_cols].copy()
        val_df = test_df.copy()

        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        train_df.to_csv(output_dir / f'fold_{fold}_train.csv', index=False)
        val_df.to_csv(output_dir / f'fold_{fold}_val.csv', index=False)
        test_df.to_csv(output_dir / f'fold_{fold}_test.csv', index=False)


def print_fold_summary(df: pd.DataFrame, n_splits: int = 5) -> None:
    print(f'Total sequences: {len(df)}')
    print(f'Patients: {df["patient_id"].nunique()}')
    print('Overall label distribution:')
    print(df['label'].value_counts().sort_index().to_string())
    print('\nPer-fold label distribution:')
    for fold in range(n_splits):
        sub = df[df['fold'] == fold]
        print(f'\nFold {fold}: {len(sub)} sequences, {sub["patient_id"].nunique()} patients')
        print(sub['label'].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description='Build REMAP SitToStand splits')
    parser.add_argument(
        '--label-file',
        type=Path,
        default=Path('data/REMAP/SitToStand/Data/STS_human_labels/.~SitToStand_human_labels.xls'),
        help='Path to the REMAP label workbook or its zero-byte temp lock file.',
    )
    parser.add_argument(
        '--skeleton-dir',
        type=Path,
        default=Path('data/REMAP/SitToStand/Data/STS_2D_skeletons_coarsened'),
        help='Directory with the SitToStand skeleton CSVs.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/REMAP/SitToStand/splits'),
        help='Directory to write fold CSVs.',
    )
    args = parser.parse_args()

    label_file = _resolve_workbook(args.label_file)
    df = load_metadata(label_file, args.skeleton_dir)
    df = assign_folds(df)
    write_splits(df, args.output_dir)
    print_fold_summary(df)
    print(f'\nWrote splits to: {args.output_dir}')


if __name__ == '__main__':
    main()
