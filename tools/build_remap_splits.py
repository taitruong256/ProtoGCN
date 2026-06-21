#!/usr/bin/env python3
"""Build 5-fold REMAP SitToStand splits grouped by patient.

The script reads the label workbook, removes rows with missing
`MDS-UPDRS_score_3.9 _arising_from_chair`, assigns each patient to one of five
folds with stratification on the label distribution, and writes per-fold
train/val/test CSVs for the ProtoGCN REMAP config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


LABEL_COL = 'MDS-UPDRS_score_3.9 _arising_from_chair'


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


def assign_folds(df: pd.DataFrame, random_state: int = 42, n_splits: int = 5) -> pd.DataFrame:
    groups = df['patient_id'].to_numpy()
    y = df['label'].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_assignment = np.full(len(df), -1, dtype=np.int64)
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        fold_assignment[test_idx] = fold_idx
    if np.any(fold_assignment < 0):
        raise RuntimeError('Some samples were not assigned to a fold.')
    df = df.copy()
    df['fold'] = fold_assignment
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
        val_fold = (fold + 1) % n_splits
        train_mask = ~df['fold'].isin([test_fold, val_fold])
        val_mask = df['fold'] == val_fold
        test_mask = df['fold'] == test_fold

        common_cols = [
            'file', 'frame_dir', 'patient_id', 'Participant ID number', 'cohort',
            'PD_or_C', 'Transition ID', 'label', LABEL_COL, 'total_frames', 'fold'
        ]
        train_df = df.loc[train_mask, common_cols].copy()
        val_df = df.loc[val_mask, common_cols].copy()
        test_df = df.loc[test_mask, common_cols].copy()

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
