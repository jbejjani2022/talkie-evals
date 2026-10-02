"""Portable paths; payloads default outside the code checkout."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get('TRAIT_BUNDLE_ROOT', REPO / 'data')).resolve()
ROOT = Path(os.environ.get('TRAIT_ARTIFACT_ROOT', REPO / 'artifacts')).expanduser().resolve()

def require_gpu():
    import shutil
    if shutil.which('sinfo') and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('On SLURM hosts, submit GPU commands with sbatch.')

def model_path(family):
    return ROOT / 'models' / f'{family}-atomic'
