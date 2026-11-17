"""Reproduce all manuscript analyses and Figures 1–10."""
from pathlib import Path
import subprocess
import sys

CODE_DIRECTORY = Path(__file__).resolve().parent / 'code'
SCRIPTS = [
    'verify_numerics.py',
    'analyses_model.py',
    'analyses_scen.py',
    'analyses_burnin.py',
    'rp_curve.py',
    'sens.py',
    'revision_checks.py',
    'make_figures.py',
]

if __name__ == '__main__':
    for index, script in enumerate(SCRIPTS, start=1):
        print(f'[{index}/{len(SCRIPTS)}] Running {script}', flush=True)
        subprocess.run([sys.executable, script], cwd=CODE_DIRECTORY, check=True)
    print('Completed. Figures are saved in figs/.', flush=True)
