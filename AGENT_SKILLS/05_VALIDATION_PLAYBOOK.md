# 05_VALIDATION_PLAYBOOK

## Purpose
Any change must demonstrate:
- No silent degradation in “no-utils” mode
- No default radical creation (cut bonds healed or run fails explicitly)
- Neutralization adds whole components, not atoms
- Cluster+vacuum does not use MIC wrap causing fake clashes
- Density sanity does not falsely block non-3D periodic systems

## Minimum self-checks (no external files)
Prefer implementing `--self_check` (or developer-only block removed before final merge).

### 1) Cut-bond toy
- Construct a simple covalent chain where a spherical selection would cut a bond.
- Expected:
  - `heal` reduces cut bonds to 0, OR
  - run fails with a clear error stating why (max_atoms reached), unless overridden.

### 2) Neutralization toy
- Construct a multi-atom anion component without residue metadata.
- Expected:
  - neutralization selects and adds whole connected components
  - `remaining_charge` is reported correctly
  - post-check verification is recorded

### 3) PBC/MIC toy
- Create a cluster with a vacuum box.
- Expected:
  - distance/clash checks use `mic=False`
  - no “across vacuum” minimum-image artifacts

## CLI regressions (lightweight)
- `python setup_aimd_ase.py --help` contains any new flags
- Output files exist when relevant:
  - `model_meta.json`
  - `selected_indices.txt`
  - `cut_bonds_report.txt` (if cut-bond detection triggers)

## Cleanup rule
If any TEST or temporary validation files are created during development:
- delete the TEST-related files after code modifications are complete.
