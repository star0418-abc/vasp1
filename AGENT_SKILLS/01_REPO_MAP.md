# 01_REPO_MAP

## A. Responsibility map (by dataflow)

### 1) Structure cutting & preprocessing
- `setup_aimd_ase.py`
  - Inputs: `POSCAR/CONTCAR/PDB/GRO/XYZ` (anything ASE can read)
  - Outputs: cluster/bulk structure file(s), `model_meta.json`, `selected_indices.txt`, (recommended) `cut_bonds_report.txt`
  - Core invariants:
    - No dangling bonds by default (prevent radicals)
    - Neutralization never selects single atoms
    - Cluster+vacuum must not use MIC for distance/clash
    - Density/volume checks only apply to true 3D periodic bulk
    - Fallbacks must not silently disable bonding protection

### 2) AIMD input generation
- `make_incar_aimd.py`
  - Inputs: CLI args + optional `INCAR.base`
  - Outputs: INCAR for equilibration/production stages, optional KPOINTS guidance
  - Invariants:
    - Merge/override precedence is explicit and logged
    - Enforced must-have settings (e.g., symmetry off for AIMD) are traceable

### 3) Run & resume orchestration
- `run_vasp.sh`
  - Handles: RESUME behavior, core-count policy, disk checks, archiving/backups
  - Invariants:
    - RESUME never silently reuses stale inputs
    - Any fallback (e.g., downscaling MPI ranks) is logged clearly
    - Archiving prevents accidental overwrite across reruns

### 4) Post-processing
- `aimd_msd.py`
  - Diffusion: MSD/MTO, unwrap, COM removal, running-D plateau detection, uncertainty
  - Invariants:
    - Defaults avoid “pseudo-diffusion”
    - All key parameters are recorded in outputs

### 5) Electronic properties
- `setup_electronic.py` / `analyze_electronic.py`
  - Work function / DOS workflows; file chain depends on `CHGCAR/LOCPOT/OUTCAR`
  - Invariants:
    - Parsers must not silently produce numbers without sanity checks
    - Assumptions (vacuum layer, dipole correction, reference energy) are recorded

## B. File contracts (must remain stable)

### `model_meta.json` must include (minimum)
- provenance:
  - command line, timestamp, script version (and ideally input file hash)
- selection:
  - `center_index_original`
  - `center_index_cluster`
  - selected atom count
  - `allow_exceed_max_atoms`
- cut bonds:
  - policy, initial count, healing/capping actions, final count, report path
- neutralization:
  - enabled, method (residue/component), added components, `remaining_charge`, verified
- pbc/mic:
  - `bulk_density_valid`, `mic_valid`, `cluster_vacuum_mic_used`
- warnings:
  - explicit list of any degraded assumptions or overrides
