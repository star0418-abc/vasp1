# 04_LOGGING_METADATA_CONTRACT

## 1) Logging requirements
Every run must print (or write) explicit statements for:
- PBC/MIC decisions:
  - `bulk_density_valid`, `mic_valid`, and whether MIC was actually used
  - cluster+vacuum detection and enforced `mic=False`
- Cut-bonds:
  - initial cut-bond count
  - policy selected
  - healing/capping actions taken
  - final cut-bond count (or failure reason)
- Neutralization:
  - method used (residue/component/none)
  - list of added components (size, charge, distance metric)
  - `remaining_charge`
  - verification result and threshold used
- Density:
  - only compute/validate if true 3D bulk; otherwise print “skipped (not 3D periodic bulk)”

## 2) `model_meta.json` minimum fields (must)
- `command_line`, `timestamp`, `script_version` (if available)
- `input`: path, format, and optional hash
- `selection`:
  - `center_index_original`
  - `center_index_cluster`
  - `selected_atom_count`
  - `allow_exceed_max_atoms`
- `cut_bonds`:
  - `policy`
  - `initial_n`, `final_n`
  - `healed`/`capped` booleans
  - report paths
- `neutralization`:
  - `enabled`
  - `method_used`
  - `added_components` (id, charge, size, distance)
  - `remaining_charge`
  - `verified`
- `pbc_mic`:
  - `bulk_density_valid`
  - `mic_valid`
  - `cluster_vacuum_mic_used`
- `warnings`: array of strings

## 3) Structure/arrays stability
- When copying ASE arrays, use `Atoms.new_array()` rather than raw dict assignment.
- If indices matter for debugging, support writing an `index_map.json`:
  - `original_index -> cluster_index` and vice versa.
