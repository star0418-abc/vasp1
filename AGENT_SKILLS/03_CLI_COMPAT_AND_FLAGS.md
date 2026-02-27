# 03_CLI_COMPAT_AND_FLAGS

## 1) Backward-compatibility principle
- Do not change existing flag meaning casually.
- If a default must change for safety (e.g., cut-bond handling):
  - document behavior change in README
  - provide an explicit flag to restore legacy behavior
  - record the chosen policy in metadata

## 2) Recommended new flags (minimal set; introduce only if needed)
- `--cut_bond_policy {heal,warn,error}`
- `--density_check {strict,warn,skip}`
- `--mic_mode {auto,on,off}` (or enforce `off` in cluster+vacuum)
- `--neutralize {none,nearest_counterions}`
- `--self_check` (developer-only, optional; removed before final merge)
- `--write_index_map` (write original→cluster index mapping)

## 3) Requirements for any new flags
- Must appear in `--help` with clear, example-driven descriptions
- Must be recorded in `model_meta.json` so runs are reproducible
- Defaults should optimize for physical safety and explicit failure
