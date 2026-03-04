# VASP Scripts - 凝胶电解质计算工作流

针对 VASP (oneAPI + Intel MPI) 环境的计算辅助脚本集合，适用于凝胶电解质、AIMD 等计算任务。

---

## ⚠️ 使用边界（Scope）

> **在开始使用前，请理解 DFT-AIMD 的适用范围**

| 性质 | 推荐方法 | 说明 |
|------|----------|------|
| **输运性质（D, σ）** | 经典 MD (ns 级) | AIMD 时间尺度太短 (ps)，扩散系数仅供趋势参考 |
| **局域结构/配位** | DFT-AIMD | AIMD 优势：电子结构精确 |
| **反应趋势/过渡态** | DFT-AIMD / NEB | 反应能垒、过渡态搜索 |
| **电化学稳定窗口 (ESW)** | 片段氧化/还原能 | PBE 带隙不能直接当 ESW！ |
| **长程扩散 (ns 级)** | 经典 MD | AIMD 不适用 |

### 模型类型说明

- **bulk 模式**：周期性子胞，适合局域性质分析
- **cluster 模式**：有限真空簇，**不能用于 bulk 性质**，表面效应显著

---

## 🚨 常见陷阱（必读）

### 1. Cluster Trap（真空簇误用）
❌ **错误**：用 cluster 模式计算扩散系数并与 bulk 比较  
✅ **正确**：cluster 仅用于局域电子结构分析；扩散用 bulk 模式或经典 MD

### 2. False Diffusion Trap（伪扩散）
❌ **错误**：AIMD 几 ps 直接用 r(t)-r(0) 拟合 MSD 得到 D  
✅ **正确**：使用 `aimd_msd.py` v2.3（默认 MTO）：
  - Multiple Time Origins (MTO): 对每个 lag τ 平均所有起点
  - 检查 log-log 斜率 α ≈ 1（正常扩散）
  - 检查 D(t) 平台稳定
  - α < 0.8 表示亚扩散/caging，D 不可信

### 6. Bulk Density Trap（bulk 密度失控）
❌ **错误**：用 bounding box + buffer 定 bulk 子胞盒子 → 密度远低于凝胶  
✅ **正确**：`setup_aimd_ase.py` v2.2 按原体系密度反推 V = M_sub / ρ_orig


### 3. Langevin Gamma Trap（热浴干扰）
❌ **错误 1**：gamma=20 跑全程，扩散被抑制  
❌ **错误 2**：LANGEVIN_GAMMA 写标量（如 `LANGEVIN_GAMMA = 10`），只热浴第一种元素！  
✅ **正确**：
  - 平衡段 gamma=10-20，生产段 gamma=1-5
  - LANGEVIN_GAMMA **必须是向量**，每种原子类型一个值：`LANGEVIN_GAMMA = 5.0 10.0 3.0`
  - 使用 `make_incar_aimd.py` v3.0.2 自动生成正确格式

### 4. Band Gap ≠ ESW Trap（带隙误用）
❌ **错误**：PBE 带隙 = 电化学稳定窗口  
✅ **正确**：ESW 需要片段氧化/还原能或反应自由能分析

### 5. Recipe Rounding Trap（凑整误差）
❌ **错误**：200 原子体系严格保持 wt%  
✅ **正确**：使用 `soft_atoms` 软约束凑整（默认），同时看原子数偏差和 wt% 偏差；必要时增大 `target_atoms`

---

## 文件列表

| 文件 | 用途 |
|------|------|
| `vasp_env.sh` | VASP 运行环境配置（oneAPI、自检、线程控制） |
| `run_vasp.sh` | VASP 运行脚本（备份、日志、续算、核数检查） |
| `check_vasp.sh` | 检查计算状态（完成标志、能量、费米能级） |
| `aimd_watch.sh` | AIMD 实时监控（温度、离子步、能量） |
| `aimd_msd.py` | MSD 计算与扩散系数拟合 v2.3（MTO + α 判定 + 分段误差 + traceability） |
| `aimd_post.py` | AIMD 热力学后处理 v2.0（E0/T/F/P → CSV、能量漂移、平衡跳过、traceability） |
| `clean_vasp.sh` | 安全清理大文件（WAVECAR、CHGCAR 等） |
| `recipe.yaml` | 配方定义文件示例（8 类组分 + 模拟条件） |
| `recipe_validate.py` | 配方验证工具（校验 wt% 总和、温度、格式） |
| `recipe_to_counts.py` | 配方换算工具（wt% → 分子/原子数） |
| `make_incar_aimd.py` | AIMD INCAR 生成器（核心） |
| `aimd_setup.sh` | AIMD 一键设置脚本 |
| `setup_electronic.py` | 电子性质输入生成（功函数/DOS） |
| `analyze_electronic.py` | 电子性质后处理（功函数/DOS） |
| `setup_aimd_ase.py` | 从大体系切割 AIMD 子体系 v2.4.1（显式 MIC 控制 + 真实元数据 + 拓扑安全重成像） |
| `examples/` | 示例文件目录 |

## 快速开始

### 1. 安装

```bash
cd ~/vasp_scripts
chmod +x *.sh *.py
source ~/.bashrc
```

### 2. 依赖安装

```bash
pip install numpy pyyaml ase matplotlib
```

### 3. 命令行检查

```bash
python3 setup_aimd_ase.py --help
```

---

## 环境配置 (vasp_env.sh)

### 功能

- 加载 oneAPI 环境（失败时明确报错，不静默）
- 自检 VASP 可执行文件（vasp_std/gam/ncl）
- 自检 mpirun 可用性
- 检查 vasp_std 动态库依赖
- 设置默认纯 MPI 模式（OMP_NUM_THREADS=1）

### 用法

```bash
source ~/vasp_scripts/vasp_env.sh
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `VASP_BIN` | VASP 可执行文件目录 |
| `VASP_PP_PATH` | POTCAR 路径（用于自动拼接脚本） |
| `OMP_NUM_THREADS` | OpenMP 线程数（默认 1） |
| `MKL_NUM_THREADS` | MKL 线程数（默认 1） |
| `I_MPI_ADJUST_REDUCE` | Intel MPI 参数（不强制覆盖，尊重用户设置） |

---

## VASP 运行脚本 (run_vasp.sh)

### 功能

- ✅ 核数检查（仅 WSL 应用 `RESERVE_CORES` 预留；非 WSL 不预留）
- ✅ 磁盘空间预检查（POSIX 兼容）
- ✅ AIMD 续算支持（CONTCAR → POSCAR + 轨迹合并）
- ✅ stdout/stderr 分离输出（支持 stdbuf 行缓冲，崩溃场景显式等待日志落盘）
- ✅ mpirun 返回码可靠捕获（管道过滤不影响真实退出码）
- ✅ 完成判定绑定本轮 OUTCAR 片段（避免旧完成标记误判）
- ✅ 运行耗时统计
- ✅ 自动备份输入文件
- ✅ 条件化归档旧输出文件（保护续算用文件）
- ✅ 并行参数提示（NCORE/KPAR，支持 `A=1 ; B=2` 行内解析）
- ✅ **v2.0** 重启文件守卫（ISTART/ICHARG 验证）
- ✅ **v2.0** HPC 线程保护（防止 MKL/OMP 过载）
- ✅ **v2.0** XDATCAR 轨迹合并（aimd_msd.py 连续性）
- ✅ **v2.0** XDATCAR 合并仅在“本轮被接受”后执行（失败/未完成不合并）
- ✅ **v2.0** RESUME 日志轮转保护（OUTCAR/OSZICAR 防无限增长）

### 用法

```bash
# 基本运行
NP=16 EXE=vasp_std run_vasp.sh

# 续算模式（保留 WAVECAR/CHGCAR/XDATCAR）
RESUME=1 NP=16 run_vasp.sh

# 续算但归档旧 WAVECAR（强制重新计算波函数）
RESUME=1 KEEP_RESTART=0 NP=16 run_vasp.sh

# 严格核数检查（超限报错）
STRICT_NP=1 NP=32 run_vasp.sh

# 强制忽略磁盘检查
FORCE_DISK=1 run_vasp.sh

# 禁用线程保护（仅当你知道自己在做什么）
THREAD_GUARD=0 NP=64 run_vasp.sh

# WSL 下显式启用 OMP 绑核（默认关闭）
FORCE_OMP_BIND=1 THREAD_GUARD=1 NP=64 run_vasp.sh

# RESUME 时保留 OUTCAR/OSZICAR 追加（默认按阈值轮转）
RESUME=1 KEEP_APPEND_LOGS=1 NP=16 run_vasp.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NP` | 8 | MPI 进程数 |
| `EXE` | vasp_std | 可执行文件 |
| `OUT` | vasp.out | stdout 文件 |
| `ERR` | vasp.err | stderr 文件 |
| `RESUME` | 0 | 续算模式：1=自动 cp CONTCAR→POSCAR |
| `STRICT_NP` | 0 | 严格核数：1=超限报错，0=自动下调 |
| `RESERVE_CORES` | 2 | WSL 预留核数（仅 WSL 生效） |
| `MIN_FREE_GB` | 20 | 最小磁盘空间 (GB) |
| `FORCE_DISK` | 0 | 忽略磁盘检查（默认启用检查） |

**v2.0 新增变量（AIMD 续算 / HPC 安全）**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KEEP_RESTART` | auto | 保留 WAVECAR/CHGCAR/CHG：RESUME=1 时默认 1，否则 0 |
| `KEEP_TRAJECTORY` | auto | 保留 XDATCAR：仅在有效续算语义下生效（RESUME=1 且 CONTCAR 校验通过） |
| `FORCE_RESTART` | 0 | 跳过 WAVECAR/CHGCAR 存在性检查（强制运行） |
| `THREAD_GUARD` | 1 | 线程保护：1=强制 OMP/MKL 线程限制，防止 HPC 过载 |
| `STRICT_INPUT` | 0 | 严格输入：1=KPOINTS/KSPACING 缺失时报错退出 |
| `FORCE_OMP_BIND` | 0 | WSL 下是否强制 `OMP_PROC_BIND/OMP_PLACES`（1=恢复旧行为） |
| `KEEP_APPEND_LOGS` | 0 | RESUME 时保留 OUTCAR/OSZICAR 持续追加（1=不轮转） |
| `LOG_ROTATE_GB` | 2 | RESUME 时 OUTCAR/OSZICAR 轮转阈值（GB） |

### AIMD 续算行为（v2.0 重要变更）

> [!IMPORTANT]
> **续算 (RESUME=1) 时，WAVECAR/CHGCAR/XDATCAR 默认保留在工作目录**，不再移动到 `old/`。
> 这确保：
> 1. 电子结构从 WAVECAR 正确重启（避免"能量冲击"）
> 2. XDATCAR 轨迹连续，`aimd_msd.py` 可计算长程 MSD

**续算保护机制**：
- 如果 INCAR 中 `ISTART ≥ 1` 但 WAVECAR 不存在 → 脚本中止
- 如果 INCAR 中 `ICHARG = 1` 但 CHGCAR 不存在 → 脚本中止
- `RESUME=1` 时会严格校验 CONTCAR（晶格/原子计数/坐标块完整性）；可疑或截断文件会直接中止
- 设置 `FORCE_RESTART=1` 可跳过检查（强制从头计算）

**XDATCAR 轨迹合并**：
- 仅在“有效续算”下备份旧 XDATCAR（`RESUME=1` 且 `CONTCAR -> POSCAR` 校验通过）
- `KEEP_TRAJECTORY=1` 但非有效续算时，会自动禁用轨迹保留并归档旧 XDATCAR，避免跨任务误拼接
- 从 POSCAR 计算 `NATOMS`，提取 `XDATCAR.prev` 最后一帧与新 XDATCAR 第一帧
- 若两帧坐标（去空白后）完全相同，则跳过新 XDATCAR 的第一帧再追加，避免重复 frame 0
- 若帧解析失败，则回退为 header-skip 追加，并在 `run.log` 写入 `WARNING`
- `Direct configuration=` 检测支持前导空白
- **仅当本轮被接受**（`mpirun rc=0` 且本轮 OUTCAR 片段检测到完成标志）才执行合并
- 本轮失败/未完成时会跳过合并，防止污染历史轨迹
- 保证 `aimd_msd.py` 可处理完整轨迹

**完成判定（RESUME 关键）**：
- 完成检查仅扫描本轮新增 OUTCAR 片段，不会被旧追加段的完成标志“误判成功”
- `mpirun` 非零返回码不会出现最终 `[OK]` 完成提示

**RESUME 日志轮转（OUTCAR/OSZICAR）**：
- 默认 `RESUME=1` 时按 `LOG_ROTATE_GB`（默认 2 GB）检查 OUTCAR/OSZICAR
- 超过阈值会移动到 `old/<name>.resume.<timestamp>`，新一轮计算从空文件开始
- 设置 `KEEP_APPEND_LOGS=1` 可保留旧的持续追加行为
- 轮转/保留动作都会记录到 `run.log`

### INCAR 参数解析（run_vasp.sh）

脚本读取 `NCORE/KPAR/ISTART/ICHARG` 时：
- 忽略 `#` 和 `!` 注释
- 支持同一行多参数（`;` 分隔），例如：`NBLOCK = 1 ; KPAR = 4 ; NCORE=8`
- 同一参数出现多次时取最后一次（覆盖式写法）

### HPC 线程安全（v2.0）

> [!WARNING]
> **无 THREAD_GUARD 保护时，MKL/OpenMP 可能导致严重过载**
> 例如：64 MPI ranks × 64 threads = 4096 线程，导致系统崩溃

默认 `THREAD_GUARD=1` 时，脚本设置：
```bash
OMP_NUM_THREADS=1      # 或用户设置的值
MKL_NUM_THREADS=$OMP_NUM_THREADS
OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
```

- 非 WSL：默认补充 `OMP_PROC_BIND=close`、`OMP_PLACES=cores`
- WSL：默认不强制这两个变量（避免部分 Hyper-V/WSL 拓扑下性能下降）
- 若你在 WSL 明确要使用绑核策略，设置 `FORCE_OMP_BIND=1`

### 输出监控

推荐使用以下命令实时监控：
```bash
tail -f vasp.out        # stdout（stdbuf 启用时低延迟）
tail -f vasp.err        # stderr（真实错误信息）
aimd_watch.sh           # AIMD 专用监控
```

### 输出文件

- `vasp.out` - 标准输出（真实 stdout）
- `vasp.err` - 标准错误（真实 stderr，非 stdout 副本）
- `run.log` - 运行日志（含耗时、状态、线程配置）
- `snapshots/<timestamp>/` - 输入文件备份
- `old/` - 旧输出文件归档（受 KEEP_* 控制）
- `XDATCAR.prev.<timestamp>` - 续算时的轨迹备份
- `XDATCAR.merged.<timestamp>` - 合并后的完整轨迹（临时）
- `XDATCAR.new.<timestamp>` - 合并前的新轨迹备份
- `old/OUTCAR.resume.<timestamp>` / `old/OSZICAR.resume.<timestamp>` - RESUME 轮转日志

---

## 配方层 (Recipe Layer)

### 配方组分（8 类固定顺序）

| 序号 | 类别 | 中文名 | 必需性 |
|------|------|--------|--------|
| 1 | `salt_solution` | 盐溶液 | 主要 |
| 2 | `polymer_matrix` | 聚合物基质 | 主要 |
| 3 | `crosslinker` | 交联剂 | 主要 |
| 4 | `photoinitiator` | 引发剂 | 主要 |
| 5 | `plasticizer_solvent` | 增塑剂/溶剂 | 可选 |
| 6 | `functional_monomer` | 功能单体 | 可选 |
| 7 | `stabilizer` | 稳定剂 | 可选 |
| 8 | `functional_filler` | 功能填料 | 可选 |

### recipe.yaml 结构

```yaml
# 模拟条件
simulation:
  mode: aimd
  temperature_C: 60       # 摄氏度 → 自动转 K
  dt_fs: 1.0              # 时间步长 = POTIM
  nsteps: 10000           # 总步数 = NSW
  ensemble: nvt
  thermostat: langevin
  gamma_1ps: 10.0         # Langevin 摩擦系数

  # AIMD 稳定性参数
  isym: 0                 # AIMD 必须关闭对称性
  maxmix: 40              # 电荷密度混合历史，建议 40-80

  # 建模参数（非 VASP 参数）
  density_g_cm3: 1.25     # 体系密度，用于建盒子
  builder: none           # 建模工具: none/packmol
  target_atoms: 200       # AIMD 目标原子数（recipe_to_counts 可直接读取）
  allow_drop_low_fraction_components: true

# 组分（wt%）
salt_solution:
  - name: "LiTFSI（双三氟甲磺酰亚胺锂）"
    wt_pct: 15.0
    kind: salt
    mw_g_mol: 287.09
    atoms_per_entity: 15

polymer_matrix:
  - name: "PEGDA（聚乙二醇二丙烯酸酯）"
    wt_pct: 40.0
    kind: polymer
    ...

# 可选组分为空
plasticizer_solvent: []
functional_monomer: []
stabilizer: []
functional_filler: []
```

### 拆分离子建模（ion_group + stoich）

> [!IMPORTANT]
> **离子配平问题**：若将盐拆分为阳离子/阴离子（如 Li+ 和 TFSI-），必须使用 `ion_group` + `stoich` 确保整数分配时严格配平，否则可能产生非中性体系。

**方式 1：中性盐实体（推荐，简单情况）**
```yaml
salt_solution:
  - name: "LiTFSI（双三氟甲磺酰亚胺锂）"
    wt_pct: 15.0
    kind: salt
    charge: 0              # 中性，无需 ion_group
    mw_g_mol: 287.09
    atoms_per_entity: 15
```

**方式 2：拆分离子（需要独立控制阳/阴离子）**
```yaml
salt_solution:
  # 阳离子
  - name: "Li+（锂离子）"
    wt_pct: 2.4            # 按摩尔比计算的 wt%
    kind: ion
    charge: 1              # 带电 → 必须有 ion_group
    ion_group: "LiTFSI"    # 同组离子一起分配
    stoich: 1              # 化学计量数（默认 1）
    mw_g_mol: 6.94
    atoms_per_entity: 1
  
  # 阴离子
  - name: "TFSI-（双三氟甲磺酰亚胺根）"
    wt_pct: 12.6
    kind: ion
    charge: -1
    ion_group: "LiTFSI"    # 同组
    stoich: 1
    mw_g_mol: 280.15
    atoms_per_entity: 14
```

**约束分配原理**：
- 同一 `ion_group` 的成员作为整体分配单一 k 值
- 每个成员的 `scaled_count = k × stoich`
- 保证 `sum(charge_i × count_i) = 0`（组内电荷平衡）
- `v3.1` 默认 `soft_atoms` 模式不再强制“固定实体总数”，而是优化以下软目标：
  - 总原子数接近 `target_atoms`
  - 实际 wt% 接近有效目标 wt%
  - 整数 count 接近连续解 `float_count`

> [!IMPORTANT]
> `require_neutral` 开启时：
> 1. 每个 `ion_group` 必须满足 `sum(charge * stoich) == 0`（按组单位检查）  
> 2. 任何带电条目都必须放入某个 `ion_group`（否则直接报错）

**多价离子示例（如 MgCl₂）**：
```yaml
salt_solution:
  - name: "Mg2+（镁离子）"
    charge: 2
    ion_group: "MgCl2"
    stoich: 1              # 1 个 Mg2+
    ...
  - name: "Cl-（氯离子）"
    charge: -1
    ion_group: "MgCl2"
    stoich: 2              # 2 个 Cl-
    ...
```

### 命名规范

**v2.0 验证规则**（接受以下任一模式）：

| 模式 | 示例 | 说明 |
|------|------|------|
| 缩写（中文全称） | `"LiTFSI（双三氟甲磺酰亚胺锂）"` | 推荐格式 |
| 缩写(中文全称) | `"PEGDA(聚乙二醇二丙烯酸酯)"` | 半角括号也可 |
| name_en + name_cn | `name_en: "LiTFSI"`, `name_cn: "双三氟甲磺酰亚胺锂"` | 分开字段 |

> 使用 `--strict_schema` 或 `STRICT_SCHEMA=1` 可将未知类别视为错误

### 关于 density 和 target_atoms

- `density_g_cm3` 仅用于建盒子/Packmol/recipe_to_counts，非 VASP 参数
- `target_atoms` 用于验证/提示，AIMD 代表性小胞可能无法严格保持实验 wt%
- **v3.0**: `recipe_to_counts.py` 可直接从 `simulation.target_atoms` 读取，无需命令行参数
- `allow_drop_low_fraction_components: true` 允许小体系中 optional 组分 count=0

### 低 wt% 组分处理

当使用 `--allow_missing_low_wt` 跳过缺少 mw/atoms 的低 wt% 组分时：

- **v3.0 行为**：自动重归一化剩余组分 wt% 至 100%
- 报告中会显示 `skipped_wt_total` 和被跳过的组分列表
- 使用 `--no_renormalize_skipped` 可禁用重归一化
- **v3.1 报告语义修正**：误差按 `target_wt_pct_effective` 计算（不再对原始 wt% 产生系统偏差）
- 输出会同时保留 `target_wt_pct_original` 和 `target_wt_pct_effective` 以便追溯

### min_count 可行性检查（v3.1）

`recipe_to_counts.py` 在分配前会检查 `min_count` 对应的最小强制原子数：

- 若 `min_atoms_total > target_atoms * 1.05`，脚本会直接失败并给出导致不可行的条目/组
- 推荐修正路径：
  - 增大 `target_atoms`
  - 降低 `min_count`
  - 用更小聚合物表示（例如低聚体单元，而非整条长链）

---

## 从大体系切割 AIMD 子体系（ASE）

### 概述

`setup_aimd_ase.py` 用于从大体系结构（GROMACS/Packmol 输出）中切割出一个可用于 VASP AIMD 的局部量子区域。

**v2.4.1 关键点**：
- ✅ `--mic_mode {auto,on,off}` 显式控制 MIC，默认 `auto`
- ✅ `resolve_use_mic()` 负责 MIC 决策；建盒后会基于**当前** `cell/pbc` 重新解析（避免使用过期 MIC 状态）
- ✅ `--mic_mode on` 在输入 cell/pbc 无效时会直接报错退出，不再静默回退
- ✅ `model_meta.json` 记录真实的 `density_input_is_3d_valid`、`mic_mode`、`mic_used_for_selection`、`mic_cell_valid`
- ✅ `model_meta.json` 记录中和前后电荷与可靠性：`charge_before/after`、`reliable_before/after`
- ✅ cluster 输出盒子始终设置 `pbc=[False,False,False]`，并在元数据中显式记录
- ✅ bulk 盒子仍按 `V_target = M_sub / ρ_target` 构建，必要时自动扩胞避免自交叠
- ✅ `cell_shape=scale_parent` 在输入不是有效 3D 周期密度来源时会明确警告并回退到 `cubic`
- ✅ `--charge_map_file` 显式提供但加载失败时会直接报错退出（不再静默回退）
- ✅ `parse_poscar_element_order()` 兼容 VASP5/VASP4（含确定性回退路径）
- ✅ 无 `utils` 模块时，检测到切断键仍会写出 `cut_bonds_report.txt`

**物理原理**：
```
Bulk 盒子体积:
  V_target = M_sub / ρ_target
  
  其中 ρ_target 默认取原体系密度 ρ_orig
  这保证了子体系密度与原体系一致，物理合理

错误做法（旧版）:
  V = bbox + buffer  →  密度远低于实际，产生"低压气相"
```

**典型场景**：凝胶电解质体系有数千原子，AIMD 只能算几百原子，需要切割一个以目标离子为中心的小子体系。

### 基本用法

```bash
# bulk 模式（默认，周期性子胞，推荐用于凝胶电解质）
python3 setup_aimd_ase.py --src equilibrated.pdb --center_atom Li --radius 8 --mode bulk --outdir aimd_bulk

# cluster 模式（真空簇，需显式指定）
python3 setup_aimd_ase.py --src equilibrated.pdb --center_atom Li --radius 8 --mode cluster --vacuum 20 --outdir aimd_cluster

# 显式要求 MIC；若输入没有有效周期性 cell，会直接失败
python3 setup_aimd_ase.py --src equilibrated.pdb --center_atom Li --radius 8 --mic_mode on

# 保留完整分子 + 电荷中和
python3 setup_aimd_ase.py --src system.pdb --center_atom Li --radius 8 --selection molecule --neutralize nearest_counterions --outdir aimd_mol
```

> ⚠️ **重要**：默认为 bulk 模式。cluster 模式必须显式指定 `--mode cluster`。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--src` | 必填 | 输入结构文件（.pdb/.gro/.xyz/.cif） |
| `--center_atom` | 必填 | 中心原子（索引或元素符号如 Li） |
| `--mode` | bulk | 模式: bulk（周期性）/ cluster（真空簇） |
| `--radius` | 8.0 | 切割半径 Å |
| `--selection` | sphere | 选择模式: sphere / molecule |
| `--bond_hops` | 0 | 键跳扩展步数（避免切断聚合物链） |
| `--one_based` | False | 将 `--center_atom` 解释为 1-based 索引 |
| `--vacuum` | 20.0 | 真空层 Å（仅 cluster 模式） |
| `--density_g_cm3` | None | 目标密度 g/cm³（bulk 模式；未提供时尝试使用输入 3D 周期体系密度） |
| `--cell_shape` | scale_parent | 盒子形状: scale_parent / cubic |
| `--bulk_span_padding` | 4.0 | bulk 最小边界留白 Å（防止 PBC 自交叠） |
| `--bulk_density_warn_pct` | 10.0 | bulk 盒子被迫放大后密度偏差警告阈值 (%) |
| `--bulk_clash_max_expand_iter` | 5 | bulk 自动扩胞消除重叠的最大迭代次数 |
| `--neutralize` | none | 电荷中和: none / nearest_counterions |
| `--target_charge` | 0 | 中和目标电荷 |
| `--charge_map_file` | - | 自定义残基电荷映射文件 |
| `--max_atoms` | 400 | 最大原子数限制 |
| `--allow_exceed_max_atoms` | False | 允许扩展后超过 `--max_atoms` |
| `--temp` | 350.0 | AIMD 温度 K |
| `--steps` | 2000 | AIMD 步数 |
| `--potim` | 1.0 | 时间步长 fs（含 H 建议 0.5-1.0） |
| `--thermostat` | langevin | 恒温器: langevin / nose |
| `--gamma_1ps` | 10.0 | Langevin gamma (1/ps) |
| `--kpoints` | `"1 1 1"` | K 点网格字符串，必须是 3 个整数 |
| `--ncore` | None | 可选 NCORE 设置 |
| `--encut` | 400.0 | ENCUT (eV) |
| `--outdir` | aimd_sub | 输出目录 |
| `--overwrite` | False | 若输出目录已存在则先备份再覆盖 |
| `--write_relax_inputs` | False | 生成 `INCAR.relax`、`RELAX_GUIDE.txt`、`run_relax_then_aimd.sh` |
| `--clash_threshold_scale` | 0.75 | 碰撞检测阈值比例 |
| `--force_density` | False | 已弃用；等价于跳过密度异常警告 |
| `--cut_bond_policy` | heal | 切断键策略: heal / warn / error |
| `--density_check` | warn | 密度检查: strict / warn / skip |
| `--mic_mode` | auto | MIC 模式: `auto`=输入 cell 有效时启用，`on`=强制启用并校验，`off`=全流程禁用 |
| `--write_index_map` | False | 写入 `index_map.json`（original ↔ cluster 映射） |

### MIC 行为

- `--mic_mode auto`：仅当输入结构通过 `has_valid_cell_for_mic()` 检查时启用 MIC
- `--mic_mode on`：要求输入结构具备有效周期性 cell；否则会以非零状态退出，并打印检测到的 `pbc`、`cell`、失败原因和修复建议
- `--mic_mode off`：选择、重成像、中和距离排序、碰撞检测全部使用非 MIC 距离
- 建盒后会按**当前**结构重新判断 MIC：`bulk` 有效周期盒会启用 MIC；`cluster+vacuum`（`pbc=False`）会禁用 MIC

### 选择模式

- **sphere**：纯半径选择，最快
- **molecule**：半径内的原子所属分子整体保留，避免切断分子（需要结构文件包含分子信息，如 .pdb 的 residue）
- **切断键检测**：`sphere/molecule` 两种模式都会执行（不再在 sphere 提前返回）
- **重成像规则**：
  - `sphere + bond_hops=0`：仅在 `radius <= 0.45 * min(L_pbc)` 时使用中心 MIC 重成像；超过阈值直接报错
  - `molecule` 或 `bond_hops>0`：使用键图拓扑展开（BFS），保持跨边界聚合物拓扑连续

> **建议**：凝胶电解质体系使用 `--selection molecule` 更物理合理

### 输出文件

```
aimd_Li8A/
├── POSCAR               # VASP 结构文件
├── INCAR                # AIMD 参数
├── KPOINTS              # K 点（Gamma-only）
├── POTCAR               # 赝势（如 VASP_PP_PATH 已设置）
├── cluster_visual.xyz   # 可视化文件（OVITO/VMD）
├── model_meta.json      # 元数据（含 density/MIC 决策、中和轨迹、警告）
├── selected_indices.txt # 选中的原子索引
├── index_map.json       # (--write_index_map) original ↔ cluster 映射
├── INCAR.cluster_hint   # (cluster 模式) 偶极修正建议（LDIPOL/IDIPOL/DIPOL）
└── cut_bonds_report.txt # 切断键报告（若有）
```

**model_meta.json 关键字段**:
```json
{
  "pbc_mic": {
    "mic_mode": "auto",
    "mic_used_for_selection": true,
    "mic_cell_valid": true,
    "mic_decision_reason": "auto_cell_valid",
    "cluster_pbc_after_box": null
  },
  "density": {
    "original_g_cm3": 1.18,
    "target_g_cm3": 1.18,
    "achieved_g_cm3": 1.17,
    "density_input_is_3d_valid": true,
    "density_decision_reason": "3d_valid"
  },
  "neutralization": {
    "enabled": true,
    "charge_before": 1,
    "reliable_before": true,
    "charge_after": 0,
    "reliable_after": true,
    "remaining_charge": 0,
    "verified": true
  },
  "estimated_charge": 0,
  "cut_bonds": {
    "policy": "heal",
    "initial_count": 0,
    "final_count": 0
  },
  "warnings": []
}
```

### 电荷检查与中和安全

脚本会自动估算 cluster 总电荷，优先级如下：
1. 残基映射（最可靠，支持 `--charge_map_file`）
2. 连通分量公式匹配（TFSI / PF6 / BF4 / ClO4 / NO3 + 单原子离子）

安全策略：
- 对多原子分量，不做“元素求和”电荷回退（避免化学上不安全的估计）
- 若 `--neutralize nearest_counterions` 且电荷估计不可靠，脚本会直接报错并终止：
  - `Provide residue info in the input OR supply --charge_map_file OR disable --neutralize.`
- 若显式提供 `--charge_map_file` 但文件不存在/解析失败，脚本会直接报错退出（不再静默回退）
- 中和时会避免“超量添加”反离子分量；若无法精确达到目标电荷，会给出剩余电荷诊断
- 若启用中和，`model_meta.json` 会记录中和前后电荷、可靠性和剩余电荷，便于回溯

### 完整工作流

```bash
# 1. 从 GROMACS 输出切割 cluster
python3 setup_aimd_ase.py --src equilibrated.pdb --center_atom Li --radius 8 \
    --selection molecule --vacuum 20 --temp 350 --steps 2000 --outdir aimd_Li8A

# 2. 检查 cluster（可选）
# 用 OVITO 打开 aimd_Li8A/cluster_visual.xyz

# 3. 准备 POTCAR（如未自动生成）
export VASP_PP_PATH=/path/to/potentials
# 或手动拼接

# 4. 运行 AIMD
cd aimd_Li8A
NP=16 EXE=vasp_std run_vasp.sh

# 5. 监控
aimd_watch.sh

# 6. 后处理
python3 aimd_post.py
python3 aimd_msd.py --specie Li --dt_fs 2.0
```

### 注意事项

- **.gro 文件问题**：ASE 对 .gro 支持有限，建议先转换：
  ```bash
  gmx trjconv -f input.gro -o output.pdb
  ```

- **含 H 原子**：时间步长建议 0.5-1.0 fs（默认 2.0 可能过大）

- **PBC/MIC 处理**：默认 `--mic_mode auto`。选择阶段先按输入结构判断 MIC；建盒后会按当前 `cell/pbc` 重新判断（bulk 周期盒启用，cluster+vacuum 禁用）。用 `--mic_mode on/off` 可显式覆盖输入阶段策略。

- **bulk 密度来源约束**：当 `--mode bulk` 且未指定 `--density_g_cm3` 时，输入必须是有效 3D 周期体系（`pbc=[T,T,T]` 且 cell 有效）；否则会报错

- **bulk 防边界重叠**：按密度建盒后会自动检查跨度与碰撞，必要时扩胞并在 `model_meta.json` 写入警告（密度可能低于目标）

- **cluster 偶极修正**：cluster 模式会生成 `INCAR.cluster_hint`，建议对有净偶极的体系测试：
  - `LDIPOL = .TRUE.`
  - `IDIPOL = 1/2/3`（按主偶极方向）
  - `DIPOL` 用晶胞中心或质心作为初值

- **molecule 模式**：需要结构文件包含分子信息（如 .pdb 的 residue），否则回退到 sphere

---

## AIMD 工作流

### 完整流程

```bash
# 1. 准备配方
cp ~/vasp_scripts/recipe.yaml ./
vim recipe.yaml  # 修改温度、组分

# 2. 验证配方
python3 recipe_validate.py

# 3. 生成 INCAR
python3 make_incar_aimd.py --out INCAR
# 或使用一键设置
aimd_setup.sh

# 4. 运行 AIMD
NP=16 EXE=vasp_std run_vasp.sh

# 5. 监控
aimd_watch.sh
# 或
tail -f vasp.out vasp.err

# 6. 续算（如需要）
RESUME=1 NP=16 run_vasp.sh

# 7. 后处理
python3 aimd_post.py
python3 aimd_msd.py --specie Li --dt_fs 1.0 --t_skip_ps 2.0
```

### MSD 分析 (v2.3)

```bash
# 基本用法（默认 MTO）
python3 aimd_msd.py --specie Li --dt_fs 1.0

# 快速模式（stride=2 降低计算量）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --stride 2

# 旧版兼容模式（单一时间原点，仅用于对比）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --msd_method single_origin

# 指定拟合区间
python3 aimd_msd.py --specie Li --dt_fs 1.0 --t_skip_ps 2.0 --t_fit_start_ps 5.0

# 分段独立误差估计
python3 aimd_msd.py --specie Li --dt_fs 1.0 --n_blocks 4

# 2D 扩散体系（slab/平面约束）建议显式设置 d=2
python3 aimd_msd.py --specie Li --dt_fs 1.0 --diff_dim 2

# 线性 COM 漂移去除（保留涨落，推荐）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --remove_com all_linear

# 聚合物骨架参考（多物种）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --remove_com selected_linear --com_selection C,O,H

# 聚合物骨架参考（精确索引，覆盖 --com_selection）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --remove_com selected_linear --com_index_file scaffold_indices.txt

# 允许输出亚扩散时的 D（非严格模式）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --allow_unreliable_D

# 显式关闭严格模式（默认 strict=True）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --no_strict

# 关闭 unwrap 一致性检查（不推荐）
python3 aimd_msd.py --specie Li --dt_fs 1.0 --no_unwrap_check

# 运行内置轻量自检
python3 aimd_msd.py --self_check
```

**输出文件**：
- `msd_Li.dat`: MSD 数据（lag_ps, MSD_A2, n_samples）
- `D_running_Li.dat`: Running-D（由 `--runningD` 控制：`ratio`/`derivative`/`both`）
- `alpha_Li.dat`: log-log 斜率 α(t)
- `msd_report.txt`: 完整分析报告

**物理原理**：
```
MTO MSD 公式:
  MSD(τ) = ⟨|r(t₀+τ) - r(t₀)|²⟩_{t₀,ions}
  
  对每个 lag τ，平均所有时间起点 t₀ 和目标离子

α(t) 判定 (凝胶体系典型 3 阶段):
  t < ~0.5 ps:    弹道区, α ≈ 2
  ~0.5 - 50 ps:    caging/亚扩散, α < 1
  t > ~50 ps:    扩散区, α ≈ 1
  
  只有 α ≈ 1 且 D(t) 平稳时，扩散系数才可信
```

**v2.3 分析语义（重要）**：
- `stride` 下不再假设 `t = arange(N)*dt`，统一使用 `lag_frames * dt` 时间轴。
- Bootstrap 与主流程使用**同一组 lag 帧**与同一物理 `max_lag_frames`，避免 stride 下的时间轴错误；SEM 使用**有效 bootstrap 样本数**。
- Trajectory blocks 使用每个 block 的**本地可用时间窗口**做拟合窗校验，不再因全局 fit_end 导致短 block 被整体误跳过。
- Block/Bootstrap 不再静默丢弃非正 `D` 样本（避免均值上偏）；若出现非正样本会在日志中显式报告保留数量。
- XDATCAR 解析改为严格 frame 边界：遇到损坏/截断帧会**明确报错退出**，不会继续“滑动重同步”并伪造后续轨迹。
- 读取 XDATCAR 后（应用 `--skip` 之后）会自动删除**连续重复帧**（常见于 RESUME 拼接），并打印删除数量与位置。
- `remove_com=all/selected` 为每帧锚定，可能抑制真实长波涨落；更推荐 `all_linear/selected_linear` 仅去除线性漂移。
- OUTCAR 变胞检测现在同时检查**晶格长度 / 3x3 矩阵分量 / 体积 / 角度（默认启用）**，阈值由 `--cell_rel_std_tol` 控制；状态会明确区分为 `constant confirmed` / `variable detected` / `unknown`。
- `--require_valid_cell_check` 可在 cell-check 为 unknown（OUTCAR 缺失/损坏/不足）时强制失败，避免流水线误把 unknown 当 constant。
- `--t_skip_ps` 若被 `<=25%` lag-window 安全上限截断，会在终端和 `msd_report.txt` 同时记录明确 warning。
- `msd_report.txt` 现在包含重复帧去重摘要、COM 参考来源、strict 标志、cell check 消息以及最终 `SUMMARY` 机器可读行。
- Einstein 分母支持显式维度 `d`：`D = slope / (2d)`（默认 `d=3`，2D 体系可用 `--diff_dim 2`）。
- `--runningD` 与 `--plateau_method` 现在会真实控制计算/写出/判定路径，不再总是固定走 “both”。

> ⚠️ **重要 (v2.3)**：严格模式下，只有当启用了 alpha 判定（`--plateau_method alpha/both`）且检测到亚扩散（α < 0.8）时，脚本才会将 D 设为 NaN 并以退出码 2 终止。其他不可靠原因（如 drifting/α不可用）不会被误标为 subdiffusion exit(2)。

**v2.3 关键改进**：

| 功能 | v2.2 | v2.3 |
|------|------|------|
| 亚扩散防护 | 默认 exit(2) + D=NaN | **CLI 改为默认 strict + `--no_strict`** |
| NPT 检测 | 晶格长度 | **+ 3x3 矩阵 / 体积 / 角度** |
| Cell 阈值 | 固定 0.01 | **`--cell_rel_std_tol` 可调** |
| Skip 语义 | 隐式 clamp | **显式 warning + 报告留痕** |
| Traceability | 基本报告 | **重复帧/COM 来源/strict/cell/SUMMARY** |
| 自检 | 无 | **`--self_check` 内置轻量检查** |

**新增/更新参数 (v2.3)**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no_strict` | False | 关闭严格模式；默认 strict=True |
| `--allow_unreliable_D` | False | 允许输出亚扩散时的 D（有 UNRELIABLE 标签） |
| `--cell_source` | xdatcar | 晶格来源（仅支持恒定胞） |
| `--cell_rel_std_tol` | 0.01 | OUTCAR 变胞检测阈值（相对标准差） |
| `--no_cell_angle_check` | False | 不使用晶胞角度波动参与变胞检测 |
| `--min_block_time_ps` | 5.0 | Block 误差估计最小物理时间 (ps) |
| `--com_selection` | - | COM 参考物种（支持逗号分隔多物种，selected* 模式） |
| `--com_index_file` | - | COM 参考索引文件（覆盖 `--com_selection`） |
| `--msd_method` | mto | mto / single_origin |
| `--stride` | 1 | MTO lag 步进（2/5 可降低计算量） |
| `--runningD` | both | Running-D 输出路径: ratio / derivative / both |
| `--plateau_method` | both | 平台判定路径: D_derivative / alpha / both |
| `--diff_dim` | 3 | 扩散维度 d（Einstein 分母 2d） |
| `--max_lag_ps` | min(T×0.5, 10) | 智能默认，避免噪声 |
| `--unwrap_check` | 启用 | 检测 \|d\|>0.5 分数坐标跳跃 |
| `--remove_com` | all | COM 漂移去除: none/all/selected/all_linear/selected_linear |
| `--duplicate_tol` | 1e-12 | 连续重复帧判定阈值（分数坐标） |
| `--alpha_window` | 21 | α(t) 滑窗大小 |
| `--block_mode` | trajectory_blocks | trajectory_blocks/bootstrap |
| `--require_valid_cell_check` | False | 若 OUTCAR 校验不可用（unknown）则失败退出 |
| `--seed` | - | Bootstrap 随机种子 |
| `--self_check` | False | 运行内置轻量自检并退出 |

**COM 模式选择建议（凝胶聚合物电解质）**：
- `--remove_com all` / `selected`：每帧减去 COM(t)-COM(0)，会抑制体系涨落；仅建议做对比诊断。
- `--remove_com all_linear`：只去除整体线性漂移，保留热涨落与长波模式，通常比 `all` 更物理。
- `--remove_com selected_linear`：推荐用于聚合物电解质，将参考定义为聚合物/交联骨架。
- 骨架定义优先级：`--com_index_file`（精确原子索引） > `--com_selection`（物种列表，如 `C,O,H` 或 `PEGDA,PVDF`）。

**log-log 斜率 α(t) 解释**：

| α 值 | 状态 | 含义 | D 输出 |
|------|------|------|--------|
| α ≈ 1 | diffusive | 正常扩散 | ✓ 有效 |
| α < 0.8 | subdiffusive | 受限/亚扩散（caging） | **默认 NaN + exit(2)** |
| α > 1.2 | superdiffusive | 弹道/超扩散（早期或漂移） | ⚠️ 可疑 |

> ⚠️ **NPT/变胞限制**：本脚本假定 XDATCAR 晶格恒定（NVT/NVE）。若检测到 OUTCAR 显示变胞（默认检查长度/3x3 分量/体积/角度），脚本将**终止并提示**。变胞体系需预处理轨迹或使用专用工具。

### 热力学数据后处理 aimd_post.py (v2.0)

从 OSZICAR/OUTCAR 提取离子步热力学数据，并进行物理 QA 分析。

**v2.0 关键改进**：
- ✅ **OUTCAR 离子步正确解析**：使用离子步号（非电子迭代）
- ✅ **Iteration a(b) 自动判别**：扫描前 200 个匹配，按跨度 + 变化频率判断离子步；支持 `--iteration_order`
- ✅ **拼接/重启处理**：自动检测步号重置，生成单调 global_step
- ✅ **traceability 保留**：内部始终保留 `global_step`；记录 `raw_step` 和 `segment`
- ✅ **Fortran 星号溢出处理**：`************` 转为 None + 警告（严格模式报错退出）
- ✅ **能量漂移监控**：线性回归计算 meV/atom/ps，阈值警告
- ✅ **平衡期跳过**：支持 `--t_skip_ps/--t_skip_steps`，分离全程/生产段统计；跳过过多时会明确提示生产段为空
- ✅ **时间步安全**：使用 `--t_skip_ps` 时，必须能从 `--dt_fs` 或 `INCAR:POTIM` 确定时间步；不会静默回退到 1.0 fs
- ✅ **压力监控**：从 OUTCAR 提取 external pressure，可选扩展 CSV
- ✅ **轻量自检**：`--self_check` 用内置 mock OUTCAR 验证两种 Iteration 顺序

```bash
# 基本用法（后向兼容）
python3 aimd_post.py

# 跳过平衡期 2 ps，只统计生产段
python3 aimd_post.py --t_skip_ps 2.0 --dt_fs 1.0

# 严格模式（星号溢出或高漂移时退出）
python3 aimd_post.py --strict

# 扩展 CSV（含压力列）
python3 aimd_post.py --extended_csv

# 手动指定原子数（用于漂移归一化）
python3 aimd_post.py --n_atoms 200

# 手动指定 OUTCAR Iteration 顺序
python3 aimd_post.py --iteration_order electron_first

# 运行内置轻量自检
python3 aimd_post.py --self_check
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--oszicar` | OSZICAR | OSZICAR 文件路径 |
| `--outcar` | OUTCAR | OUTCAR 文件路径 |
| `--output` | aimd_thermo.csv | 输出 CSV 路径 |
| `--incar` | INCAR | 用于读取 POTIM |
| `--t_skip_steps` | 0 | 跳过的平衡步数 |
| `--t_skip_ps` | - | 跳过的平衡时间 ps；要求 `--dt_fs` 或 `INCAR:POTIM` 可用 |
| `--dt_fs` | from INCAR | 时间步长 fs；未指定且未用 `--t_skip_ps` 时才回退到 1.0 |
| `--n_atoms` | from OUTCAR | 原子数（漂移归一化用） |
| `--drift_warn_mev_atom_ps` | 1.0 | 漂移警告阈值 |
| `--drift_strict_mev_atom_ps` | 3.0 | 严格模式漂移阈值 |
| `--strict` | False | 严格模式：星号/高漂移时退出 |
| `--extended_csv` | False | 输出 `step,raw_step,segment,...`；若有压力再追加 `P_kB` |
| `--renumber` | True | 将 `global_step` 压缩为 1..N；关闭时保留拼接感知的单调 `global_step` |
| `--dedup` | keep_last | 重复步策略 |
| `--iteration_order` | auto | `Iteration a(b)` 顺序：`auto/electron_first/ionic_first` |
| `--self_check` | False | 运行内置轻量自检并退出 |

**输出格式**：

默认 CSV（后向兼容）：
```text
step,E0_eV,T_K,F_eV
1,-123.45600000,300.00,-123.45670000
2,-123.43900000,305.00,-123.44000000
```

扩展 CSV（`--extended_csv`）：
有压力数据时：
```text
step,raw_step,segment,E0_eV,T_K,F_eV,P_kB
1,1,1,-123.45600000,300.00,-123.45670000,
2,2,1,-123.43900000,305.00,-123.44000000,10.50
```

无压力数据时：
```text
step,raw_step,segment,E0_eV,T_K,F_eV
1,1,1,-123.45600000,300.00,-123.45670000
2,2,1,-123.43900000,305.00,-123.44000000
```

说明：
- `step` 始终表示内部单调的 `global_step`，用于时间轴和漂移分析
- `raw_step` 保留原始离子步号
- `segment` 标记拼接/重启段
- 终端摘要会打印一行简短 debug 信息，说明 OUTCAR `Iteration a(b)` 选择了哪种顺序

**能量漂移解释**：

| 漂移值 (meV/atom/ps) | 说明 |
|---------------------|------|
| < 1.0 | 正常（NVT 控温波动） |
| 1.0 - 3.0 | 偏高，检查 SCF 收敛/ENCUT/PREC |
| > 3.0 | 过大，可能有物理问题（严格模式报错） |

> [!NOTE]
> 平衡期建议是**仅供参考**的建议，不会自动应用。必须使用 `--t_skip_ps` 或 `--t_skip_steps` 显式指定才会生效。
> 若 `--t_skip_steps/--t_skip_ps` 覆盖了全部数据，脚本会保留全量 CSV，但生产段统计为空，漂移会报告数据不足。

### make_incar_aimd.py 功能 (v3.0.2+)

**基础功能**：
- 从 recipe.yaml 读取模拟条件
- 温度自动转换：°C → K
- 强制设置 `ISYM = 0`（AIMD 必须）
- 设置 `MAXMIX`（默认 40）
- 添加 `LASPH = .TRUE.` 和 `ADDGRID = .TRUE.`
- 支持 INCAR.base 继承

**最新关键改进（正确性/一致性）**：
- ✅ **修复恒温器映射（关键物理修复）**：
  - Langevin: `MDALGO = 3` + `LANGEVIN_GAMMA` 向量
  - Nosé-Hoover: `MDALGO = 2` + `SMASS`
- ✅ **鲁棒 POSCAR 解析**：支持 VASP4/VASP5/Selective dynamics，首行注释可为空
- ✅ **LANGEVIN_GAMMA 输出向量**：每种原子类型一个值（VASP 要求）
- ✅ **`simulation.has_h` 严格布尔归一化**：支持 `true/false/0/1`，非法值会明确报错
- ✅ **`dt_fs` 可选且 `null` 安全**：未指定、`dt_fs: null`、`dt_fs: \"null\"` 都会按“未指定”处理
- ✅ **拒绝非有限数值**：`temperature_C/dt_fs/gamma_1ps/smass/ediff/...` 中的 `NaN/Inf/-Inf` 直接报错
- ✅ **两段式 prod 预生成重启意图**：默认 `ISTART=1, ICHARG=0`（不在生成时检查 WAVECAR）
- ✅ **ALGO/LREAL/LWAVE/LCHARG 确定性优先级**：stage > global > INCAR.base > 默认值（`null` 不会截断回退链）
- ✅ **gamma 键名兼容增强**：支持 `gamma_list_eq_1ps` 和 `gamma_eq_list_1ps` 两种风格
- ✅ **高 gamma 保护**：`max(gamma) >= 50` 默认硬错误（可用 `allow_high_gamma: true` 放行）
- ✅ **INCAR.base 多参数解析**：支持 `A=1 ; B=2`，重复键后者覆盖前者
- ✅ **H 检测回退增强**：POTCAR 可识别 `H_h/H_s/H.5` 等常见变体；扫描不确定时走安全默认而非误判“无 H”
- ✅ **thermostat 别名规范化**：`nose` / `nose-hoover` / `nh` 都会规范化为 `nose_hoover`
- ✅ **两段式摘要按阶段分别打印**：EQ / PROD 会分别显示各自的 gamma、`dt_fs`、`ISTART/ICHARG`、`LWAVE/LCHARG`
- ✅ **INCAR 头部 traceability**：自动写入 `has_h` 来源、`dt_fs resolved_by`、`ISTART/ICHARG resolved_by`
- ✅ **`simulation.stages` 优先级修正**：只要定义了 `simulation.stages`，即使未传 `--two_stage` 也按 stages 生成
- ✅ **`NBLOCK/KBLOCK` 显式处理**：不再隐式继承；会写入 INCAR 与摘要/警告，避免下游时间轴误读

> [!IMPORTANT]
> **LANGEVIN_GAMMA 必须是向量**，每种原子类型（NTYP）一个值。
> 写标量常常**只热浴第一种元素**，导致 NVT 静默失效！
> v3.0.2 会自动将标量复制为长度=NTYP 的向量，或显式指定列表/字典。

### recipe.yaml simulation 段配置

```yaml
simulation:
  # === 必需参数 ===
  temperature_C: 60       # 摄氏度 → 自动转 K
  nsteps: 10000           # 总步数 = NSW
  
  # === 可选：时间步长 ===
  # 如不指定：含 H → POTIM=1.0 fs，否则 → POTIM=2.0 fs
  dt_fs: 1.0              # 时间步长 (fs)
  # dt_fs: null           # 与省略等效：自动按 has_h 规则解析
  # dt_fs: "null"         # 也会按“未指定”处理（常见占位符兼容）
  
  # === 可选：强制 H 检测结果 ===
  # has_h: true           # 覆盖自动检测（支持 true/false/0/1）
  
  # === 恒温器 ===
  thermostat: langevin    # langevin / nose_hoover / nose / nose-hoover / nh
  # 映射：
  #   langevin    -> MDALGO=3 + LANGEVIN_GAMMA
  #   nose_hoover -> MDALGO=2 + SMASS
  
  # === Langevin gamma（关键！）===
  # 方式 1: 标量（复制到所有原子类型）
  gamma_1ps: 10.0
  
  # 方式 2: 显式列表（必须与 NTYP 一致）
  # gamma_list_1ps: [5.0, 10.0, 3.0]
  
  # 方式 3: 按元素（需要 POSCAR 包含元素符号）
  # gamma_by_element_1ps:
  #   Li: 5.0
  #   C: 10.0
  #   H: 3.0
  
  # === 两段式 gamma ===
  gamma_eq_1ps: 10.0      # 平衡段
  gamma_prod_1ps: 5.0     # 生产段
  # 同样支持两种命名风格（eq/prod 均可）：
  #   旧风格：gamma_list_eq_1ps / gamma_by_element_prod_1ps
  #   新风格：gamma_eq_list_1ps / gamma_prod_by_element_1ps
  # 高阻尼保护（默认 false）：max(gamma)>=50 时硬错误
  # allow_high_gamma: true
  
  # === ISTART / ICHARG（stage-aware）===
  # 全局默认
  istart: 0
  icharg: 2
  # 平衡段覆盖
  istart_eq: 0
  icharg_eq: 2
  # 生产段覆盖（两段式预生成默认重启意图：1/0）
  istart_prod: 1
  icharg_prod: 0
  
  # === ALGO / LREAL（stage-aware，确定性优先级）===
  algo: Fast              # 全局
  lreal: Auto             # 全局（默认：natoms<=200 → False，否则 Auto）
  algo_eq: VeryFast       # 平衡段
  algo_prod: Fast         # 生产段（更安全）
  # 支持 base 继承与覆盖：stage > global > INCAR.base > 默认
  
  # === 输出文件控制（stage-aware）===
  # 优先级同上：lwave_eq/lwave_prod > lwave > INCAR.base > 默认
  # 两段式默认：eq/prod 均 LWAVE=.TRUE.（便于重启）
  lwave_eq: true
  lwave_prod: true
  # LCHARG 默认 .FALSE.（节省磁盘）
  # lcharg_prod: true
  
  # === 其他 ===
  smass: -3               # Nosé-Hoover 默认参数（保留现有策略）
  nelm: 100
  ediff: 1.0e-5
  maxmix: 40
  # nblock: 2             # 可选：显式输出步频控制（影响轨迹时间轴）
  # kblock: 1             # 可选：显式输出步频控制（影响轨迹时间轴）
  # encut: 400            # 如不指定，需在 INCAR.base 中提供
```

> [!IMPORTANT]
> 在 `simulation.stages` 模式中，阶段步数请使用 `stage.nsteps`。  
> `nsteps_eq` / `nsteps_prod` 在 stages 模式下会报错，避免阶段意图被静默忽略。

### 两段式 AIMD（平衡/生产分离）

- 若 `simulation.stages` 已定义：脚本会优先按 YAML stages 逐段生成（即使未传 `--two_stage`）
- 若同时传 `--two_stage` 且存在 `simulation.stages`：会打印提示并以 YAML stages 为准
- `simulation.stages` 必须是非空列表；`null` 或 `[]` 会直接报错（避免静默回退到单阶段）
- 自定义阶段名（非 `eq/prod`）在多阶段模式下会按顺序赋予默认 profile：
  - 第一段按 `eq` 默认
  - 其余阶段按 `prod` 默认
  - 终端和 INCAR 头部会写出 `stage_profile` 便于追溯
- 单阶段自定义 stage 不会强制套用 `eq/prod` profile 默认

```bash
# 生成两段式 INCAR
python3 make_incar_aimd.py --two_stage

# 输出：
#   INCAR.eq   - 平衡段，ISTART=0, gamma 较大
#   INCAR.prod - 生产段，默认 ISTART=1/ICHARG=0（期望使用平衡段 WAVECAR），gamma 较小
# 不会再额外生成 INCAR.aimd
# 终端会分别打印 EQ / PROD 摘要，避免把平衡段默认 gamma=10 显示成生产段默认值

# 1. 平衡段（强控温）
cp INCAR.eq INCAR
NP=16 run_vasp.sh

# 2. 切换到生产段
cp CONTCAR POSCAR
# 保留/复制平衡段产生的 WAVECAR（如需 ICHARG=1 还需 CHGCAR）
cp INCAR.prod INCAR
NP=16 run_vasp.sh

# 3. 扩散分析只用生产段
python3 aimd_msd.py --specie Li --dt_fs 1.0
```

> ⚠️ **扩散系数只能用生产段数据**，平衡段 gamma 较大会抑制动力学
>
> 若无 WAVECAR，请把 `INCAR.prod` 改为 `ISTART=0, ICHARG=2` 再运行。

### POTIM / dt_fs 语义（v3.0.2+）

- `dt_fs` 现在是**可选**的（v2.x 版本曾要求必填）
- `dt_fs: null` 与“未指定”语义相同，不会再因 `float(None)` 崩溃
- 如果未指定：
  - 检测到 H → `POTIM = 1.0 fs`（保守值）
  - 未检测到 H → `POTIM = 2.0 fs`
- 如果指定且含 H 且 > 1.5 fs → 发出警告
- H 检测优先级：YAML `has_h` > POSCAR 元素 > POTCAR 扫描
- POTCAR 扫描支持常见 H 变体（如 `H_h/H_s/H.5`）
- 若 POTCAR 扫描不确定（无法解析到可靠元素符号）或上述信息都不可用：
  - 按 `has_h = true` 处理，避免静默使用不安全的大时间步
  - 头部会写明 `source=potcar_inconclusive_safe_default` 或 `source=unknown_safe_default`
- 生成的 INCAR 头部会记录：
  - `# has_h=<bool> source=<source>`
  - `# NOTE: H 检测不确定，按 has_h=true 保守处理...`（仅安全默认路径出现）
  - `# dt_fs resolved_by=user|auto_has_h|auto_no_h`

### ISTART / ICHARG 智能默认（v3.0.2）

| 场景 | ISTART | ICHARG | 说明 |
|------|--------|--------|------|
| eq | 0 | 2 | 从头开始 |
| `--two_stage` 预生成 `INCAR.prod` | 1 | 0 | 确定性“重启意图”，不检查生成时 WAVECAR |
| 非 two_stage / 运行时智能模式 | 1 (if WAVECAR) | 0 | 检测到 WAVECAR 时重启 |
| 非 two_stage / 运行时回退 | 0 | 2 | WAVECAR 缺失时回退冷启动 |

可通过 `istart_eq`, `icharg_eq`, `istart_prod`, `icharg_prod` 覆盖。

生成的 INCAR 头部会额外写入：
- `# ISTART/ICHARG resolved_by=user_override|two_stage_default|wavecar_present|wavecar_missing_fallback|default_cold_start`

### 生成的 INCAR 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| IBRION | 0 | MD 模式 |
| NSW | nsteps | 离子步数 |
| POTIM | dt_fs | 时间步长（自动或指定） |
| TEBEG/TEEND | T_K | 温度 (K) |
| ISYM | 0 | 关闭对称性（强制） |
| MAXMIX | 40 | 电荷混合历史 |
| MDALGO | 3/2 | Langevin/Nosé-Hoover |
| LANGEVIN_GAMMA | `g1 g2 ...` | **向量**，每种原子类型一个值 |
| SMASS | 默认 `-3`（仅 Nosé-Hoover） | 保持现有默认策略；可显式覆盖 |
| ISTART | 0/1 | stage-aware |
| ICHARG | 0/2 | stage-aware |
| ALGO | VeryFast/Fast | stage-aware |
| LREAL | `natoms<=200 -> .FALSE.`，否则 `Auto` | 当前阈值策略保留 |
| LWAVE | staged 生成默认 `.TRUE.` | 便于跨阶段重启（可配置） |
| NBLOCK/KBLOCK | 默认不写；如存在则显式写入 | 会影响轨迹写出步频与时间轴解释 |
| LCHARG | .FALSE. | 不写 CHGCAR |

### 命令行参数

```bash
python3 make_incar_aimd.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--recipe` | recipe.yaml | 配方文件路径 |
| `--out` | INCAR.aimd | 输出 INCAR 路径（单阶段模式） |
| `--two_stage` | - | 未定义 `simulation.stages` 时生成默认 INCAR.eq/INCAR.prod；若已定义 stages 则仅作为兼容入口并以 stages 为准 |
| `--poscar` | POSCAR | POSCAR 文件路径（用于 NTYP 和 H 检测） |
| `--potcar` | POTCAR | POTCAR 文件路径（H 检测回退） |
| `--exe` | - | 可执行文件名（仅记录到注释） |

---

## 电子性质计算（功函数/DOS）v2.2

### 概述

支持两种电子性质计算：
- **功函数 (Work Function)**: Φ = V_vac − E_F，从 LOCPOT + OUTCAR 计算
- **DOS/PDOS**: 态密度，使用两步法（SCF → CHGCAR → NSCF）

> [!CAUTION]
> **PBE 带隙低估警告**: PBE（及类似 GGA）系统性低估带隙。基于带边的电化学
> 稳定窗口 (ESW) 可能小于实验值。如需定量设计，推荐 HSE06 或剪刀修正。
> PBE 趋势仍适用于相对比较。

**v2.2 关键改进**：
- ✅ **无平台默认硬失败**：未检测到真空平台时默认退出码 `2`，拒绝输出可疑功函数
- ✅ **显式不可靠回退**：仅 `--allow_fraction_fallback` 才允许旧式端点平均，并标记 `UNRELIABLE_FALLBACK`
- ✅ **物理化平台检测**：使用 `np.gradient(planar_avg, coords)`（单位 eV/Å）+ 连续 run-length 检测
- ✅ **平台参数可调**：`--plateau_grad_tol_evA` / `--plateau_end_fraction` / `--plateau_min_points`
- ✅ **机器可读摘要**：新增 `wf_summary.json` / `dos_summary.json`
- ✅ **DOSCAR 更鲁棒**：容忍额外空行并验证 `NEDOS` 数据完整性
- ✅ **自旋极化 DOS**：正确报告 DOS_up, DOS_down, DOS_total
- ✅ **绝缘体功函数**：标注"数值 EF 参考"，建议 VBM 参考
- ✅ **鲁棒 LOCPOT 解析**：支持 VASP4/5、选择动力学、空行

**v2.0 改进**：
- ✅ **自动 k-点选择**：根据结构类型（bulk/slab/wire/cluster）自动选择合适的 k-mesh
- ✅ **Bulk 保护**：功函数模式对 bulk 结构**硬性拒绝**，防止真空切割产生伪表面
- ✅ **真空轴自动检测**：IDIPOL 与检测到的真空方向一致，不再硬编码 z
- ✅ **ISMEAR 分离配置**：SCF 和 NSCF 步骤的 ISMEAR 可独立设置

> [!IMPORTANT]
> **Breaking Change**: `--kpts_dos` 默认值从 `"12 12 1"` 改为自动检测（bulk → `"12 12 12"`）。
> 这修复了 bulk 体系 DOS k-点维度坍缩的严重错误。

### 功函数计算

```bash
# 生成输入（slab 结构，自动添加真空层）
python3 setup_electronic.py --src CONTCAR --mode wf --vacuum 20 --ncore 8

# 运行 VASP
cd calc_electronic/wf_static
NP=16 EXE=vasp_std run_vasp.sh

# 后处理（计算功函数，绘制电势剖面）
python3 analyze_electronic.py --calcdir calc_electronic/wf_static --mode wf

# v2.2: 指定真空轴为 x
python3 analyze_electronic.py --calcdir calc_wf/wf_static --mode wf --axis x

# v2.2: 如必须使用旧式端点平均，需显式允许回退
python3 analyze_electronic.py --calcdir calc_wf/wf_static --mode wf --allow_fraction_fallback
```

**关键 INCAR 参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| LVHAR | .TRUE. | 输出 LOCPOT（静电势） |
| LDIPOL | .TRUE. | 启用偶极修正 |
| IDIPOL | 1/2/3 | **v2.0**: 自动匹配真空方向 (x/y/z) |
| ISYM | 0 | **v2.0**: 偶极修正时关闭对称性 |
| ISMEAR | 0 | Gaussian 展宽 |

**输出**:
- `vacuum_potential.dat`: 真空方向平面平均电势（含 V_vac_left/right、平台索引/坐标范围、回退标记）
- `wf_summary.json`: 机器可读摘要（`Phi`, `E_F`, `V_vac`, `method`, `plateau_found`, `fallback_used` 等）
- `wf_profile.png`: 电势剖面图
- 终端打印 E_F, V_vac, Φ（含左/右两侧）

> [!CAUTION]
> **无真空平台默认直接失败**：v2.2 默认要求平台检测成功。
> 对 bulk、无真空、真空过薄或电势未收敛的情况，`analyze_electronic.py` 会以退出码 `2`
> 拒绝输出功函数。只有显式加入 `--allow_fraction_fallback` 时才会使用端点平均，
> 且所有输出都会标记为 `UNRELIABLE_FALLBACK`。

> [!NOTE]
> **绝缘体功函数参考**: 对于绝缘体/半导体，OUTCAR 中的 E_F 是数值占据平衡点，
> 不一定物理有意义。更可靠的参考是 VBM (价带顶): Φ_VBM = V_vac - E_VBM。

### DOS 计算（两步法）

```bash
# 生成输入（SCF + NSCF 两步）
python3 setup_electronic.py --src CONTCAR --mode dos --two_step

# 步骤 1: SCF 自洽
cd calc_electronic/dos_scf
NP=16 EXE=vasp_std run_vasp.sh

# 步骤 2: 拷贝 CHGCAR
cp CHGCAR ../dos_nscf/

# 步骤 3: NSCF 计算 DOS
cd ../dos_nscf
NP=16 EXE=vasp_std run_vasp.sh

# 后处理
python3 analyze_electronic.py --calcdir calc_electronic/dos_nscf --mode dos
```

**关键 INCAR 参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| ICHARG | 11 | 从 CHGCAR 读取电荷（NSCF） |
| LORBIT | 11 | 输出 PDOS（投影态密度） |
| NEDOS | 3000 | DOS 采样点数 |
| ISMEAR | -5 (NSCF) | 四面体法（半导体/绝缘体） |
| ISMEAR | 0 (SCF) | Gaussian 展宽 |

**输出**:
- `dos_total.csv`: 能量-DOS 数据（自旋极化时含 DOS_up, DOS_down）
- `dos_summary.json`: 机器可读摘要（`E_F`, `NEDOS`, `spin_polarized`, `DOS_at_EF` 等）
- `dos.png`: DOS 图（自旋极化时上下自旋分开显示）
- 推荐使用 `sumo` 或 `p4vasp` 进行更详细的 PDOS 分析

### 结构类型自动检测

v2.0 会自动检测输入结构的维度类型：

| 类型 | 检测条件 | 推荐 k-points (DOS) | 推荐 k-points (WF) |
|------|----------|---------------------|---------------------|
| bulk | 无真空方向 | 12 12 12 | 8 8 8 |
| slab | 1 个真空方向 | 12 12 1 (z真空) | 8 8 1 |
| wire | 2 个真空方向 | 1 1 12 (z周期) | 1 1 8 |
| cluster | 3 个真空方向 | 1 1 1 (Γ-only) | 1 1 1 |

**检测原理**：计算原子跨度与 cell 长度的比值，ratio < 0.6 认为存在真空。

> [!NOTE]
> 检测是启发式的。如果自动检测不准确，可使用 `--kpts_*` 或 `--vacuum_axis` 覆盖。

### setup_electronic.py 参数

```bash
python3 setup_electronic.py --help
```

**基本参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--src` | 必填 | 输入结构文件 |
| `--mode` | 必填 | wf 或 dos |
| `--outdir` | calc_electronic | 输出目录 |

**功函数参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vacuum` | 20 | 真空层厚度 Å |
| `--vacuum_axis` | auto | **v2.0** 真空方向: auto/x/y/z |
| `--force_slab` | False | **v2.0** [危险] 强制对 bulk 应用 WF |
| `--ismear_wf` | 0 | 功函数 ISMEAR |

**K 点参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--kpts_wf` | 自动 | 功函数 K 点 |
| `--kpts_dos` | 自动 | **v2.0** DOS K 点（不再是 12 12 1） |
| `--auto_kpts` | True | **v2.0** 根据结构类型自动选择 k-点 |
| `--no_auto_kpts` | - | 禁用自动 k-点选择 |
| `--no_auto_fix` | False | **v2.0** 允许 bulk + 1D k-mesh（仅警告） |
| `--gamma` | True | Gamma-centered |

**ISMEAR/SIGMA 参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ismear_scf` | 0 | DOS SCF 步骤的 ISMEAR |
| `--ismear_nscf` | -5 | DOS NSCF 步骤的 ISMEAR |
| `--sigma_scf` | 自动 | **v2.1** SCF 的 SIGMA（ISMEAR=-5 时 0.01，否则 0.05） |
| `--sigma_nscf` | 自动 | **v2.1** NSCF 的 SIGMA |

**泛函元数据**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--functional_tag` | PBE | 泛函标签（用于输出注释） |
| `--scissor_ev` | - | 剪刀修正值 eV（仅注释） |

**通用参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ncore` | - | NCORE 并行参数 |
| `--encut` | 500 | 截断能 eV |
| `--ediff` | 1e-6 | 收敛判据 |
| `--xc` | PBE | 交换关联泛函 |
| `--two_step` | True | DOS 使用两步法 |

### analyze_electronic.py 参数 (v2.2)

```bash
python3 analyze_electronic.py --help
```

**基本参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--calcdir` | 必填 | VASP 计算目录 |
| `--mode` | 必填 | wf 或 dos |
| `--self_check` | False | 运行内建合成数据自检并退出 |

**功函数参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--axis` | z | **v2.2** 真空方向: x/y/z（支持非正交晶胞） |
| `--one_side` | False | 只使用右侧真空平台/端点窗口；默认使用两端平均 |
| `--allow_fraction_fallback` | False | **v2.2** 显式允许旧式端点平均回退，并标记 `UNRELIABLE_FALLBACK` |
| `--vac_fraction` | 0.15 | 端点平均回退时的取样比例 |
| `--no_plateau` | False | 跳过平台检测并直接使用端点平均；必须配合 `--allow_fraction_fallback` |
| `--plateau_grad_tol_evA` | 0.05 | **v2.2** 平台判据：`|dV/dx| < tol`，单位 eV/Å |
| `--plateau_end_fraction` | 0.2 | **v2.2** 两端搜索平台的比例 |
| `--plateau_min_points` | 5 | **v2.2** 平台连续最少网格点数 |

**泛函元数据**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--functional_tag` | PBE | 泛函标签（用于输出注释） |
| `--scissor_ev` | - | 剪刀修正值 eV（仅注释） |

### 示例

```bash
# 功函数：slab 结构，真空在 x 方向
python3 setup_electronic.py --src slab_x.vasp --mode wf --vacuum_axis x

# DOS：bulk 结构，自动使用 12 12 12
python3 setup_electronic.py --src bulk.vasp --mode dos

# DOS：金属体系，NSCF 使用 Gaussian 展宽
python3 setup_electronic.py --src metal.vasp --mode dos --ismear_nscf 0

# 覆盖自动 k-点（高精度 DOS）
python3 setup_electronic.py --src bulk.vasp --mode dos --no_auto_kpts --kpts_dos "16 16 16"

# 功函数：调宽平台阈值
python3 analyze_electronic.py --calcdir calc_wf/wf_static --mode wf \
  --plateau_grad_tol_evA 0.08 --plateau_min_points 8

# 功函数：明确允许不可靠回退
python3 analyze_electronic.py --calcdir calc_wf/wf_static --mode wf \
  --allow_fraction_fallback

# 内建自检
python3 analyze_electronic.py --self_check
```

### 注意事项

- **功函数需要可识别真空平台**：v2.2 默认在无平台时以退出码 `2` 失败，避免静默输出错误结果
- **端点平均已降级为显式回退**：仅 `--allow_fraction_fallback` 才会启用，且输出统一标记 `UNRELIABLE_FALLBACK`
- **k-点维度验证**：bulk 结构如果 k-mesh 有维度 = 1，会报错退出（防止物理错误）
- **VASP_PP_PATH**：需设置环境变量用于生成 POTCAR
  ```bash
  export VASP_PP_PATH=/path/to/potentials
  ```
- **DOS ISMEAR**: 
  - 半导体/绝缘体/凝胶电解质：-5（四面体法）
  - 金属：0（Gaussian）
  - v2.0 允许 SCF/NSCF 使用不同 ISMEAR

---

## 其他工具

### 检查计算状态

```bash
check_vasp.sh
```

### 清理文件

```bash
# 预览
DRYRUN=1 clean_vasp.sh

# 执行
clean_vasp.sh
```

### 配方换算 (v3.1)

```bash
# 从 YAML 读取 target_atoms（推荐）
python3 recipe_to_counts.py --recipe recipe.yaml

# 显式指定目标原子数
python3 recipe_to_counts.py --target_atoms 200

# 按总质量
python3 recipe_to_counts.py --total_mass_g 1.0 --scale_to_atoms 5000

# 显式切换旧版固定实体总数凑整（复现实验）
python3 recipe_to_counts.py --target_atoms 200 --rounding_mode legacy_total

# 允许跳过低 wt% 缺失字段的组分（自动重归一化）
python3 recipe_to_counts.py --target_atoms 200 --allow_missing_low_wt 0.5

# 严格模式（不允许带电体系）
python3 recipe_to_counts.py --target_atoms 200 --require_neutral
```

**v3.1 新特性**:
- 默认 `--rounding_mode soft_atoms`（移除“固定实体总数”硬约束）
- 保留 `--rounding_mode legacy_total` 兼容旧版结果复现
- 报告误差以有效 wt% 为基准，并同时输出 original/effective 两套目标
- 增加 `min_count` 可行性前置检查（防止小盒子被硬约束挤爆）
- `ion_group` + `stoich` 约束分配，保证拆分离子严格配平
- 从 `simulation.target_atoms` 读取默认值，无需 CLI 参数
- 默认 `--wt_tol 1e-3`（更严格）
- 跳过组分后自动重归一化 wt%

---

## 常见问题

**Q: oneAPI 加载失败？**
```bash
# 查看日志
cat /tmp/oneapi_setvars_$(id -u).log
```

**Q: 核数超限警告？**
```bash
# 自动下调（默认）
NP=32 run_vasp.sh

# 严格模式（报错退出）
STRICT_NP=1 NP=32 run_vasp.sh
```
说明：`RESERVE_CORES` 仅在 WSL 下生效；非 WSL 仅按 `nproc` 上限检查。

**Q: 如何续算 AIMD？**
```bash
# v2.0 推荐方式（自动保留重启文件和轨迹）
RESUME=1 NP=16 run_vasp.sh
```

**Q: 续算时 WAVECAR/XDATCAR 被移走了？（v2.0 前的行为）**

v2.0 起，`RESUME=1` 时默认保留 WAVECAR/CHGCAR/XDATCAR，不再移到 `old/`。如需恢复旧行为：
```bash
RESUME=1 KEEP_RESTART=0 KEEP_TRAJECTORY=0 run_vasp.sh
```

**Q: 报错 "ISTART=1 需要 WAVECAR"？**

INCAR 指定从波函数重启但 WAVECAR 不存在。两种解决方式：
```bash
# 方式 1: 修改 INCAR 为 ISTART=0
sed -i 's/ISTART.*=.*/ISTART = 0/' INCAR

# 方式 2: 强制忽略检查（从头计算波函数）
FORCE_RESTART=1 run_vasp.sh
```

**Q: aimd_msd.py 报错"轨迹不连续"？**

续算时 XDATCAR 可能被分割或出现重复 frame 0。`RESUME=1` 下脚本会自动：
- 对比 `XDATCAR.prev` 最后一帧与新 XDATCAR 第一帧
- 重复时跳过新文件第一帧，避免零速度假段
- 解析失败时在 `run.log` 写 `WARNING` 并回退到 header-skip 合并
- 仅在本轮被接受（`mpirun rc=0` + 本轮 OUTCAR 完成标志）时才合并
- 本轮失败/未完成会跳过合并，避免污染旧轨迹

必要时可手动合并：
```bash
# 跳过第二个文件的头部（通常 7-8 行）
cat XDATCAR.part1 > XDATCAR
tail -n +8 XDATCAR.part2 >> XDATCAR
```

**Q: HPC 节点卡死/过载？**

可能是 MKL/OpenMP 线程爆炸。检查是否禁用了线程保护：
```bash
# 确保 THREAD_GUARD=1（默认）
THREAD_GUARD=1 NP=64 run_vasp.sh

# 检查运行时线程配置
grep -E "(OMP|MKL)_NUM_THREADS" run.log
```

在 WSL 上，默认不强制 `OMP_PROC_BIND/OMP_PLACES`。如需恢复旧绑核行为：
```bash
FORCE_OMP_BIND=1 THREAD_GUARD=1 NP=64 run_vasp.sh
```

**Q: RESUME 时 OUTCAR/OSZICAR 太大怎么办？**

默认会按 `LOG_ROTATE_GB`（默认 2 GB）自动轮转到 `old/`，并写入 `run.log`。如需持续追加：
```bash
RESUME=1 KEEP_APPEND_LOGS=1 run_vasp.sh
```

**Q: 磁盘空间不足？**
```bash
# 强制继续
FORCE_DISK=1 run_vasp.sh

# 清理旧文件
clean_vasp.sh
```

**Q: ISYM 为什么强制为 0？**

AIMD 必须关闭对称性（ISYM=0），否则可能导致错误的轨迹或崩溃。脚本会强制覆盖 INCAR.base 中的 ISYM 设置。

**Q: MAXMIX 是什么？**

MAXMIX 控制电荷密度混合的历史长度，AIMD 中建议 40-80，有助于电子步收敛。

---

## 目录结构

```
~/vasp_scripts/
├── vasp_env.sh            # 环境配置（含自检）
├── run_vasp.sh            # 运行脚本（含续算/核数检查）
├── check_vasp.sh          # 状态检查
├── aimd_watch.sh          # AIMD 监控
├── aimd_msd.py            # MSD 分析
├── aimd_post.py           # 热力学后处理
├── clean_vasp.sh          # 文件清理
├── recipe.yaml            # 配方示例
├── recipe_validate.py     # 配方验证
├── recipe_to_counts.py    # 配方换算
├── make_incar_aimd.py     # AIMD INCAR 生成器
├── aimd_setup.sh          # AIMD 一键设置
├── setup_electronic.py    # 电子性质输入生成
├── analyze_electronic.py  # 电子性质后处理
├── setup_aimd_ase.py      # 大体系切割 AIMD cluster
└── README.md              # 本文档
```

## 版本信息

- VASP: 6.4.3
- Intel oneAPI: setvars.sh
- Python: 3.x (需要 numpy, pyyaml)
