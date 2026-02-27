#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aimd_msd.py v2.3 - 从 XDATCAR 计算 MSD 并拟合扩散系数（MTO + 统计误差 + α 判定）

================================================================================
Change Summary (v2.3):
================================================================================
1. [CRITICAL] Variable-cell detection is now stricter and more traceable
   - Checks full 3x3 cell matrix variability, volume variability, and angles
   - New CLI: --cell_rel_std_tol and optional --no_cell_angle_check
   - Cell-check result is always written to msd_report.txt

2. [HIGH] Fit/skip window semantics are explicit for automation
   - Warns when --t_skip_ps is clamped by the <=25% lag-window safety cap
   - Same note is written into msd_report.txt for reproducible pipeline logs

3. [HIGH] Report traceability for automated pipelines
   - Records duplicate-frame removal summary, COM reference source, strict flags
   - Adds a final machine-readable SUMMARY line to console + msd_report.txt

4. [MEDIUM] CLI cleanup and built-in self-check
   - Default strict=True, public opt-out is now --no_strict
   - New --self_check validates ps_to_index(), cell parsing, and a tiny end-to-end run

================================================================================
Change Summary (v2.2):
================================================================================
1. [CRITICAL] Stride semantics fix - consistent time axis handling
   - New helpers: build_time_axis_and_lags(), ps_to_index()
   - All ps→index conversions now use searchsorted on actual t_ps array
   - Bootstrap uses same lag sampling as main fit
   
2. [CRITICAL] Subdiffusion guardrail - protect pipelines from invalid D
   - --strict (default True): exit code 2 if α<0.8, D set to NaN
   - --allow_unreliable_D: override to output D with UNRELIABLE label
   
3. [CRITICAL] NPT/Variable cell detection
   - Detects variable cell from OUTCAR and aborts with clear message
   - Prevents silent errors from constant-cell assumption violation
   
4. [HIGH] Physical minimum block time for error estimation
   - --min_block_time_ps (default 5.0 ps) replaces frame-based check
   - Returns NaN (not 0.0) if blocks too short

5. [MEDIUM] COM removal semantics clarification
   - --com_selection for explicit COM species when remove_com='selected'
   - Warning if target species used as own COM reference

================================================================================
Change Summary (v2.1):
================================================================================
1. [CRITICAL] Multiple Time Origins (MTO) - default enabled, reduces noise
   - MSD(τ) = ⟨|r(t₀+τ) - r(t₀)|²⟩ averaged over all t₀ and selected ions
   - CLI: --time_origin {single,multi}, default=multi
   
2. [CRITICAL] Correct error estimation - independent trajectory blocks
   - Split trajectory into n_blocks NON-OVERLAPPING segments
   - Each block: independent MTO MSD → fit D_block
   - Report: D_mean ± STD, ± SEM
   - This replaces the WRONG "time-window slope std" approach
   
3. [NEW] log-log slope α(t) = d log(MSD) / d log(t)
   - α ≈ 1: diffusive (Fickian)
   - α < 1: subdiffusive / caging / network confinement
   - α > 1: ballistic / superdiffusive / drift / early-time
   
4. [IMPROVED] Unwrap robustness
   - Detects |d_frac| > 0.5 jumps (PBC crossing)
   - Warns on suspicious Cartesian jumps > threshold
   - Reports frame range, max jump, possible causes
   
5. [NEW] COM drift removal (default: all atoms)
   - Prevents global drift from inflating MSD
   
6. [NEW] Both D_ratio and D_derivative
   - D_ratio = MSD/(6t)
   - D_deriv = (1/6) d(MSD)/dt (more reliable for plateau)

================================================================================

用法:
    # 默认 MTO 模式（推荐）
    python3 aimd_msd.py --specie Li --dt_fs 1.0
    
    # 指定时间原点数和最大 lag
    python3 aimd_msd.py --specie Li --dt_fs 1.0 --n_origins 30 --max_lag_ps 15.0
    
    # 旧版兼容（单一时间原点，仅用于对比）
    python3 aimd_msd.py --specie Li --dt_fs 1.0 --time_origin single
    
    # 分段独立轨迹误差估计
    python3 aimd_msd.py --specie Li --dt_fs 1.0 --n_blocks 4
    
    # Bootstrap 误差估计
    python3 aimd_msd.py --specie Li --dt_fs 1.0 --block_mode bootstrap --seed 42

输出:
    - msd_<specie>.dat: t_ps (ps), MSD_A2 (Å²), n_samples
    - D_running_<specie>.dat: t_ps (ps), D_ratio (cm²/s), D_deriv (cm²/s)
    - alpha_<specie>.dat: t_ps (ps), alpha (dimensionless)
    - msd_<specie>.png: MSD 曲线图
    - D_running_<specie>.png: Running-D 曲线图（ratio + derivative）
    - alpha_<specie>.png: log-log 斜率 α(t) 图
    - msd_report.txt: 完整分析报告

物理说明:
    - MTO: 对每个 lag τ，平均所有起点 t0 的 |r(t0+τ)-r(t0)|²
    - 误差估计: 将轨迹切成 n_blocks 段，每段独立做 MTO → D_i，再统计
    - α(t) = d log(MSD) / d log(t): α≈1 正常扩散，α<1 亚扩散，α>1 弹道
    - 只有 α≈1 且 D(t) 平稳时，扩散系数才可信
    
注意:
    - 本脚本假定 XDATCAR 晶格恒定
    - NPT/变胞体系请先预处理为 unwrapped Cartesian 轨迹

依赖:
    pip install numpy matplotlib (matplotlib 可选)

作者: STAR0418-ABC
版本: v2.3
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Tuple, Optional, Dict, Any

try:
    import numpy as np
except ImportError:
    print("[ERROR] 需要 numpy 库: pip install numpy")
    sys.exit(1)

# 可选 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

VERSION = "v2.3"

# ==============================================================================
# 共享辅助函数 (v2.3)
# ==============================================================================

def build_time_axis_and_lags(
    max_lag_frames: int,
    stride: int,
    dt_ps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build consistent time axis and lag indices for strided MSD.
    
    Args:
        max_lag_frames: Maximum lag in frames (physical)
        stride: Stride for downsampling
        dt_ps: Time step in ps
    
    Returns:
        stride_lags: Array of lag indices to use (0, stride, 2*stride, ...)
        t_ps: Corresponding time array in ps
    """
    stride_lags = np.arange(0, max_lag_frames + 1, stride)
    t_ps = stride_lags * dt_ps
    return stride_lags, t_ps


def ps_to_index(
    t_target_ps: float,
    t_ps_array: np.ndarray,
    side: str = 'left'
) -> int:
    """
    Convert time in ps to index using searchsorted on actual time array.
    
    This ensures consistent ps→index conversion regardless of stride.
    
    Args:
        t_target_ps: Target time in ps
        t_ps_array: The actual time array (may be strided)
        side: search side semantics:
            - 'left': first index with t >= target
            - 'right': first index with t > target (exclusive boundary)
    
    Returns:
        Index into t_ps_array.
        For side='left': returns [0, len(t_ps_array)-1]
        For side='right': returns [0, len(t_ps_array)] (exclusive upper bound allowed)
    """
    if len(t_ps_array) == 0:
        return 0
    if side not in ('left', 'right'):
        raise ValueError(f"Invalid side='{side}', expected 'left' or 'right'")

    idx = int(np.searchsorted(t_ps_array, t_target_ps, side=side))
    if side == 'left':
        return min(max(0, idx), len(t_ps_array) - 1)
    return min(max(0, idx), len(t_ps_array))


def extract_outcar_lattice_snapshots(outcar_path: str) -> List[np.ndarray]:
    """Extract 3x3 direct lattice matrices from OUTCAR."""
    float_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
    snapshots: List[np.ndarray] = []
    pending_vectors = 0
    vectors: List[List[float]] = []

    with open(outcar_path, 'r') as f:
        for line in f:
            if pending_vectors > 0:
                values = float_pattern.findall(line)
                if len(values) >= 3:
                    vectors.append([float(values[0]), float(values[1]), float(values[2])])
                    pending_vectors -= 1
                    if pending_vectors == 0 and len(vectors) == 3:
                        snapshots.append(np.array(vectors, dtype=float))
                    continue
                pending_vectors = 0
                vectors = []

            if 'direct lattice vectors' in line.lower():
                pending_vectors = 3
                vectors = []

    return snapshots


def compute_cell_angles_deg(cell_matrix: np.ndarray) -> np.ndarray:
    """Return cell angles [alpha, beta, gamma] in degrees."""
    a_vec, b_vec, c_vec = cell_matrix

    def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom <= 0.0:
            return float('nan')
        cos_theta = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    return np.array([
        angle_deg(b_vec, c_vec),
        angle_deg(a_vec, c_vec),
        angle_deg(a_vec, b_vec)
    ], dtype=float)


def analyze_cell_variability(
    outcar_path: str,
    tolerance: float = 0.01,
    check_angles: bool = True
) -> Dict[str, Any]:
    """
    Analyze OUTCAR cell variability for constant-cell compatibility checks.

    Metrics:
        - a/b/c length relative std
        - 3x3 direct lattice component std relative to a characteristic cell length
        - cell volume relative std
        - optional cell-angle relative std
    """
    details: Dict[str, Any] = {
        'outcar_path': outcar_path,
        'tolerance': float(tolerance),
        'check_angles': bool(check_angles),
        'found_outcar': os.path.isfile(outcar_path),
        'n_snapshots': 0,
        'is_variable': False,
        'max_rel_std': 0.0,
        'trigger_metrics': [],
        'message': "",
        'length_rel_std_max': 0.0,
        'matrix_rel_std_max': 0.0,
        'volume_rel_std': 0.0,
        'angle_rel_std_max': None,
    }

    if not details['found_outcar']:
        details['message'] = (
            f"OUTCAR not found, assuming constant cell "
            f"(cell_rel_std_tol={tolerance:.4f})"
        )
        return details

    try:
        cell_matrices = extract_outcar_lattice_snapshots(outcar_path)
        details['n_snapshots'] = len(cell_matrices)
        if len(cell_matrices) < 2:
            details['message'] = (
                f"Insufficient lattice data in OUTCAR "
                f"(snapshots={len(cell_matrices)}, cell_rel_std_tol={tolerance:.4f})"
            )
            return details

        matrices = np.asarray(cell_matrices, dtype=float)
        cell_lengths = np.linalg.norm(matrices, axis=2)
        length_means = np.mean(cell_lengths, axis=0)
        length_stds = np.std(cell_lengths, axis=0)
        length_rel_stds = length_stds / np.maximum(length_means, 1e-12)
        length_rel_std_max = float(np.max(length_rel_stds))

        characteristic_length = float(max(np.max(length_means), 1e-12))
        matrix_component_stds = np.std(matrices, axis=0)
        matrix_rel_std_max = float(np.max(matrix_component_stds / characteristic_length))

        volumes = np.abs(np.linalg.det(matrices))
        volume_rel_std = float(
            np.std(volumes) / max(abs(float(np.mean(volumes))), 1e-12)
        )

        angle_rel_std_max: Optional[float] = None
        angle_metrics: List[Tuple[str, float]] = []
        if check_angles:
            angles = np.asarray([compute_cell_angles_deg(m) for m in matrices], dtype=float)
            if np.all(np.isfinite(angles)):
                angle_means = np.mean(angles, axis=0)
                angle_stds = np.std(angles, axis=0)
                angle_rel_stds = angle_stds / np.maximum(np.abs(angle_means), 1e-12)
                angle_rel_std_max = float(np.max(angle_rel_stds))
                angle_metrics.append(("angles", angle_rel_std_max))

        metrics = [
            ("lengths", length_rel_std_max),
            ("matrix", matrix_rel_std_max),
            ("volume", volume_rel_std),
        ] + angle_metrics
        max_metric_name, max_rel_std = max(metrics, key=lambda item: item[1])
        trigger_metrics = [name for name, value in metrics if value > tolerance]
        is_variable = len(trigger_metrics) > 0

        summary_terms = [
            f"lengths={length_rel_std_max:.4f}",
            f"matrix={matrix_rel_std_max:.4f}",
            f"volume={volume_rel_std:.4f}",
        ]
        if angle_rel_std_max is not None:
            summary_terms.append(f"angles={angle_rel_std_max:.4f}")

        relation = ">" if is_variable else "<="
        trigger_text = (
            f"; trigger={','.join(trigger_metrics)}"
            if trigger_metrics else
            f"; dominant={max_metric_name}"
        )
        status = "Variable cell detected" if is_variable else "Constant cell"
        details.update({
            'is_variable': is_variable,
            'max_rel_std': float(max_rel_std),
            'trigger_metrics': trigger_metrics,
            'length_rel_std_max': length_rel_std_max,
            'matrix_rel_std_max': matrix_rel_std_max,
            'volume_rel_std': volume_rel_std,
            'angle_rel_std_max': angle_rel_std_max,
            'message': (
                f"{status}: max_rel_std={max_rel_std:.4f} {relation} {tolerance:.4f} "
                f"({', '.join(summary_terms)}; snapshots={len(cell_matrices)}{trigger_text})"
            ),
        })
        return details

    except Exception as exc:
        details['message'] = f"Error parsing OUTCAR: {exc}"
        return details


def detect_variable_cell(
    outcar_path: str,
    tolerance: float = 0.01,
    check_angles: bool = True
) -> Tuple[bool, float, str]:
    """
    Backward-compatible variable-cell detector.

    Returns:
        is_variable, max_rel_std, message
    """
    details = analyze_cell_variability(
        outcar_path,
        tolerance=tolerance,
        check_angles=check_angles
    )
    return details['is_variable'], details['max_rel_std'], details['message']


def format_positions_sample(positions: List[int], limit: int = 20) -> str:
    """Format a compact sample of integer positions for logs/reports."""
    if len(positions) == 0:
        return "none"
    preview = ", ".join(str(p) for p in positions[:limit])
    if len(positions) > limit:
        return f"{preview} ..."
    return preview


def format_summary_line(
    status: str,
    verdict: str,
    D_mean: float,
    D_std: float,
    D_sem: float,
    alpha_mean: float,
    alpha_std: float,
    drift_ratio: float,
    fit_start_ps: float,
    fit_end_ps: float,
    skip_ps_used: float,
    stride: int,
    max_lag_ps: float,
    strict: bool,
    allow_unreliable_D: bool,
    cell_is_variable: bool,
    cell_rel_std: float,
    cell_rel_std_tol: float,
    duplicate_frames_removed: int,
    reason: Optional[str] = None
) -> str:
    """Build a single machine-readable SUMMARY line for pipeline logs/reports."""
    fields = [
        f"status={status}",
        f"verdict={verdict}",
        f"D_cm2_s={D_mean:.6e}" if np.isfinite(D_mean) else "D_cm2_s=nan",
        f"D_std_cm2_s={D_std:.6e}" if np.isfinite(D_std) else "D_std_cm2_s=nan",
        f"D_sem_cm2_s={D_sem:.6e}" if np.isfinite(D_sem) else "D_sem_cm2_s=nan",
        f"alpha_mean={alpha_mean:.6f}" if np.isfinite(alpha_mean) else "alpha_mean=nan",
        f"alpha_std={alpha_std:.6f}" if np.isfinite(alpha_std) else "alpha_std=nan",
        f"drift_pct={drift_ratio * 100.0:.3f}",
        f"fit_start_ps={fit_start_ps:.6f}",
        f"fit_end_ps={fit_end_ps:.6f}",
        f"skip_ps={skip_ps_used:.6f}",
        f"stride={stride}",
        f"max_lag_ps={max_lag_ps:.6f}",
        f"strict={int(strict)}",
        f"allow_unreliable_D={int(allow_unreliable_D)}",
        f"cell_variable={int(cell_is_variable)}",
        f"cell_max_rel_std={cell_rel_std:.6f}",
        f"cell_rel_std_tol={cell_rel_std_tol:.6f}",
        f"duplicate_frames_removed={duplicate_frames_removed}",
    ]
    if reason is not None:
        fields.append(f"reason={reason}")
    return "SUMMARY " + " ".join(fields)


# ==============================================================================
# 轨迹解析
# ==============================================================================

def parse_xdatcar(filepath: str) -> Tuple[np.ndarray, List[str], List[int], List[np.ndarray]]:
    """
    解析 XDATCAR 文件
    
    返回: (lattice, species, counts, frames)
        - lattice: (3,3) 晶格矢量
        - species: 元素符号列表
        - counts: 每种元素原子数
        - frames: 分数坐标帧列表
    """
    if not os.path.isfile(filepath):
        print(f"[ERROR] 文件不存在: {filepath}")
        sys.exit(1)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 8:
        print(f"[ERROR] XDATCAR 文件格式错误: {filepath}")
        sys.exit(1)

    scale = float(lines[1].strip())
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    natoms = sum(counts)

    frames = []
    idx = 7

    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("Direct") or line.startswith("direct") or "configuration" in line.lower():
            idx += 1
            continue

        if idx + natoms > len(lines):
            break

        frame = np.zeros((natoms, 3))
        valid_frame = True
        for i in range(natoms):
            try:
                coords = lines[idx + i].split()[:3]
                frame[i] = [float(c) for c in coords]
            except (ValueError, IndexError):
                valid_frame = False
                break

        if valid_frame:
            frames.append(frame)
            idx += natoms
        else:
            idx += 1

    if len(frames) == 0:
        print(f"[ERROR] 未能从 XDATCAR 读取任何帧")
        sys.exit(1)

    return lattice, species, counts, frames


def get_species_indices(species: List[str], counts: List[int], target: str) -> np.ndarray:
    """获取目标物种的原子索引"""
    indices = []
    offset = 0
    found = False

    for sp, cnt in zip(species, counts):
        if sp == target:
            indices.extend(range(offset, offset + cnt))
            found = True
        offset += cnt

    if not found:
        print(f"[ERROR] 未找到物种: {target}")
        print(f"[INFO] 可用物种: {species}")
        sys.exit(1)

    return np.array(indices)


def get_species_indices_multi(
    species: List[str],
    counts: List[int],
    selection_text: str
) -> np.ndarray:
    """
    Parse comma-separated species names and return merged atom indices.

    Example:
        "C,O,H" -> indices of species C + O + H
    """
    tokens = [tok.strip() for tok in selection_text.split(',') if tok.strip()]
    if len(tokens) == 0:
        print("[ERROR] --com_selection 为空")
        sys.exit(1)

    merged: List[np.ndarray] = []
    for token in tokens:
        merged.append(get_species_indices(species, counts, token))

    return np.unique(np.concatenate(merged)).astype(int)


def read_index_file(index_file: str, natoms: int) -> np.ndarray:
    """
    Read atom indices from text file.

    Supports whitespace/comma-separated integers and inline comments (#...).
    If all indices are >=1 and no 0 exists, assumes 1-based indices and converts.
    """
    if not os.path.isfile(index_file):
        print(f"[ERROR] COM index file not found: {index_file}")
        sys.exit(1)

    values: List[int] = []
    with open(index_file, 'r') as f:
        for line in f:
            line_clean = line.split('#', 1)[0].strip()
            if not line_clean:
                continue
            for token in re.split(r"[\s,]+", line_clean):
                if token:
                    try:
                        values.append(int(token))
                    except ValueError:
                        print(f"[ERROR] Invalid index token '{token}' in {index_file}")
                        sys.exit(1)

    if len(values) == 0:
        print(f"[ERROR] No indices found in {index_file}")
        sys.exit(1)

    idx = np.array(values, dtype=int)
    if np.any(idx < 0):
        print(f"[ERROR] Negative indices are not allowed in {index_file}")
        sys.exit(1)

    # Support common 1-based index files.
    if np.min(idx) >= 1 and not np.any(idx == 0):
        idx = idx - 1
        print(f"    [INFO] COM index file interpreted as 1-based and converted to 0-based")

    if np.any(idx >= natoms):
        bad_max = int(np.max(idx))
        print(f"[ERROR] Index {bad_max} out of range for natoms={natoms}")
        sys.exit(1)

    return np.unique(idx)


def remove_consecutive_duplicate_frames(
    frames: List[np.ndarray],
    atol: float = 1e-12
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Remove consecutive duplicate frames in fractional coordinates (O(N)).

    Returns:
        cleaned_frames
        removed_positions: frame positions (0-based in current sliced trajectory)
    """
    if len(frames) <= 1:
        return frames, []

    cleaned = [frames[0]]
    removed_positions: List[int] = []

    for i in range(1, len(frames)):
        if np.allclose(frames[i], cleaned[-1], rtol=0.0, atol=atol):
            removed_positions.append(i)
            continue
        cleaned.append(frames[i])

    return cleaned, removed_positions


# ==============================================================================
# 轨迹预处理
# ==============================================================================

def unwrap_trajectory_robust(
    frames: List[np.ndarray],
    lattice: np.ndarray,
    jump_threshold_A: float = 5.0,
    check_frac_consistency: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    稳健的分数坐标 unwrapping，带跳变检测和一致性检查
    
    Args:
        frames: 分数坐标帧列表
        lattice: 晶格矢量
        jump_threshold_A: 跳变检测阈值 (Å)
        check_frac_consistency: 是否检查分数坐标一致性 (|d|>0.5)
    
    Returns:
        unwrapped: (nframes, natoms, 3) unwrapped 分数坐标
        diagnostics: 诊断信息字典
    """
    nframes = len(frames)
    natoms = frames[0].shape[0]
    
    # 计算晶格参数用于自适应阈值
    cell_lengths = np.linalg.norm(lattice, axis=1)
    auto_threshold = 0.5 * np.min(cell_lengths)
    effective_threshold = min(jump_threshold_A, auto_threshold)
    
    unwrapped = np.zeros((nframes, natoms, 3))
    unwrapped[0] = frames[0].copy()
    
    # 跳变检测
    suspicious_jumps = []
    max_jump_A = 0.0
    max_jump_frame = 0
    
    # 分数坐标一致性检查
    frac_consistency_issues = 0
    frac_issue_frames = []
    
    for t in range(1, nframes):
        d_frac = frames[t] - frames[t - 1]
        
        # 检查分数坐标一致性：|d|>0.5 意味着发生了 PBC 跳跃
        if check_frac_consistency:
            large_frac_jumps = np.abs(d_frac) > 0.5
            if np.any(large_frac_jumps):
                n_issues = np.sum(large_frac_jumps)
                frac_consistency_issues += n_issues
                if len(frac_issue_frames) < 10:
                    frac_issue_frames.append({
                        'frame': t,
                        'n_components': int(n_issues),
                        'max_frac_jump': float(np.max(np.abs(d_frac)))
                    })
        
        d_frac -= np.round(d_frac)  # 最小镜像
        unwrapped[t] = unwrapped[t - 1] + d_frac
        
        # 转换为笛卡尔坐标检查跳变
        d_cart = np.dot(d_frac, lattice)
        jump_distances = np.linalg.norm(d_cart, axis=1)
        max_jump_this_frame = np.max(jump_distances)
        
        if max_jump_this_frame > effective_threshold:
            atom_idx = np.argmax(jump_distances)
            suspicious_jumps.append({
                'frame': t,
                'atom': int(atom_idx),
                'distance_A': float(max_jump_this_frame)
            })
        
        if max_jump_this_frame > max_jump_A:
            max_jump_A = max_jump_this_frame
            max_jump_frame = t
    
    # 检测是否为 cluster/vacuum 体系（可能不需要 unwrap）
    is_vacuum_system = False
    if len(suspicious_jumps) == 0 and frac_consistency_issues == 0:
        # 检查原子是否都在盒子中央（典型 cluster 特征）
        all_frac = np.array([f for f in frames])
        frac_range = all_frac.max(axis=(0, 1)) - all_frac.min(axis=(0, 1))
        if np.all(frac_range < 0.5):
            is_vacuum_system = True
    
    diagnostics = {
        'n_suspicious_jumps': len(suspicious_jumps),
        'suspicious_jumps': suspicious_jumps[:10],
        'max_jump_A': float(max_jump_A),
        'max_jump_frame': int(max_jump_frame),
        'effective_threshold_A': float(effective_threshold),
        'cell_lengths': cell_lengths.tolist(),
        'has_warnings': len(suspicious_jumps) > 0,
        # 新增一致性检查结果
        'frac_consistency_issues': frac_consistency_issues,
        'frac_issue_frames': frac_issue_frames,
        'is_vacuum_system': is_vacuum_system,
    }
    
    return unwrapped, diagnostics


def compute_com_trajectory(
    unwrapped_frac: np.ndarray,
    lattice: np.ndarray,
    indices: Optional[np.ndarray] = None,
    masses: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算质心 (COM) 轨迹
    
    Args:
        unwrapped_frac: (nframes, natoms, 3) unwrapped 分数坐标
        lattice: 晶格矢量
        indices: 用于 COM 计算的原子索引（None = 全部）
        masses: 原子质量（None = 等权重）
    
    Returns:
        com_cart: (nframes, 3) COM 笛卡尔坐标
    """
    nframes = unwrapped_frac.shape[0]
    
    if indices is None:
        traj = unwrapped_frac
    else:
        traj = unwrapped_frac[:, indices, :]
    
    # 转换为笛卡尔坐标
    cart = np.tensordot(traj, lattice, axes=([2], [0]))  # (nframes, natoms, 3)
    
    if masses is None:
        # 等权重平均
        com_cart = np.mean(cart, axis=1)
    else:
        if indices is not None:
            masses = masses[indices]
        total_mass = np.sum(masses)
        com_cart = np.sum(cart * masses[np.newaxis, :, np.newaxis], axis=1) / total_mass
    
    return com_cart


def remove_com_drift(
    unwrapped_frac: np.ndarray,
    lattice: np.ndarray,
    mode: str = 'all',
    selected_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    去除 COM 漂移
    
    Args:
        unwrapped_frac: (nframes, natoms, 3) unwrapped 分数坐标
        lattice: 晶格矢量
        mode: 'none', 'all', 'selected', 'all_linear', 'selected_linear'
        selected_indices: mode='selected' 时使用
    
    Returns:
        corrected_frac: 修正后的分数坐标
        com_drift: COM 漂移量 (Å)
    """
    if mode == 'none':
        return unwrapped_frac.copy(), np.zeros(3)
    
    if mode in ('selected', 'selected_linear') and selected_indices is not None:
        com_indices = selected_indices
    else:
        com_indices = None  # 使用全部原子
    
    com_cart = compute_com_trajectory(unwrapped_frac, lattice, com_indices)
    
    # Per-frame anchoring: subtract instantaneous COM(t)-COM(0)
    if mode in ('all', 'selected'):
        com_cart_shift = com_cart - com_cart[0]
    elif mode in ('all_linear', 'selected_linear'):
        # Linear drift removal only: keep COM fluctuations around linear trend.
        t = np.arange(com_cart.shape[0], dtype=float)
        com_cart_shift = np.zeros_like(com_cart)
        for dim in range(3):
            slope, intercept = np.polyfit(t, com_cart[:, dim], 1)
            trend = slope * t + intercept
            com_cart_shift[:, dim] = trend - trend[0]
    else:
        raise ValueError(f"Unknown COM mode: {mode}")

    com_drift = com_cart_shift[-1]

    # Convert COM correction to fractional coordinates and remove from all atoms.
    lattice_inv = np.linalg.inv(lattice)
    com_frac_drift = np.dot(com_cart_shift, lattice_inv)  # (nframes, 3)
    corrected_frac = unwrapped_frac - com_frac_drift[:, np.newaxis, :]
    
    return corrected_frac, com_drift


# ==============================================================================
# MSD 计算 - Multiple Time Origins
# ==============================================================================

def compute_msd_single_origin(
    unwrapped_frac: np.ndarray,
    lattice: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    单一时间原点 (t0=0) MSD 计算（旧版兼容）
    
    Returns:
        msd: (nframes,) MSD 数组
    """
    traj = unwrapped_frac[:, indices, :]
    cart = np.tensordot(traj, lattice, axes=([2], [0]))
    disp = cart - cart[0]
    msd = np.mean(np.sum(disp ** 2, axis=2), axis=1)
    return msd


def compute_msd_mto(
    unwrapped_frac: np.ndarray,
    lattice: np.ndarray,
    indices: np.ndarray,
    origins: np.ndarray,
    max_lag: Optional[int] = None,
    lags: Optional[np.ndarray] = None,
    min_samples: int = 5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Multiple Time Origins (MTO) MSD 计算
    
    Args:
        unwrapped_frac: (nframes, natoms, 3) unwrapped 分数坐标
        lattice: 晶格矢量
        indices: 目标原子索引
        origins: 时间原点帧索引数组
        max_lag: 最大 lag（None = 自动）
        lags: 显式 lag 帧数组（可选，推荐与 stride 配合）
        min_samples: 每个 lag 最少样本数
    
    Returns:
        msd: (n_lags,) MSD 数组（按 dense lag 或输入 lags 顺序）
        n_samples: (n_lags,) 每个 lag 的样本数
        effective_max_lag: 实际使用的最大 lag（单位: frames）
    """
    nframes = unwrapped_frac.shape[0]
    
    # 提取目标原子的笛卡尔坐标
    traj = unwrapped_frac[:, indices, :]
    cart = np.tensordot(traj, lattice, axes=([2], [0]))  # (nframes, n_indices, 3)
    
    if max_lag is None:
        max_lag = nframes - 1
    max_lag = min(max_lag, nframes - 1)

    if lags is None:
        lag_values = np.arange(max_lag + 1, dtype=int)
    else:
        lag_values = np.asarray(lags, dtype=int)
        lag_values = lag_values[(lag_values >= 0) & (lag_values <= max_lag)]
        lag_values = np.unique(lag_values)
        if len(lag_values) == 0:
            return np.array([]), np.array([], dtype=int), 0

    msd_sum = np.zeros(len(lag_values), dtype=float)
    n_samples = np.zeros(len(lag_values), dtype=int)

    # Outer loop on origins only; lag dimension is vectorized.
    for t0 in origins:
        if t0 < 0 or t0 >= nframes:
            continue

        max_lag_t0 = min(max_lag, nframes - 1 - t0)
        n_valid_lags = int(np.searchsorted(lag_values, max_lag_t0, side='right'))
        if n_valid_lags == 0:
            continue

        disp = cart[t0:t0 + max_lag_t0 + 1] - cart[t0]  # (n_lag_t0+1, n_indices, 3)
        sq_all = np.sum(disp ** 2, axis=2).mean(axis=1)  # mean over atoms

        selected_lags = lag_values[:n_valid_lags]
        msd_sum[:n_valid_lags] += sq_all[selected_lags]
        n_samples[:n_valid_lags] += 1

    valid_mask = n_samples >= min_samples
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.array([]), np.array([], dtype=int), 0

    effective_last_pos = int(valid_indices[-1])
    effective_max_lag = int(lag_values[effective_last_pos])

    msd = np.zeros(len(lag_values), dtype=float)
    msd[valid_mask] = msd_sum[valid_mask] / n_samples[valid_mask]

    return (
        msd[:effective_last_pos + 1],
        n_samples[:effective_last_pos + 1],
        effective_max_lag
    )


def select_time_origins(
    nframes: int,
    n_origins: int = 20,
    origin_stride: Optional[int] = None,
    max_lag: Optional[int] = None
) -> np.ndarray:
    """
    选择时间原点
    
    Args:
        nframes: 总帧数
        n_origins: 目标原点数
        origin_stride: 原点间隔（优先于 n_origins）
        max_lag: 最大 lag（用于确保原点有足够的后续帧）
    
    Returns:
        origins: 时间原点帧索引数组
    """
    if max_lag is None:
        max_lag = nframes // 2
    
    # 原点只能在 [0, nframes - max_lag) 范围内
    max_origin = max(1, nframes - max_lag)
    
    if origin_stride is not None and origin_stride > 0:
        origins = np.arange(0, max_origin, origin_stride)
    else:
        n_origins = min(n_origins, max_origin)
        if n_origins <= 1:
            origins = np.array([0])
        else:
            origins = np.linspace(0, max_origin - 1, n_origins, dtype=int)
    
    return np.unique(origins)


# ==============================================================================
# Running D 计算
# ==============================================================================

def compute_running_D_ratio(t_ps: np.ndarray, msd: np.ndarray) -> np.ndarray:
    """
    计算 Running D (ratio): D(t) = MSD(t) / (6t)
    
    单位: cm²/s (1 Å²/ps = 1e-4 cm²/s)
    """
    D_ratio = np.zeros_like(msd)
    D_ratio[0] = 0
    D_ratio[1:] = msd[1:] / (6.0 * t_ps[1:]) * 1e-4
    return D_ratio


def compute_running_D_derivative(
    t_ps: np.ndarray,
    msd: np.ndarray,
    smooth_window: int = 5
) -> np.ndarray:
    """
    计算 Running D (derivative): D(t) = (1/6) d(MSD)/dt
    
    使用中心差分 + 可选平滑
    
    单位: cm²/s
    """
    n = len(msd)
    D_deriv = np.zeros(n)
    
    if n < 3:
        return D_deriv
    
    dt = t_ps[1] - t_ps[0] if len(t_ps) > 1 else 1.0
    
    # 中心差分
    D_deriv[1:-1] = (msd[2:] - msd[:-2]) / (2 * dt) / 6.0
    
    # 边界处理
    D_deriv[0] = 0
    D_deriv[-1] = (msd[-1] - msd[-2]) / dt / 6.0
    
    # 可选平滑
    if smooth_window > 1 and n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        D_deriv_smooth = np.convolve(D_deriv, kernel, mode='same')
        # 保持边界原值
        half_w = smooth_window // 2
        D_deriv_smooth[:half_w] = D_deriv[:half_w]
        D_deriv_smooth[-half_w:] = D_deriv[-half_w:]
        D_deriv = D_deriv_smooth
    
    # 转换单位: Å²/ps -> cm²/s
    D_deriv *= 1e-4
    
    return D_deriv


# ==============================================================================
# log-log 斜率 α(t) 计算
# ==============================================================================

def compute_alpha(
    t_ps: np.ndarray,
    msd: np.ndarray,
    window: int = 21,
    min_t_ps: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 log-log 斜率 α(t) = d log(MSD) / d log(t)
    
    α ≈ 1: 正常扩散
    α < 1: 受限/亚扩散（caging）
    α > 1: 弹道/超扩散
    
    Args:
        t_ps: 时间数组
        msd: MSD 数组
        window: 滑动窗口大小（点数，建议奇数）
        min_t_ps: 忽略小于此时间的点
    
    Returns:
        t_alpha: 有效时间点
        alpha: 对应的 α 值
    """
    # 过滤有效点
    valid_mask = (t_ps > min_t_ps) & (msd > 0)
    t_valid = t_ps[valid_mask]
    msd_valid = msd[valid_mask]
    
    if len(t_valid) < window:
        return np.array([]), np.array([])
    
    log_t = np.log(t_valid)
    log_msd = np.log(msd_valid)
    
    n = len(log_t)
    half_w = window // 2
    
    alpha = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        
        if end - start < 3:
            alpha[i] = np.nan
            continue
        
        # 局部线性回归
        x = log_t[start:end]
        y = log_msd[start:end]
        
        # 最小二乘
        n_pts = len(x)
        sx = np.sum(x)
        sy = np.sum(y)
        sxx = np.sum(x * x)
        sxy = np.sum(x * y)
        
        denom = n_pts * sxx - sx * sx
        if abs(denom) < 1e-15:
            alpha[i] = np.nan
        else:
            alpha[i] = (n_pts * sxy - sx * sy) / denom
    
    return t_valid, alpha


def interpret_alpha(alpha_mean: float, alpha_std: float) -> Tuple[str, str]:
    """
    解释 α 值的物理意义
    
    Returns:
        regime: 扩散状态
        description: 描述
    """
    if np.isnan(alpha_mean):
        return "unknown", "α 无法计算"
    
    if 0.8 <= alpha_mean <= 1.2:
        if alpha_std < 0.2:
            regime = "diffusive"
            desc = f"正常扩散 (α={alpha_mean:.2f}±{alpha_std:.2f})"
        else:
            regime = "diffusive_noisy"
            desc = f"可能为扩散，但噪声较大 (α={alpha_mean:.2f}±{alpha_std:.2f})"
    elif alpha_mean < 0.8:
        regime = "subdiffusive"
        desc = f"亚扩散/受限 (α={alpha_mean:.2f}±{alpha_std:.2f}): 可能存在 caging 或网络限制"
    else:  # alpha_mean > 1.2
        regime = "superdiffusive"
        desc = f"超扩散/弹道 (α={alpha_mean:.2f}±{alpha_std:.2f}): 可能处于早期弹道区或存在数值漂移"
    
    return regime, desc


# ==============================================================================
# 误差估计
# ==============================================================================

def estimate_D_with_trajectory_blocks(
    unwrapped_frac: np.ndarray,
    lattice: np.ndarray,
    indices: np.ndarray,
    dt_ps: float,
    n_blocks: int,
    fit_start_frac: float = 0.3,
    fit_end_frac: float = 0.9,
    n_origins_per_block: int = 10,
    min_block_time_ps: float = 5.0,
    fit_start_ps: Optional[float] = None,
    fit_end_ps: Optional[float] = None,
    skip_ps: float = 0.0,
    stride: int = 1,
    max_lag_frames: Optional[int] = None,
    min_samples: int = 5
) -> Tuple[float, float, float, List[float]]:
    """
    Trajectory blocks 误差估计（正确实现）
    
    将轨迹切成互不重叠的 blocks，每个 block 内独立做 MTO MSD，
    分别拟合得到多个 D，再统计。
    
    Args:
        unwrapped_frac: (nframes, natoms, 3)
        lattice: 晶格矢量
        indices: 目标原子索引
        dt_ps: 时间步长 (ps)
        n_blocks: 分块数
        fit_start_frac: 拟合起始比例（仅在 fit_start_ps=None 时用于兼容）
        fit_end_frac: 拟合终止比例（仅在 fit_end_ps=None 时用于兼容）
        n_origins_per_block: 每个 block 内的时间原点数
        min_block_time_ps: Minimum block duration in ps (v2.2)
        fit_start_ps: 拟合起始时间（绝对时间，ps）
        fit_end_ps: 拟合终止时间（绝对时间，ps，右边界 exclusive）
        skip_ps: 跳过时间（ps，left semantics）
        stride: lag stride
        max_lag_frames: 主流程使用的最大物理 lag（frames）
        min_samples: 每个 lag 的最小样本数
    
    Returns:
        D_mean: 平均扩散系数 (cm²/s), NaN if blocks too short
        D_std: 标准差
        D_sem: 标准误差
        D_blocks: 各 block 的 D 值列表
    """
    nframes = unwrapped_frac.shape[0]
    if n_blocks <= 0:
        print(f"[WARN] Invalid n_blocks={n_blocks}; must be > 0")
        return np.nan, np.nan, np.nan, []
    block_size = nframes // n_blocks

    if block_size <= 1:
        print(f"[WARN] block_size={block_size} too small for n_blocks={n_blocks}")
        return np.nan, np.nan, np.nan, []

    reference_max_lag_frames = max_lag_frames if max_lag_frames is not None else (nframes // 2)
    reference_max_lag_frames = max(1, min(reference_max_lag_frames, nframes - 1))
    _, global_t_ps = build_time_axis_and_lags(reference_max_lag_frames, stride, dt_ps)
    if len(global_t_ps) == 0:
        return np.nan, np.nan, np.nan, []

    if fit_start_ps is None:
        fit_start_idx_global = min(len(global_t_ps) - 1, int(len(global_t_ps) * fit_start_frac))
        fit_start_ps = float(global_t_ps[fit_start_idx_global])
    if fit_end_ps is None:
        fit_end_idx_global = min(len(global_t_ps) - 1, max(1, int(len(global_t_ps) * fit_end_frac)))
        fit_end_ps = float(global_t_ps[fit_end_idx_global])

    fit_start_ps = max(float(skip_ps), float(fit_start_ps))
    fit_end_ps = max(fit_end_ps, fit_start_ps)

    print(f"    [INFO] Trajectory blocks fit window (absolute): {fit_start_ps:.3f} ~ {fit_end_ps:.3f} ps")

    # v2.2: Physical minimum time check instead of frame-based
    min_block_frames = int(np.ceil(min_block_time_ps / dt_ps))
    block_time_ps = block_size * dt_ps
    
    if block_size < min_block_frames:
        print(f"[WARN] Block duration ({block_time_ps:.2f} ps) < min_block_time ({min_block_time_ps} ps)")
        print(f"[WARN] Block uncertainty unavailable due to insufficient block duration")
        print(f"[WARN] Increase trajectory length or reduce n_blocks")
        return np.nan, np.nan, np.nan, []
    
    D_blocks = []
    
    for i in range(n_blocks):
        start_frame = i * block_size
        end_frame = start_frame + block_size
        
        # 提取 block
        block_frac = unwrapped_frac[start_frame:end_frame]
        block_nframes = block_frac.shape[0]
        if block_nframes <= 2:
            print(f"    [WARN] block {i+1}/{n_blocks} skipped: too few frames ({block_nframes})")
            continue

        block_max_lag = block_nframes - 1 if max_lag_frames is None else min(max_lag_frames, block_nframes - 1)
        block_lags, _ = build_time_axis_and_lags(block_max_lag, stride, dt_ps)
        if len(block_lags) < 2:
            print(f"    [WARN] block {i+1}/{n_blocks} skipped: insufficient lag points")
            continue
        
        # Block 内的时间原点
        block_origins = select_time_origins(
            block_nframes,
            n_origins=n_origins_per_block,
            max_lag=block_max_lag
        )
        
        # 计算 block MSD
        msd_block, _, max_lag_block = compute_msd_mto(
            block_frac,
            lattice,
            indices,
            block_origins,
            max_lag=block_max_lag,
            lags=block_lags,
            min_samples=min_samples
        )
        if len(msd_block) < 2:
            print(f"    [WARN] block {i+1}/{n_blocks} skipped: no valid MSD points after min_samples")
            continue

        lag_block_used = block_lags[block_lags <= max_lag_block]
        t_block_ps = lag_block_used * dt_ps
        if len(t_block_ps) != len(msd_block):
            # Defensive: keep aligned with MSD length
            t_block_ps = lag_block_used[:len(msd_block)] * dt_ps

        if len(t_block_ps) == 0:
            print(f"    [WARN] block {i+1}/{n_blocks} skipped: empty time axis")
            continue

        if t_block_ps[-1] + 1e-12 < fit_end_ps:
            print(
                f"    [INFO] block {i+1}/{n_blocks} skipped: "
                f"max time {t_block_ps[-1]:.3f} ps < fit_end {fit_end_ps:.3f} ps"
            )
            continue

        skip_idx_block = ps_to_index(skip_ps, t_block_ps, side='left')
        fit_start_idx = max(skip_idx_block, ps_to_index(fit_start_ps, t_block_ps, side='left'))
        fit_end_idx = ps_to_index(fit_end_ps, t_block_ps, side='right')
        fit_end_idx = min(fit_end_idx, len(msd_block))

        if fit_end_idx - fit_start_idx < 5:
            print(f"    [INFO] block {i+1}/{n_blocks} skipped: fit window has <5 points")
            continue
        
        # 线性拟合
        t_fit = t_block_ps[fit_start_idx:fit_end_idx]
        msd_fit = msd_block[fit_start_idx:fit_end_idx]
        
        _, slope, _ = linear_fit(t_fit, msd_fit)
        D_block = slope / 6.0 * 1e-4  # Å²/ps -> cm²/s
        
        if D_block > 0:
            D_blocks.append(D_block)
    
    if len(D_blocks) < 2:
        print(f"    [WARN] only {len(D_blocks)} valid blocks remain; need >=2 for uncertainty")
        return np.nan, np.nan, np.nan, D_blocks
    
    D_mean = np.mean(D_blocks)
    D_std = np.std(D_blocks, ddof=1)
    D_sem = D_std / np.sqrt(len(D_blocks))
    
    return D_mean, D_std, D_sem, D_blocks


def estimate_D_with_bootstrap(
    unwrapped_frac: np.ndarray,
    lattice: np.ndarray,
    indices: np.ndarray,
    dt_ps: float,
    all_origins: np.ndarray,
    lag_frames: np.ndarray,
    fit_start_idx: int,
    fit_end_idx: int,
    n_bootstrap: int = 100,
    seed: Optional[int] = None,
    min_samples: int = 5
) -> Tuple[float, float, float, float]:
    """
    Bootstrap 误差估计
    
    从时间原点中重采样，计算多组 MSD 与 D
    
    Returns:
        D_mean: 平均扩散系数 (cm²/s)
        D_std: 标准差
        D_ci_low: 95% CI 下界
        D_ci_high: 95% CI 上界
    """
    if seed is not None:
        np.random.seed(seed)

    lag_frames = np.asarray(lag_frames, dtype=int)
    lag_frames = np.unique(lag_frames[lag_frames >= 0])
    if len(lag_frames) == 0:
        return np.nan, np.nan, np.nan, np.nan

    max_lag_frames = int(np.max(lag_frames))
    
    n_origins = len(all_origins)
    D_bootstrap = []
    
    for _ in range(n_bootstrap):
        # 有放回重采样
        resampled_origins = np.random.choice(all_origins, size=n_origins, replace=True)
        
        # 计算 MSD
        msd, _, max_lag_boot = compute_msd_mto(
            unwrapped_frac,
            lattice,
            indices,
            resampled_origins,
            max_lag=max_lag_frames,
            lags=lag_frames,
            min_samples=min_samples
        )

        if len(msd) == 0:
            continue

        lag_used = lag_frames[lag_frames <= max_lag_boot]
        if len(lag_used) != len(msd):
            lag_used = lag_used[:len(msd)]
        t_ps = lag_used * dt_ps

        fit_end_idx_eff = min(fit_end_idx, len(msd))
        if fit_end_idx_eff <= fit_start_idx + 2:
            continue

        t_fit = t_ps[fit_start_idx:fit_end_idx_eff]
        msd_fit = msd[fit_start_idx:fit_end_idx_eff]
        _, slope, _ = linear_fit(t_fit, msd_fit)
        D = slope / 6.0 * 1e-4
        
        if D > 0:
            D_bootstrap.append(D)
    
    if len(D_bootstrap) < 10:
        return np.nan, np.nan, np.nan, np.nan
    
    D_mean = np.mean(D_bootstrap)
    D_std = np.std(D_bootstrap, ddof=1)
    D_ci_low = np.percentile(D_bootstrap, 2.5)
    D_ci_high = np.percentile(D_bootstrap, 97.5)
    
    return D_mean, D_std, D_ci_low, D_ci_high


# ==============================================================================
# 辅助函数
# ==============================================================================

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    线性最小二乘拟合 y = a + b*x
    
    返回: (intercept, slope, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)
    syy = np.sum(y * y)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    # R²
    ss_tot = syy - sy * sy / n
    ss_res = syy - 2 * slope * sxy - 2 * intercept * sy + slope * slope * sxx + 2 * slope * intercept * sx + n * intercept * intercept
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return intercept, slope, max(0, min(1, r_squared))


def check_plateau_combined(
    D_deriv: np.ndarray,
    t_ps: np.ndarray,
    alpha: np.ndarray,
    t_alpha: np.ndarray,
    fit_start_idx: int,
    fit_end_idx: int
) -> Tuple[str, float, float, float, str]:
    """
    综合平台判定（D_derivative + alpha）
    
    Returns:
        verdict: 判定结果等级
        drift_ratio: D_deriv 漂移率
        alpha_mean: 拟合区间 α 均值
        alpha_std: 拟合区间 α 标准差
        message: 判定消息
    """
    # 1. D_deriv 漂移率
    D_fit = D_deriv[fit_start_idx:fit_end_idx]
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    
    if len(D_fit) < 3:
        return "insufficient_data", 1.0, np.nan, np.nan, "拟合区间数据不足"
    
    _, slope, _ = linear_fit(t_fit, D_fit)
    mean_D = np.mean(D_fit)
    
    if mean_D <= 0:
        drift_ratio = 1.0
    else:
        time_span = t_fit[-1] - t_fit[0]
        total_drift = abs(slope * time_span)
        drift_ratio = total_drift / abs(mean_D)
    
    # 2. Alpha 在拟合区间的统计
    if len(t_alpha) > 0 and len(alpha) > 0:
        t_fit_start = t_ps[fit_start_idx]
        t_fit_end = t_ps[fit_end_idx - 1] if fit_end_idx > 0 else t_ps[-1]
        
        alpha_mask = (t_alpha >= t_fit_start) & (t_alpha <= t_fit_end) & (~np.isnan(alpha))
        alpha_in_fit = alpha[alpha_mask]
        
        if len(alpha_in_fit) > 0:
            alpha_mean = np.mean(alpha_in_fit)
            alpha_std = np.std(alpha_in_fit)
        else:
            alpha_mean = np.nan
            alpha_std = np.nan
    else:
        alpha_mean = np.nan
        alpha_std = np.nan
    
    # 3. 综合判定
    drift_ok = drift_ratio < 0.2
    alpha_ok = (not np.isnan(alpha_mean)) and (0.8 <= alpha_mean <= 1.2)
    
    if drift_ok and alpha_ok:
        verdict = "credible"
        message = f"可信 (diffusive): drift={drift_ratio*100:.1f}%%, α={alpha_mean:.2f}±{alpha_std:.2f}"
    elif drift_ok and not alpha_ok:
        if np.isnan(alpha_mean):
            verdict = "uncertain"
            message = f"不确定: drift={drift_ratio*100:.1f}%%, α 无法计算"
        elif alpha_mean < 0.8:
            verdict = "subdiffusive"
            message = f"亚扩散/受限: drift={drift_ratio*100:.1f}%%, α={alpha_mean:.2f} (< 0.8)"
        else:
            verdict = "superdiffusive"
            message = f"超扩散/弹道: drift={drift_ratio*100:.1f}%%, α={alpha_mean:.2f} (> 1.2)"
    elif not drift_ok and alpha_ok:
        verdict = "drifting"
        message = f"D(t) 漂移: drift={drift_ratio*100:.1f}%% (> 20%%), α={alpha_mean:.2f}"
    else:
        verdict = "unreliable"
        if np.isnan(alpha_mean):
            message = f"不可靠: drift={drift_ratio*100:.1f}%%, α 无法计算"
        else:
            message = f"不可靠: drift={drift_ratio*100:.1f}%%, α={alpha_mean:.2f}"
    
    return verdict, drift_ratio, alpha_mean, alpha_std, message


# ==============================================================================
# 输出与绘图
# ==============================================================================

def save_msd_data(
    filepath: str,
    t_ps: np.ndarray,
    msd: np.ndarray,
    n_samples: Optional[np.ndarray] = None
):
    """保存 MSD 数据"""
    with open(filepath, 'w') as f:
        f.write("# MSD 数据\n")
        f.write("# 单位: t_ps (ps), MSD_A2 (Å²)\n")
        if n_samples is not None:
            f.write("# t_ps  MSD_A2  n_samples\n")
            for t, m, n in zip(t_ps, msd, n_samples):
                f.write(f"{t:.6f}  {m:.6f}  {int(n)}\n")
        else:
            f.write("# t_ps  MSD_A2\n")
            for t, m in zip(t_ps, msd):
                f.write(f"{t:.6f}  {m:.6f}\n")


def save_running_D_data(
    filepath: str,
    t_ps: np.ndarray,
    D_ratio: np.ndarray,
    D_deriv: np.ndarray
):
    """保存 Running-D 数据"""
    with open(filepath, 'w') as f:
        f.write("# Running Diffusion Coefficient\n")
        f.write("# 单位: t_ps (ps), D_ratio (cm²/s), D_deriv (cm²/s)\n")
        f.write("# D_ratio = MSD/(6t), D_deriv = (1/6) d(MSD)/dt\n")
        f.write("# t_ps  D_ratio  D_deriv\n")
        for t, dr, dd in zip(t_ps[1:], D_ratio[1:], D_deriv[1:]):
            f.write(f"{t:.6f}  {dr:.6e}  {dd:.6e}\n")


def save_alpha_data(filepath: str, t_ps: np.ndarray, alpha: np.ndarray):
    """保存 alpha 数据"""
    with open(filepath, 'w') as f:
        f.write("# log-log 斜率 α(t) = d log(MSD) / d log(t)\n")
        f.write("# α ≈ 1: 正常扩散; α < 1: 亚扩散/受限; α > 1: 弹道/超扩散\n")
        f.write("# 单位: t_ps (ps), alpha (无量纲)\n")
        f.write("# t_ps  alpha\n")
        for t, a in zip(t_ps, alpha):
            if not np.isnan(a):
                f.write(f"{t:.6f}  {a:.4f}\n")


def plot_msd(
    t_ps: np.ndarray,
    msd: np.ndarray,
    specie: str,
    fit_start_idx: int,
    fit_end_idx: int,
    slope: float,
    intercept: float,
    outdir: str = "."
):
    """绘制 MSD 曲线"""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t_ps, msd, 'b-', linewidth=1, label='MSD')
    
    # 拟合线
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    msd_fit = intercept + slope * t_fit
    ax.plot(t_fit, msd_fit, 'r--', linewidth=2, label=f'Linear fit (slope={slope:.4f} Å²/ps)')
    
    # 拟合区间标记
    ax.axvline(x=t_ps[fit_start_idx], color='gray', linestyle=':', alpha=0.5)
    if fit_end_idx > 0:
        ax.axvline(x=t_ps[fit_end_idx-1], color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('MSD (Å²)')
    ax.set_title(f'MSD of {specie} (MTO)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'msd_{specie}.png'), dpi=150)
    plt.close()


def plot_running_D(
    t_ps: np.ndarray,
    D_ratio: np.ndarray,
    D_deriv: np.ndarray,
    specie: str,
    fit_start_idx: int,
    fit_end_idx: int,
    D_mean: float,
    D_std: float,
    outdir: str = "."
):
    """绘制 Running-D 曲线（ratio + derivative）"""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 跳过 t=0
    ax.plot(t_ps[1:], D_ratio[1:], 'b-', linewidth=1, alpha=0.7, label='D_ratio = MSD/(6t)')
    ax.plot(t_ps[1:], D_deriv[1:], 'g-', linewidth=1, alpha=0.7, label='D_deriv = (1/6)dMSD/dt')
    
    # 平均值线
    ax.axhline(y=D_mean, color='r', linestyle='--', linewidth=2, 
               label=f'D = ({D_mean:.2e} ± {D_std:.2e}) cm²/s')
    
    # 误差带
    if D_std > 0:
        ax.axhspan(D_mean - D_std, D_mean + D_std, alpha=0.15, color='red')
    
    # 拟合区间
    ax.axvline(x=t_ps[fit_start_idx], color='gray', linestyle=':', alpha=0.5)
    if fit_end_idx > 0 and fit_end_idx <= len(t_ps):
        ax.axvline(x=t_ps[fit_end_idx-1], color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('D(t) (cm²/s)')
    ax.set_title(f'Running Diffusion Coefficient of {specie}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'D_running_{specie}.png'), dpi=150)
    plt.close()


def plot_alpha(
    t_alpha: np.ndarray,
    alpha: np.ndarray,
    specie: str,
    fit_start_ps: float,
    fit_end_ps: float,
    outdir: str = "."
):
    """绘制 α(t) 曲线"""
    if not HAS_MPL or len(t_alpha) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    valid_mask = ~np.isnan(alpha)
    ax.plot(t_alpha[valid_mask], alpha[valid_mask], 'b-', linewidth=1, label='α(t)')
    
    # 参考线
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='α=1 (diffusive)')
    ax.axhline(y=0.8, color='orange', linestyle=':', alpha=0.7, label='α=0.8 (subdiffusive threshold)')
    ax.axhline(y=1.2, color='orange', linestyle=':', alpha=0.7, label='α=1.2 (superdiffusive threshold)')
    
    # 拟合区间
    ax.axvline(x=fit_start_ps, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=fit_end_ps, color='gray', linestyle=':', alpha=0.5)
    ax.axvspan(fit_start_ps, fit_end_ps, alpha=0.1, color='blue', label='Fit window')
    
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('α = d log(MSD) / d log(t)')
    ax.set_title(f'log-log slope α(t) of {specie}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'alpha_{specie}.png'), dpi=150)
    plt.close()


def write_report(
    specie: str,
    nframes: int,
    dt_fs: float,
    t_total_ps: float,
    t_fit_start: float,
    t_fit_end: float,
    D_mean: float,
    D_std: float,
    D_sem: float,
    r2: float,
    verdict: str,
    drift_ratio: float,
    alpha_mean: float,
    alpha_std: float,
    plateau_msg: str,
    time_origin_mode: str,
    n_origins: int,
    max_lag_ps: float,
    min_samples: int,
    com_mode: str,
    com_drift: np.ndarray,
    block_mode: str,
    n_blocks: int,
    unwrap_diagnostics: Dict[str, Any],
    cell_check: Dict[str, Any],
    duplicate_summary: Dict[str, Any],
    com_reference: Dict[str, Any],
    strict_flags: Dict[str, bool],
    skip_window_note: Optional[str],
    summary_line: str,
    outdir: str = "."
):
    """写入完整分析报告"""
    report_path = os.path.join(outdir, 'msd_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"MSD/扩散系数分析报告 (aimd_msd.py {VERSION})\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("--- 基本信息 ---\n")
        f.write(f"物种: {specie}\n")
        f.write(f"总帧数: {nframes}\n")
        f.write(f"时间步长: {dt_fs} fs\n")
        f.write(f"总模拟时间: {t_total_ps:.3f} ps\n\n")
        
        f.write("--- 时间原点配置 ---\n")
        f.write(f"模式: {time_origin_mode}\n")
        if time_origin_mode == 'multi':
            f.write(f"时间原点数: {n_origins}\n")
            f.write(f"最大 lag: {max_lag_ps:.3f} ps\n")
            f.write(f"最小样本数阈值: {min_samples}\n")
        f.write("\n")

        f.write("--- Traceability ---\n")
        f.write(
            f"严格模式: strict={strict_flags['strict']}, "
            f"allow_unreliable_D={strict_flags['allow_unreliable_D']}\n"
        )
        f.write(
            f"Cell check (tol={cell_check['tolerance']:.4f}, "
            f"angles={'on' if cell_check['check_angles'] else 'off'}): "
            f"{cell_check['message']}\n"
        )
        f.write(
            f"连续重复帧去重: removed={duplicate_summary['count']}, "
            f"sample_global_positions={duplicate_summary['sample_text']}\n"
        )
        f.write(
            f"COM reference source: {com_reference['source']} "
            f"({com_reference['description']})\n"
        )
        if skip_window_note is not None:
            f.write(f"Skip/fix note: {skip_window_note}\n")
        f.write("\n")
        
        f.write("--- COM 漂移去除 ---\n")
        f.write(f"模式: {com_mode}\n")
        if com_mode != 'none':
            f.write(f"总 COM 漂移: ({com_drift[0]:.4f}, {com_drift[1]:.4f}, {com_drift[2]:.4f}) Å\n")
            f.write(f"COM 漂移长度: {np.linalg.norm(com_drift):.4f} Å\n")
        f.write("\n")
        
        f.write("--- Unwrap 诊断 ---\n")
        f.write(f"跳变检测阈值: {unwrap_diagnostics['effective_threshold_A']:.2f} Å\n")
        f.write(f"晶格参数: {unwrap_diagnostics['cell_lengths']}\n")
        f.write(f"最大单步位移: {unwrap_diagnostics['max_jump_A']:.4f} Å (帧 {unwrap_diagnostics['max_jump_frame']})\n")
        f.write(f"可疑跳变数: {unwrap_diagnostics['n_suspicious_jumps']}\n")
        
        # 分数坐标一致性检查
        frac_issues = unwrap_diagnostics.get('frac_consistency_issues', 0)
        if frac_issues > 0:
            f.write(f"分数坐标 |d|>0.5 跳跃次数: {frac_issues} (已 MIC 修正)\n")
        
        if unwrap_diagnostics.get('is_vacuum_system', False):
            f.write("\n[INFO] 检测到可能是 cluster/vacuum 体系\n")
            f.write("说明: 原子未跨越 PBC，unwrap 影响较小\n")
        
        if unwrap_diagnostics['has_warnings']:
            f.write("\n!!! 警告: 检测到可疑跳变 !!!\n")
            f.write("可能原因: XDATCAR 错误 / 步长过大 / NPT 变胞 / 周期边界问题\n")
            f.write("建议: 检查 XDATCAR/CONTCAR，确认 POTIM 设置\n")
            for jump in unwrap_diagnostics['suspicious_jumps'][:5]:
                f.write(f"  帧 {jump['frame']}, 原子 {jump['atom']}, 距离 {jump['distance_A']:.2f} Å\n")
        f.write("\n")
        
        f.write("--- 拟合参数 ---\n")
        f.write(f"拟合区间: {t_fit_start:.3f} ~ {t_fit_end:.3f} ps\n")
        f.write(f"R²: {r2:.6f}\n\n")
        
        f.write("--- 扩散系数 ---\n")
        f.write(f"D = ({D_mean:.6e} ± {D_std:.6e}) cm²/s (±STD)\n")
        f.write(f"D = ({D_mean:.6e} ± {D_sem:.6e}) cm²/s (±SEM)\n")
        f.write(f"D = {D_mean * 1e4:.6e} Å²/ps\n\n")
        
        f.write("--- 误差估计方法 ---\n")
        f.write(f"方法: {block_mode}\n")
        if block_mode == 'trajectory_blocks':
            f.write(f"分块数: {n_blocks}\n")
            f.write("说明: 轨迹被切成互不重叠的 blocks，每个 block 独立计算 MTO MSD 并拟合 D\n")
        else:
            f.write("说明: Bootstrap 重采样时间原点\n")
        f.write("\n")
        
        f.write("--- 平台判定 ---\n")
        f.write(f"判定等级: {verdict}\n")
        f.write(f"D_deriv 漂移率: {drift_ratio*100:.1f}%%\n")
        if not np.isnan(alpha_mean):
            f.write(f"α 均值: {alpha_mean:.3f} ± {alpha_std:.3f}\n")
            regime, regime_desc = interpret_alpha(alpha_mean, alpha_std)
            f.write(f"扩散状态: {regime_desc}\n")
        f.write(f"综合判定: {plateau_msg}\n\n")
        
        if verdict not in ['credible', 'diffusive']:
            f.write("!" * 70 + "\n")
            f.write("!!! 警告: 扩散系数可能不可靠 !!!\n")
            if verdict == 'subdiffusive':
                f.write("!!! 检测到亚扩散行为，可能存在 caging 或受限运动 !!!\n")
            elif verdict == 'superdiffusive':
                f.write("!!! 检测到超扩散行为，可能处于早期弹道区或存在数值漂移 !!!\n")
            elif verdict == 'drifting':
                f.write("!!! D(t) 漂移严重，AIMD 时间可能不足 !!!\n")
            f.write("!!! 建议: 延长模拟时间 / 检查体系平衡 / 检查 POTIM !!!\n")
            f.write("!" * 70 + "\n")
        
        f.write("\n--- 输出文件 ---\n")
        f.write(f"msd_{specie}.dat: MSD 数据 (t_ps, MSD_A2, n_samples)\n")
        f.write(f"D_running_{specie}.dat: Running-D 数据 (D_ratio, D_deriv)\n")
        f.write(f"alpha_{specie}.dat: log-log 斜率 α(t)\n")
        f.write(f"msd_{specie}.png: MSD 图\n")
        f.write(f"D_running_{specie}.png: Running-D 图 (ratio + derivative)\n")
        f.write(f"alpha_{specie}.png: α(t) 图\n")
        
        f.write("\n--- 注意事项 ---\n")
        f.write("1. 本脚本假定 XDATCAR 晶格恒定；NPT/变胞请先预处理\n")
        f.write("2. AIMD 时间尺度有限（ps 级），扩散系数仅供趋势参考\n")
        f.write("3. 长程扩散（ns 级）请使用经典 MD\n")
        f.write("4. α ≈ 1 才能称为正常扩散；α < 1 表示受限运动\n")
        f.write("\n")
        f.write(summary_line + "\n")
        f.write("=" * 70 + "\n")


def write_abort_report(
    specie: str,
    dt_fs: float,
    abort_reason: str,
    strict_flags: Dict[str, bool],
    cell_check: Dict[str, Any],
    summary_line: str,
    outdir: str = "."
):
    """Write a minimal report for early-abort pipeline cases."""
    report_path = os.path.join(outdir, 'msd_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"MSD/扩散系数分析报告 (aimd_msd.py {VERSION})\n")
        f.write("=" * 70 + "\n\n")
        f.write("--- Status ---\n")
        f.write("运行状态: aborted\n")
        f.write(f"原因: {abort_reason}\n")
        f.write(f"物种: {specie}\n")
        f.write(f"时间步长: {dt_fs} fs\n\n")

        f.write("--- Traceability ---\n")
        f.write(
            f"严格模式: strict={strict_flags['strict']}, "
            f"allow_unreliable_D={strict_flags['allow_unreliable_D']}\n"
        )
        f.write(
            f"Cell check (tol={cell_check['tolerance']:.4f}, "
            f"angles={'on' if cell_check['check_angles'] else 'off'}): "
            f"{cell_check['message']}\n"
        )
        f.write("\n")
        f.write(summary_line + "\n")
        f.write("=" * 70 + "\n")


def _write_mock_outcar(path: str, cells: List[np.ndarray]) -> None:
    lines: List[str] = []
    for cell in cells:
        lines.append("  direct lattice vectors                 reciprocal lattice vectors\n")
        for vec in cell:
            lines.append(
                f" {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}"
                "   0.0000000000   0.0000000000   0.0000000000\n"
            )
    with open(path, 'w') as f:
        f.writelines(lines)


def _write_synthetic_xdatcar(path: str) -> None:
    rng = np.random.default_rng(12345)
    lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    species = ["Li", "O"]
    counts = [4, 4]
    nframes = 64
    natoms = sum(counts)

    li_unwrapped = np.array([
        [0.10, 0.10, 0.10],
        [0.20, 0.25, 0.15],
        [0.35, 0.30, 0.25],
        [0.45, 0.15, 0.35],
    ], dtype=float)
    o_positions = np.array([
        [0.65, 0.65, 0.65],
        [0.72, 0.68, 0.61],
        [0.60, 0.75, 0.70],
        [0.78, 0.58, 0.73],
    ], dtype=float)

    frames = np.zeros((nframes, natoms, 3), dtype=float)
    for t in range(nframes):
        if t > 0:
            li_unwrapped += rng.normal(loc=0.0, scale=0.012, size=li_unwrapped.shape)
        frames[t, :counts[0], :] = np.mod(li_unwrapped, 1.0)
        frames[t, counts[0]:, :] = o_positions

    lines = [
        "Synthetic XDATCAR for aimd_msd self-check\n",
        "1.0\n",
    ]
    for vec in lattice:
        lines.append(f" {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n")
    lines.append(" " + " ".join(species) + "\n")
    lines.append(" " + " ".join(str(c) for c in counts) + "\n")
    for idx, frame in enumerate(frames, start=1):
        lines.append(f"Direct configuration= {idx:6d}\n")
        for atom in frame:
            lines.append(f" {atom[0]:.10f} {atom[1]:.10f} {atom[2]:.10f}\n")

    with open(path, 'w') as f:
        f.writelines(lines)


def run_self_check() -> int:
    """Run lightweight internal checks and exit."""
    print("[SELF_CHECK] Starting aimd_msd internal checks")

    t_ps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
    assert ps_to_index(0.0, t_ps, side='left') == 0
    assert ps_to_index(0.75, t_ps, side='left') == 2
    assert ps_to_index(1.0, t_ps, side='right') == 3
    assert ps_to_index(9.9, t_ps, side='left') == len(t_ps) - 1
    print("[SELF_CHECK] ps_to_index ok")

    with tempfile.TemporaryDirectory(prefix="aimd_msd_selfcheck_") as tmpdir:
        const_cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])
        variable_cell = np.array([
            [10.0, 0.0, 0.0],
            [0.2, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])

        const_outcar = os.path.join(tmpdir, "OUTCAR.constant")
        var_outcar = os.path.join(tmpdir, "OUTCAR.variable")
        _write_mock_outcar(const_outcar, [const_cell, const_cell])
        _write_mock_outcar(var_outcar, [const_cell, variable_cell])

        const_check = analyze_cell_variability(const_outcar, tolerance=0.01)
        variable_check = analyze_cell_variability(var_outcar, tolerance=1e-4)
        assert not const_check['is_variable'], const_check['message']
        assert variable_check['is_variable'], variable_check['message']
        print("[SELF_CHECK] cell detection ok")

        xdatcar_path = os.path.join(tmpdir, "XDATCAR")
        outcar_path = os.path.join(tmpdir, "OUTCAR")
        outdir = os.path.join(tmpdir, "out")
        os.makedirs(outdir, exist_ok=True)
        _write_synthetic_xdatcar(xdatcar_path)
        _write_mock_outcar(outcar_path, [const_cell, const_cell, const_cell])

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--xdatcar", xdatcar_path,
            "--specie", "Li",
            "--dt_fs", "500.0",
            "--remove_com", "none",
            "--block_mode", "bootstrap",
            "--n_bootstrap", "8",
            "--seed", "123",
            "--max_lag_ps", "12.0",
            "--no_strict",
            "--outdir", outdir,
        ]
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Self-check end-to-end run failed:\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        report_path = os.path.join(outdir, "msd_report.txt")
        if not os.path.isfile(report_path):
            raise AssertionError("Self-check end-to-end run did not create msd_report.txt")
        with open(report_path, 'r') as f:
            report_text = f.read()
        if "SUMMARY " not in completed.stdout or "SUMMARY " not in report_text:
            raise AssertionError("Self-check end-to-end run missing SUMMARY output")
        print("[SELF_CHECK] end-to-end run ok")

    print("[SELF_CHECK] PASS")
    return 0


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--self_check", action='store_true')
    pre_args, remaining_argv = pre_parser.parse_known_args()
    if pre_args.self_check and "--help" not in remaining_argv and "-h" not in remaining_argv:
        sys.exit(run_self_check())

    parser = argparse.ArgumentParser(
        description=f"从 XDATCAR 计算 MSD 并拟合扩散系数 (aimd_msd.py {VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        示例:
            # 默认 MTO 模式（推荐）
            python3 aimd_msd.py --specie Li --dt_fs 1.0

            # 指定时间原点数和最大 lag
            python3 aimd_msd.py --specie Li --dt_fs 1.0 --n_origins 30 --max_lag_ps 15.0

            # 旧版兼容（单一时间原点，仅用于对比）
            python3 aimd_msd.py --specie Li --dt_fs 1.0 --time_origin single

            # 使用 stride 降低计算量
            python3 aimd_msd.py --specie Li --dt_fs 1.0 --stride 2

            # 分块误差估计（默认）
            python3 aimd_msd.py --specie Li --dt_fs 1.0 --n_blocks 5

            # Bootstrap 误差估计
            python3 aimd_msd.py --specie Li --dt_fs 1.0 --block_mode bootstrap --seed 42

            # 不去除 COM 漂移
            python3 aimd_msd.py --specie Li --dt_fs 1.0 --remove_com none

            # 运行内置自检
            python3 aimd_msd.py --self_check

        注意:
            - AIMD 时间尺度有限（ps 级），扩散系数仅供趋势参考
            - 长程扩散（ns 级）请使用经典 MD
            - α ≈ 1 才可称为正常扩散；α < 1 说明受限/亚扩散
            - 本脚本假定晶格恒定；NPT/变胞请先预处理
        """)
    )

    parser.set_defaults(strict=True, cell_angle_check=True)
    parser.add_argument("--self_check", action='store_true',
                        help="运行轻量级内置自检并退出")

    # 基本参数
    parser.add_argument("--xdatcar", default="XDATCAR",
                        help="XDATCAR 文件路径 (默认: XDATCAR)")
    parser.add_argument("--specie", default=None,
                        help="目标物种符号，如 Li/Zn/K/Al")
    parser.add_argument("--dt_fs", type=float, default=None,
                        help="时间步长 POTIM (fs)")
    
    # 跳过参数
    parser.add_argument("--skip", type=int, default=0,
                        help="跳过前 N 帧 (默认: 0)")
    parser.add_argument("--t_skip_ps", type=float, default=1.0,
                        help="跳过初始弹道区时间 (ps, 默认: 1.0)")
    
    # 拟合区间
    parser.add_argument("--t_fit_start_ps", type=float, default=None,
                        help="拟合起始时间 (ps, 默认: 总时间的 30%%)")
    parser.add_argument("--t_fit_end_ps", type=float, default=None,
                        help="拟合终止时间 (ps, 默认: 总时间的 90%%)")
    parser.add_argument("--fit_start", type=float, default=0.3,
                        help="拟合起始位置 (0~1 比例, 默认: 0.3)")
    parser.add_argument("--fit_end", type=float, default=0.9,
                        help="拟合终止位置 (0~1 比例, 默认: 0.9)")
    
    # === v2.1 MSD 方法参数 ===
    parser.add_argument("--time_origin", choices=['single', 'multi'], default='multi',
                        help="时间原点模式: multi (MTO, 默认) / single (t0=0 旧版)")
    parser.add_argument("--msd_method", choices=['mto', 'single_origin'], default=None,
                        help="[兼容参数] 等价于 --time_origin (mto=multi, single_origin=single)")
    parser.add_argument("--n_origins", type=int, default=20,
                        help="MTO 模式时间原点数 (默认: 20)")
    parser.add_argument("--origin_stride", type=int, default=None,
                        help="时间原点间隔帧数 (若提供则覆盖 n_origins)")
    parser.add_argument("--stride", type=int, default=1,
                        help="MTO lag 步进 (默认: 1; 设 2/5 可降低计算量)")
    parser.add_argument("--max_lag_ps", type=float, default=None,
                        help="MTO 最大 lag (ps, 默认: min(总时长×0.5, 10))")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="每个 lag 最少样本数 (默认: 5)")
    
    # === v2.1 Unwrap 检查 ===
    parser.add_argument("--unwrap_check", dest='unwrap_check', action='store_true', default=True,
                        help="启用 unwrap 一致性检查与跳变检测 (默认)")
    parser.add_argument("--no_unwrap_check", dest='unwrap_check', action='store_false',
                        help="禁用 unwrap 一致性检查")
    
    # === v2.1 新增：COM 漂移 ===
    parser.add_argument(
        "--remove_com",
        choices=['none', 'all', 'selected', 'all_linear', 'selected_linear'],
        default='all',
        help="COM 漂移去除: none/all/selected/all_linear/selected_linear (默认: all)"
    )
    
    # === v2.1 新增：Running-D 模式 ===
    parser.add_argument("--runningD", choices=['ratio', 'derivative', 'both'], default='both',
                        help="Running-D 计算方式 (默认: both)")
    
    # === v2.1 新增：平台判定 ===
    parser.add_argument("--plateau_method", choices=['D_derivative', 'alpha', 'both'], default='both',
                        help="平台判定方法 (默认: both)")
    parser.add_argument("--alpha_window", type=int, default=21,
                        help="α(t) 滑窗大小 (点数, 默认: 21)")
    
    # === v2.1 改进：误差估计 ===
    parser.add_argument("--n_blocks", type=int, default=5,
                        help="Trajectory blocks 分块数 (默认: 5)")
    parser.add_argument("--block_mode", choices=['trajectory_blocks', 'bootstrap'], default='trajectory_blocks',
                        help="误差估计方法 (默认: trajectory_blocks)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Bootstrap 随机种子")
    parser.add_argument("--n_bootstrap", type=int, default=100,
                        help="Bootstrap 重采样次数 (默认: 100)")
    
    # === v2.2 新增：物理安全和防护 ===
    parser.add_argument("--no_strict", dest='strict', action='store_false',
                        help="禁用严格模式（亚扩散时不再 exit(2)）")
    parser.add_argument("--strict", dest='strict', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument("--allow_unreliable_D", action='store_true', default=False,
                        help="Allow outputting D even when regime is subdiffusive/unreliable")
    parser.add_argument("--cell_source", choices=['xdatcar'], default='xdatcar',
                        help="Cell source: xdatcar (constant cell assumption, default)")
    parser.add_argument("--cell_rel_std_tol", type=float, default=0.01,
                        help="OUTCAR 变胞检测相对标准差阈值 (默认: 0.01)")
    parser.add_argument("--no_cell_angle_check", dest='cell_angle_check', action='store_false',
                        help="变胞检测时不使用晶胞角度波动判据")
    parser.add_argument("--min_block_time_ps", type=float, default=5.0,
                        help="Minimum block duration for error estimation (ps, default: 5.0)")
    parser.add_argument("--com_selection", default=None,
                        help="COM 参考物种，可用逗号分隔多物种（remove_com=selected* 时）")
    parser.add_argument("--com_index_file", default=None,
                        help="COM 参考原子索引文件（覆盖 --com_selection）")
    parser.add_argument("--duplicate_tol", type=float, default=1e-12,
                        help="连续重复帧判定阈值（分数坐标, 默认: 1e-12）")
    
    # 输出
    parser.add_argument("--outdir", default=".",
                        help="输出目录 (默认: 当前目录)")

    args = parser.parse_args(remaining_argv)

    if args.stride <= 0:
        print("[ERROR] --stride 必须 > 0")
        sys.exit(1)
    if args.n_blocks <= 0:
        print("[ERROR] --n_blocks 必须 > 0")
        sys.exit(1)
    if args.duplicate_tol < 0:
        print("[ERROR] --duplicate_tol 必须 >= 0")
        sys.exit(1)
    if args.cell_rel_std_tol < 0:
        print("[ERROR] --cell_rel_std_tol 必须 >= 0")
        sys.exit(1)
    if args.specie is None:
        parser.error("--specie is required unless --self_check is used")
    if args.dt_fs is None:
        parser.error("--dt_fs is required unless --self_check is used")

    # 处理参数兼容: --msd_method (旧) vs --time_origin (新, 首选)
    if args.msd_method is not None:
        # 旧参数指定时优先使用
        msd_method = args.msd_method
    else:
        # 新参数: time_origin 映射到 msd_method
        msd_method = 'mto' if args.time_origin == 'multi' else 'single_origin'
    
    print("=" * 70)
    print(f"aimd_msd.py {VERSION} - MSD 计算与扩散系数拟合")
    print("=" * 70)
    print(f"XDATCAR: {args.xdatcar}")
    print(f"物种: {args.specie}")
    print(f"dt: {args.dt_fs} fs")
    print(f"MSD 方法: {msd_method}")
    if msd_method == 'mto':
        print(f"stride: {args.stride}")
    print(f"COM 漂移去除: {args.remove_com}")
    print(f"Unwrap 检查: {'启用' if args.unwrap_check else '禁用'}")
    print(f"严格模式: {'启用' if args.strict else '禁用'}")
    print("=" * 70)

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    strict_flags = {
        'strict': bool(args.strict),
        'allow_unreliable_D': bool(args.allow_unreliable_D),
    }

    # 1. 读取 XDATCAR
    print("\n>>> 读取 XDATCAR...")
    lattice, species, counts, frames = parse_xdatcar(args.xdatcar)
    print(f"    物种: {species}")
    print(f"    原子数: {counts} (总计 {sum(counts)})")
    print(f"    总帧数: {len(frames)}")
    
    # 1.5 v2.2: NPT/Variable cell detection
    xdatcar_dir = os.path.dirname(os.path.abspath(args.xdatcar))
    outcar_path = os.path.join(xdatcar_dir, 'OUTCAR')
    cell_check = analyze_cell_variability(
        outcar_path,
        tolerance=args.cell_rel_std_tol,
        check_angles=args.cell_angle_check
    )
    print(f"    Cell check: {cell_check['message']}")
    
    if cell_check['is_variable']:
        summary_line = format_summary_line(
            status='error',
            verdict='not_run',
            D_mean=np.nan,
            D_std=np.nan,
            D_sem=np.nan,
            alpha_mean=np.nan,
            alpha_std=np.nan,
            drift_ratio=0.0,
            fit_start_ps=0.0,
            fit_end_ps=0.0,
            skip_ps_used=0.0,
            stride=args.stride,
            max_lag_ps=0.0,
            strict=args.strict,
            allow_unreliable_D=args.allow_unreliable_D,
            cell_is_variable=cell_check['is_variable'],
            cell_rel_std=cell_check['max_rel_std'],
            cell_rel_std_tol=args.cell_rel_std_tol,
            duplicate_frames_removed=0,
            reason='variable_cell',
        )
        print("\n" + "!" * 70)
        print("[ERROR] Variable cell detected (NPT/ISIF=3)!")
        print(f"[ERROR] {cell_check['message']}")
        print("[ERROR] This script assumes constant cell volume (NVT/NVE).")
        print("[ERROR] Variable cell MSD requires fractional-coordinate MIC with")
        print("[ERROR] reference cell mapping. Pre-process trajectory or use")
        print("[ERROR] compatible analysis tool.")
        print("!" * 70)
        write_abort_report(
            args.specie,
            args.dt_fs,
            abort_reason='variable_cell',
            strict_flags=strict_flags,
            cell_check={
                'tolerance': args.cell_rel_std_tol,
                'check_angles': args.cell_angle_check,
                'message': cell_check['message'],
            },
            summary_line=summary_line,
            outdir=args.outdir
        )
        print(f"    [OK] {os.path.join(args.outdir, 'msd_report.txt')}")
        print(summary_line)
        sys.exit(1)

    # 2. 跳过初始帧
    if args.skip >= len(frames):
        print(f"[ERROR] skip ({args.skip}) >= 总帧数 ({len(frames)})")
        sys.exit(1)

    frames = frames[args.skip:]
    nframes = len(frames)
    print(f"    有效帧数: {nframes} (跳过 {args.skip})")

    # 2.5 连续重复帧去重（常见于 XDATCAR stitching）
    frames, removed_dup_pos = remove_consecutive_duplicate_frames(frames, atol=args.duplicate_tol)
    removed_global_pos = [args.skip + p for p in removed_dup_pos]
    duplicate_summary = {
        'count': len(removed_dup_pos),
        'global_positions': removed_global_pos,
        'sample_text': format_positions_sample(removed_global_pos),
    }
    if len(removed_dup_pos) > 0:
        print(f"    [WARN] Removed {len(removed_dup_pos)} consecutive duplicate frames")
        print(
            "    [WARN] Removed frame positions (global): "
            f"{duplicate_summary['sample_text']}"
        )
    nframes = len(frames)
    if nframes < 2:
        print("[ERROR] Not enough frames after duplicate-frame removal")
        sys.exit(1)
    print(f"    去重后有效帧数: {nframes}")

    # 3. 获取目标物种索引
    print(f"\n>>> 查找物种 {args.specie}...")
    indices = get_species_indices(species, counts, args.specie)
    print(f"    找到 {len(indices)} 个 {args.specie} 原子")

    # 4. Unwrap 轨迹（带诊断）
    print("\n>>> Unwrapping 轨迹...")
    unwrapped, unwrap_diag = unwrap_trajectory_robust(
        frames, lattice, check_frac_consistency=args.unwrap_check
    )
    print(f"    最大单步位移: {unwrap_diag['max_jump_A']:.4f} Å")
    
    if unwrap_diag.get('is_vacuum_system', False):
        print("    [INFO] 检测到可能是 cluster/vacuum 体系（原子未跨 PBC）")
    
    if unwrap_diag['has_warnings']:
        print(f"    [WARN] 检测到 {unwrap_diag['n_suspicious_jumps']} 个可疑跳变！")
        print(f"    [WARN] 请检查 XDATCAR / POTIM / 体系设置")
    
    if args.unwrap_check and unwrap_diag.get('frac_consistency_issues', 0) > 0:
        n_issues = unwrap_diag['frac_consistency_issues']
        print(f"    [INFO] 检测到 {n_issues} 次分数坐标 |d|>0.5 跳跃（已通过 MIC 修正）")

    # 5. COM 漂移去除
    print(f"\n>>> COM 漂移处理 (mode={args.remove_com})...")
    if args.remove_com in ('all', 'selected'):
        print("    [WARN] Per-frame COM anchoring may suppress legitimate MSD fluctuations")
        print("    [WARN] Consider --remove_com all_linear/selected_linear for drift-only correction")

    uses_selected_reference = args.remove_com in ('selected', 'selected_linear')
    natoms_total = sum(counts)
    com_reference = {
        'source': 'not_used',
        'description': f"remove_com={args.remove_com}",
    }

    if uses_selected_reference:
        if args.com_index_file is not None:
            com_indices = read_index_file(args.com_index_file, natoms_total)
            print(f"    COM 参考: index file {args.com_index_file} ({len(com_indices)} 原子)")
            com_reference = {
                'source': 'index_file',
                'description': f"path={args.com_index_file}, natoms={len(com_indices)}",
            }
        elif args.com_selection is not None:
            try:
                com_indices = get_species_indices_multi(species, counts, args.com_selection)
                print(f"    COM 参考物种: {args.com_selection} ({len(com_indices)} 原子)")
                com_reference = {
                    'source': 'species_list',
                    'description': f"selection={args.com_selection}, natoms={len(com_indices)}",
                }
            except SystemExit:
                print(f"[ERROR] Invalid --com_selection: {args.com_selection}")
                sys.exit(1)
        else:
            print("    [WARN] selected COM mode uses target species as COM reference")
            print("    [WARN] This may suppress target-species diffusion signal")
            print("    [WARN] Prefer scaffold reference via --com_selection or --com_index_file")
            com_indices = indices
            com_reference = {
                'source': 'fallback_target',
                'description': f"specie={args.specie}, natoms={len(com_indices)}",
            }
    else:
        com_indices = None
        if args.remove_com == 'none':
            com_reference = {
                'source': 'not_used',
                'description': "remove_com=none",
            }
        else:
            com_reference = {
                'source': 'all_atoms',
                'description': f"remove_com={args.remove_com}, natoms={natoms_total}",
            }

    if args.com_index_file is not None and not uses_selected_reference:
        print("    [WARN] --com_index_file is ignored unless remove_com is selected/selected_linear")
    if args.com_selection is not None and not uses_selected_reference:
        print("    [WARN] --com_selection is ignored unless remove_com is selected/selected_linear")
    
    unwrapped, com_drift = remove_com_drift(unwrapped, lattice, args.remove_com, com_indices)
    
    if args.remove_com != 'none':
        print(f"    COM 漂移: {np.linalg.norm(com_drift):.4f} Å")

    # 6. 时间轴
    dt_ps = args.dt_fs / 1000.0
    t_ps_full = np.arange(nframes) * dt_ps
    t_total_ps = t_ps_full[-1]

    # 7. 计算 MSD
    print(f"\n>>> 计算 MSD (method={msd_method})...")
    
    if msd_method == 'single_origin':
        # 旧版兼容：单一时间原点
        msd = compute_msd_single_origin(unwrapped, lattice, indices)
        t_ps = t_ps_full
        n_samples = np.ones(len(msd), dtype=int)  # 每个点只有1个样本
        lag_frames_used = np.arange(len(msd), dtype=int)
        n_origins_used = 1
        max_lag_used = nframes - 1
        max_lag_frames_used = nframes - 1
        print(f"    模式: 单一时间原点 (t0=0)")
        print(f"    [WARN] single_origin 仅用于快速检查/对比，推荐使用 --msd_method mto")
    else:
        # MTO 模式
        # 智能 max_lag 默认: min(总时长×0.5, 10 ps)
        if args.max_lag_ps is not None:
            max_lag_ps_target = args.max_lag_ps
        else:
            max_lag_ps_target = min(t_total_ps * 0.5, 10.0)
        
        max_lag_frames = int(max_lag_ps_target / dt_ps)
        max_lag_frames = max(1, min(max_lag_frames, nframes - 1))
        lag_frames_requested, _ = build_time_axis_and_lags(max_lag_frames, args.stride, dt_ps)
        
        print(f"    max_lag: {max_lag_ps_target:.2f} ps ({max_lag_frames} 帧)")
        print(f"    stride: {args.stride}")
        
        origins = select_time_origins(
            nframes,
            n_origins=args.n_origins,
            origin_stride=args.origin_stride,
            max_lag=max_lag_frames
        )
        
        msd, n_samples, effective_max_lag_frames = compute_msd_mto(
            unwrapped,
            lattice,
            indices,
            origins,
            max_lag=max_lag_frames,
            lags=lag_frames_requested,
            min_samples=args.min_samples
        )
        if len(msd) == 0:
            print("[ERROR] MTO MSD has no valid lag points under current --min_samples/--max_lag_ps")
            sys.exit(1)

        lag_frames_used = lag_frames_requested[lag_frames_requested <= effective_max_lag_frames]
        if len(lag_frames_used) != len(msd):
            lag_frames_used = lag_frames_used[:len(msd)]
        t_ps = lag_frames_used * dt_ps
        
        n_origins_used = len(origins)
        max_lag_used = len(msd) - 1
        max_lag_frames_used = int(lag_frames_used[-1])
        
        print(f"    模式: Multiple Time Origins (MTO)")
        print(f"    时间原点数: {n_origins_used}")
        print(f"    有效最大 lag: {max_lag_frames_used} 帧 ({t_ps[-1] if len(t_ps) > 0 else 0:.3f} ps)")
        if len(n_samples) > 0 and np.any(n_samples > 0):
            print(f"    最小样本数: {np.min(n_samples[n_samples > 0])}")

    # 8. 计算 Running-D
    print("\n>>> 计算 Running-D...")
    D_ratio = compute_running_D_ratio(t_ps, msd)
    D_deriv = compute_running_D_derivative(t_ps, msd)
    print(f"    D_ratio: MSD/(6t)")
    print(f"    D_deriv: (1/6) d(MSD)/dt")

    # 9. 计算 α(t)
    print("\n>>> 计算 log-log 斜率 α(t)...")
    t_alpha, alpha = compute_alpha(t_ps, msd, window=args.alpha_window)
    if len(t_alpha) > 0:
        valid_alpha = alpha[~np.isnan(alpha)]
        if len(valid_alpha) > 0:
            print(f"    α 范围: {np.min(valid_alpha):.2f} ~ {np.max(valid_alpha):.2f}")
        else:
            print(f"    [WARN] 无法计算有效 α")
    else:
        print(f"    [WARN] 数据点不足，无法计算 α")

    # 10. 确定拟合区间
    print("\n>>> 确定拟合区间...")
    
    npts = len(msd)
    requested_skip_idx = ps_to_index(args.t_skip_ps, t_ps, side='left')
    skip_cap_idx = npts // 4
    skip_idx = max(1, min(requested_skip_idx, skip_cap_idx))
    skip_window_note = None
    skip_ps_used = float(t_ps[min(skip_idx, len(t_ps) - 1)])
    if requested_skip_idx > skip_cap_idx:
        cap_ps = float(t_ps[min(skip_cap_idx, len(t_ps) - 1)])
        skip_window_note = (
            f"[WARN] Requested --t_skip_ps={args.t_skip_ps:.3f} ps was clamped to "
            f"{skip_ps_used:.3f} ps by the <=25% lag-window safety cap "
            f"(cap_time={cap_ps:.3f} ps)."
        )
        print(f"    {skip_window_note}")
    
    # 拟合起点
    if args.t_fit_start_ps is not None:
        fit_start_idx = ps_to_index(args.t_fit_start_ps, t_ps, side='left')
    else:
        fit_start_idx = int(npts * args.fit_start)
    fit_start_idx = max(skip_idx, fit_start_idx)
    
    # 拟合终点
    if args.t_fit_end_ps is not None:
        fit_end_idx = ps_to_index(args.t_fit_end_ps, t_ps, side='right')
    else:
        fit_end_idx = int(npts * args.fit_end)
    fit_end_idx = min(npts, fit_end_idx)
    
    # 检查拟合区间
    if fit_end_idx <= fit_start_idx + 5:
        print("[WARN] 用户指定的拟合区间太短，自动调整")
        fit_start_idx = max(skip_idx, npts // 4)
        fit_end_idx = min(npts, npts * 9 // 10)
    
    fit_start_ps = t_ps[fit_start_idx] if fit_start_idx < len(t_ps) else 0
    fit_end_ps = t_ps[fit_end_idx - 1] if fit_end_idx > 0 and fit_end_idx <= len(t_ps) else t_ps[-1]
    
    print(f"    弹道区跳过: {skip_ps_used:.3f} ps")
    print(f"    拟合区间: {fit_start_ps:.3f} ~ {fit_end_ps:.3f} ps")
    print(f"    拟合点数: {fit_end_idx - fit_start_idx}")

    # 11. 平台判定
    print("\n>>> 平台判定...")
    verdict, drift_ratio, alpha_mean, alpha_std, plateau_msg = check_plateau_combined(
        D_deriv, t_ps, alpha, t_alpha, fit_start_idx, fit_end_idx
    )
    print(f"    {plateau_msg}")
    
    if verdict not in ['credible']:
        print("")
        print("!" * 70)
        if verdict == 'subdiffusive':
            print("!!! 警告: 检测到亚扩散行为 (α < 0.8) !!!")
            print("!!! 可能存在 caging 或受限运动，D 值仅供参考 !!!")
        elif verdict == 'superdiffusive':
            print("!!! 警告: 检测到超扩散行为 (α > 1.2) !!!")
            print("!!! 可能处于早期弹道区或存在数值漂移 !!!")
        elif verdict == 'drifting':
            print("!!! 警告: D(t) 漂移严重 !!!")
            print("!!! AIMD 模拟时间可能不足 !!!")
        else:
            print("!!! 警告: 扩散系数可能不可靠 !!!")
        print("!" * 70)

    # 12. 扩散系数估算
    print("\n>>> 估算扩散系数...")
    
    # 全局线性拟合（用于绘图和基准）
    t_fit = t_ps[fit_start_idx:fit_end_idx]
    msd_fit = msd[fit_start_idx:fit_end_idx]
    intercept, slope, r2 = linear_fit(t_fit, msd_fit)
    D_global = slope / 6.0 * 1e-4
    fit_start_ps_used = float(t_ps[min(fit_start_idx, len(t_ps) - 1)])
    fit_end_ps_used = float(t_ps[fit_end_idx - 1]) if fit_end_idx > 0 else float(t_ps[-1])
    
    # 误差估计
    if args.block_mode == 'trajectory_blocks':
        print(f"    误差方法: Trajectory Blocks (n_blocks={args.n_blocks})")
        D_mean, D_std, D_sem, D_blocks = estimate_D_with_trajectory_blocks(
            unwrapped, lattice, indices, dt_ps, args.n_blocks,
            args.fit_start, args.fit_end, n_origins_per_block=10,
            min_block_time_ps=args.min_block_time_ps,
            fit_start_ps=fit_start_ps_used,
            fit_end_ps=fit_end_ps_used,
            skip_ps=skip_ps_used,
            stride=args.stride if msd_method == 'mto' else 1,
            max_lag_frames=max_lag_frames_used,
            min_samples=args.min_samples
        )
        
        # v2.2: NaN check (blocks too short returns NaN)
        if np.isnan(D_mean) or len(D_blocks) == 0:
            print("    [WARN] Block 方法失败，回退到全局拟合")
            D_mean = D_global
            D_std = 0.0
            D_sem = 0.0
        else:
            print(f"    有效 blocks: {len(D_blocks)}")
    else:
        print(f"    误差方法: Bootstrap (n={args.n_bootstrap}, seed={args.seed})")
        if msd_method == 'mto':
            origins = select_time_origins(
                nframes, args.n_origins, args.origin_stride, max_lag_frames_used
            )
            D_mean, D_std, D_ci_low, D_ci_high = estimate_D_with_bootstrap(
                unwrapped,
                lattice,
                indices,
                dt_ps,
                origins,
                lag_frames_used,
                fit_start_idx,
                fit_end_idx,
                args.n_bootstrap,
                args.seed,
                min_samples=args.min_samples
            )
            if np.isnan(D_mean):
                print("    [WARN] Bootstrap 有效重采样不足，回退到全局拟合")
                D_mean = D_global
                D_std = 0.0
                D_sem = 0.0
            else:
                D_sem = D_std / np.sqrt(args.n_bootstrap)
                print(f"    95%% CI: [{D_ci_low:.6e}, {D_ci_high:.6e}] cm²/s")
        else:
            print("    [WARN] Bootstrap 需要 MTO 模式，回退到全局拟合")
            D_mean = D_global
            D_std = 0.0
            D_sem = 0.0

    # 13. 保存数据
    print("\n>>> 保存数据...")
    
    # MSD 数据
    msd_path = os.path.join(args.outdir, f"msd_{args.specie}.dat")
    save_msd_data(msd_path, t_ps, msd, n_samples if msd_method == 'mto' else None)
    print(f"    [OK] {msd_path}")
    
    # Running-D 数据
    D_path = os.path.join(args.outdir, f"D_running_{args.specie}.dat")
    save_running_D_data(D_path, t_ps, D_ratio, D_deriv)
    print(f"    [OK] {D_path}")
    
    # Alpha 数据
    if len(t_alpha) > 0:
        alpha_path = os.path.join(args.outdir, f"alpha_{args.specie}.dat")
        save_alpha_data(alpha_path, t_alpha, alpha)
        print(f"    [OK] {alpha_path}")

    # 14. 绘图
    print("\n>>> 绘图...")
    plot_msd(t_ps, msd, args.specie, fit_start_idx, fit_end_idx, slope, intercept, args.outdir)
    plot_running_D(t_ps, D_ratio, D_deriv, args.specie, fit_start_idx, fit_end_idx, D_mean, D_std, args.outdir)
    plot_alpha(t_alpha, alpha, args.specie, fit_start_ps, fit_end_ps, args.outdir)
    
    if HAS_MPL:
        print(f"    [OK] msd_{args.specie}.png")
        print(f"    [OK] D_running_{args.specie}.png")
        if len(t_alpha) > 0:
            print(f"    [OK] alpha_{args.specie}.png")
    else:
        print("    [SKIP] matplotlib 未安装，跳过绘图")

    # 15. 写入报告
    max_lag_ps_used = float(t_ps[-1]) if len(t_ps) > 0 else 0.0
    
    # 确定时间原点模式名称
    time_origin_mode_str = 'multi (MTO)' if msd_method == 'mto' else 'single (t0=0)'
    is_subdiffusive = verdict in ['subdiffusive', 'unreliable']
    is_strict = args.strict and not args.allow_unreliable_D
    summary_D_mean = np.nan if (is_subdiffusive and is_strict) else D_mean
    summary_D_std = np.nan if (is_subdiffusive and is_strict) else D_std
    summary_D_sem = np.nan if (is_subdiffusive and is_strict) else D_sem
    summary_line = format_summary_line(
        status='ok',
        verdict=verdict,
        D_mean=summary_D_mean,
        D_std=summary_D_std,
        D_sem=summary_D_sem,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        drift_ratio=drift_ratio,
        fit_start_ps=fit_start_ps,
        fit_end_ps=fit_end_ps,
        skip_ps_used=skip_ps_used,
        stride=args.stride,
        max_lag_ps=max_lag_ps_used,
        strict=args.strict,
        allow_unreliable_D=args.allow_unreliable_D,
        cell_is_variable=cell_check['is_variable'],
        cell_rel_std=cell_check['max_rel_std'],
        cell_rel_std_tol=args.cell_rel_std_tol,
        duplicate_frames_removed=duplicate_summary['count'],
    )
    
    write_report(
        args.specie, nframes, args.dt_fs, t_total_ps,
        fit_start_ps, fit_end_ps,
        D_mean, D_std, D_sem, r2,
        verdict, drift_ratio, alpha_mean, alpha_std, plateau_msg,
        time_origin_mode_str, n_origins_used, max_lag_ps_used, args.min_samples,
        args.remove_com, com_drift,
        args.block_mode, args.n_blocks,
        unwrap_diag,
        {
            'tolerance': args.cell_rel_std_tol,
            'check_angles': args.cell_angle_check,
            'message': cell_check['message'],
        },
        duplicate_summary,
        com_reference,
        strict_flags,
        skip_window_note,
        summary_line,
        args.outdir
    )
    print(f"    [OK] msd_report.txt")

    # 16. 输出摘要
    print("\n" + "=" * 70)
    print("计算结果摘要")
    print("=" * 70)
    print(f"总帧数: {nframes}")
    print(f"时间步长 dt: {args.dt_fs} fs = {dt_ps} ps")
    print(f"总模拟时间: {t_total_ps:.3f} ps")
    print(f"时间原点模式: {time_origin_mode_str} (n_origins={n_origins_used})")
    print(f"MSD 末值: {msd[-1]:.4f} Å²")
    print(f"拟合区间: {fit_start_ps:.3f} ~ {fit_end_ps:.3f} ps")
    print(f"拟合 R²: {r2:.6f}")
    print("-" * 70)
    
    if is_subdiffusive and is_strict:
        print(f"扩散系数 D = NaN (D_UNDEFINED: α<0.8 subdiffusive regime)")
        print(f"           [STRICT MODE] D is not well-defined for subdiffusive motion")
    else:
        if is_subdiffusive:
            print(f"扩散系数 D = ({D_mean:.4e} ± {D_std:.4e}) cm²/s (±STD) [UNRELIABLE]")
            print(f"           = ({D_mean:.4e} ± {D_sem:.4e}) cm²/s (±SEM) [UNRELIABLE]")
        else:
            print(f"扩散系数 D = ({D_mean:.4e} ± {D_std:.4e}) cm²/s (±STD)")
            print(f"           = ({D_mean:.4e} ± {D_sem:.4e}) cm²/s (±SEM)")
    
    print("-" * 70)
    print(f"平台判定: {verdict}")
    print(f"D_deriv 漂移率: {drift_ratio*100:.1f}%%")
    if not np.isnan(alpha_mean):
        print(f"α (拟合区间): {alpha_mean:.3f} ± {alpha_std:.3f}")
        regime, regime_desc = interpret_alpha(alpha_mean, alpha_std)
        print(f"扩散状态: {regime_desc}")
    print("=" * 70)

    print(summary_line)
    
    if verdict not in ['credible']:
        print("\n[WARN] 扩散系数可能不可靠，请检查报告详情！")
        print("[INFO] 建议: 延长 AIMD 模拟时间 / 检查体系平衡状态")
    
    # v2.2: Exit code logic for subdiffusion guardrail
    if is_subdiffusive and is_strict:
        print("\n" + "!" * 70)
        print("[STRICT MODE] Exiting with code 2: D is undefined under subdiffusion")
        print("[STRICT MODE] To output D anyway, use: --allow_unreliable_D")
        print("[STRICT MODE] To disable strict mode: --no_strict")
        print("!" * 70)
        sys.exit(2)


if __name__ == "__main__":
    main()
