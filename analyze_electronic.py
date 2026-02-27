#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_electronic.py - VASP 电子性质后处理 (v2.2)

功能：
  - 功函数 (wf): 从 LOCPOT/OUTCAR 计算功函数，绘制真空电势剖面
  - DOS (dos): 解析 DOSCAR，绘制 total DOS（支持自旋极化）
  - 自检 (self_check): 合成数据验证真空平台检测与退出码

v2.2 关键改进：
  - 无真空平台时默认硬失败（退出码 2），避免静默输出错误功函数
  - 新增 --allow_fraction_fallback 显式启用旧式端点平均回退
  - 平台检测改为基于坐标梯度 eV/Å + 连续 run-length 检测
  - 平台检测参数可配置：--plateau_grad_tol_evA / --plateau_end_fraction / --plateau_min_points
  - DOSCAR 解析更鲁棒：容忍空行并验证 NEDOS 数据完整性
  - 新增 wf_summary.json / dos_summary.json 机器可读摘要

依赖：
  pip install numpy matplotlib
"""

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# 检查 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 无头模式
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，无法生成图片")
    print("[INFO] 安装: pip install matplotlib")


SCRIPT_VERSION = "v2.2"
AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}
UNRELIABLE_FALLBACK_TAG = "UNRELIABLE_FALLBACK"


class PlateauDetectionError(RuntimeError):
    """Raised when vacuum plateau detection fails under strict mode."""


def axis_name(idx: int) -> str:
    return ['x', 'y', 'z'][idx]


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write('\n')


def format_plateau_summary(plateau: Dict[str, Any]) -> str:
    side = plateau['side']
    if not plateau['found']:
        return (
            f"{side}: NOT_FOUND "
            f"(search_idx={plateau['search_start_idx']}-{plateau['search_end_idx']}, "
            f"search_coord={plateau['search_start_coord']:.4f}-{plateau['search_end_coord']:.4f} Å)"
        )
    return (
        f"{side}: idx={plateau['start_idx']}-{plateau['end_idx']}, "
        f"coord={plateau['start_coord']:.4f}-{plateau['end_coord']:.4f} Å, "
        f"n={plateau['npoints']}, V={plateau['v_vac']:.6f} eV"
    )


# =============================================================================
# OUTCAR Parsing
# =============================================================================

def parse_efermi(outcar_path: str) -> float:
    """
    从 OUTCAR 解析费米能级 E-fermi
    """
    if not os.path.isfile(outcar_path):
        raise FileNotFoundError(f"OUTCAR 不存在: {outcar_path}")

    efermi = None
    pattern = re.compile(r'E-fermi\s*:\s*([-+]?\d+\.?\d*)')

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                efermi = float(match.group(1))

    if efermi is None:
        raise ValueError("未在 OUTCAR 中找到 E-fermi")

    return efermi


# =============================================================================
# LOCPOT Parsing
# =============================================================================

def parse_locpot(locpot_path: str, axis: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    解析 LOCPOT 文件

    支持：
      - VASP4/VASP5 格式（有/无元素名行）
      - Selective dynamics
      - 空行处理
      - 数据长度验证
    """
    if not os.path.isfile(locpot_path):
        raise FileNotFoundError(f"LOCPOT 不存在: {locpot_path}")

    with open(locpot_path, 'r', encoding='utf-8', errors='ignore') as handle:
        lines = handle.readlines()

    if len(lines) < 9:
        raise ValueError(f"LOCPOT 文件过短，无法解析: {locpot_path}")

    scale = float(lines[1].strip())
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    # 使用向量范数支持非正交晶胞
    cell_length = np.linalg.norm(lattice[axis])

    idx = 5
    first_tokens = lines[idx].strip().split()
    if not first_tokens:
        raise ValueError("LOCPOT 原子类型/数量行为空，无法解析")

    if not first_tokens[0].isdigit():
        idx += 1

    atom_counts = [int(x) for x in lines[idx].split()]
    natoms = sum(atom_counts)
    idx += 1

    coord_line = lines[idx].strip()
    if coord_line and coord_line[0].upper() == 'S':
        idx += 1
        coord_line = lines[idx].strip()

    if coord_line and coord_line[0].upper() in 'CD':
        idx += 1

    idx += natoms

    while idx < len(lines) and lines[idx].strip() == '':
        idx += 1

    if idx >= len(lines):
        raise ValueError("LOCPOT 中未找到网格大小行")

    grid_parts = lines[idx].strip().split()
    if len(grid_parts) < 3:
        raise ValueError(f"无法解析网格大小行: {lines[idx]}")

    try:
        nx, ny, nz = int(grid_parts[0]), int(grid_parts[1]), int(grid_parts[2])
    except ValueError as exc:
        raise ValueError(f"网格大小解析失败: {grid_parts}") from exc

    idx += 1
    expected_data = nx * ny * nz

    potential_data = []
    for line in lines[idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith('augmentation'):
            break
        try:
            potential_data.extend(float(x) for x in stripped.split())
        except ValueError:
            break

    if len(potential_data) < expected_data:
        raise ValueError(
            f"LOCPOT 数据不足: 期望 {expected_data} 点 ({nx}×{ny}×{nz}), "
            f"实际 {len(potential_data)} 点"
        )

    if len(potential_data) > expected_data:
        print(f"[WARN] LOCPOT 数据超出网格大小，截断 {len(potential_data)} -> {expected_data}")
        potential_data = potential_data[:expected_data]

    potential_3d = np.array(potential_data).reshape((nz, ny, nx))

    if axis == 0:
        planar_avg = np.mean(potential_3d, axis=(0, 1))
        npts = nx
    elif axis == 1:
        planar_avg = np.mean(potential_3d, axis=(0, 2))
        npts = ny
    else:
        planar_avg = np.mean(potential_3d, axis=(1, 2))
        npts = nz

    coords = np.linspace(0, cell_length, npts, endpoint=False)
    return coords, planar_avg, lattice, cell_length


# =============================================================================
# Vacuum Plateau Detection
# =============================================================================

def _longest_true_run(mask: np.ndarray, prefer: str) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    start = None

    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
            continue
        if not flag and start is not None:
            candidate = (start, idx - 1)
            best = _select_better_run(best, candidate, prefer)
            start = None

    if start is not None:
        candidate = (start, len(mask) - 1)
        best = _select_better_run(best, candidate, prefer)

    return best


def _select_better_run(current: Optional[Tuple[int, int]],
                       candidate: Tuple[int, int],
                       prefer: str) -> Tuple[int, int]:
    if current is None:
        return candidate

    current_len = current[1] - current[0] + 1
    candidate_len = candidate[1] - candidate[0] + 1

    if candidate_len > current_len:
        return candidate
    if candidate_len < current_len:
        return current

    if prefer == 'left':
        return candidate if candidate[0] < current[0] else current
    return candidate if candidate[1] > current[1] else current


def _build_plateau_result(side: str,
                          coords: np.ndarray,
                          planar_avg: np.ndarray,
                          abs_gradient: np.ndarray,
                          search_start: int,
                          search_end: int,
                          run: Optional[Tuple[int, int]],
                          min_points: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'side': side,
        'found': False,
        'search_start_idx': int(search_start),
        'search_end_idx': int(search_end),
        'search_start_coord': float(coords[search_start]),
        'search_end_coord': float(coords[search_end]),
        'start_idx': None,
        'end_idx': None,
        'start_coord': None,
        'end_coord': None,
        'npoints': 0,
        'v_vac': None,
        'mean_abs_grad_evA': None,
        'max_abs_grad_evA': None,
        'min_points_required': int(min_points),
    }

    if run is None:
        return result

    start_idx, end_idx = run
    npoints = end_idx - start_idx + 1
    if npoints < min_points:
        return result

    values = planar_avg[start_idx:end_idx + 1]
    grads = abs_gradient[start_idx:end_idx + 1]

    result.update({
        'found': True,
        'start_idx': int(start_idx),
        'end_idx': int(end_idx),
        'start_coord': float(coords[start_idx]),
        'end_coord': float(coords[end_idx]),
        'npoints': int(npoints),
        'v_vac': float(np.mean(values)),
        'mean_abs_grad_evA': float(np.mean(grads)),
        'max_abs_grad_evA': float(np.max(grads)),
    })
    return result


def find_vacuum_plateau(coords: np.ndarray,
                        planar_avg: np.ndarray,
                        grad_tol_evA: float = 0.05,
                        min_plateau_points: int = 5,
                        end_fraction: float = 0.2) -> Dict[str, Any]:
    """
    在电势曲线两端寻找连续真空平台。

    判据：
      - 梯度使用实际坐标：np.gradient(planar_avg, coords)，单位 eV/Å
      - 在每个端部搜索窗口内寻找 |grad| < tol 的最长连续 run
      - run 长度必须 >= min_plateau_points
    """
    if len(coords) != len(planar_avg):
        raise ValueError("coords 与 planar_avg 长度不一致")
    if len(planar_avg) == 0:
        raise ValueError("planar_avg 为空")
    if grad_tol_evA <= 0:
        raise ValueError("plateau_grad_tol_evA 必须 > 0")
    if min_plateau_points < 1:
        raise ValueError("plateau_min_points 必须 >= 1")
    if not 0 < end_fraction <= 0.5:
        raise ValueError("plateau_end_fraction 必须在 (0, 0.5] 范围内")

    npts = len(planar_avg)
    n_search = min(npts, max(min_plateau_points, int(np.ceil(npts * end_fraction))))

    if npts == 1:
        gradient = np.zeros_like(planar_avg)
    else:
        gradient = np.gradient(planar_avg, coords)
    abs_gradient = np.abs(gradient)

    left_search = (0, n_search - 1)
    right_search = (npts - n_search, npts - 1)

    left_mask = abs_gradient[left_search[0]:left_search[1] + 1] < grad_tol_evA
    right_mask = abs_gradient[right_search[0]:right_search[1] + 1] < grad_tol_evA

    left_run = _longest_true_run(left_mask, prefer='left')
    right_run = _longest_true_run(right_mask, prefer='right')

    left_global = None if left_run is None else (
        left_search[0] + left_run[0],
        left_search[0] + left_run[1],
    )
    right_global = None if right_run is None else (
        right_search[0] + right_run[0],
        right_search[0] + right_run[1],
    )

    return {
        'gradient_evA': gradient,
        'abs_gradient_evA': abs_gradient,
        'grad_tol_evA': float(grad_tol_evA),
        'min_plateau_points': int(min_plateau_points),
        'end_fraction': float(end_fraction),
        'n_search': int(n_search),
        'left': _build_plateau_result(
            'left', coords, planar_avg, abs_gradient,
            left_search[0], left_search[1], left_global, min_plateau_points
        ),
        'right': _build_plateau_result(
            'right', coords, planar_avg, abs_gradient,
            right_search[0], right_search[1], right_global, min_plateau_points
        ),
    }


def estimate_vacuum_potential(coords: np.ndarray,
                              planar_avg: np.ndarray,
                              fraction: float = 0.15,
                              both_sides: bool = True,
                              use_plateau: bool = True,
                              allow_fraction_fallback: bool = False,
                              plateau_grad_tol_evA: float = 0.05,
                              plateau_end_fraction: float = 0.2,
                              plateau_min_points: int = 5) -> Dict[str, Any]:
    """
    估计真空电势 V_vac。

    默认严格要求平台检测成功。仅在 allow_fraction_fallback=True 时
    才允许回退到传统端点平均。
    """
    result: Dict[str, Any] = {
        'v_vac': None,
        'v_vac_left': None,
        'v_vac_right': None,
        'method': None,
        'plateau_found': False,
        'plateau_found_left': False,
        'plateau_found_right': False,
        'fallback_used': False,
        'reliability_tag': 'PLATEAU_OK',
        'required_plateaus': 'both' if both_sides else 'right',
        'fraction_window': None,
        'plateau': None,
    }

    if fraction <= 0 or fraction > 0.5:
        raise ValueError("vac_fraction 必须在 (0, 0.5] 范围内")

    if use_plateau:
        plateau_info = find_vacuum_plateau(
            coords,
            planar_avg,
            grad_tol_evA=plateau_grad_tol_evA,
            min_plateau_points=plateau_min_points,
            end_fraction=plateau_end_fraction,
        )
        left = plateau_info['left']
        right = plateau_info['right']
        result['plateau'] = plateau_info
        result['v_vac_left'] = left['v_vac']
        result['v_vac_right'] = right['v_vac']
        result['plateau_found_left'] = left['found']
        result['plateau_found_right'] = right['found']

        if both_sides and left['found'] and right['found']:
            result['v_vac'] = (left['v_vac'] + right['v_vac']) / 2.0
            result['method'] = 'plateau_both'
            result['plateau_found'] = True
            return result

        if (not both_sides) and right['found']:
            result['v_vac'] = right['v_vac']
            result['method'] = 'plateau_right'
            result['plateau_found'] = True
            return result

    if not allow_fraction_fallback:
        return result

    npts = len(planar_avg)
    n_sample = min(npts, max(1, int(np.ceil(npts * fraction))))

    left_avg = float(np.mean(planar_avg[:n_sample]))
    right_avg = float(np.mean(planar_avg[-n_sample:]))

    result['v_vac_left'] = left_avg
    result['v_vac_right'] = right_avg
    result['fallback_used'] = True
    result['reliability_tag'] = UNRELIABLE_FALLBACK_TAG
    result['fraction_window'] = {
        'fraction': float(fraction),
        'npoints': int(n_sample),
        'left': {
            'start_idx': 0,
            'end_idx': n_sample - 1,
            'start_coord': float(coords[0]),
            'end_coord': float(coords[n_sample - 1]),
            'v_vac': left_avg,
        },
        'right': {
            'start_idx': npts - n_sample,
            'end_idx': npts - 1,
            'start_coord': float(coords[npts - n_sample]),
            'end_coord': float(coords[-1]),
            'v_vac': right_avg,
        },
    }

    if both_sides:
        result['v_vac'] = (left_avg + right_avg) / 2.0
        result['method'] = 'fraction_both'
    else:
        result['v_vac'] = right_avg
        result['method'] = 'fraction_right'

    return result


# =============================================================================
# Work Function Analysis
# =============================================================================

def _write_vacuum_profile(path: str,
                          coords: np.ndarray,
                          planar_avg: np.ndarray,
                          efermi: float,
                          phi: float,
                          phi_left: Optional[float],
                          phi_right: Optional[float],
                          axis: int,
                          functional_tag: str,
                          scissor_ev: Optional[float],
                          vac_result: Dict[str, Any]) -> None:
    plateau = vac_result.get('plateau')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(f"# Vacuum potential analysis ({SCRIPT_VERSION})\n")
        handle.write(f"# Functional: {functional_tag}\n")
        if scissor_ev is not None:
            handle.write(f"# Scissor correction: {scissor_ev} eV\n")
        handle.write(f"# Axis: {axis_name(axis)}\n")
        handle.write(f"# E-fermi = {efermi:.6f} eV\n")
        handle.write(f"# Reliability: {vac_result['reliability_tag']}\n")
        handle.write(f"# Required plateau(s): {vac_result['required_plateaus']}\n")
        if plateau is not None:
            handle.write(
                f"# Plateau settings: grad_tol={plateau['grad_tol_evA']:.6f} eV/Å, "
                f"end_fraction={plateau['end_fraction']:.6f}, "
                f"min_points={plateau['min_plateau_points']}\n"
            )
            handle.write(f"# Plateau left: {format_plateau_summary(plateau['left'])}\n")
            handle.write(f"# Plateau right: {format_plateau_summary(plateau['right'])}\n")
        if vac_result['fraction_window'] is not None:
            fw = vac_result['fraction_window']
            handle.write(
                f"# Fallback window: fraction={fw['fraction']:.6f}, "
                f"npoints={fw['npoints']} [{UNRELIABLE_FALLBACK_TAG}]\n"
            )
            handle.write(
                "# Fallback left: "
                f"idx={fw['left']['start_idx']}-{fw['left']['end_idx']}, "
                f"coord={fw['left']['start_coord']:.4f}-{fw['left']['end_coord']:.4f} Å\n"
            )
            handle.write(
                "# Fallback right: "
                f"idx={fw['right']['start_idx']}-{fw['right']['end_idx']}, "
                f"coord={fw['right']['start_coord']:.4f}-{fw['right']['end_coord']:.4f} Å\n"
            )
        handle.write(
            f"# V_vac = {vac_result['v_vac']:.6f} eV "
            f"(method: {vac_result['method']})\n"
        )
        if vac_result['v_vac_left'] is not None:
            handle.write(f"# V_vac_left = {vac_result['v_vac_left']:.6f} eV\n")
        if vac_result['v_vac_right'] is not None:
            handle.write(f"# V_vac_right = {vac_result['v_vac_right']:.6f} eV\n")
        handle.write(f"# Work function Phi = {phi:.6f} eV\n")
        if phi_left is not None:
            handle.write(f"# Phi_left = {phi_left:.6f} eV\n")
        if phi_right is not None:
            handle.write(f"# Phi_right = {phi_right:.6f} eV\n")
        handle.write(f"#\n")
        handle.write(f"# {axis_name(axis)} (Angstrom)    V({axis_name(axis)}) (eV)\n")
        for coord, value in zip(coords, planar_avg):
            handle.write(f"{coord:12.6f}  {value:12.6f}\n")


def _plot_work_function(calcdir: str,
                        coords: np.ndarray,
                        planar_avg: np.ndarray,
                        efermi: float,
                        phi: float,
                        axis: int,
                        functional_tag: str,
                        vac_result: Dict[str, Any]) -> None:
    if not HAS_MPL:
        return

    fig_path = os.path.join(calcdir, 'wf_profile.png')
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(coords, planar_avg, linewidth=1.5, color='blue')
    ax.axhline(y=efermi, linestyle='--', linewidth=1, color='red', label=f'E_F = {efermi:.2f} eV')
    ax.axhline(
        y=vac_result['v_vac'],
        linestyle=':',
        linewidth=1,
        color='green',
        label=f"V_vac = {vac_result['v_vac']:.2f} eV",
    )

    plateau = vac_result.get('plateau')
    if plateau is not None:
        for side in ('left', 'right'):
            entry = plateau[side]
            if entry['found']:
                ax.axvspan(
                    entry['start_coord'],
                    entry['end_coord'],
                    alpha=0.15,
                    color='green',
                )

    if vac_result['fraction_window'] is not None:
        fw = vac_result['fraction_window']
        ax.axvspan(
            fw['left']['start_coord'],
            fw['left']['end_coord'],
            alpha=0.10,
            color='orange',
        )
        ax.axvspan(
            fw['right']['start_coord'],
            fw['right']['end_coord'],
            alpha=0.10,
            color='orange',
        )

    ax.set_xlabel(f'{axis_name(axis)} (Å)')
    ax.set_ylabel('Planar Average Potential (eV)')
    title = f'Work Function Profile (Φ = {phi:.2f} eV)'
    if functional_tag != "PBE":
        title += f' [{functional_tag}]'
    if vac_result['fallback_used']:
        title += f' [{UNRELIABLE_FALLBACK_TAG}]'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f">>> 图已保存: {fig_path}")


def analyze_work_function(calcdir: str,
                          axis: int = 2,
                          both_sides: bool = True,
                          vac_fraction: float = 0.15,
                          use_plateau: bool = True,
                          allow_fraction_fallback: bool = False,
                          plateau_grad_tol_evA: float = 0.05,
                          plateau_end_fraction: float = 0.2,
                          plateau_min_points: int = 5,
                          functional_tag: str = "PBE",
                          scissor_ev: Optional[float] = None) -> Dict[str, Any]:
    """
    分析功函数。

    默认模式下，只有在检测到所需真空平台时才会输出功函数。
    """
    print("\n>>> 分析功函数...")
    print(f"    真空方向: {axis_name(axis)}")

    outcar_path = os.path.join(calcdir, 'OUTCAR')
    locpot_path = os.path.join(calcdir, 'LOCPOT')

    print(f"    读取 OUTCAR: {outcar_path}")
    efermi = parse_efermi(outcar_path)
    print(f"    E-fermi = {efermi:.4f} eV")

    print(f"    读取 LOCPOT: {locpot_path}")
    coords, planar_avg, lattice, cell_length = parse_locpot(locpot_path, axis)
    print(f"    Cell {axis_name(axis)} = {cell_length:.2f} Å (向量范数)")
    print(f"    Grid n{axis_name(axis)} = {len(coords)}")

    vac_result = estimate_vacuum_potential(
        coords,
        planar_avg,
        fraction=vac_fraction,
        both_sides=both_sides,
        use_plateau=use_plateau,
        allow_fraction_fallback=allow_fraction_fallback,
        plateau_grad_tol_evA=plateau_grad_tol_evA,
        plateau_end_fraction=plateau_end_fraction,
        plateau_min_points=plateau_min_points,
    )

    plateau = vac_result.get('plateau')
    if plateau is not None:
        print(
            "    平台参数: "
            f"grad_tol={plateau['grad_tol_evA']:.4f} eV/Å, "
            f"end_fraction={plateau['end_fraction']:.3f}, "
            f"min_points={plateau['min_plateau_points']}"
        )
        print(f"    左侧平台: {format_plateau_summary(plateau['left'])}")
        print(f"    右侧平台: {format_plateau_summary(plateau['right'])}")

    if vac_result['v_vac'] is None:
        required = "左右两端" if both_sides else "右侧"
        raise PlateauDetectionError(
            "\n".join([
                "未检测到满足要求的真空平台，已拒绝输出功函数。",
                f"要求平台: {required}",
                f"左侧结果: {format_plateau_summary(plateau['left']) if plateau else '未执行平台检测'}",
                f"右侧结果: {format_plateau_summary(plateau['right']) if plateau else '未执行平台检测'}",
                "常见原因：bulk/无真空、真空层过薄、电势尚未收敛或阈值过严。",
                "可调参数：--plateau_grad_tol_evA / --plateau_end_fraction / --plateau_min_points",
                f"如确需旧式端点平均，请显式添加 --allow_fraction_fallback（结果将标记为 {UNRELIABLE_FALLBACK_TAG}）。",
            ])
        )

    print(f"    真空电势检测方法: {vac_result['method']}")
    if vac_result['v_vac_left'] is not None:
        print(f"    V_vac (左) = {vac_result['v_vac_left']:.4f} eV")
    if vac_result['v_vac_right'] is not None:
        print(f"    V_vac (右) = {vac_result['v_vac_right']:.4f} eV")
    print(f"    V_vac (最终) = {vac_result['v_vac']:.4f} eV")
    if vac_result['fallback_used']:
        print(f"    [WARN] {UNRELIABLE_FALLBACK_TAG}: 使用了端点平均回退")

    phi = vac_result['v_vac'] - efermi
    phi_left = vac_result['v_vac_left'] - efermi if vac_result['v_vac_left'] is not None else None
    phi_right = vac_result['v_vac_right'] - efermi if vac_result['v_vac_right'] is not None else None

    print(f"    Φ = V_vac - E_F = {vac_result['v_vac']:.4f} - {efermi:.4f} = {phi:.4f} eV")

    data_path = os.path.join(calcdir, 'vacuum_potential.dat')
    _write_vacuum_profile(
        data_path,
        coords,
        planar_avg,
        efermi,
        phi,
        phi_left,
        phi_right,
        axis,
        functional_tag,
        scissor_ev,
        vac_result,
    )
    print(f"\n>>> 数据已保存: {data_path}")

    wf_summary = {
        'Phi': float(phi),
        'Phi_left': float(phi_left) if phi_left is not None else None,
        'Phi_right': float(phi_right) if phi_right is not None else None,
        'E_F': float(efermi),
        'V_vac': float(vac_result['v_vac']),
        'V_vac_left': float(vac_result['v_vac_left']) if vac_result['v_vac_left'] is not None else None,
        'V_vac_right': float(vac_result['v_vac_right']) if vac_result['v_vac_right'] is not None else None,
        'method': vac_result['method'],
        'plateau_found': bool(vac_result['plateau_found']),
        'plateau_found_left': bool(vac_result['plateau_found_left']),
        'plateau_found_right': bool(vac_result['plateau_found_right']),
        'fallback_used': bool(vac_result['fallback_used']),
        'axis': axis_name(axis),
        'both_sides': bool(both_sides),
        'functional_tag': functional_tag,
        'scissor_ev': scissor_ev,
        'reliability_tag': vac_result['reliability_tag'],
    }
    if plateau is not None:
        wf_summary['plateau_settings'] = {
            'grad_tol_evA': plateau['grad_tol_evA'],
            'end_fraction': plateau['end_fraction'],
            'min_points': plateau['min_plateau_points'],
        }
        wf_summary['plateau_ranges'] = {
            'left': plateau['left'],
            'right': plateau['right'],
        }
    if vac_result['fraction_window'] is not None:
        wf_summary['fraction_window'] = vac_result['fraction_window']

    wf_summary_path = os.path.join(calcdir, 'wf_summary.json')
    write_json(wf_summary_path, wf_summary)
    print(f">>> 摘要已保存: {wf_summary_path}")

    _plot_work_function(
        calcdir,
        coords,
        planar_avg,
        efermi,
        phi,
        axis,
        functional_tag,
        vac_result,
    )

    print("\n" + "=" * 60)
    print(f"功函数计算结果 [{functional_tag}]")
    print("=" * 60)
    print(f"E_F      = {efermi:.4f} eV")
    reliability_suffix = f", {UNRELIABLE_FALLBACK_TAG}" if vac_result['fallback_used'] else ""
    print(f"V_vac    = {vac_result['v_vac']:.4f} eV ({vac_result['method']}{reliability_suffix})")
    if phi_left is not None and phi_right is not None:
        print(f"Φ_left   = {phi_left:.4f} eV")
        print(f"Φ_right  = {phi_right:.4f} eV")
    print(f"Φ        = {phi:.4f} eV")

    print("\n" + "-" * 60)
    print("[NOTE] 绝缘体功函数参考")
    print("-" * 60)
    print("""
对于绝缘体/半导体：
  - OUTCAR 中的 E_F 是数值占据平衡点，不一定物理有意义
  - 更可靠的参考是 VBM (价带顶)
  - Φ_VBM = V_vac - E_VBM

当前报告的 Φ 基于 E_F，标记为"数值 EF 参考"。
如需 Φ_VBM，请从 DOS/EIGENVAL 提取 VBM 能量。
""")

    if functional_tag.upper() in ['PBE', 'GGA', 'LDA']:
        print("-" * 60)
        print("⚠️  PBE 带隙低估警告")
        print("-" * 60)
        print("PBE 系统性低估带隙。带边相关的 ESW 可能小于实验值。")

    print("=" * 60)
    return wf_summary


# =============================================================================
# DOS Parsing
# =============================================================================

def parse_doscar(doscar_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray],
                                             Optional[np.ndarray], float, int, bool]:
    """
    解析 DOSCAR 文件，容忍额外空行并验证 NEDOS 完整性。
    """
    if not os.path.isfile(doscar_path):
        raise FileNotFoundError(f"DOSCAR 不存在: {doscar_path}")

    with open(doscar_path, 'r', encoding='utf-8', errors='ignore') as handle:
        lines = handle.readlines()

    if len(lines) < 6:
        raise ValueError(f"DOSCAR 头部不足 6 行，无法解析: {doscar_path}")

    header_idx = None
    header_parts = None
    for idx in range(5, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            float(parts[0])
            float(parts[1])
            int(float(parts[2]))
            float(parts[3])
        except ValueError:
            continue
        header_idx = idx
        header_parts = parts
        break

    if header_idx is None or header_parts is None:
        raise ValueError("无法在 DOSCAR 中定位 total DOS 头部（需包含 EMAX EMIN NEDOS EFERMI）")

    nedos = int(float(header_parts[2]))
    efermi = float(header_parts[3])

    data_rows = []
    for idx in range(header_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            row = [float(x) for x in parts]
        except ValueError as exc:
            raise ValueError(f"DOSCAR 数据行无法解析 (第 {idx + 1} 行): {stripped}") from exc
        if len(row) < 2:
            raise ValueError(f"DOSCAR 数据列数不足 (第 {idx + 1} 行): {stripped}")
        data_rows.append(row)
        if len(data_rows) == nedos:
            break

    if len(data_rows) < nedos:
        raise ValueError(
            f"DOSCAR 数据不足: 头部声明 NEDOS={nedos}, "
            f"但只读取到 {len(data_rows)} 行 total DOS 数据"
        )

    is_spin = len(data_rows[0]) >= 5

    energy = []
    dos_total = []
    dos_up = [] if is_spin else None
    dos_down = [] if is_spin else None

    for row_idx, row in enumerate(data_rows, start=1):
        if is_spin and len(row) < 3:
            raise ValueError(f"DOSCAR 自旋极化行列数不足 (DOS 行 {row_idx})")
        if (not is_spin) and len(row) < 2:
            raise ValueError(f"DOSCAR 非自旋行列数不足 (DOS 行 {row_idx})")

        energy.append(row[0])
        if is_spin:
            d_up = row[1]
            d_down = row[2]
            dos_up.append(d_up)
            dos_down.append(d_down)
            dos_total.append(d_up + d_down)
        else:
            dos_total.append(row[1])

    energy_arr = np.array(energy) - efermi
    dos_total_arr = np.array(dos_total)
    dos_up_arr = np.array(dos_up) if is_spin and dos_up is not None else None
    dos_down_arr = np.array(dos_down) if is_spin and dos_down is not None else None

    return energy_arr, dos_total_arr, dos_up_arr, dos_down_arr, efermi, nedos, is_spin


def analyze_dos(calcdir: str,
                functional_tag: str = "PBE",
                scissor_ev: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    分析 DOS。
    """
    print("\n>>> 分析 DOS...")

    doscar_path = os.path.join(calcdir, 'DOSCAR')
    if not os.path.isfile(doscar_path):
        raise FileNotFoundError(f"DOSCAR 不存在: {doscar_path}")

    print(f"    读取 DOSCAR: {doscar_path}")
    energy, dos_total, dos_up, dos_down, efermi, nedos, is_spin = parse_doscar(doscar_path)

    print(f"    E-fermi = {efermi:.4f} eV")
    print(f"    NEDOS = {nedos}")
    print(f"    自旋极化: {'是' if is_spin else '否'}")
    print(f"    能量范围: {energy.min():.2f} ~ {energy.max():.2f} eV (相对 E_F)")

    csv_path = os.path.join(calcdir, 'dos_total.csv')
    with open(csv_path, 'w', encoding='utf-8') as handle:
        handle.write(f"# DOS analysis ({SCRIPT_VERSION})\n")
        handle.write(f"# Functional: {functional_tag}\n")
        if scissor_ev is not None:
            handle.write(f"# Scissor correction: {scissor_ev} eV (annotation only)\n")
        handle.write(f"# E_F = {efermi:.6f} eV\n")
        handle.write(f"# Spin-polarized: {is_spin}\n")

        if is_spin:
            handle.write("# Energy (eV rel E_F), DOS_total (states/eV), DOS_up, DOS_down\n")
            for e_val, d_tot, d_up, d_down in zip(energy, dos_total, dos_up, dos_down):
                handle.write(f"{e_val:12.6f},{d_tot:12.6f},{d_up:12.6f},{d_down:12.6f}\n")
        else:
            handle.write("# Energy (eV rel E_F), DOS_total (states/eV)\n")
            for e_val, d_val in zip(energy, dos_total):
                handle.write(f"{e_val:12.6f},{d_val:12.6f}\n")

    print(f"\n>>> 数据已保存: {csv_path}")

    if HAS_MPL:
        fig_path = os.path.join(calcdir, 'dos.png')
        fig, ax = plt.subplots(figsize=(8, 5))

        if is_spin:
            ax.plot(energy, dos_up, linewidth=1, color='blue', label='Spin up')
            ax.plot(energy, -dos_down, linewidth=1, color='red', label='Spin down')
            ax.fill_between(energy, dos_up, where=(energy <= 0), alpha=0.3, color='blue')
            ax.fill_between(energy, -dos_down, where=(energy <= 0), alpha=0.3, color='red')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('DOS (states/eV, down = negative)')
        else:
            ax.plot(energy, dos_total, linewidth=1, color='blue')
            ax.fill_between(energy, dos_total, where=(energy <= 0), alpha=0.3, color='blue')
            ax.set_ylabel('DOS (states/eV)')
            ax.set_ylim(0, None)

        ax.axvline(x=0, linestyle='--', linewidth=0.8, alpha=0.7, color='gray', label='E_F')
        ax.set_xlabel('Energy - E_F (eV)')

        title = 'Density of States'
        if is_spin:
            title += ' (Spin-polarized)'
        if functional_tag != "PBE":
            title += f' [{functional_tag}]'
        ax.set_title(title)
        ax.set_xlim(energy.min(), energy.max())
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f">>> 图已保存: {fig_path}")

    ef_idx = int(np.abs(energy).argmin())
    dos_summary: Dict[str, Any] = {
        'E_F': float(efermi),
        'NEDOS': int(nedos),
        'spin_polarized': bool(is_spin),
        'DOS_at_EF': float(dos_total[ef_idx]),
        'functional_tag': functional_tag,
        'scissor_ev': scissor_ev,
    }
    if is_spin and dos_up is not None and dos_down is not None:
        dos_summary['DOS_up_at_EF'] = float(dos_up[ef_idx])
        dos_summary['DOS_down_at_EF'] = float(dos_down[ef_idx])

    dos_summary_path = os.path.join(calcdir, 'dos_summary.json')
    write_json(dos_summary_path, dos_summary)
    print(f">>> 摘要已保存: {dos_summary_path}")

    print("\n" + "=" * 60)
    print(f"DOS 分析结果 [{functional_tag}]")
    print("=" * 60)
    print(f"E_F = {efermi:.4f} eV")
    print(f"NEDOS = {nedos}")
    print(f"自旋极化: {'是' if is_spin else '否'}")
    print(f"DOS @ E_F = {dos_total[ef_idx]:.4f} states/eV")

    if is_spin and dos_up is not None and dos_down is not None:
        print(f"  DOS_up @ E_F = {dos_up[ef_idx]:.4f}")
        print(f"  DOS_down @ E_F = {dos_down[ef_idx]:.4f}")

    if functional_tag.upper() in ['PBE', 'GGA', 'LDA']:
        print("\n" + "-" * 60)
        print("⚠️  PBE 带隙低估警告")
        print("-" * 60)
        print("PBE 系统性低估带隙。如需定量分析，推荐 HSE06 或剪刀修正。")

    print("=" * 60)
    print("\n推荐进一步分析:")
    print("  - PDOS: 使用 sumo-dosplot 或 p4vasp")
    print("  - 带结构: 使用 sumo-bandplot")
    return dos_summary


# =============================================================================
# Self Check
# =============================================================================

def _write_mock_outcar(path: str, efermi: float) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(" random header\n")
        handle.write(f" E-fermi : {efermi:10.6f} XC(G=0): 0.000000 alpha+bet : 0.000000\n")


def _write_mock_locpot(path: str,
                       planar_avg: np.ndarray,
                       cell_lengths: Tuple[float, float, float] = (5.0, 5.0, 30.0),
                       grid_xy: Tuple[int, int] = (2, 2)) -> None:
    nx, ny = grid_xy
    nz = len(planar_avg)
    potential_3d = np.zeros((nz, ny, nx))
    for idx, value in enumerate(planar_avg):
        potential_3d[idx, :, :] = value

    flat_data = potential_3d.reshape(-1)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write("Mock LOCPOT\n")
        handle.write("1.0\n")
        handle.write(f"{cell_lengths[0]:12.6f} 0.0 0.0\n")
        handle.write(f"0.0 {cell_lengths[1]:12.6f} 0.0\n")
        handle.write(f"0.0 0.0 {cell_lengths[2]:12.6f}\n")
        handle.write("H\n")
        handle.write("1\n")
        handle.write("Direct\n")
        handle.write("0.0 0.0 0.0\n")
        handle.write("\n")
        handle.write(f"{nx} {ny} {nz}\n")
        for start in range(0, len(flat_data), 5):
            chunk = flat_data[start:start + 5]
            handle.write(" ".join(f"{val:15.8f}" for val in chunk) + "\n")


def _make_mock_wf_calcdir(base_dir: str,
                          name: str,
                          planar_avg: np.ndarray,
                          efermi: float = 1.25) -> str:
    calcdir = os.path.join(base_dir, name)
    os.makedirs(calcdir, exist_ok=True)
    _write_mock_outcar(os.path.join(calcdir, 'OUTCAR'), efermi)
    _write_mock_locpot(os.path.join(calcdir, 'LOCPOT'), planar_avg)
    return calcdir


def run_self_check(parser: argparse.ArgumentParser) -> int:
    print("=" * 70)
    print(f"analyze_electronic.py {SCRIPT_VERSION} - self_check")
    print("=" * 70)

    coords = np.linspace(0.0, 30.0, 60, endpoint=False)
    plateau_profile = np.full_like(coords, 5.0)
    middle_mask = (coords >= 6.0) & (coords < 24.0)
    plateau_profile[middle_mask] = 2.0 + 0.6 * np.cos((coords[middle_mask] - 15.0) / 9.0 * np.pi)

    plateau_result = find_vacuum_plateau(
        coords,
        plateau_profile,
        grad_tol_evA=0.05,
        min_plateau_points=5,
        end_fraction=0.2,
    )
    if not (plateau_result['left']['found'] and plateau_result['right']['found']):
        print("[FAIL] 合成平台用例未能检测到左右两侧平台")
        return 1

    ramp_profile = np.linspace(0.0, 6.0, len(coords))
    ramp_result = find_vacuum_plateau(
        coords,
        ramp_profile,
        grad_tol_evA=0.05,
        min_plateau_points=5,
        end_fraction=0.2,
    )
    if ramp_result['left']['found'] or ramp_result['right']['found']:
        print("[FAIL] 无平台用例被误判为平台")
        return 1

    with tempfile.TemporaryDirectory(prefix='analyze_electronic_self_check_') as tmpdir:
        good_dir = _make_mock_wf_calcdir(tmpdir, 'wf_good', plateau_profile)
        bad_dir = _make_mock_wf_calcdir(tmpdir, 'wf_bad', ramp_profile)

        good_args = parser.parse_args(['--calcdir', good_dir, '--mode', 'wf'])
        good_code = run_analysis(good_args, show_banner=False)
        if good_code != 0:
            print(f"[FAIL] 平台用例退出码错误: 期望 0，实际 {good_code}")
            return 1

        bad_args = parser.parse_args(['--calcdir', bad_dir, '--mode', 'wf'])
        bad_code = run_analysis(bad_args, show_banner=False)
        if bad_code != 2:
            print(f"[FAIL] 无平台用例退出码错误: 期望 2，实际 {bad_code}")
            return 1

        fallback_args = parser.parse_args([
            '--calcdir', bad_dir, '--mode', 'wf', '--allow_fraction_fallback'
        ])
        fallback_code = run_analysis(fallback_args, show_banner=False)
        if fallback_code != 0:
            print(f"[FAIL] 回退用例退出码错误: 期望 0，实际 {fallback_code}")
            return 1

    print("[PASS] plateau detection 与退出码自检通过")
    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"VASP 电子性质后处理（功函数/DOS）{SCRIPT_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{SCRIPT_VERSION} 改进:
  - 无真空平台默认硬失败（退出码 2）
  - 显式回退: --allow_fraction_fallback
  - 平台检测参数: --plateau_grad_tol_evA / --plateau_end_fraction / --plateau_min_points
  - JSON 摘要: wf_summary.json / dos_summary.json

示例:
  # 功函数后处理（默认严格要求真空平台）
  python3 analyze_electronic.py --calcdir calc_wf/wf_static --mode wf

  # 若必须使用旧式端点平均，需显式允许回退
  python3 analyze_electronic.py --calcdir calc_wf/wf_static --mode wf --allow_fraction_fallback

  # DOS 后处理
  python3 analyze_electronic.py --calcdir calc_dos/dos_nscf --mode dos

  # 内建自检
  python3 analyze_electronic.py --self_check
        """
    )

    parser.add_argument(
        "--self_check",
        action="store_true",
        help="运行内建合成数据自检并退出",
    )
    parser.add_argument(
        "--calcdir",
        help="VASP 计算目录（包含 OUTCAR, LOCPOT 或 DOSCAR）",
    )
    parser.add_argument(
        "--mode",
        choices=['wf', 'dos'],
        help="分析模式: wf=功函数, dos=DOS",
    )

    parser.add_argument(
        "--axis",
        choices=['x', 'y', 'z'],
        default='z',
        help="真空/平面平均方向 (默认: z)",
    )
    parser.add_argument(
        "--one_side",
        action="store_true",
        help="功函数: 只使用右侧真空平台/端点窗口；默认使用两端平均",
    )
    parser.add_argument(
        "--vac_fraction",
        type=float,
        default=0.15,
        help="功函数: 端点平均回退时的取样比例 (默认: 0.15)",
    )
    parser.add_argument(
        "--allow_fraction_fallback",
        action="store_true",
        help=f"功函数: 允许在平台检测失败时回退到端点平均，并标记为 {UNRELIABLE_FALLBACK_TAG}",
    )
    parser.add_argument(
        "--no_plateau",
        action="store_true",
        help="功函数: 跳过平台检测，直接使用端点平均；必须配合 --allow_fraction_fallback",
    )
    parser.add_argument(
        "--plateau_grad_tol_evA",
        type=float,
        default=0.05,
        help="功函数: 平台判据 |dV/dx| < tol，单位 eV/Å (默认: 0.05)",
    )
    parser.add_argument(
        "--plateau_end_fraction",
        type=float,
        default=0.2,
        help="功函数: 在两端各自搜索平台的比例 (默认: 0.2)",
    )
    parser.add_argument(
        "--plateau_min_points",
        type=int,
        default=5,
        help="功函数: 平台连续最少网格点数 (默认: 5)",
    )

    parser.add_argument(
        "--functional_tag",
        default="PBE",
        help="泛函标签 (默认: PBE)",
    )
    parser.add_argument(
        "--scissor_ev",
        type=float,
        default=None,
        help="剪刀修正值 eV (仅用于注释)",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_check:
        return

    if not args.calcdir or not args.mode:
        parser.error("--calcdir 和 --mode 为必填参数（除非使用 --self_check）")

    if args.no_plateau and not args.allow_fraction_fallback:
        parser.error("--no_plateau 必须配合 --allow_fraction_fallback 使用")

    if args.plateau_grad_tol_evA <= 0:
        parser.error("--plateau_grad_tol_evA 必须 > 0")

    if not 0 < args.plateau_end_fraction <= 0.5:
        parser.error("--plateau_end_fraction 必须在 (0, 0.5] 范围内")

    if args.plateau_min_points < 1:
        parser.error("--plateau_min_points 必须 >= 1")

    if not 0 < args.vac_fraction <= 0.5:
        parser.error("--vac_fraction 必须在 (0, 0.5] 范围内")


def run_analysis(args: argparse.Namespace, show_banner: bool = True) -> int:
    if show_banner:
        print("=" * 70)
        print(f"analyze_electronic.py {SCRIPT_VERSION} - VASP 电子性质后处理")
        print("=" * 70)
        if args.self_check:
            print("运行模式: self_check")
        else:
            print(f"计算目录: {args.calcdir}")
            print(f"分析模式: {args.mode}")
            print(f"泛函标签: {args.functional_tag}")
            if args.scissor_ev is not None:
                print(f"剪刀修正: {args.scissor_ev} eV")

    if args.self_check:
        parser = build_parser()
        return run_self_check(parser)

    if not os.path.isdir(args.calcdir):
        print(f"[ERROR] 目录不存在: {args.calcdir}")
        return 1

    try:
        if args.mode == 'wf':
            axis_idx = AXIS_MAP[args.axis]
            analyze_work_function(
                args.calcdir,
                axis=axis_idx,
                both_sides=not args.one_side,
                vac_fraction=args.vac_fraction,
                use_plateau=not args.no_plateau,
                allow_fraction_fallback=args.allow_fraction_fallback,
                plateau_grad_tol_evA=args.plateau_grad_tol_evA,
                plateau_end_fraction=args.plateau_end_fraction,
                plateau_min_points=args.plateau_min_points,
                functional_tag=args.functional_tag,
                scissor_ev=args.scissor_ev,
            )
        elif args.mode == 'dos':
            analyze_dos(
                args.calcdir,
                functional_tag=args.functional_tag,
                scissor_ev=args.scissor_ev,
            )
        else:
            raise ValueError(f"未知模式: {args.mode}")
    except PlateauDetectionError as exc:
        print("\n" + "=" * 70)
        print("[ERROR] 真空平台检测失败")
        print("=" * 70)
        print(exc)
        print("=" * 70)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return run_analysis(args, show_banner=True)


if __name__ == "__main__":
    sys.exit(main())
