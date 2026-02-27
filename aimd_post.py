#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aimd_post.py - AIMD 热力学数据后处理 (v2.0)

从 OSZICAR 或 OUTCAR 提取每个离子步的热力学数据：
- 时间步索引 (step)
- 总能量 E0 (eV)
- 温度 T (K)
- 自由能 F (eV)
- 外压力 P (kB) [可选，extended_csv 模式]

用法:
    # 基本用法（保持后向兼容）
    python3 aimd_post.py [--oszicar OSZICAR] [--outcar OUTCAR] [--output aimd_thermo.csv]

    # 跳过平衡期，只统计生产段
    python3 aimd_post.py --t_skip_ps 2.0 --dt_fs 1.0

    # 严格模式（遇到星号溢出或高能量漂移时退出）
    python3 aimd_post.py --strict

    # 扩展 CSV（含 raw_step/segment；若有压力再加 P_kB）
    python3 aimd_post.py --extended_csv

    # 显式指定 OUTCAR Iteration 顺序
    python3 aimd_post.py --iteration_order electron_first

    # 运行内置轻量自检
    python3 aimd_post.py --self_check

输出:
    - aimd_thermo.csv: 热力学数据 CSV 文件
    - 终端摘要: 统计信息、能量漂移、压力统计、Iteration 顺序调试信息

版本: 2.0
变更:
    - v2.0: 修复 OUTCAR 离子步解析、拼接处理、星号溢出
           添加平衡跳过、能量漂移、压力监控

依赖:
    - 标准库 (无需额外安装)

作者: STAR0418-ABC
"""

import argparse
import sys
import os
import re
import math
import tempfile
from typing import List, Dict, Optional, Tuple, Any

# ============================================================================
# Constants
# ============================================================================

# Default thresholds for energy drift
DEFAULT_DRIFT_WARN_MEV_ATOM_PS = 1.0
DEFAULT_DRIFT_STRICT_MEV_ATOM_PS = 3.0
DEFAULT_ITERATION_SCAN_LIMIT = 200

# ============================================================================
# Utility Functions
# ============================================================================

def parse_numeric_value(s: str, context: str = "") -> Tuple[Optional[float], bool]:
    """
    Parse a numeric value, handling Fortran star overflow.
    
    Args:
        s: String to parse (e.g., "3.14159", "-1.23E+02", "************")
        context: Context string for error messages
    
    Returns:
        (value, is_stars): value is None if parsing failed or stars detected
                          is_stars is True if the field contains stars
    """
    s = s.strip()
    if not s:
        return None, False
    
    # Check for Fortran star overflow
    if '*' in s:
        return None, True
    
    try:
        return float(s), False
    except ValueError:
        return None, False


def read_potim_from_incar(incar_path: str) -> Optional[float]:
    """
    Read POTIM value from INCAR file.
    
    Args:
        incar_path: Path to INCAR file
    
    Returns:
        POTIM value in fs, or None if not found
    """
    if not os.path.isfile(incar_path):
        return None
    
    pattern = re.compile(r'^\s*POTIM\s*=\s*([\d.E+-]+)', re.IGNORECASE)
    
    try:
        with open(incar_path, 'r') as f:
            for line in f:
                # Skip comments
                line = line.split('#')[0].split('!')[0]
                m = pattern.match(line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    
    return None


def read_nions_from_outcar(outcar_path: str) -> Optional[int]:
    """
    Read NIONS (number of atoms) from OUTCAR.
    
    Args:
        outcar_path: Path to OUTCAR file
    
    Returns:
        Number of atoms, or None if not found
    """
    if not os.path.isfile(outcar_path):
        return None
    
    pattern = re.compile(r'NIONS\s*=\s*(\d+)', re.IGNORECASE)
    
    try:
        with open(outcar_path, 'r') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    return int(m.group(1))
                # NIONS is near the top of OUTCAR, stop after a few hundred lines
                if 'POTCAR:' in line:
                    break
        # Retry: sometimes NIONS is later
        with open(outcar_path, 'r') as f:
            for i, line in enumerate(f):
                if i > 5000:
                    break
                m = pattern.search(line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    
    return None


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    Simple linear regression y = a + b*x without numpy.
    
    Args:
        x: Independent variable values
        y: Dependent variable values
    
    Returns:
        (intercept, slope, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-30:
        return 0.0, 0.0, 0.0
    
    b = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n
    
    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    
    return a, b, r_squared


def summarize_iteration_component(values: List[int]) -> Dict[str, float]:
    """
    Summarize one component of OUTCAR Iteration a(b) markers.

    A larger span and slower change rate indicate the ionic-step component.
    """
    if not values:
        return {
            'span': 0.0,
            'change_count': 0.0,
            'change_rate': 0.0,
            'avg_run_length': 0.0,
            'score': 0.0,
        }

    span = float(max(values) - min(values))
    change_count = float(sum(1 for prev, curr in zip(values, values[1:]) if curr != prev))
    change_rate = change_count / max(1, len(values) - 1)
    avg_run_length = len(values) / (change_count + 1.0)
    score = (span + 1.0) * avg_run_length
    return {
        'span': span,
        'change_count': change_count,
        'change_rate': change_rate,
        'avg_run_length': avg_run_length,
        'score': score,
    }


def detect_iteration_order_from_matches(
    matches: List[Tuple[int, int]],
    iteration_order: str = 'auto'
) -> Dict[str, Any]:
    """
    Decide how to interpret OUTCAR "Iteration a(b)" markers.

    Returns:
        Dict with order, ionic_group, electronic_group, source, and stats.
    """
    if iteration_order == 'ionic_first':
        return {
            'order': 'ionic_first',
            'ionic_group': 1,
            'electronic_group': 2,
            'source': 'override',
            'inspected_matches': len(matches),
            'debug_label': 'ionic_first (override)',
        }

    if iteration_order == 'electron_first':
        return {
            'order': 'electron_first',
            'ionic_group': 2,
            'electronic_group': 1,
            'source': 'override',
            'inspected_matches': len(matches),
            'debug_label': 'electron_first (override)',
        }

    if not matches:
        return {
            'order': 'electron_first',
            'ionic_group': 2,
            'electronic_group': 1,
            'source': 'auto-fallback',
            'inspected_matches': 0,
            'reason': 'no iteration markers found',
            'debug_label': 'electron_first (auto-fallback)',
        }

    first_values = [a for a, _ in matches]
    second_values = [b for _, b in matches]
    first_stats = summarize_iteration_component(first_values)
    second_stats = summarize_iteration_component(second_values)

    if first_stats['score'] > second_stats['score']:
        order = 'ionic_first'
    elif second_stats['score'] > first_stats['score']:
        order = 'electron_first'
    elif first_stats['change_rate'] < second_stats['change_rate']:
        order = 'ionic_first'
    elif second_stats['change_rate'] < first_stats['change_rate']:
        order = 'electron_first'
    elif first_stats['span'] > second_stats['span']:
        order = 'ionic_first'
    else:
        order = 'electron_first'

    ionic_group = 1 if order == 'ionic_first' else 2
    electronic_group = 2 if ionic_group == 1 else 1
    return {
        'order': order,
        'ionic_group': ionic_group,
        'electronic_group': electronic_group,
        'source': 'auto',
        'inspected_matches': len(matches),
        'first_stats': first_stats,
        'second_stats': second_stats,
        'debug_label': f"{order} (auto)",
    }


# ============================================================================
# OSZICAR Parsing
# ============================================================================

def parse_oszicar(filepath: str, allow_stars: bool = True, strict: bool = False
                  ) -> Tuple[List[Dict], int, int]:
    """
    Parse OSZICAR file, extracting thermo data for each ionic step.
    
    Uses order-independent key-value parsing to handle format variations.
    
    Args:
        filepath: Path to OSZICAR file
        allow_stars: If True, treat star fields as None; if False, raise error
        strict: If True, exit on stars encountered
    
    Returns:
        (data_list, stars_count, segment_count)
    """
    if not os.path.isfile(filepath):
        return [], 0, 0
    
    data = []
    stars_count = 0
    segment_count = 1
    prev_step = 0
    segment_offset = 0
    stars_lines = []
    
    # Pattern for ionic step line start (step number at beginning)
    pattern_step = re.compile(r'^\s*(\d+)\s+')
    
    # Pattern for key-value pairs: T=, E=, F=, E0=
    # Match key= followed by the value (which may be stars or numeric)
    pattern_kv = re.compile(r'(T|E0|F|E)\s*=\s*(\S+)', re.IGNORECASE)
    
    with open(filepath, 'r', errors='replace') as f:
        for lineno, line in enumerate(f, 1):
            # Check for ionic step line
            m_step = pattern_step.match(line)
            if not m_step:
                continue
            
            step = int(m_step.group(1))
            
            # Detect step reset (concatenation)
            if step <= prev_step and len(data) > 0:
                # New segment starts
                segment_offset = data[-1]['global_step']
                segment_count += 1
            prev_step = step
            
            global_step = segment_offset + step
            
            # Extract key-value pairs
            kv_matches = pattern_kv.findall(line)
            if not kv_matches:
                continue
            
            values = {'T': None, 'E': None, 'F': None, 'E0': None}
            line_has_stars = False
            
            for key, val_str in kv_matches:
                key_upper = key.upper()
                val, is_stars = parse_numeric_value(val_str, f"OSZICAR line {lineno}")
                if is_stars:
                    line_has_stars = True
                    stars_count += 1
                    stars_lines.append((lineno, line.strip()[:80]))
                values[key_upper] = val
            
            if line_has_stars:
                if strict:
                    print(f"[ERROR] Fortran star overflow in OSZICAR line {lineno}:")
                    print(f"        {line.strip()[:100]}")
                    print("        Use --allow_stars or remove --strict to continue.")
                    sys.exit(2)
                elif not allow_stars:
                    print(f"[ERROR] Fortran star overflow in OSZICAR line {lineno}:")
                    print(f"        {line.strip()[:100]}")
                    sys.exit(2)
            
            data.append({
                'step': step,
                'raw_step': step,
                'global_step': global_step,
                'segment': segment_count,
                'E0': values['E0'],
                'T': values['T'],
                'F': values['F'],
                'source': 'oszicar'
            })
    
    # Print warning summary for stars
    if stars_count > 0:
        print(f"[WARNING] OSZICAR contains {stars_count} star overflow field(s):")
        for lineno, content in stars_lines[:5]:
            print(f"          Line {lineno}: {content}")
        if len(stars_lines) > 5:
            print(f"          ... and {len(stars_lines) - 5} more")
    
    return data, stars_count, segment_count


# ============================================================================
# OUTCAR Parsing
# ============================================================================

def parse_outcar_thermo(
    filepath: str,
    allow_stars: bool = True,
    strict: bool = False,
    iteration_order: str = 'auto',
    scan_limit: int = DEFAULT_ITERATION_SCAN_LIMIT
) -> Tuple[List[Dict], int, int, Dict[str, Any]]:
    """
    Parse OUTCAR for temperature, energy, and pressure data.

    The ionic-step component of "Iteration a(b)" is determined by
    --iteration_order or by auto-detecting which component has the larger span
    and slower change rate in the first scan_limit matches.

    Args:
        filepath: Path to OUTCAR file
        allow_stars: If True, treat star fields as None
        strict: If True, exit on stars encountered
        iteration_order: auto, electron_first, or ionic_first
        scan_limit: Number of iteration markers to inspect for auto detection

    Returns:
        (data_list, stars_count, segment_count, iteration_info)
    """
    if not os.path.isfile(filepath):
        return [], 0, 0, {
            'order': 'electron_first',
            'ionic_group': 2,
            'electronic_group': 1,
            'source': 'missing-file',
            'inspected_matches': 0,
            'debug_label': 'electron_first (missing-file)',
        }
    
    data = []
    stars_count = 0
    segment_count = 1
    prev_ionic_step = 0
    segment_offset = 0
    stars_lines = []
    
    current_ionic_step = 0
    current_E0 = None
    current_T = None
    current_F = None
    current_P_kB = None
    
    # Regex patterns
    pattern_ionic_step = re.compile(r'---+\s*Iteration\s+(\d+)\s*\(\s*(\d+)\s*\)', re.IGNORECASE)
    pattern_toten = re.compile(r'free\s+energy\s+TOTEN\s*=\s*(\S+)', re.IGNORECASE)
    pattern_e0 = re.compile(r'energy\s+without\s+entropy\s*=\s*(\S+)', re.IGNORECASE)
    pattern_ekin_lat = re.compile(r'EKIN_LAT\s*=\s*\S+\s*\(\s*temperature\s+(\S+)\s*K\s*\)', re.IGNORECASE)
    pattern_temp = re.compile(r'temperature\s+(\S+)\s*K', re.IGNORECASE)
    pattern_pressure = re.compile(r'external\s+pressure\s*=\s*(\S+)\s*kB', re.IGNORECASE)

    iteration_matches: List[Tuple[int, int]] = []
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            m = pattern_ionic_step.search(line)
            if not m:
                continue
            iteration_matches.append((int(m.group(1)), int(m.group(2))))
            if len(iteration_matches) >= scan_limit:
                break

    iteration_info = detect_iteration_order_from_matches(
        iteration_matches,
        iteration_order=iteration_order
    )
    ionic_group = iteration_info['ionic_group']
    
    def flush_step():
        """Save current ionic step data and reset."""
        nonlocal current_ionic_step, current_E0, current_T, current_F, current_P_kB
        nonlocal prev_ionic_step, segment_offset, segment_count, data
        
        if current_ionic_step > 0 and current_E0 is not None:
            # Detect step reset
            if current_ionic_step <= prev_ionic_step and len(data) > 0:
                segment_offset = data[-1]['global_step']
                segment_count += 1
            prev_ionic_step = current_ionic_step
            
            global_step = segment_offset + current_ionic_step
            
            data.append({
                'step': current_ionic_step,
                'raw_step': current_ionic_step,
                'global_step': global_step,
                'segment': segment_count,
                'E0': current_E0,
                'T': current_T,
                'F': current_F,
                'P_kB': current_P_kB,
                'source': 'outcar'
            })
    
    with open(filepath, 'r', errors='replace') as f:
        for lineno, line in enumerate(f, 1):
            # VASP format varies between Iteration ionic(electronic) and
            # Iteration electronic(ionic); use the chosen ionic_group.
            m = pattern_ionic_step.search(line)
            if m:
                ionic_step = int(m.group(ionic_group))
                
                # If ionic step changed, flush previous step
                if ionic_step != current_ionic_step:
                    flush_step()
                    current_ionic_step = ionic_step
                    current_E0 = None
                    current_T = None
                    current_F = None
                    current_P_kB = None
                # If same ionic step but different electronic step, keep updating
                continue
            
            # TOTEN (as F)
            m = pattern_toten.search(line)
            if m:
                val, is_stars = parse_numeric_value(m.group(1), f"OUTCAR line {lineno}")
                if is_stars:
                    stars_count += 1
                    stars_lines.append((lineno, line.strip()[:80]))
                    if strict:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                    if not allow_stars:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                current_F = val
                continue
            
            # E0 (energy without entropy)
            m = pattern_e0.search(line)
            if m:
                val, is_stars = parse_numeric_value(m.group(1), f"OUTCAR line {lineno}")
                if is_stars:
                    stars_count += 1
                    stars_lines.append((lineno, line.strip()[:80]))
                    if strict:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                    if not allow_stars:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                current_E0 = val
                continue
            
            # Temperature (AIMD) - prefer EKIN_LAT over generic temperature
            m = pattern_ekin_lat.search(line)
            if m:
                val, is_stars = parse_numeric_value(m.group(1), f"OUTCAR line {lineno}")
                if is_stars:
                    stars_count += 1
                    stars_lines.append((lineno, line.strip()[:80]))
                    if strict or not allow_stars:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                current_T = val
                continue
            
            m = pattern_temp.search(line)
            if m and current_T is None:
                val, is_stars = parse_numeric_value(m.group(1), f"OUTCAR line {lineno}")
                if is_stars:
                    stars_count += 1
                    stars_lines.append((lineno, line.strip()[:80]))
                    if strict or not allow_stars:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                current_T = val
                continue
            
            # External pressure
            m = pattern_pressure.search(line)
            if m:
                val, is_stars = parse_numeric_value(m.group(1), f"OUTCAR line {lineno}")
                if is_stars:
                    stars_count += 1
                    stars_lines.append((lineno, line.strip()[:80]))
                    if strict or not allow_stars:
                        print(f"[ERROR] Fortran star overflow in OUTCAR line {lineno}")
                        sys.exit(2)
                current_P_kB = val
                continue
    
    # Flush last ionic step
    flush_step()
    
    # Print warning summary for stars
    if stars_count > 0:
        print(f"[WARNING] OUTCAR contains {stars_count} star overflow field(s):")
        for lineno, content in stars_lines[:5]:
            print(f"          Line {lineno}: {content}")
        if len(stars_lines) > 5:
            print(f"          ... and {len(stars_lines) - 5} more")
    
    return data, stars_count, segment_count, iteration_info


# ============================================================================
# Data Merge
# ============================================================================

def merge_data(oszicar_data: List[Dict], outcar_data: List[Dict],
               include_outcar_only: bool = False, dedup: str = 'keep_last'
               ) -> List[Dict]:
    """
    Merge OSZICAR and OUTCAR data, filling missing fields.
    
    Priority: OSZICAR values preferred; OUTCAR fills gaps.
    
    Args:
        oszicar_data: Data from OSZICAR
        outcar_data: Data from OUTCAR
        include_outcar_only: If True, include steps only in OUTCAR
        dedup: How to handle duplicates: keep_last, keep_first, error
    
    Returns:
        Merged data list sorted by global_step
    """
    if not oszicar_data and not outcar_data:
        return []
    
    if not oszicar_data:
        return sorted(outcar_data, key=lambda d: d.get('global_step', d['step']))
    
    if not outcar_data:
        return sorted(oszicar_data, key=lambda d: d.get('global_step', d['step']))
    
    # Build OUTCAR lookup with dedup handling
    outcar_dict: Dict[int, Dict] = {}
    for d in outcar_data:
        gs = d.get('global_step', d['step'])
        if gs in outcar_dict:
            if dedup == 'error':
                print(f"[ERROR] Duplicate global_step {gs} in OUTCAR data")
                sys.exit(1)
            elif dedup == 'keep_first':
                continue  # Keep existing
            # keep_last: overwrite (fall through)
        outcar_dict[gs] = d
    
    # Build merged data from OSZICAR, filling from OUTCAR
    merged_dict: Dict[int, Dict] = {}
    for d in oszicar_data:
        gs = d.get('global_step', d['step'])
        result = d.copy()
        
        if gs in outcar_dict:
            out_d = outcar_dict[gs]
            # Fill missing fields from OUTCAR
            if result.get('T') is None and out_d.get('T') is not None:
                result['T'] = out_d['T']
            if result.get('E0') is None and out_d.get('E0') is not None:
                result['E0'] = out_d['E0']
            if result.get('F') is None and out_d.get('F') is not None:
                result['F'] = out_d['F']
            # Always take pressure from OUTCAR if available
            if out_d.get('P_kB') is not None:
                result['P_kB'] = out_d['P_kB']
        
        if gs in merged_dict:
            if dedup == 'error':
                print(f"[ERROR] Duplicate global_step {gs} after merge")
                sys.exit(1)
            elif dedup == 'keep_first':
                continue
        merged_dict[gs] = result
    
    # Optionally add OUTCAR-only steps
    if include_outcar_only:
        for gs, d in outcar_dict.items():
            if gs not in merged_dict:
                merged_dict[gs] = d.copy()
    
    # Sort by global_step
    merged = sorted(merged_dict.values(), key=lambda d: d.get('global_step', d['step']))
    
    return merged


# ============================================================================
# CSV Output
# ============================================================================

def write_csv(data: List[Dict], filepath: str, extended: bool = False):
    """
    Write data to CSV file.
    
    Args:
        data: Data list
        filepath: Output path
        extended: If True, include pressure columns
    """
    with open(filepath, 'w') as f:
        # Header
        if extended:
            has_pressure = any(d.get('P_kB') is not None for d in data)
            header = "step,raw_step,segment,E0_eV,T_K,F_eV"
            if has_pressure:
                header += ",P_kB"
            f.write(header + "\n")
        else:
            f.write("step,E0_eV,T_K,F_eV\n")
        
        for d in data:
            step = d.get('global_step', d['step'])
            raw_step = d.get('raw_step', d.get('step', step))
            segment = d.get('segment', 1)
            E0 = f"{d['E0']:.8f}" if d.get('E0') is not None else ""
            T = f"{d['T']:.2f}" if d.get('T') is not None else ""
            F = f"{d['F']:.8f}" if d.get('F') is not None else ""
            
            if extended:
                row = f"{step},{raw_step},{segment},{E0},{T},{F}"
                if has_pressure:
                    P = f"{d['P_kB']:.2f}" if d.get('P_kB') is not None else ""
                    row += f",{P}"
                f.write(row + "\n")
            else:
                f.write(f"{step},{E0},{T},{F}\n")


# ============================================================================
# Statistics and Summary
# ============================================================================

def compute_drift(data: List[Dict], dt_ps: float, n_atoms: Optional[int] = None
                  ) -> Dict[str, Any]:
    """
    Compute energy drift via linear regression of E0 vs time.
    
    Args:
        data: Data list (should be production region only)
        dt_ps: Timestep in ps
        n_atoms: Number of atoms for per-atom normalization
    
    Returns:
        Dict with slope_eV_ps, slope_meV_atom_ps, r_squared
    """
    # Filter for valid E0 values
    valid = [(d.get('global_step', d['step']), d['E0']) 
             for d in data if d.get('E0') is not None]
    
    if len(valid) < 2:
        return {'slope_eV_ps': 0.0, 'slope_meV_atom_ps': None, 'r_squared': 0.0, 'n_points': 0}
    
    # Convert steps to time (ps)
    t0 = valid[0][0]
    times = [(step - t0) * dt_ps for step, _ in valid]
    energies = [e0 for _, e0 in valid]
    
    intercept, slope, r_squared = linear_regression(times, energies)
    
    # slope is in eV/ps
    slope_eV_ps = slope
    slope_meV_atom_ps = None
    if n_atoms and n_atoms > 0:
        slope_meV_atom_ps = slope * 1000.0 / n_atoms  # eV/ps -> meV/atom/ps
    
    return {
        'slope_eV_ps': slope_eV_ps,
        'slope_meV_atom_ps': slope_meV_atom_ps,
        'r_squared': r_squared,
        'n_points': len(valid)
    }


def compute_stats(data: List[Dict], label: str = "") -> Dict[str, Any]:
    """
    Compute statistics for temperature, energy, and pressure.
    
    Args:
        data: Data list
        label: Label for display
    
    Returns:
        Statistics dictionary
    """
    stats = {'label': label, 'n_total': len(data)}
    
    # Temperature
    temps = [d['T'] for d in data if d.get('T') is not None]
    if temps:
        stats['T'] = {
            'n': len(temps),
            'mean': sum(temps) / len(temps),
            'min': min(temps),
            'max': max(temps),
            'std': math.sqrt(sum((t - sum(temps)/len(temps))**2 for t in temps) / len(temps)) if len(temps) > 1 else 0.0
        }
    
    # E0 (energy)
    energies = [d['E0'] for d in data if d.get('E0') is not None]
    if energies:
        E_mean = sum(energies) / len(energies)
        stats['E0'] = {
            'n': len(energies),
            'mean': E_mean,
            'min': min(energies),
            'max': max(energies),
            'std': math.sqrt(sum((e - E_mean)**2 for e in energies) / len(energies)) if len(energies) > 1 else 0.0
        }
    
    # Pressure
    pressures = [d['P_kB'] for d in data if d.get('P_kB') is not None]
    if pressures:
        P_mean = sum(pressures) / len(pressures)
        stats['P_kB'] = {
            'n': len(pressures),
            'mean': P_mean,
            'min': min(pressures),
            'max': max(pressures),
            'std': math.sqrt(sum((p - P_mean)**2 for p in pressures) / len(pressures)) if len(pressures) > 1 else 0.0
        }
    
    return stats


def suggest_equilibration(data: List[Dict], dt_ps: float, window_steps: int = 50
                          ) -> Dict[str, Any]:
    """
    Suggest equilibration skip based on temperature and energy stabilization.
    
    This is advisory only - does not automatically apply.
    
    Args:
        data: Full data list
        dt_ps: Timestep in ps
        window_steps: Window size for rolling average
    
    Returns:
        Suggestion dictionary with skip_steps, skip_ps, and metrics
    """
    if len(data) < 2 * window_steps:
        return {'suggested_skip_steps': 0, 'suggested_skip_ps': 0.0, 'reason': 'insufficient data'}
    
    # Calculate rolling variance of temperature
    valid_temp_points = [(idx, d['T']) for idx, d in enumerate(data) if d.get('T') is not None]
    valid_indices = [idx for idx, _ in valid_temp_points]
    valid_temps = [temp for _, temp in valid_temp_points]
    if len(valid_temps) < window_steps:
        return {'suggested_skip_steps': 0, 'suggested_skip_ps': 0.0, 'reason': 'insufficient temperature data'}
    
    # Find where variance stabilizes (simple heuristic)
    final_mean_T = sum(valid_temps[-window_steps:]) / window_steps
    final_std_T = math.sqrt(sum((t - final_mean_T)**2 for t in valid_temps[-window_steps:]) / window_steps)
    
    # Look for first point where rolling mean is within 1.5*std of final mean
    suggested_temp_idx = 0
    for i in range(window_steps, len(valid_temps) - window_steps):
        window_data = valid_temps[i - window_steps//2:i + window_steps//2]
        if not window_data:
            continue
        window_mean = sum(window_data) / len(window_data)
        if abs(window_mean - final_mean_T) < 1.5 * final_std_T:
            suggested_temp_idx = i
            break
    
    if suggested_temp_idx == 0:
        suggested_temp_idx = min(window_steps, len(valid_indices) - 1)

    # Map back to actual data index before converting to step count
    actual_data_idx = valid_indices[min(suggested_temp_idx, len(valid_indices) - 1)]
    actual_step = data[actual_data_idx].get('global_step', actual_data_idx + 1)
    first_step = data[0].get('global_step', 1)
    skip_steps = max(0, actual_step - first_step)
    skip_ps = skip_steps * dt_ps
    
    return {
        'suggested_skip_steps': skip_steps,
        'suggested_skip_ps': skip_ps,
        'final_T_mean': final_mean_T,
        'final_T_std': final_std_T,
        'reason': 'temperature stabilization'
    }


def print_summary(data: List[Dict], production_data: List[Dict],
                  drift_info: Dict[str, Any], stars_total: int,
                  dt_ps: float, n_atoms: Optional[int],
                  drift_warn_threshold: float, drift_strict_threshold: float,
                  strict: bool, suggestion: Dict[str, Any],
                  segment_count: int, applied_skip_steps: int = 0,
                  iteration_info: Optional[Dict[str, Any]] = None,
                  n_lines: int = 5):
    """
    Print comprehensive summary to terminal.
    """
    print("\n" + "=" * 70)
    print("AIMD 热力学数据摘要 / AIMD Thermodynamic Summary")
    print("=" * 70)
    
    # Data overview
    print(f"\n>>> 数据概览 / Data Overview:")
    print(f"    总数据点 / Total points: {len(data)}")
    print(f"    段数 / Segments: {segment_count}")
    if iteration_info is not None:
        print(f"    OUTCAR Iteration order: {iteration_info.get('debug_label', iteration_info.get('order', 'N/A'))}")
    if stars_total > 0:
        print(f"    [WARNING] 星号溢出 / Star overflows: {stars_total}")
    
    # Full run stats
    full_stats = compute_stats(data, "Full run")
    print(f"\n>>> 全程统计 / Full Run Statistics:")
    if 'T' in full_stats:
        s = full_stats['T']
        print(f"    温度 T: 平均 {s['mean']:.1f} K, 范围 [{s['min']:.1f}, {s['max']:.1f}] K, σ={s['std']:.1f} K")
    if 'E0' in full_stats:
        s = full_stats['E0']
        print(f"    能量 E0: 平均 {s['mean']:.4f} eV, 范围 [{s['min']:.4f}, {s['max']:.4f}] eV")
    if 'P_kB' in full_stats:
        s = full_stats['P_kB']
        print(f"    压力 P: 平均 {s['mean']:.2f} kB, 范围 [{s['min']:.2f}, {s['max']:.2f}] kB")
    
    # Production stats (if different from full)
    if applied_skip_steps > 0 and len(production_data) > 0:
        skip_ps = applied_skip_steps * dt_ps
        print(f"\n>>> 生产段统计 / Production Statistics (skip {applied_skip_steps} steps = {skip_ps:.2f} ps):")
        prod_stats = compute_stats(production_data, "Production")
        if 'T' in prod_stats:
            s = prod_stats['T']
            print(f"    温度 T: 平均 {s['mean']:.1f} K, 范围 [{s['min']:.1f}, {s['max']:.1f}] K, σ={s['std']:.1f} K")
        if 'E0' in prod_stats:
            s = prod_stats['E0']
            print(f"    能量 E0: 平均 {s['mean']:.4f} eV, 范围 [{s['min']:.4f}, {s['max']:.4f}] eV")
        if 'P_kB' in prod_stats:
            s = prod_stats['P_kB']
            print(f"    压力 P: 平均 {s['mean']:.2f} kB, 范围 [{s['min']:.2f}, {s['max']:.2f}] kB")
    elif applied_skip_steps > 0 and len(production_data) == 0:
        skip_ps = applied_skip_steps * dt_ps
        print(f"\n>>> 生产段统计 / Production Statistics:")
        print(f"    [WARNING] 跳过 {applied_skip_steps} steps = {skip_ps:.2f} ps 后无剩余数据 / Production window is empty")
    
    # Energy drift
    print(f"\n>>> 能量漂移 / Energy Drift (production region):")
    if drift_info['n_points'] < 2:
        print(f"    [INFO] 数据不足，无法计算漂移 / Insufficient data for drift calculation")
    else:
        slope_eV_ps = drift_info['slope_eV_ps']
        r2 = drift_info['r_squared']
        print(f"    斜率 / Slope: {slope_eV_ps:.6f} eV/ps (R² = {r2:.4f})")
        
        if drift_info['slope_meV_atom_ps'] is not None:
            slope_meV = drift_info['slope_meV_atom_ps']
            print(f"    归一化 / Normalized: {slope_meV:.3f} meV/atom/ps (n_atoms = {n_atoms})")
            
            abs_slope = abs(slope_meV)
            if abs_slope > drift_strict_threshold:
                msg = f"    [ERROR] 漂移过大 / High drift: |{slope_meV:.3f}| > {drift_strict_threshold} meV/atom/ps"
                print(msg)
                if strict:
                    print("            严格模式退出 / Strict mode exit")
                    sys.exit(2)
            elif abs_slope > drift_warn_threshold:
                print(f"    [WARNING] 漂移偏高 / Elevated drift: |{slope_meV:.3f}| > {drift_warn_threshold} meV/atom/ps")
                print(f"              检查 SCF 收敛、ENCUT、PREC 设置")
            else:
                print(f"    [OK] 漂移正常 / Drift OK: |{slope_meV:.3f}| < {drift_warn_threshold} meV/atom/ps")
        else:
            print(f"    [INFO] n_atoms 未知，无法计算每原子漂移")
    
    # Equilibration suggestion (advisory only)
    if suggestion and suggestion.get('suggested_skip_steps', 0) > 0:
        print(f"\n>>> 平衡期建议 / Equilibration Suggestion (advisory only):")
        print(f"    建议跳过 / Suggested skip: {suggestion['suggested_skip_steps']} steps = {suggestion['suggested_skip_ps']:.2f} ps")
        print(f"    依据 / Reason: {suggestion.get('reason', 'N/A')}")
        print(f"    使用 --t_skip_steps 或 --t_skip_ps 应用 / Use --t_skip_steps or --t_skip_ps to apply")
    
    # Last N lines
    print(f"\n>>> 最后 {n_lines} 步数据 / Last {n_lines} Steps:")
    print("-" * 70)
    print(f"{'Step':>8}  {'E0 (eV)':>16}  {'T (K)':>10}  {'F (eV)':>16}  {'P (kB)':>10}")
    print("-" * 70)
    
    for d in data[-n_lines:]:
        step = d.get('global_step', d['step'])
        E0 = f"{d['E0']:.6f}" if d.get('E0') is not None else "N/A"
        T = f"{d['T']:.2f}" if d.get('T') is not None else "N/A"
        F = f"{d['F']:.6f}" if d.get('F') is not None else "N/A"
        P = f"{d['P_kB']:.2f}" if d.get('P_kB') is not None else "N/A"
        print(f"{step:>8}  {E0:>16}  {T:>10}  {F:>16}  {P:>10}")
    
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def run_self_check() -> int:
    """
    Run a lightweight parser self-check against mocked OUTCAR snippets.
    """
    electron_first_text = """\
--- Iteration    1(   1) ---------------------------------
 free energy    TOTEN  =      -10.100000 eV
 energy  without entropy=     -10.000000
 EKIN_LAT =  0.1234 (temperature 300.00 K)
 external pressure =    1.00 kB
--- Iteration    2(   1) ---------------------------------
 free energy    TOTEN  =      -10.000000 eV
 energy  without entropy=      -9.900000
 EKIN_LAT =  0.1234 (temperature 301.00 K)
--- Iteration    1(   2) ---------------------------------
 free energy    TOTEN  =       -9.900000 eV
 energy  without entropy=      -9.800000
 EKIN_LAT =  0.1234 (temperature 302.00 K)
--- Iteration    2(   2) ---------------------------------
 free energy    TOTEN  =       -9.800000 eV
 energy  without entropy=      -9.700000
 EKIN_LAT =  0.1234 (temperature 303.00 K)
"""

    ionic_first_text = """\
--- Iteration    1(   1) ---------------------------------
 free energy    TOTEN  =      -20.100000 eV
 energy  without entropy=     -20.000000
 EKIN_LAT =  0.1234 (temperature 400.00 K)
 external pressure =    2.00 kB
--- Iteration    1(   2) ---------------------------------
 free energy    TOTEN  =      -20.000000 eV
 energy  without entropy=     -19.900000
 EKIN_LAT =  0.1234 (temperature 401.00 K)
--- Iteration    2(   1) ---------------------------------
 free energy    TOTEN  =      -19.900000 eV
 energy  without entropy=     -19.800000
 EKIN_LAT =  0.1234 (temperature 402.00 K)
--- Iteration    2(   2) ---------------------------------
 free energy    TOTEN  =      -19.800000 eV
 energy  without entropy=     -19.700000
 EKIN_LAT =  0.1234 (temperature 403.00 K)
"""

    cases = [
        {
            'label': 'electron_first override',
            'filename': 'electron_first.OUTCAR',
            'text': electron_first_text,
            'iteration_order': 'electron_first',
            'expected_order': 'electron_first',
            'expected_e0': [-9.9, -9.7],
        },
        {
            'label': 'electron_first auto',
            'filename': 'electron_first_auto.OUTCAR',
            'text': electron_first_text,
            'iteration_order': 'auto',
            'expected_order': 'electron_first',
            'expected_e0': [-9.9, -9.7],
        },
        {
            'label': 'ionic_first override',
            'filename': 'ionic_first.OUTCAR',
            'text': ionic_first_text,
            'iteration_order': 'ionic_first',
            'expected_order': 'ionic_first',
            'expected_e0': [-19.9, -19.7],
        },
        {
            'label': 'ionic_first auto',
            'filename': 'ionic_first_auto.OUTCAR',
            'text': ionic_first_text,
            'iteration_order': 'auto',
            'expected_order': 'ionic_first',
            'expected_e0': [-19.9, -19.7],
        },
    ]

    with tempfile.TemporaryDirectory(prefix='aimd_post_self_check_') as tmpdir:
        for case in cases:
            path = os.path.join(tmpdir, case['filename'])
            with open(path, 'w') as f:
                f.write(case['text'])

            data, _, _, iteration_info = parse_outcar_thermo(
                path,
                iteration_order=case['iteration_order']
            )

            raw_steps = [d.get('raw_step') for d in data]
            global_steps = [d.get('global_step') for d in data]
            e0_values = [d.get('E0') for d in data]

            if len(data) != 2:
                raise AssertionError(f"{case['label']}: expected 2 ionic steps, got {len(data)}")
            if raw_steps != [1, 2]:
                raise AssertionError(f"{case['label']}: expected raw_steps [1, 2], got {raw_steps}")
            if global_steps != [1, 2]:
                raise AssertionError(f"{case['label']}: expected global_steps [1, 2], got {global_steps}")
            if iteration_info.get('order') != case['expected_order']:
                raise AssertionError(
                    f"{case['label']}: expected order {case['expected_order']}, got {iteration_info.get('order')}"
                )
            if any(abs(actual - expected) > 1e-9 for actual, expected in zip(e0_values, case['expected_e0'])):
                raise AssertionError(f"{case['label']}: expected E0 {case['expected_e0']}, got {e0_values}")

    print("[SELF_CHECK] aimd_post.py parser self-check passed")
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="AIMD 热力学数据后处理 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
    # 基本用法
    python3 aimd_post.py

    # 跳过平衡期 2 ps
    python3 aimd_post.py --t_skip_ps 2.0 --dt_fs 1.0

    # 严格模式
    python3 aimd_post.py --strict

    # 扩展 CSV（含 raw_step/segment；若有压力再加 P）
    python3 aimd_post.py --extended_csv

    # 显式指定 OUTCAR Iteration 顺序
    python3 aimd_post.py --iteration_order electron_first

    # 运行内置轻量自检
    python3 aimd_post.py --self_check
        """
    )
    
    # File paths
    parser.add_argument("--oszicar", default="OSZICAR",
                        help="OSZICAR 文件路径 (默认: OSZICAR)")
    parser.add_argument("--outcar", default="OUTCAR",
                        help="OUTCAR 文件路径 (默认: OUTCAR)")
    parser.add_argument("--output", default="aimd_thermo.csv",
                        help="输出 CSV 文件名 (默认: aimd_thermo.csv)")
    parser.add_argument("--incar", default="INCAR",
                        help="INCAR 文件路径，用于读取 POTIM (默认: INCAR)")
    
    # Concatenation handling
    parser.add_argument("--renumber", type=lambda x: x.lower() != 'false', default=True,
                        help="将 global_step 压缩为连续编号 1..N (默认: True)")
    parser.add_argument("--dedup", choices=['keep_last', 'keep_first', 'error'],
                        default='keep_last',
                        help="重复步处理策略 (默认: keep_last)")
    parser.add_argument("--iteration_order",
                        choices=['auto', 'electron_first', 'ionic_first'],
                        default='auto',
                        help="OUTCAR Iteration a(b) 解析顺序 (默认: auto)")
    
    # Stars handling
    parser.add_argument("--strict", action="store_true",
                        help="严格模式：星号溢出或高漂移时退出")
    parser.add_argument("--allow_stars", type=lambda x: x.lower() != 'false', default=True,
                        help="允许星号溢出（值设为 None）(默认: True)")
    
    # Equilibration skip
    parser.add_argument("--t_skip_steps", type=int, default=0,
                        help="跳过的平衡步数 (默认: 0)")
    parser.add_argument("--t_skip_ps", type=float, default=None,
                        help="跳过的平衡时间 ps (需要 --dt_fs 或 INCAR POTIM)")
    parser.add_argument("--dt_fs", type=float, default=None,
                        help="时间步长 fs (优先于 INCAR POTIM)")
    
    # Energy drift
    parser.add_argument("--n_atoms", type=int, default=None,
                        help="原子数 (覆盖 OUTCAR 自动检测)")
    parser.add_argument("--drift_warn_mev_atom_ps", type=float, 
                        default=DEFAULT_DRIFT_WARN_MEV_ATOM_PS,
                        help=f"漂移警告阈值 meV/atom/ps (默认: {DEFAULT_DRIFT_WARN_MEV_ATOM_PS})")
    parser.add_argument("--drift_strict_mev_atom_ps", type=float,
                        default=DEFAULT_DRIFT_STRICT_MEV_ATOM_PS,
                        help=f"严格模式漂移阈值 meV/atom/ps (默认: {DEFAULT_DRIFT_STRICT_MEV_ATOM_PS})")
    
    # Output options
    parser.add_argument("--extended_csv", action="store_true",
                        help="扩展 CSV 格式（含 raw_step/segment；若有压力再加 P_kB）")
    parser.add_argument("--include_outcar_only_steps", action="store_true",
                        help="包含仅在 OUTCAR 中的步")
    parser.add_argument("--self_check", action="store_true",
                        help="运行内置轻量自检并退出")
    
    args = parser.parse_args()

    if args.self_check:
        try:
            sys.exit(run_self_check())
        except AssertionError as exc:
            print(f"[SELF_CHECK][ERROR] {exc}")
            sys.exit(2)
    
    print("=" * 70)
    print("aimd_post.py v2.0 - AIMD 热力学数据后处理")
    print("=" * 70)
    print(f"OSZICAR: {args.oszicar}")
    print(f"OUTCAR: {args.outcar}")
    print(f"输出文件: {args.output}")
    print("=" * 70)
    
    # Check file existence
    oszicar_exists = os.path.isfile(args.oszicar)
    outcar_exists = os.path.isfile(args.outcar)
    
    if not oszicar_exists and not outcar_exists:
        print(f"[ERROR] OSZICAR 和 OUTCAR 都不存在")
        sys.exit(1)
    
    # Determine dt_fs
    dt_fs = args.dt_fs
    if dt_fs is None:
        dt_fs = read_potim_from_incar(args.incar)
        if dt_fs is not None:
            print(f"    从 INCAR 读取 POTIM = {dt_fs} fs")
    if dt_fs is None:
        if args.t_skip_ps is not None:
            print("    [ERROR] 使用 --t_skip_ps 时必须通过 --dt_fs 或 INCAR POTIM 提供时间步长")
            sys.exit(2)
        dt_fs = 1.0  # Default fallback only when t_skip_ps is not used
        print(f"    [INFO] 使用默认 dt_fs = {dt_fs} fs")
    dt_ps = dt_fs / 1000.0  # Convert to ps
    
    # Calculate skip steps
    skip_steps = args.t_skip_steps
    if args.t_skip_ps is not None:
        skip_from_ps = int(args.t_skip_ps / dt_ps)
        skip_steps = max(skip_steps, skip_from_ps)
        print(f"    跳过 {args.t_skip_ps} ps = {skip_from_ps} steps")
    
    # Determine n_atoms
    n_atoms = args.n_atoms
    if n_atoms is None and outcar_exists:
        n_atoms = read_nions_from_outcar(args.outcar)
        if n_atoms is not None:
            print(f"    从 OUTCAR 读取 NIONS = {n_atoms}")
    if n_atoms is None:
        print(f"    [INFO] n_atoms 未知，漂移将以 eV/ps 报告")
    
    # Parse data
    print("\n>>> 解析数据...")
    
    oszicar_data = []
    outcar_data = []
    outcar_iteration_info = None
    total_stars = 0
    total_segments = 0
    
    if oszicar_exists:
        print(f"    读取 OSZICAR...")
        oszicar_data, stars, segs = parse_oszicar(
            args.oszicar, allow_stars=args.allow_stars, strict=args.strict
        )
        total_stars += stars
        total_segments = max(total_segments, segs)
        print(f"    从 OSZICAR 读取 {len(oszicar_data)} 个离子步 ({segs} 段)")
    
    if outcar_exists:
        print(f"    读取 OUTCAR...")
        outcar_data, stars, segs, outcar_iteration_info = parse_outcar_thermo(
            args.outcar,
            allow_stars=args.allow_stars,
            strict=args.strict,
            iteration_order=args.iteration_order
        )
        total_stars += stars
        total_segments = max(total_segments, segs)
        print(f"    从 OUTCAR 读取 {len(outcar_data)} 个离子步 ({segs} 段)")
        print(f"    OUTCAR Iteration order: {outcar_iteration_info.get('debug_label', 'N/A')}")
    
    # Merge data
    data = merge_data(oszicar_data, outcar_data,
                      include_outcar_only=args.include_outcar_only_steps,
                      dedup=args.dedup)
    
    if len(data) == 0:
        print("[ERROR] 未能解析到任何热力学数据")
        sys.exit(1)
    
    print(f"\n>>> 合并后共 {len(data)} 个数据点")
    
    if args.renumber:
        data = sorted(data, key=lambda d: d.get('global_step', d.get('raw_step', d['step'])))
        for i, d in enumerate(data):
            d['global_step'] = i + 1
    else:
        data = sorted(data, key=lambda d: d.get('global_step', d.get('raw_step', d['step'])))
    
    # Check data completeness
    n_with_T = sum(1 for d in data if d.get('T') is not None)
    n_with_E0 = sum(1 for d in data if d.get('E0') is not None)
    n_with_F = sum(1 for d in data if d.get('F') is not None)
    n_with_P = sum(1 for d in data if d.get('P_kB') is not None)
    
    print(f"    有 E0 数据: {n_with_E0}/{len(data)}")
    print(f"    有 T 数据: {n_with_T}/{len(data)}")
    print(f"    有 F 数据: {n_with_F}/{len(data)}")
    print(f"    有 P 数据: {n_with_P}/{len(data)}")
    
    if n_with_T == 0:
        print("    [INFO] 无温度数据，可能是静态计算或结构优化")
    
    # Split into full and production data
    if skip_steps >= len(data):
        production_data = []
        print(f"    [WARNING] 跳过步数 {skip_steps} >= 总数据点 {len(data)}，生产段为空")
    else:
        production_data = data[skip_steps:]
    
    # Compute energy drift on production region
    drift_info = compute_drift(production_data, dt_ps, n_atoms)
    
    # Generate equilibration suggestion (advisory only)
    suggestion = suggest_equilibration(data, dt_ps)
    
    # Write CSV
    print(f"\n>>> 保存到 {args.output}...")
    write_csv(data, args.output, extended=args.extended_csv)
    print(f"[OK] 已保存 {len(data)} 行数据到 {args.output}")
    
    # Print summary
    print_summary(
        data=data,
        production_data=production_data,
        drift_info=drift_info,
        stars_total=total_stars,
        dt_ps=dt_ps,
        n_atoms=n_atoms,
        drift_warn_threshold=args.drift_warn_mev_atom_ps,
        drift_strict_threshold=args.drift_strict_mev_atom_ps,
        strict=args.strict,
        suggestion=suggestion,
        segment_count=total_segments,
        applied_skip_steps=skip_steps,
        iteration_info=outcar_iteration_info,
        n_lines=5
    )


if __name__ == "__main__":
    main()
