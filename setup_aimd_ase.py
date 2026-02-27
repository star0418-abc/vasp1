#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_aimd_ase.py v2.4.1 - 从大体系结构切割 AIMD 子体系并生成 VASP 输入

================================================================================
CHANGELOG v2.4.1:
================================================================================
1. [NEW] 显式 MIC 模式: --mic_mode {auto,on,off}
2. [FIX] MIC 决策统一由 resolve_use_mic() 控制，并贯穿选择/重成像/中和/碰撞检测
3. [FIX] model_meta.json 不再硬编码 mic_valid/bulk_density_valid，改为真实检查结果
4. [FIX] neutralize 元数据追加 charge_before/after 与 reliable_before/after
5. [DOC] CLI/README 与实际实现保持一致，移除 setup_aimd_ase.py 的 phantom 选项

================================================================================
CHANGELOG v2.4.0:
================================================================================
1. [PERF] Clash detection speedup: use get_all_distances(mic=True) once,
   instead of O(N^2) individual safe_get_distances() calls
2. [FIX] Multi-frame file: force read(filepath, index=0) to ensure first frame
3. [FIX] LANGEVIN_GAMMA order: parse POSCAR line 6 to match exact element order
   - New function: parse_poscar_element_order()
   - Guarantees NTYP order consistency with POSCAR
4. [FIX] Relax script: backup INCAR to INCAR.aimd before overwriting
5. [FIX] Cut-bond graph: use has_valid_cell() for cell/pbc consistency
6. [DOC] Clarified clash threshold logic comments

================================================================================
CHANGELOG v2.3:
================================================================================
1. [FIX] get_distances() return order guard (ASE version compatibility)
2. [FIX] Safer cell checks: atoms.pbc.any() AND cell.volume > 1e-8
3. [FIX] LANGEVIN_GAMMA format: per-element-type (NTYP) instead of per-atom
4. [FIX] Density sanity checks: warn if <0.5 or >3.0 g/cm³
5. [FIX] Neutralization verification: re-check charge after adding counterions
6. [FIX] Multi-frame file handling: safely take first frame from trajectories
7. [FIX] Strict --kpoints validation: must be exactly 3 integers
8. [NEW] Clash detection: detect atomic overlaps after density compression
9. [NEW] Relaxation guidance: RELAX_GUIDE.txt + optional INCAR.relax

关键改进 (v2.4.0 → v2.4.1):
  - ✅ MIC 使用可由 --mic_mode 显式控制
  - ✅ --mic_mode on 在无效 cell/pbc 下会 fail loudly，不再静默回退
  - ✅ model_meta.json 记录真实的 density/MIC 决策与中和轨迹
  - ✅ README 的 setup_aimd_ase.py 选项与 argparse 一致

用法:
  # bulk 模式（默认，按原体系密度定盒子）
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --mode bulk

  # bulk 模式，指定目标密度（带弛豫输入）
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 \\
      --density_g_cm3 1.2 --write_relax_inputs

  # cluster 模式（真空簇，需显式指定）
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 \\
      --mode cluster --vacuum 20

  # 使用 bond_hops 避免切断聚合物链
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --bond_hops 3

依赖:
  pip install ase numpy

作者: STAR0418-ABC
版本: v2.4.1
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import json
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Any
from datetime import datetime

import numpy as np

# 检查 ASE
try:
    from ase import Atoms
    from ase.io import read, write
    from ase.geometry import get_distances
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("[ERROR] 需要 ASE 库: pip install ase")
    sys.exit(1)

# 导入本地工具模块
try:
    from utils.connectivity import (
        build_bond_graph, find_connected_components,
        expand_by_bond_hops, detect_cut_bonds, write_cut_bonds_report
    )
    from utils.charges import (
        ELEMENT_CHARGE_MAP, RESIDUE_CHARGE_MAP,
        estimate_charge_by_residue, estimate_charge_by_element,
        load_charge_map_file,
        CATION_RESIDUES, ANION_RESIDUES
    )
    from utils.units import (
        compute_mass, compute_density, volume_from_density,
        scale_cell_to_volume, ATOMIC_MASSES
    )
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    print("[WARN] utils 模块未找到，使用内置简化版本")

VERSION = "v2.4.1"

# ==============================================================================
# 共价半径表（用于碰撞检测）
# ==============================================================================
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'S': 1.05, 'P': 1.07,
    'Li': 1.28, 'Na': 1.66, 'K': 2.03, 'Mg': 1.41, 'Ca': 1.76, 'Zn': 1.22, 'Al': 1.21,
    'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'Si': 1.11, 'B': 0.84,
}

# 常见离子组成（分量级电荷匹配，避免多原子离子按元素求和失真）
KNOWN_COMPONENT_FORMULA_CHARGES: Dict[Tuple[Tuple[str, int], ...], Tuple[int, str]] = {
    tuple(sorted({'C': 2, 'F': 6, 'N': 1, 'O': 4, 'S': 2}.items())): (-1, 'TFSI'),
    tuple(sorted({'P': 1, 'F': 6}.items())): (-1, 'PF6'),
    tuple(sorted({'B': 1, 'F': 4}.items())): (-1, 'BF4'),
    tuple(sorted({'Cl': 1, 'O': 4}.items())): (-1, 'ClO4'),
    tuple(sorted({'N': 1, 'O': 3}.items())): (-1, 'NO3'),
}

# ==============================================================================
# 内置简化版工具（当 utils 不可用时）- v2.4 修正: 真实键图实现
# ==============================================================================

# 尝试使用 ASE 的原子质量表（更完整）
try:
    from ase.data import atomic_masses as ASE_ATOMIC_MASSES
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
    _HAS_ASE_DATA = True
except ImportError:
    _HAS_ASE_DATA = False

# 内置原子质量表（ASE 不可用时的回退）
_INTERNAL_ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'S': 32.07,
    'Li': 6.941, 'Na': 22.99, 'K': 39.10, 'Mg': 24.31, 'Ca': 40.08,
    'Zn': 65.38, 'Al': 26.98, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90,
    'P': 30.97, 'Si': 28.09, 'B': 10.81, 'Fe': 55.85, 'Cu': 63.55,
    'Pt': 195.08, 'Au': 196.97, 'Ag': 107.87, 'Ti': 47.87, 'Ni': 58.69,
}

def _get_atomic_mass(symbol: str) -> float:
    """获取原子质量，优先使用 ASE 数据"""
    if _HAS_ASE_DATA:
        z = ASE_ATOMIC_NUMBERS.get(symbol)
        if z is not None and z >= 1:
            return ASE_ATOMIC_MASSES[z]
    # 回退到内置表
    mass = _INTERNAL_ATOMIC_MASSES.get(symbol)
    if mass is not None:
        return mass
    # 最终回退（警告并使用默认值）
    print(f"[WARN] 未知元素 '{symbol}' 的原子质量，使用 10.0 g/mol")
    return 10.0

# 键判定容差因子
_BOND_TOLERANCE = 1.3  # 距离 < (r1 + r2) * tolerance 视为成键
_DEFAULT_COVALENT_RADIUS = 1.5

def _get_covalent_radius(symbol: str) -> float:
    """获取元素的共价半径"""
    return COVALENT_RADII.get(symbol, _DEFAULT_COVALENT_RADIUS)

if not HAS_UTILS:
    ELEMENT_CHARGE_MAP = {
        'Li': +1, 'Na': +1, 'K': +1, 'Mg': +2, 'Ca': +2, 'Zn': +2, 'Al': +3,
        'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
        'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'P': 0,
    }

    def estimate_charge_by_element(
        symbols: List[str],
        charge_map: Optional[Dict[str, int]] = None
    ) -> Tuple[int, Dict[str, int], bool]:
        """按元素估算电荷（内置回退，避免 utils 缺失时 NameError）"""
        table = charge_map if charge_map else ELEMENT_CHARGE_MAP
        element_counts: Dict[str, int] = {}
        total_charge = 0
        is_reliable = True

        for sym in symbols:
            element_counts[sym] = element_counts.get(sym, 0) + 1
            charge = table.get(sym, None)
            if charge is None:
                charge = 0
                is_reliable = False
            total_charge += int(charge)

        return total_charge, element_counts, is_reliable

    def load_charge_map_file(filepath: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        """加载自定义电荷映射（简化回退版，支持 JSON / YAML）"""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"电荷映射文件不存在: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        with open(filepath, 'r', encoding='utf-8') as f:
            if ext in ['.yaml', '.yml']:
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("读取 YAML 需要 pyyaml: pip install pyyaml")
            else:
                data = json.load(f)

        residue_charges = {str(k).upper(): int(v) for k, v in data.get('residue_charges', {}).items()}
        element_charges = {str(k): int(v) for k, v in data.get('element_charges', {}).items()}
        return residue_charges, element_charges
    
    # 使用统一的原子质量获取函数
    ATOMIC_MASSES = _INTERNAL_ATOMIC_MASSES  # 保持向后兼容
    
    def compute_mass(symbols):
        return sum(_get_atomic_mass(s) for s in symbols)
    
    def compute_density(symbols, volume_A3):
        mass_g = compute_mass(symbols) / 6.022e23
        volume_cm3 = volume_A3 * 1e-24
        return mass_g / volume_cm3
    
    def volume_from_density(symbols, density):
        mass_g = compute_mass(symbols) / 6.022e23
        volume_cm3 = mass_g / density
        return volume_cm3 / 1e-24
    
    def scale_cell_to_volume(cell, target_vol, mode='scale_proportional'):
        if mode == 'cubic':
            L = target_vol ** (1/3)
            return np.diag([L, L, L])
        current_vol = abs(np.linalg.det(cell))
        if current_vol < 1e-10:
            L = target_vol ** (1/3)
            return np.diag([L, L, L])
        scale = (target_vol / current_vol) ** (1/3)
        return cell * scale
    
    def build_bond_graph(positions, symbols, cell=None, pbc=None, 
                         tolerance=_BOND_TOLERANCE, max_bond_length=3.5, **kwargs):
        """
        构建化学键图（邻接表）- 内置版本
        
        v2.4: 真实实现，不再返回空图
        """
        from collections import deque
        n_atoms = len(positions)
        graph = {i: set() for i in range(n_atoms)}
        
        # 预计算共价半径
        radii = np.array([_get_covalent_radius(s) for s in symbols])
        
        # 计算距离矩阵
        use_mic = cell is not None and pbc is not None and np.any(pbc)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # 计算距离
                if use_mic:
                    # 最小镜像距离
                    d_cart = positions[j] - positions[i]
                    try:
                        cell_inv = np.linalg.inv(cell)
                        d_frac = np.dot(d_cart, cell_inv)
                        d_frac -= np.round(d_frac)
                        d_cart = np.dot(d_frac, cell)
                    except np.linalg.LinAlgError:
                        pass  # 使用原始 d_cart
                    dist = np.linalg.norm(d_cart)
                else:
                    dist = np.linalg.norm(positions[j] - positions[i])
                
                # 键判定
                bond_cutoff = (radii[i] + radii[j]) * tolerance
                
                if dist < min(bond_cutoff, max_bond_length):
                    graph[i].add(j)
                    graph[j].add(i)
        
        return graph
    
    def find_connected_components(graph):
        """寻找图的连通分量（分子）- 内置版本"""
        from collections import deque
        visited = set()
        components = []
        
        for start in graph:
            if start in visited:
                continue
            
            component = set()
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    def detect_cut_bonds(graph, selected):
        """检测被切断的化学键 - 内置版本"""
        cut_bonds = []
        selected_set = set(selected)
        
        for atom_in in selected_set:
            for neighbor in graph.get(atom_in, set()):
                if neighbor not in selected_set:
                    bond = tuple(sorted([atom_in, neighbor]))
                    if bond not in cut_bonds:
                        cut_bonds.append((atom_in, neighbor))
        
        return cut_bonds
    
    def expand_by_bond_hops(graph, seed, max_hops, max_size):
        """从种子原子出发，按键跳扩展 - 内置版本"""
        expanded = set(seed)
        frontier = set(seed)
        
        for hop in range(max_hops):
            if len(expanded) >= max_size:
                break
            
            new_frontier = set()
            for atom in frontier:
                for neighbor in graph.get(atom, set()):
                    if neighbor not in expanded:
                        new_frontier.add(neighbor)
                        expanded.add(neighbor)
                        
                        if len(expanded) >= max_size:
                            return expanded
            
            frontier = new_frontier
            
            if not frontier:
                break
        
        return expanded
    
    def write_cut_bonds_report(filepath, cut_bonds, symbols, positions, cell=None):
        """写入切断键报告 - 内置版本"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 切断键报告 (Cut Bonds Report)\n")
            f.write(f"# 检测到 {len(cut_bonds)} 个被切断的化学键\n")
            f.write("# ⚠️ 警告: 切断化学键可能引入不物理的自由基/断链！\n")
            f.write("# 建议: 增大切割半径 / 使用 --bond_hops / 使用 molecule 模式\n")
            f.write("#\n")
            f.write("# atom_in_idx  atom_in_elem  atom_out_idx  atom_out_elem  distance_A\n")
            
            for atom_in, atom_out in cut_bonds:
                d = positions[atom_out] - positions[atom_in]
                if cell is not None:
                    try:
                        cell_inv = np.linalg.inv(cell)
                        d_frac = np.dot(d, cell_inv)
                        d_frac -= np.round(d_frac)
                        d = np.dot(d_frac, cell)
                    except:
                        pass
                dist = np.linalg.norm(d)
                
                f.write(f"{atom_in:6d}  {symbols[atom_in]:4s}  {atom_out:6d}  {symbols[atom_out]:4s}  {dist:.3f}\n")
    
    # 打印一次警告，记录使用内置版本
    print("[INFO] 使用内置键图实现 (无 utils 模块)")
else:
    # 当 HAS_UTILS=True 时也使用统一的原子质量获取
    def _get_atomic_mass_with_utils(symbol: str) -> float:
        """使用 utils.ATOMIC_MASSES 或 ASE 回退"""
        if symbol in ATOMIC_MASSES:
            return ATOMIC_MASSES[symbol]
        return _get_atomic_mass(symbol)


# ==============================================================================
# 辅助函数：安全检查与兼容性
# ==============================================================================

def has_valid_cell(atoms: Atoms) -> bool:
    """
    安全检查原子对象是否有有效的周期性晶胞
    
    替代不安全的 atoms.cell.any() 检查
    
    注意: 这是通用检查，对于特定用途请使用:
    - has_valid_cell_for_density(): 3D bulk 密度检查
    - has_valid_cell_for_mic(): MIC 距离计算
    """
    if not atoms.pbc.any():
        return False
    try:
        vol = atoms.cell.volume
        return vol > 1e-8
    except:
        return False


def has_valid_cell_for_density(atoms: Atoms) -> bool:
    """
    检查是否适合进行 3D bulk 密度计算
    
    要求: pbc.all() == True 且 cell.volume > 1e-8
    
    v2.4: 用于密度健全性检查，避免对 2D/1D/cluster 误报
    """
    if not np.all(atoms.pbc):  # 必须是三维周期性
        return False
    try:
        vol = atoms.cell.volume
        return vol > 1e-8
    except:
        return False


def has_valid_cell_for_mic(atoms: Atoms) -> bool:
    """
    检查是否可以使用 MIC (Minimum Image Convention) 计算距离
    
    要求: 任意 pbc 方向为 True 且对应 cell 分量有效
    
    v2.4: 用于距离/碰撞检测，cluster+vacuum 模式应返回 False
    """
    if not atoms.pbc.any():
        return False
    try:
        # 检查 cell 是否有足够的体积或有效的分量
        cell_lengths = atoms.cell.lengths()
        # 至少有一个方向有有效的周期长度
        valid_lengths = cell_lengths > 1e-6
        # PBC 方向必须有有效的 cell 长度
        for i in range(3):
            if atoms.pbc[i] and not valid_lengths[i]:
                return False
        return atoms.pbc.any() and atoms.cell.volume > 1e-8
    except:
        return False


def _cell_matrix_as_list(atoms: Atoms) -> List[List[float]]:
    """返回便于打印/写入元数据的 3x3 cell 矩阵"""
    return np.asarray(atoms.cell.array, dtype=float).round(6).tolist()


def _explain_invalid_mic(atoms: Atoms) -> str:
    """给出 MIC 无效的明确原因，供 fail-loudly 报错使用"""
    reasons: List[str] = []
    pbc = [bool(x) for x in atoms.pbc]

    if not any(pbc):
        reasons.append("pbc 没有任何 True 方向")

    try:
        cell_lengths = np.asarray(atoms.cell.lengths(), dtype=float)
        invalid_axes = [i for i in range(3) if pbc[i] and cell_lengths[i] <= 1e-6]
        if invalid_axes:
            axes_str = ", ".join(str(i) for i in invalid_axes)
            reasons.append(f"周期方向 {axes_str} 的 cell 长度无效 (<= 1e-6 Å)")
        volume = float(atoms.cell.volume)
        if volume <= 1e-8:
            reasons.append(f"cell 体积无效 (<= 1e-8 Å^3, 当前 {volume:.3e})")
    except Exception as exc:
        reasons.append(f"无法解析 cell 信息: {exc}")

    if not reasons:
        reasons.append("cell/pbc 未通过 has_valid_cell_for_mic() 检查")

    return "; ".join(reasons)


def _build_invalid_mic_error(atoms: Atoms, mic_mode: str) -> str:
    """构建 MIC fail-loudly 错误消息"""
    pbc = [bool(x) for x in atoms.pbc]
    cell_matrix = _cell_matrix_as_list(atoms)
    reason = _explain_invalid_mic(atoms)
    return (
        f"--mic_mode {mic_mode} 要求输入结构具备有效周期性 cell 才能使用 MIC。\n"
        f"检测到 pbc={pbc}\n"
        f"检测到 cell={cell_matrix}\n"
        f"MIC 无效原因: {reason}\n"
        "处理建议: 使用 --mic_mode off 禁用 MIC，或提供有效的周期性 cell/pbc。"
    )


def resolve_use_mic(atoms: Atoms, mic_mode: str) -> bool:
    """
    统一解析是否启用 MIC。

    这是 MIC 使用的单一决策入口，供选择/重成像/中和/碰撞检测复用。
    """
    mic_cell_valid = has_valid_cell_for_mic(atoms)

    if mic_mode == "off":
        return False
    if mic_mode == "on":
        if not mic_cell_valid:
            raise ValueError(_build_invalid_mic_error(atoms, mic_mode))
        return True
    if mic_mode == "auto":
        return mic_cell_valid

    raise ValueError(f"未知 mic_mode: {mic_mode}")


def get_mic_decision_reason(atoms: Atoms, mic_mode: str) -> str:
    """返回元数据所需的 MIC 决策原因字符串"""
    if mic_mode == "off":
        return "user_off"
    if mic_mode == "on":
        resolve_use_mic(atoms, mic_mode)
        return "user_on"
    return "auto_cell_valid" if has_valid_cell_for_mic(atoms) else "auto_cell_invalid"


def compute_center_vectors_and_distances(
    atoms: Atoms,
    center_idx: int,
    use_mic: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算中心到所有原子的位移向量和距离。

    use_mic=True 时使用 ASE MIC；否则使用普通笛卡尔距离。
    """
    positions = atoms.get_positions()
    center_pos = positions[center_idx]

    if use_mic:
        try:
            return safe_get_distances(center_pos, positions, cell=atoms.cell, pbc=atoms.pbc)
        except Exception as exc:
            raise RuntimeError(f"MIC 距离计算失败: {exc}") from exc

    vectors = positions - center_pos
    distances = np.linalg.norm(vectors, axis=1)
    return vectors, distances


def to_jsonable(value: Any) -> Any:
    """递归转换 numpy 标量/数组，确保可写入 JSON"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def safe_get_distances(
    center_pos: np.ndarray,
    positions: np.ndarray,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASE get_distances() 的安全包装，处理返回值顺序歧义
    
    不同 ASE 版本可能返回 (vectors, distances) 或 (distances, vectors)
    通过形状检测来确定正确顺序:
    - vectors: shape (1, n_atoms, 3) 或 (n_atoms, 3)，最后一维 = 3
    - distances: shape (1, n_atoms) 或 (n_atoms,)，是标量数组
    
    Returns:
        mic_vectors: (n_atoms, 3) MIC 位移向量
        distances: (n_atoms,) 距离数组
    """
    result = get_distances([center_pos], positions, cell=cell, pbc=pbc)
    
    ret0, ret1 = result[0], result[1]
    
    # 检测哪个是向量（最后一维是 3）
    shape0 = np.array(ret0).shape
    shape1 = np.array(ret1).shape
    
    # vectors 的最后一维应该是 3
    if len(shape0) >= 2 and shape0[-1] == 3:
        vectors = np.array(ret0)
        distances = np.array(ret1)
    elif len(shape1) >= 2 and shape1[-1] == 3:
        vectors = np.array(ret1)
        distances = np.array(ret0)
    else:
        # 回退：假设 (vectors, distances) 顺序
        vectors = np.array(ret0)
        distances = np.array(ret1)
    
    # 归一化形状
    if vectors.ndim == 3:
        vectors = vectors[0]  # (1, n, 3) -> (n, 3)
    if distances.ndim == 2:
        distances = distances[0]  # (1, n) -> (n,)
    
    return vectors, distances


def validate_kpoints(kpts_str: str) -> Tuple[int, int, int]:
    """验证 K 点字符串，必须是 3 个整数"""
    parts = kpts_str.split()
    if len(parts) != 3:
        raise ValueError(f"--kpoints 必须是 3 个整数 (如 '1 1 1')，收到: '{kpts_str}'")
    try:
        kpts = tuple(int(x) for x in parts)
    except ValueError:
        raise ValueError(f"--kpoints 必须是整数，收到: '{kpts_str}'")
    if any(k < 1 for k in kpts):
        raise ValueError(f"K 点值必须 >= 1，收到: {kpts}")
    return kpts


def safe_read_structure(filepath: str) -> Atoms:
    """
    安全读取结构文件，强制取第一帧
    
    v2.3.1: 使用 index=0 确保读取第一帧，避免不同 reader 行为差异
    """
    try:
        # 优先使用 index=0 强制取第一帧
        result = read(filepath, index=0)
    except TypeError:
        # 某些格式可能不支持 index 参数
        result = read(filepath)
    
    # 双重保险：如果仍然返回 list，取第一个
    if isinstance(result, list):
        if len(result) == 0:
            raise ValueError(f"文件 {filepath} 不包含任何结构")
        print(f"[INFO] 文件包含 {len(result)} 帧，使用第一帧")
        return result[0]
    
    return result


def parse_poscar_element_order(poscar_path: str) -> List[str]:
    """
    解析 POSCAR 第 6 行获取元素顺序
    
    v2.3.1: 确保 LANGEVIN_GAMMA 与 POSCAR 元素顺序严格一致
    
    POSCAR 格式:
        Line 1: Comment
        Line 2: Scale
        Line 3-5: Lattice vectors
        Line 6: Element symbols (e.g., "C H O Li F S N")
        Line 7: Atom counts
        ...
    
    Returns:
        elements: 元素符号列表，按 POSCAR 顺序
    """
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 7:
        raise ValueError(f"POSCAR 格式错误: {poscar_path}")
    
    # 第 6 行是元素符号（0-indexed line 5）
    element_line = lines[5].strip()
    elements = element_line.split()
    
    # 验证：应该都是元素符号，不是数字
    for elem in elements:
        if elem.isdigit():
            raise ValueError(f"POSCAR 第 6 行应为元素符号，但发现数字: {element_line}")
    
    return elements


# ==============================================================================
# 核心函数
# ==============================================================================

def parse_center_atom(center_str: str, atoms: Atoms, one_based: bool = False) -> int:
    """解析中心原子参数"""
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    
    try:
        idx = int(center_str)
        if one_based:
            idx -= 1
        if idx == n_atoms:
            print(f"[WARN] 索引 {center_str} = 原子总数，按 1-based 解释为 {idx - 1}")
            idx -= 1
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"原子索引 {idx} 超出范围 [0, {n_atoms - 1}]")
        return idx
    except ValueError:
        pass
    
    element = center_str.strip()
    for i, sym in enumerate(symbols):
        if sym == element:
            print(f"[INFO] 找到第一个 {element} 原子，索引 {i}")
            return i
    
    raise ValueError(f"未找到元素 '{element}'，可用: {set(symbols)}")


def select_indices_with_mic_vectors(
    atoms: Atoms,
    center_idx: int,
    radius: float,
    use_mic: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    选择中心原子周围指定半径内的所有原子，返回用于后续重成像的位移向量

    Returns:
        indices: 选中的原子索引
        distances: 到中心的距离
        mic_vectors: center -> atom 位移向量（MIC 或直角坐标）
    """
    mic_vectors, distances = compute_center_vectors_and_distances(atoms, center_idx, use_mic)
    indices = np.where(distances <= radius)[0]
    return indices, distances, mic_vectors


def reimage_atoms_around_center(
    atoms: Atoms,
    selected_indices: np.ndarray,
    center_idx: int,
    mic_vectors: np.ndarray,
    use_mic: bool,
    selection_mode: str = 'sphere',
    bond_hops: int = 0,
    radius: Optional[float] = None,
    inconsistency_tol: float = 1e-3
) -> Tuple[Atoms, List[str]]:
    """
    将选中的原子重新成像到中心附近（空间连贯），并返回警告信息。

    安全策略:
    1) sphere 且 bond_hops=0 时，允许中心 MIC 重成像，但半径必须满足
       radius <= 0.45 * min(cell_lengths[pbc=True])
    2) molecule 或 bond_hops>0 时，改用键图遍历重成像，保持拓扑连续性
    """
    warnings: List[str] = []
    center_pos = atoms.get_positions()[center_idx]
    symbols = atoms.get_chemical_symbols()

    selected_set = set(int(i) for i in selected_indices)
    if center_idx not in selected_set:
        raise ValueError("中心原子不在选中集合中，无法重成像")

    positions = atoms.get_positions()
    valid_cell = use_mic and has_valid_cell_for_mic(atoms)
    cell = atoms.cell if valid_cell else None
    pbc = atoms.pbc if valid_cell else None

    # polymer/connectivity 风险模式：使用键图拓扑展开
    use_topology_unwrap = (selection_mode == 'molecule') or (bond_hops > 0)

    if use_topology_unwrap:
        graph = build_bond_graph(positions, symbols, cell, pbc)
        placed: Dict[int, np.ndarray] = {int(center_idx): np.array(center_pos, dtype=float)}
        queue: deque[int] = deque([int(center_idx)])

        while len(placed) < len(selected_set):
            while queue:
                cur = queue.popleft()
                cur_pos = placed[cur]
                for nbr in sorted(graph.get(cur, set())):
                    if nbr not in selected_set:
                        continue
                    if valid_cell:
                        disp, _ = safe_get_distances(
                            positions[cur], np.array([positions[nbr]]),
                            cell=atoms.cell, pbc=atoms.pbc
                        )
                        bond_disp = np.array(disp[0], dtype=float)
                    else:
                        bond_disp = np.array(positions[nbr] - positions[cur], dtype=float)
                    candidate = cur_pos + bond_disp
                    if nbr not in placed:
                        placed[nbr] = candidate
                        queue.append(nbr)
                    else:
                        err = np.linalg.norm(placed[nbr] - candidate)
                        if err > inconsistency_tol:
                            warnings.append(
                                f"[WARN] 拓扑重成像一致性冲突: atom {nbr}, 偏差 {err:.3e} Å，保留首次放置"
                            )

            # 若存在与中心不连通的已选分量，分别锚定并继续拓扑展开
            remaining = sorted(selected_set - set(placed.keys()))
            if not remaining:
                break
            seed = int(remaining[0])
            placed[seed] = np.array(positions[seed], dtype=float)
            queue.append(seed)
            warnings.append(
                f"[WARN] 选中集合含与中心不连通分量（seed={seed}），按原坐标锚定后局部拓扑展开"
            )

        new_positions = [placed[int(idx)] for idx in selected_indices]
        new_symbols = [symbols[int(idx)] for idx in selected_indices]
    else:
        # sphere 模式: 半径太大时，中心 MIC 重成像会折叠跨边界链段，直接报错
        if valid_cell and radius is not None and np.any(atoms.pbc):
            pbc_lengths = atoms.cell.lengths()[np.array(atoms.pbc, dtype=bool)]
            if pbc_lengths.size > 0:
                safe_limit = 0.45 * float(np.min(pbc_lengths))
                if radius > safe_limit:
                    raise ValueError(
                        "sphere 模式下中心 MIC 重成像不安全: "
                        f"radius={radius:.3f} Å > 0.45*Lmin={safe_limit:.3f} Å. "
                        "请减小 --radius、改用 --selection molecule/--bond_hops，"
                        "或使用更大父晶胞。"
                    )

        new_positions = []
        new_symbols = []
        for idx in selected_indices:
            new_pos = center_pos + mic_vectors[idx]
            new_positions.append(new_pos)
            new_symbols.append(symbols[int(idx)])
    
    # 创建新 Atoms 对象
    cluster = Atoms(
        symbols=new_symbols,
        positions=new_positions
    )
    
    # 复制数组属性（如 residuenumbers）
    if hasattr(atoms, 'arrays'):
        for key, arr in atoms.arrays.items():
            if key in ['positions', 'numbers']:
                continue
            if len(arr) == len(atoms):
                try:
                    cluster.arrays[key] = arr[selected_indices]
                except:
                    pass
    
    return cluster, warnings


def get_residue_info(atoms: Atoms) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """获取残基信息"""
    arrays = atoms.arrays if hasattr(atoms, 'arrays') else {}
    
    # 残基名称
    resnames = None
    for key in ['residuenames', 'resname', 'resnames']:
        if key in arrays:
            resnames = [str(r) for r in arrays[key]]
            break
    
    # 残基编号
    resids = None
    for key in ['residuenumbers', 'resid', 'resids', 'molid']:
        if key in arrays:
            resids = list(arrays[key])
            break
    
    return resnames, resids


def expand_selection_with_molecules(
    atoms: Atoms,
    selected_indices: np.ndarray,
    selection_mode: str,
    bond_hops: int,
    max_atoms: int,
    allow_exceed: bool,
    use_mic: bool
) -> Tuple[np.ndarray, bool, int]:
    """
    扩展选择到完整分子/键跳
    
    Returns:
        final_indices: 最终选中的索引
        was_truncated: 是否被截断
        n_cut_bonds: 切断键数量
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    valid_cell = use_mic and has_valid_cell_for_mic(atoms)
    cell = atoms.cell if valid_cell else None
    pbc = atoms.pbc if valid_cell else None
    
    # v2.4: 总是构建键图（内置版本或 utils 版本都可用）
    graph = build_bond_graph(positions, symbols, cell, pbc)
    
    selected_set = set(selected_indices)
    
    # 根据模式扩展
    if selection_mode == 'molecule':
        # 尝试基于残基信息扩展
        resnames, resids = get_residue_info(atoms)
        
        if resnames is not None and resids is not None:
            # 按残基扩展
            touched_residues = set()
            for idx in selected_indices:
                touched_residues.add((resnames[idx], resids[idx]))
            
            expanded_set = set()
            for i in range(len(atoms)):
                if (resnames[i], resids[i]) in touched_residues:
                    expanded_set.add(i)
            
            selected_set = expanded_set
        else:
            # 回退到连通分量
            print("[INFO] 无残基信息，使用键图连通性扩展")
            components = find_connected_components(graph)
            touched_components = []
            for comp in components:
                if comp & set(selected_indices):
                    touched_components.append(comp)

            for comp in touched_components:
                if len(comp) <= max_atoms or allow_exceed:
                    selected_set.update(comp)
    
    # 键跳扩展
    # v2.4: 当 allow_exceed=True 时，使用更大的内部限制
    if bond_hops > 0:
        effective_max = 10000 if allow_exceed else max_atoms
        print(f"[INFO] 执行 {bond_hops} 步键跳扩展...")
        selected_set = expand_by_bond_hops(graph, selected_set, bond_hops, effective_max)
    
    # 检查原子数限制
    was_truncated = False
    if len(selected_set) > max_atoms and not allow_exceed:
        print(f"[WARN] 扩展后原子数 ({len(selected_set)}) 超过 max_atoms ({max_atoms})")
        print("[INFO] 回退到原始 sphere 选择")
        selected_set = set(selected_indices)
        was_truncated = True
    
    final_indices = np.array(sorted(selected_set))
    
    # 检测切断键 - v2.4: 总是执行（内置 fallback 也可用）
    cut_bonds = detect_cut_bonds(graph, set(final_indices))
    n_cut_bonds = len(cut_bonds)
    
    return final_indices, was_truncated, n_cut_bonds


def heal_cut_bonds(
    atoms: Atoms,
    selected_indices: np.ndarray,
    use_mic: bool,
    max_atoms: int = 400,
    allow_exceed: bool = False,
    max_iterations: int = 10
) -> Tuple[np.ndarray, int, int, List[str]]:
    """
    修复切断键：迭代添加缺失的成键伙伴
    
    v2.4: 按 AGENTS.md 规范实现 - 扩展选择以包含切断键的完整连通分量
    
    Args:
        atoms: 原子对象
        selected_indices: 当前选中的原子索引
        max_atoms: 最大原子数
        allow_exceed: 允许超过 max_atoms
        max_iterations: 最大迭代次数
    
    Returns:
        (healed_indices, initial_cut_bonds, final_cut_bonds, heal_log)
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    valid_cell = use_mic and has_valid_cell_for_mic(atoms)
    cell = atoms.cell if valid_cell else None
    pbc = atoms.pbc if valid_cell else None
    
    # 构建键图
    graph = build_bond_graph(positions, symbols, cell, pbc)
    
    # 检测初始切断键
    selected_set = set(selected_indices)
    cut_bonds = detect_cut_bonds(graph, selected_set)
    initial_n = len(cut_bonds)
    
    if initial_n == 0:
        return selected_indices, 0, 0, ["无切断键"]
    
    heal_log = [f"初始切断键数: {initial_n}"]
    
    # 获取连通分量
    if HAS_UTILS:
        components = find_connected_components(graph)
    else:
        # 内置版本
        from collections import deque
        visited = set()
        components = []
        for start in graph:
            if start in visited:
                continue
            component = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
    
    # 建立原子到分量的映射
    atom_to_comp = {}
    for i, comp in enumerate(components):
        for atom in comp:
            atom_to_comp[atom] = i
    
    effective_max = 10000 if allow_exceed else max_atoms
    
    for iteration in range(max_iterations):
        # 检测当前切断键
        cut_bonds = detect_cut_bonds(graph, selected_set)
        if len(cut_bonds) == 0:
            heal_log.append(f"迭代 {iteration+1}: 所有切断键已修复")
            break
        
        # 找出需要添加的分量
        components_to_add = set()
        for atom_in, atom_out in cut_bonds:
            # atom_out 不在选择中，需要添加其整个分量
            comp_idx = atom_to_comp.get(atom_out)
            if comp_idx is not None:
                components_to_add.add(comp_idx)
        
        # 尝试添加分量
        added_atoms = 0
        for comp_idx in components_to_add:
            comp = components[comp_idx]
            new_size = len(selected_set) + len(comp - selected_set)
            
            if new_size <= effective_max:
                added = len(comp - selected_set)
                selected_set.update(comp)
                added_atoms += added
                heal_log.append(f"迭代 {iteration+1}: 添加分量 {comp_idx} ({added} 原子)")
            else:
                heal_log.append(f"迭代 {iteration+1}: 跳过分量 {comp_idx} (会超过 max_atoms)")
        
        if added_atoms == 0:
            heal_log.append(f"迭代 {iteration+1}: 无法添加更多原子 (达到 max_atoms 限制)")
            break
    
    # 最终检测
    final_cut_bonds = detect_cut_bonds(graph, selected_set)
    final_n = len(final_cut_bonds)
    
    healed_indices = np.array(sorted(selected_set))
    
    return healed_indices, initial_n, final_n, heal_log


def create_density_based_bulk_box(
    cluster: Atoms,
    original_density: Optional[float],
    target_density: Optional[float],
    use_mic: bool,
    cell_shape: str = 'scale_parent',
    parent_cell: Optional[np.ndarray] = None,
    min_cell_length: float = 10.0,
    span_padding: float = 4.0,
    density_warn_pct: float = 10.0,
    clash_scale: float = 0.75,
    max_expand_iters: int = 5,
    expand_factor: float = 1.08
) -> Tuple[Atoms, float, float, List[str]]:
    """
    创建基于密度的周期性盒子
    
    公式: V_target = M_sub / ρ_target
    
    Args:
        cluster: 子体系原子
        original_density: 原体系密度 (g/cm³)
        target_density: 目标密度 (g/cm³)，None 则使用 original_density
        cell_shape: 'scale_parent' 或 'cubic'
        parent_cell: 父体系晶胞（用于 scale_parent）
        min_cell_length: 最小盒子边长 (Å)
    
    Returns:
        cluster: 带新晶胞的子体系
        target_density: 使用的目标密度
        achieved_density: 实际达到的密度
        warnings: 盒子构建相关警告
    """
    symbols = cluster.get_chemical_symbols()
    warnings: List[str] = []
    positions = cluster.get_positions()
    
    # 确定目标密度
    if target_density is None:
        if original_density is not None:
            target_density = original_density
        else:
            # 默认凝胶电解质密度
            target_density = 1.2
            print(f"[WARN] 无法确定原体系密度，使用默认值 {target_density} g/cm³")
    
    # 计算目标体积
    target_volume = volume_from_density(symbols, target_density)
    
    # 确保最小盒子尺寸
    min_volume = min_cell_length ** 3
    if target_volume < min_volume:
        print(f"[WARN] 目标体积 ({target_volume:.1f} Å³) 太小，调整到最小 ({min_volume:.1f} Å³)")
        target_volume = min_volume
    
    # 生成新晶胞
    if cell_shape == 'scale_parent' and parent_cell is not None:
        new_cell = scale_cell_to_volume(parent_cell, target_volume, 'scale_proportional')
    else:
        new_cell = scale_cell_to_volume(np.eye(3) * 10, target_volume, 'cubic')

    # 防止边界自交叠: 保证盒长覆盖真实空间跨度 + padding
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    span = max_pos - min_pos
    required_lengths = span + 2.0 * span_padding
    if hasattr(new_cell, 'lengths'):
        cell_lengths = np.array(new_cell.lengths(), dtype=float)
    else:
        cell_lengths = np.linalg.norm(np.array(new_cell, dtype=float), axis=1)
    enforced_lengths = np.maximum(cell_lengths, required_lengths)
    if np.any(enforced_lengths > cell_lengths + 1e-8):
        new_cell = np.diag(enforced_lengths)
        msg = (
            "Cell enlarged to prevent PBC self-intersection "
            f"(span={span.round(3)} Å, padding={span_padding:.2f} Å)."
        )
        warnings.append(msg)
        print(f"[WARN] {msg}")
    
    # 设置晶胞并居中
    cluster.set_cell(new_cell)
    cluster.set_pbc([True, True, True])
    cluster.center()

    # 若仍存在严重重叠，自动迭代放大晶胞
    last_clash_info = detect_atomic_clashes(cluster, use_mic=use_mic, scale=clash_scale)
    for it in range(max_expand_iters):
        if not last_clash_info['has_clashes']:
            break
        old_lengths = np.array(cluster.get_cell().lengths(), dtype=float)
        new_lengths = old_lengths * expand_factor
        cluster.set_cell(np.diag(new_lengths))
        cluster.center()
        msg = (
            f"Bulk cell auto-expanded ({it+1}/{max_expand_iters}) due to clashes: "
            f"d_min={last_clash_info['d_min']:.3f} Å"
        )
        warnings.append(msg)
        print(f"[WARN] {msg}")
        last_clash_info = detect_atomic_clashes(cluster, use_mic=use_mic, scale=clash_scale)

    if last_clash_info['has_clashes']:
        raise ValueError(
            "密度定盒后仍检测到严重原子重叠，自动放大晶胞失败。"
            "请减小 --density_g_cm3、增大 --radius、或先做结构预弛豫。"
        )
    
    # 计算实际达到的密度
    actual_volume = abs(np.linalg.det(cluster.get_cell()))
    achieved_density = compute_density(symbols, actual_volume)

    density_error_pct = abs(achieved_density - target_density) / target_density * 100.0
    if density_error_pct > density_warn_pct:
        msg = (
            "Cell enlarged to prevent PBC self-intersection; achieved density lower than target. "
            f"(target={target_density:.4f}, achieved={achieved_density:.4f}, dev={density_error_pct:.1f}%)"
        )
        warnings.append(msg)
        print(f"[WARN] {msg}")

    return cluster, target_density, achieved_density, warnings


def create_vacuum_box(cluster: Atoms, vacuum: float) -> Atoms:
    """
    创建真空盒子（cluster 模式）
    
    v2.4: 设置 pbc=[False, False, False] 避免 MIC 跨真空边界的伪距离
    """
    positions = cluster.get_positions()
    
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    size = max_pos - min_pos
    
    cell = size + 2 * vacuum
    cell = np.maximum(cell, vacuum * 2)
    
    cluster.set_cell(cell)
    # v2.4: cluster 模式关闭 PBC，避免 MIC 伪距离
    cluster.set_pbc([False, False, False])
    cluster.center()
    
    return cluster


def _formula_key_from_counts(counts: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    """元素计数字典 -> 可哈希公式键"""
    return tuple(sorted((sym, int(n)) for sym, n in counts.items() if n > 0))


def _estimate_component_charge_from_formula(
    component_indices: Set[int],
    symbols: List[str],
    element_charge_map: Optional[Dict[str, int]] = None
) -> Tuple[int, bool, str]:
    """
    估算单个连通分量电荷（优先已知多原子离子公式，其次单原子离子）

    Returns:
        charge: 分量电荷
        reliable: 是否可靠
        label: 匹配标签或原因
    """
    elem_map = element_charge_map if element_charge_map else ELEMENT_CHARGE_MAP
    counts: Dict[str, int] = {}
    comp_list = sorted(component_indices)
    for idx in comp_list:
        sym = symbols[idx]
        counts[sym] = counts.get(sym, 0) + 1

    # 单原子离子/原子: 使用元素映射（Li/Na/K/F/Cl...）
    if len(comp_list) == 1:
        sym = symbols[comp_list[0]]
        charge = elem_map.get(sym, None)
        if charge is None:
            return 0, False, f"单原子 {sym} 未知电荷"
        return int(charge), True, f"monatomic:{sym}"

    # 多原子离子: 禁止直接按元素求和，必须公式匹配
    key = _formula_key_from_counts(counts)
    match = KNOWN_COMPONENT_FORMULA_CHARGES.get(key)
    if match is None:
        formula = ''.join(f"{s}{n}" for s, n in key)
        return 0, False, f"未知多原子分量公式: {formula}"

    charge, label = match
    return int(charge), True, f"formula:{label}"


def _estimate_charge_by_components(
    atoms: Atoms,
    use_mic: bool,
    element_charge_map: Optional[Dict[str, int]] = None
) -> Tuple[int, Dict[str, int], bool, List[str]]:
    """
    基于键图连通分量 + 公式匹配估算总电荷。

    对多原子分量仅接受显式公式匹配，避免危险的元素求和。
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    element_counts: Dict[str, int] = {}
    for sym in symbols:
        element_counts[sym] = element_counts.get(sym, 0) + 1

    valid_cell = use_mic and has_valid_cell_for_mic(atoms)
    cell = atoms.cell if valid_cell else None
    pbc = atoms.pbc if valid_cell else None
    graph = build_bond_graph(positions, symbols, cell, pbc)
    components = find_connected_components(graph)

    total_charge = 0
    warnings: List[str] = []
    reliable = True

    for comp in components:
        comp_charge, comp_reliable, label = _estimate_component_charge_from_formula(
            comp, symbols, element_charge_map
        )
        total_charge += comp_charge
        if not comp_reliable:
            reliable = False
            warnings.append(f"无法可靠判定分量电荷: {label}")

    return int(total_charge), element_counts, reliable, warnings


def estimate_charge_comprehensive(
    cluster: Atoms,
    use_mic: bool,
    charge_map_file: Optional[str] = None
) -> Tuple[int, Dict[str, int], bool, List[str]]:
    """
    综合电荷估计：
    1) 残基映射（最优）
    2) 连通分量公式匹配（安全回退）
    3) 明确标记不可靠（不再对多原子离子做元素求和）
    
    Returns:
        total_charge: 总电荷
        counts: 组分计数
        is_reliable: 是否可靠
        warnings: 警告信息
    """
    symbols = cluster.get_chemical_symbols()
    element_counts: Dict[str, int] = {}
    for sym in symbols:
        element_counts[sym] = element_counts.get(sym, 0) + 1
    resnames, resids = get_residue_info(cluster)
    
    # 加载自定义电荷表
    custom_res_map = None
    custom_elem_map = None
    if charge_map_file:
        try:
            custom_res_map, custom_elem_map = load_charge_map_file(charge_map_file)
            print(f"[INFO] 已加载自定义电荷表: {charge_map_file}")
        except Exception as e:
            print(f"[WARN] 加载电荷表失败: {e}")
    
    # 尝试残基估计
    if resnames is not None and resids is not None and HAS_UTILS:
        selected_set = set(range(len(cluster)))
        charge, counts, reliable, warnings = estimate_charge_by_residue(
            resnames, resids, selected_set,
            custom_res_map, symbols, custom_elem_map
        )
        if reliable:
            return charge, element_counts, reliable, warnings
        warnings = warnings + ["残基信息不完整，回退到分量公式匹配"]

    # 回退: 分量公式匹配（安全，避免多原子离子按元素求和）
    comp_charge, element_counts, comp_reliable, comp_warnings = _estimate_charge_by_components(
        cluster, use_mic, custom_elem_map
    )
    if comp_reliable:
        return comp_charge, element_counts, True, ["使用分量公式匹配估算电荷"]

    # 最终仍不可靠：显式返回不可靠状态，调用方据此决定是否允许继续
    fallback_warnings = [
        "电荷估计不可靠：存在无法匹配的分量公式",
        "未对多原子分量执行元素求和（为避免化学上不安全的电荷）"
    ]
    if 'warnings' in locals():
        fallback_warnings = warnings + fallback_warnings
    fallback_warnings.extend(comp_warnings)
    return comp_charge, element_counts, False, fallback_warnings


def neutralize_by_residue(
    atoms: Atoms,
    selected_indices: np.ndarray,
    center_idx: int,
    current_charge: int,
    target_charge: int,
    all_distances: np.ndarray,
    use_mic: bool
) -> Tuple[np.ndarray, int, List[str]]:
    """
    按残基/分子中和电荷
    
    Returns:
        new_indices: 添加反离子后的索引
        added_count: 添加的原子数
        info: 信息列表
    """
    if current_charge == target_charge:
        return selected_indices, 0, []
    
    resnames, resids = get_residue_info(atoms)
    info = []
    
    if resnames is not None and resids is not None and HAS_UTILS:
        # 确保选中集合不包含残基的“半截”
        selected_set = set(selected_indices)
        residue_groups: Dict[Tuple[str, int], Set[int]] = {}
        for idx, (resname, resid) in enumerate(zip(resnames, resids)):
            residue_groups.setdefault((resname, resid), set()).add(idx)

        expanded_selected = set()
        for (resname, resid), indices in residue_groups.items():
            if indices & selected_set:
                expanded_selected.update(indices)

        if expanded_selected != selected_set:
            selected_set = expanded_selected
            selected_indices = np.array(sorted(selected_set))
            info.append("[INFO] 已将选中集合扩展为完整残基，避免部分残基参与中和")

        needed_charge = current_charge - target_charge
        target_residues = ANION_RESIDUES if needed_charge > 0 else CATION_RESIDUES

        residue_candidates = []
        residue_groups: Dict[Tuple[str, int], Set[int]] = {}
        for idx, (resname, resid) in enumerate(zip(resnames, resids)):
            residue_groups.setdefault((str(resname).upper().strip(), int(resid)), set()).add(idx)

        for (resname, resid), group in residue_groups.items():
            if group & selected_set:
                continue
            if resname not in target_residues:
                continue

            charge = RESIDUE_CHARGE_MAP.get(resname, 0)
            if charge == 0:
                continue

            group_indices = sorted(group)
            residue_candidates.append({
                "indices": group,
                "distance": float(np.min(all_distances[group_indices])),
                "charge": abs(int(charge)),
                "label": f"{resname}:{resid}"
            })

        residue_candidates.sort(key=lambda item: item["distance"])

        added_indices = []
        remaining = abs(needed_charge)
        for cand in residue_candidates:
            if remaining <= 0:
                break
            group = cand["indices"]
            added_indices.extend(group)
            remaining -= cand["charge"]
            info.append(
                f"添加残基 {cand['label']} ({len(group)} 原子, 距离 {cand['distance']:.2f} Å)"
            )
        
        if added_indices:
            new_indices = np.concatenate([selected_indices, np.array(added_indices)])
            new_indices = np.unique(new_indices)
            return new_indices, len(added_indices), info
    
    # v2.4 回退：按连通分量（分子）中和，而非单原子
    # 这避免了从多原子阴离子中选取单个原子破坏化学计量
    info.append("[INFO] 无残基信息，使用键图连通分量中和")
    
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    selected_set = set(selected_indices)
    
    # 构建键图
    valid_cell = use_mic and has_valid_cell_for_mic(atoms)
    cell_arr = atoms.cell if valid_cell else None
    pbc_arr = atoms.pbc if valid_cell else None
    graph = build_bond_graph(positions, symbols, cell_arr, pbc_arr)
    
    # 获取连通分量
    components = find_connected_components(graph)
    
    needed_charge = current_charge - target_charge
    target_charge_sign = -1 if needed_charge > 0 else +1
    
    # 按连通分量分组候选
    component_candidates = []
    for comp in components:
        # 跳过已选中的分量
        if comp & selected_set:
            continue

        # 分量电荷必须能被可靠识别（公式匹配或单原子映射）
        comp_charge, comp_reliable, charge_label = _estimate_component_charge_from_formula(
            comp, symbols
        )
        if not comp_reliable:
            continue
        if comp_charge * target_charge_sign <= 0:
            continue  # 电荷符号不匹配
        
        # 使用与中心选择同源的距离数组排序，避免 MIC 行为不一致
        comp_indices = sorted(comp)
        dist = float(np.min(all_distances[comp_indices]))
        
        component_candidates.append({
            'indices': comp,
            'charge': abs(comp_charge),
            'signed_charge': int(comp_charge),
            'distance': dist,
            'size': len(comp),
            'label': charge_label
        })
    
    # 按距离排序
    component_candidates.sort(key=lambda x: x['distance'])
    
    added_all = []
    remaining = abs(needed_charge)
    for cand in component_candidates:
        if remaining <= 0:
            break
        added_all.extend(cand['indices'])
        remaining -= cand['charge']
        info.append(
            f"添加分量 {cand['label']} ({cand['size']} 原子, 电荷 {cand['signed_charge']:+d}, 距离 {cand['distance']:.2f} Å)"
        )
    
    if added_all:
        new_indices = np.concatenate([selected_indices, np.array(list(added_all))])
        new_indices = np.unique(new_indices)
        return new_indices, len(added_all), info
    
    info.append("[WARN] 未找到合适的反离子分量")
    return selected_indices, 0, info


def detect_atomic_clashes(
    cluster: Atoms,
    use_mic: bool,
    scale: float = 0.75,
    global_min_h: float = 0.8,
    global_min_other: float = 1.2
) -> Dict[str, Any]:
    """
    检测原子碰撞/重叠（v2.3.1 加速版）
    
    当密度压缩后，边界原子可能重叠，导致 AIMD 不稳定。
    
    v2.3.1: 使用 get_all_distances(mic=True) 一次性计算距离矩阵，
            避免 O(N^2) 次 safe_get_distances() 调用
            N=400 时加速约 100x
    
    v2.4: 使用 has_valid_cell_for_mic() 判断是否使用 MIC，
          cluster 模式 (pbc=False) 不使用 MIC，避免跨真空伪距离
    
    Args:
        cluster: 原子对象（已设置 PBC）
        scale: 共价半径比例阈值 (默认 0.75)
        global_min_h: 涉及 H 的全局最小距离阈值 (Å)
        global_min_other: 其他原子对的全局最小距离阈值 (Å)
    
    阈值逻辑说明:
        对每对原子 (i, j)，计算两种阈值:
        1. threshold_cov = (r_cov_i + r_cov_j) * scale  # 基于共价半径
        2. threshold_global = 0.8 Å (涉及H) 或 1.2 Å (其他)
        取 min(threshold_cov, threshold_global) 作为最终阈值。
        
        这是"宽松"策略：只捕捉严重重叠（d < 较小阈值），
        避免误报正常的短距离（如 H-O 氢键 ~1.8Å）。
    
    Returns:
        clash_info: 包含 d_min, clash_count, clash_pairs, has_clashes
    """
    n_atoms = len(cluster)
    symbols = cluster.get_chemical_symbols()
    
    dist_mat = cluster.get_all_distances(mic=use_mic)  # (N, N) 对称矩阵
    
    clash_pairs = []
    d_min = float('inf')
    d_min_pair = None
    
    # 遍历上三角矩阵（i < j）
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d_ij = dist_mat[i, j]
            
            # 更新最小距离
            if d_ij < d_min:
                d_min = d_ij
                d_min_pair = (i, j)
            
            # 检测碰撞
            sym_i, sym_j = symbols[i], symbols[j]
            
            # 方法1：基于共价半径比例
            r_cov_i = COVALENT_RADII.get(sym_i, 1.0)
            r_cov_j = COVALENT_RADII.get(sym_j, 1.0)
            threshold_cov = (r_cov_i + r_cov_j) * scale
            
            # 方法2：全局硬阈值
            if 'H' in (sym_i, sym_j):
                threshold_global = global_min_h
            else:
                threshold_global = global_min_other
            
            # 取较小值（宽松策略：只抓严重重叠，避免误报）
            threshold = min(threshold_cov, threshold_global)
            
            if d_ij < threshold:
                clash_pairs.append({
                    'i': int(i),
                    'j': int(j),
                    'sym_i': sym_i,
                    'sym_j': sym_j,
                    'distance': float(d_ij),
                    'threshold': float(threshold)
                })
    
    # 按距离排序
    clash_pairs.sort(key=lambda x: x['distance'])
    
    return {
        'd_min': float(d_min) if d_min < float('inf') else None,
        'd_min_pair': d_min_pair,
        'clash_count': len(clash_pairs),
        'clash_pairs': clash_pairs[:20],  # 只保留前 20 个
        'has_clashes': len(clash_pairs) > 0
    }


def write_relax_guide(
    outdir: str,
    clash_info: Dict[str, Any],
    has_h: bool,
    gamma_1ps: float,
    volume_compression_ratio: float,
    is_cluster: bool = False,
    dipol_suggestion: Optional[np.ndarray] = None
) -> None:
    """
    写入弛豫/预平衡指导文件
    
    当检测到碰撞或大体积压缩时，AIMD 前需要弛豫
    """
    guide_path = os.path.join(outdir, 'RELAX_GUIDE.txt')
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"AIMD 前处理指南 - 生成自 setup_aimd_ase.py {VERSION}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【重要警告】\n")
        f.write("-" * 70 + "\n")
        
        if clash_info['has_clashes']:
            f.write(f"⚠️  检测到 {clash_info['clash_count']} 对原子碰撞/重叠！\n")
            f.write(f"⚠️  最小原子间距: {clash_info['d_min']:.3f} Å\n")
            f.write("⚠️  直接运行 AIMD 可能导致:\n")
            f.write("    - 巨大的初始力和能量\n")
            f.write("    - SCF 不收敛\n")
            f.write("    - 模拟爆炸/崩溃\n\n")
        
        if volume_compression_ratio > 0.3:
            f.write(f"⚠️  体积压缩比例: {volume_compression_ratio*100:.1f}%\n")
            f.write("⚠️  大幅压缩可能导致原子重叠\n\n")
        
        f.write("\n【推荐工作流】\n")
        f.write("-" * 70 + "\n")
        f.write("步骤 1: 离子弛豫 (必须！)\n")
        f.write("        - 使用 INCAR.relax (如已生成)\n")
        f.write("        - IBRION = 2 (共轭梯度)\n")
        f.write("        - NSW = 200-500\n")
        f.write("        - ISIF = 2 (固定晶胞)\n")
        f.write(f"        - POTIM = {0.5 if has_h else 1.0} (含 H 用更小步长)\n")
        f.write("        - EDIFFG = -0.02 到 -0.05\n\n")
        
        f.write("步骤 2: NVT 预平衡\n")
        f.write("        - 从弛豫后的 CONTCAR 启动\n")
        f.write("        - 较小 POTIM (0.5-1.0 fs)\n")
        f.write("        - 较大 LANGEVIN_GAMMA (10-20) 快速控温\n")
        f.write("        - NSW = 2000-5000\n\n")
        
        f.write("步骤 3: 生产 AIMD\n")
        f.write("        - 从预平衡 CONTCAR 启动\n")
        f.write("        - 正常 POTIM (1.0-2.0 fs)\n")
        f.write("        - 较小 LANGEVIN_GAMMA (1-5) 减少对动力学干扰\n\n")
        
        if gamma_1ps >= 10:
            f.write(f"【注意】当前 LANGEVIN_GAMMA = {gamma_1ps}\n")
            f.write("        适合平衡段，生产段建议降至 1-5\n\n")

        if is_cluster:
            f.write("【Cluster 偶极修正建议】\n")
            f.write("-" * 70 + "\n")
            f.write("cluster+vacuum 在 VASP 中仍是 3D 周期边界，\n")
            f.write("若体系非中心对称且具有净偶极，建议显式测试偶极修正收敛:\n")
            f.write("  LDIPOL = .TRUE.\n")
            f.write("  IDIPOL = 1/2/3 (按主偶极方向选择)\n")
            if dipol_suggestion is not None:
                f.write("  # 可先用晶胞中心作为初值\n")
                f.write(f"  DIPOL = {dipol_suggestion[0]:.4f} {dipol_suggestion[1]:.4f} {dipol_suggestion[2]:.4f}\n")
            f.write("  # 也可用分子质心（见 INCAR.cluster_hint）\n\n")
        
        f.write("\n【碰撞详情】\n")
        f.write("-" * 70 + "\n")
        if clash_info['has_clashes']:
            f.write("最差的原子对 (前 20 个):\n")
            for i, cp in enumerate(clash_info['clash_pairs'][:20]):
                f.write(f"  {i+1:2d}. 原子 {cp['i']:4d} ({cp['sym_i']}) - "
                       f"原子 {cp['j']:4d} ({cp['sym_j']}): "
                       f"{cp['distance']:.3f} Å (阈值 {cp['threshold']:.3f} Å)\n")
        else:
            f.write("未检测到严重碰撞（d_min 可能仍较小，建议弛豫）\n")
        
        f.write("\n" + "=" * 70 + "\n")


def write_cluster_dipole_hint(outdir: str, cluster: Atoms) -> None:
    """
    写入 cluster 模式偶极修正提示（不强制修改 INCAR）
    """
    hint_path = os.path.join(outdir, 'INCAR.cluster_hint')
    cell_lengths = cluster.get_cell().lengths()
    cell_center = 0.5 * np.array(cell_lengths, dtype=float)
    com = cluster.get_center_of_mass()

    with open(hint_path, 'w', encoding='utf-8') as f:
        f.write(f"# Cluster dipole correction hints - Generated by setup_aimd_ase.py {VERSION}\n")
        f.write("# 仅提示，不会自动写入 INCAR。请按体系对称性与收敛性测试后采用。\n\n")
        f.write("LDIPOL = .TRUE.\n")
        f.write("# IDIPOL: 1=x, 2=y, 3=z (按主偶极方向选)\n")
        f.write("# 若需全分子修正，请按你的 VASP 版本文档确认设置。\n")
        f.write("IDIPOL = 3\n\n")
        f.write("# 方案A: 用晶胞中心\n")
        f.write(f"DIPOL = {cell_center[0]:.6f} {cell_center[1]:.6f} {cell_center[2]:.6f}\n\n")
        f.write("# 方案B: 用分子质心（可替换上行）\n")
        f.write(f"# DIPOL = {com[0]:.6f} {com[1]:.6f} {com[2]:.6f}\n")


def write_incar_relax(
    outdir: str,
    encut: float,
    ncore: Optional[int],
    has_h: bool,
    n_element_types: int
) -> None:
    """写入弛豫用 INCAR"""
    incar_path = os.path.join(outdir, 'INCAR.relax')
    
    potim = 0.5 if has_h else 1.0
    
    with open(incar_path, 'w', encoding='utf-8') as f:
        f.write(f"# INCAR for relaxation - Generated by setup_aimd_ase.py {VERSION}\n")
        f.write("# 用于密度压缩后的离子弛豫（AIMD 前必须步骤）\n\n")
        
        f.write("# ============ 基础参数 ============\n")
        f.write("PREC = Accurate\n")
        f.write(f"ENCUT = {encut}\n")
        f.write("ALGO = Normal\n")
        f.write("EDIFF = 1E-6\n")
        f.write("NELM = 200\n")
        f.write("LREAL = Auto\n\n")
        
        f.write("# ============ 展宽 ============\n")
        f.write("ISMEAR = 0\n")
        f.write("SIGMA = 0.05\n\n")
        
        f.write("# ============ 离子弛豫 ============\n")
        f.write("IBRION = 2       # 共轭梯度\n")
        f.write("ISIF = 2         # 固定晶胞，弛豫离子\n")
        f.write("NSW = 300        # 最大步数\n")
        f.write(f"POTIM = {potim}        # 步长 (含 H 用更小值)\n")
        f.write("EDIFFG = -0.03   # 力收敛标准 (eV/Å)\n")
        f.write("ISYM = 0         # 关闭对称性\n\n")
        
        f.write("# ============ 输出控制 ============\n")
        f.write("LWAVE = .FALSE.\n")
        f.write("LCHARG = .TRUE.  # 保留 CHGCAR 供后续计算\n")
        
        if ncore:
            f.write(f"\nNCORE = {ncore}\n")


def write_relax_script(outdir: str) -> None:
    """
    写入弛豫-AIMD 运行脚本
    
    v2.3.1: 自动备份 INCAR 到 INCAR.aimd
    """
    script_path = os.path.join(outdir, 'run_relax_then_aimd.sh')
    
    # v2.3.1: 如果 INCAR 存在，先备份为 INCAR.aimd
    incar_path = os.path.join(outdir, 'INCAR')
    incar_aimd_path = os.path.join(outdir, 'INCAR.aimd')
    if os.path.exists(incar_path) and not os.path.exists(incar_aimd_path):
        shutil.copy(incar_path, incar_aimd_path)
        print(f"    [INFO] 已备份 INCAR → INCAR.aimd")
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# ================================================================\n")
        f.write("# 弛豫 + AIMD 分步运行脚本\n")
        f.write(f"# 生成自 setup_aimd_ase.py {VERSION}\n")
        f.write("# ================================================================\n\n")
        
        f.write("set -e  # 出错即停\n\n")
        
        f.write("# ================== Step 1: 离子弛豫 ==================\n")
        f.write("echo '>>> Step 1: 离子弛豫'\n")
        f.write("if [ -f INCAR.relax ]; then\n")
        f.write("    # 备份 AIMD INCAR（如果尚未备份）\n")
        f.write("    if [ -f INCAR ] && [ ! -f INCAR.aimd ]; then\n")
        f.write("        cp INCAR INCAR.aimd\n")
        f.write("        echo '[INFO] 已备份 INCAR → INCAR.aimd'\n")
        f.write("    fi\n")
        f.write("    cp INCAR.relax INCAR\n")
        f.write("    echo '[INFO] 已设置 INCAR.relax → INCAR'\n")
        f.write("    echo '[INFO] 请运行 VASP 进行弛豫:'\n")
        f.write("    echo '        mpirun -np 8 vasp_std > vasp_relax.out 2>&1'\n")
        f.write("else\n")
        f.write("    echo '[ERROR] 未找到 INCAR.relax'\n")
        f.write("    exit 1\n")
        f.write("fi\n\n")
        
        f.write("# ================== Step 2: 弛豫后准备 ==================\n")
        f.write("# 弛豫完成后运行以下命令:\n")
        f.write("#   cp CONTCAR POSCAR\n")
        f.write("#   cp INCAR.aimd INCAR\n")
        f.write("#   # 然后运行 AIMD\n")
        f.write("#   mpirun -np 8 vasp_std > vasp_aimd.out 2>&1\n\n")
        
        f.write("echo ''\n")
        f.write("echo '================================================================'\n")
        f.write("echo '弛豫完成后的命令:'\n")
        f.write("echo '  cp CONTCAR POSCAR'\n")
        f.write("echo '  cp INCAR.aimd INCAR'\n")
        f.write("echo '  # 运行 AIMD'\n")
        f.write("echo '================================================================'\n")
    
    os.chmod(script_path, 0o755)


def write_model_meta(
    outdir: str,
    args: argparse.Namespace,
    n_atoms: int,
    center_idx: int,
    center_symbol: str,
    total_charge: int,
    element_counts: Dict[str, int],
    was_neutralized: bool,
    neutralized_count: int,
    selection_truncated: bool,
    cell_lengths: np.ndarray,
    final_pbc: List[bool],
    density_original: Optional[float],
    density_target: Optional[float],
    density_achieved: float,
    density_input_is_3d_valid: bool,
    density_decision_reason: str,
    mic_cell_valid: bool,
    mic_used_for_selection: bool,
    mic_decision_reason: str,
    input_pbc: List[bool],
    input_cell_matrix: List[List[float]],
    n_cut_bonds: int,
    charge_warnings: List[str],
    clash_info: Optional[Dict[str, Any]] = None,
    neutralize_verified: bool = True,
    center_idx_cluster: int = 0,
    remaining_charge: Optional[int] = None,
    healed: bool = False,
    initial_cut_bonds: int = 0,
    neutralize_charge_before: Optional[int] = None,
    neutralize_reliable_before: Optional[bool] = None,
    neutralize_charge_after: Optional[int] = None,
    neutralize_reliable_after: Optional[bool] = None
) -> None:
    """
    写入模型元数据
    """
    is_cluster = args.mode == "cluster"
    cluster_pbc_after_box = final_pbc if is_cluster else None

    meta = {
        "generator": f"setup_aimd_ase.py {VERSION}",
        "timestamp": datetime.now().isoformat(),
        "model_type": args.mode,
        "source_file": args.src,
        "selection": {
            "center_index_original": int(center_idx),
            "center_index_cluster": int(center_idx_cluster),
            "center_element": center_symbol,
            "selected_atom_count": n_atoms,
            "radius_angstrom": args.radius,
            "selection_mode": args.selection,
            "bond_hops": args.bond_hops,
            "allow_exceed_max_atoms": args.allow_exceed_max_atoms if hasattr(args, 'allow_exceed_max_atoms') else False
        },
        # 兼容旧字段
        "center_atom": {
            "index": int(center_idx),
            "element": center_symbol
        },
        "radius_angstrom": args.radius,
        "selection_mode": args.selection,
        "bond_hops": args.bond_hops,
        "n_atoms": n_atoms,
        "estimated_charge": total_charge,
        "element_counts": element_counts,
        "cell_angstrom": [float(x) for x in cell_lengths],
        "pbc": final_pbc,
        "has_vacuum": is_cluster,
        "vacuum_angstrom": args.vacuum if is_cluster else None,
        "pbc_mic": {
            "input_pbc": input_pbc,
            "input_cell_matrix_angstrom": input_cell_matrix,
            "mic_mode": args.mic_mode,
            "mic_used_for_selection": mic_used_for_selection,
            "mic_cell_valid": mic_cell_valid,
            "mic_decision_reason": mic_decision_reason,
            "pbc_after_box": final_pbc,
            "cluster_pbc_after_box": cluster_pbc_after_box
        },
        "density": {
            "original_g_cm3": density_original,
            "target_g_cm3": density_target,
            "achieved_g_cm3": density_achieved,
            "used_for_box": args.mode == "bulk",
            "density_input_is_3d_valid": density_input_is_3d_valid,
            "density_decision_reason": density_decision_reason
        },
        "neutralization": {
            "enabled": args.neutralize != "none",
            "method": args.neutralize,
            "target_charge": args.target_charge,
            "atoms_added": neutralized_count,
            "was_neutralized": was_neutralized,
            "charge_before": neutralize_charge_before,
            "reliable_before": neutralize_reliable_before,
            "charge_after": neutralize_charge_after,
            "reliable_after": neutralize_reliable_after,
            "remaining_charge": remaining_charge,
            "verified": neutralize_verified
        },
        "cut_bonds": {
            "policy": args.cut_bond_policy if hasattr(args, 'cut_bond_policy') else "warn",
            "initial_count": initial_cut_bonds,
            "final_count": n_cut_bonds,
            "healed": healed,
            "report_file": "cut_bonds_report.txt" if n_cut_bonds > 0 or initial_cut_bonds > 0 else None
        },
        "aimd_params": {
            "temperature_K": args.temp,
            "steps": args.steps,
            "potim_fs": args.potim,
            "thermostat": args.thermostat,
            "gamma_1ps": args.gamma_1ps
        },
        "warnings": []
    }
    
    # 添加碰撞检测信息
    if clash_info is not None:
        meta["clash_check"] = {
            "d_min_angstrom": clash_info['d_min'],
            "clash_count": clash_info['clash_count'],
            "has_clashes": clash_info['has_clashes'],
            "worst_pairs": clash_info['clash_pairs'][:10],
            "relaxation_recommended": clash_info['has_clashes']
        }
    
    # 添加警告
    if abs(total_charge) >= 1:
        meta["warnings"].append(f"体系可能带电 ({total_charge:+d})")
    if selection_truncated:
        meta["warnings"].append("molecule/bond_hops 模式被截断为 sphere")
    if n_cut_bonds > 0:
        meta["warnings"].append(f"检测到 {n_cut_bonds} 个切断的化学键")
    if args.mode == "cluster":
        meta["warnings"].append("CLUSTER 模式：有限簇模型，不能用于 bulk 输运性质！")
        if cluster_pbc_after_box != [False, False, False]:
            meta["warnings"].append("cluster 模式输出的 pbc 不是 [False, False, False]")
    if density_target and density_achieved:
        error = abs(density_achieved - density_target) / density_target * 100
        if error > 5:
            meta["warnings"].append(f"密度偏差 {error:.1f}%")
    if clash_info and clash_info['has_clashes']:
        meta["warnings"].append(f"检测到 {clash_info['clash_count']} 对原子碰撞！需要弛豫！")
    if not neutralize_verified and args.neutralize != 'none':
        meta["warnings"].append("中和后电荷验证失败，可能未完全中和")
    meta["warnings"].extend(charge_warnings)
    
    json_path = os.path.join(outdir, "model_meta.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(meta), f, ensure_ascii=False, indent=2)


def write_vasp_inputs(
    cluster: Atoms,
    outdir: str,
    temp: float,
    steps: int,
    potim: float,
    kpoints: Tuple[int, int, int],
    ncore: Optional[int],
    encut: float,
    is_bulk: bool,
    thermostat: str,
    gamma_1ps: float
) -> Tuple[bool, int]:
    """
    生成 VASP AIMD 输入文件
    
    v2.3.1: LANGEVIN_GAMMA 顺序从 POSCAR 元素行解析，确保一致性
    
    Returns:
        potcar_generated: 是否生成了 POTCAR
        n_element_types: 元素种类数 (NTYP)
    """
    os.makedirs(outdir, exist_ok=True)
    
    pp_path = os.environ.get('VASP_PP_PATH', '')
    has_pp = bool(pp_path and os.path.isdir(pp_path))
    potcar_generated = False
    
    # 检查含 H
    symbols = cluster.get_chemical_symbols()
    has_h = 'H' in symbols
    if has_h and potim > 1.5:
        print(f"[WARN] 体系含 H，POTIM={potim} fs 可能过大，建议 0.5-1.0 fs")
    
    poscar_path = os.path.join(outdir, 'POSCAR')
    
    if has_pp:
        try:
            from ase.calculators.vasp import Vasp
            
            mdalgo = 3 if thermostat == 'langevin' else 2
            
            calc_params = {
                'directory': outdir,
                'xc': 'PBE',
                'encut': encut,
                'prec': 'Normal',
                'algo': 'VeryFast',
                'ediff': 1e-5,
                'nelm': 200,
                'ismear': 0,
                'sigma': 0.05,
                'ibrion': 0,
                'mdalgo': mdalgo,
                'isym': 0,
                'tebeg': temp,
                'teend': temp,
                'potim': potim,
                'nsw': steps,
                'lreal': 'Auto',
                'lwave': False,
                'lcharg': False,
                'kpts': kpoints,
                'gamma': True,
            }
            
            if ncore:
                calc_params['ncore'] = ncore
            
            calc = Vasp(**calc_params)
            calc.write_input(cluster)
            potcar_generated = True
            
            # v2.3.1: 从 POSCAR 解析元素顺序，确保 LANGEVIN_GAMMA 一致
            if thermostat == 'langevin':
                poscar_elements = parse_poscar_element_order(poscar_path)
                n_element_types = len(poscar_elements)
                _append_langevin_to_incar(outdir, gamma_1ps, n_element_types, poscar_elements)
            else:
                poscar_elements = parse_poscar_element_order(poscar_path)
                n_element_types = len(poscar_elements)
            
        except Exception as e:
            print(f"[WARN] ASE Vasp 写入失败: {e}")
            has_pp = False
    
    if not has_pp:
        # 先写 POSCAR
        write(poscar_path, cluster, format='vasp')
        
        # v2.3.1: 从 POSCAR 解析元素顺序
        poscar_elements = parse_poscar_element_order(poscar_path)
        n_element_types = len(poscar_elements)
        
        _write_incar_manual(outdir, temp, steps, potim, encut, ncore, is_bulk, 
                          thermostat, gamma_1ps, n_element_types, poscar_elements)
        _write_kpoints_manual(outdir, kpoints)
        
        if not pp_path:
            print("\n[WARN] VASP_PP_PATH 未设置，未生成 POTCAR")
        print("[INFO] 请手动准备 POTCAR")
    
    write(os.path.join(outdir, 'cluster_visual.xyz'), cluster, format='xyz')
    
    return potcar_generated, n_element_types


def _append_langevin_to_incar(
    outdir: str, 
    gamma_1ps: float, 
    n_element_types: int,
    poscar_elements: Optional[List[str]] = None
):
    """
    追加 Langevin 参数到 INCAR
    
    v2.3.1: 元素顺序从 POSCAR 第 6 行解析，确保与 VASP 一致
    """
    incar_path = os.path.join(outdir, 'INCAR')
    
    element_str = " ".join(poscar_elements) if poscar_elements else f"(NTYP={n_element_types})"
    
    with open(incar_path, 'a') as f:
        f.write("\n# Langevin thermostat parameters\n")
        f.write(f"# LANGEVIN_GAMMA: per-element-type, order follows POSCAR: {element_str}\n")
        f.write(f"LANGEVIN_GAMMA = {' '.join([str(gamma_1ps)] * n_element_types)}\n")


def _write_incar_manual(
    outdir: str,
    temp: float,
    steps: int,
    potim: float,
    encut: float,
    ncore: Optional[int],
    is_bulk: bool,
    thermostat: str,
    gamma_1ps: float,
    n_element_types: int,
    poscar_elements: Optional[List[str]] = None
) -> None:
    """
    手动写入 INCAR
    
    v2.3.1: LANGEVIN_GAMMA 顺序与 POSCAR 元素行严格一致
    """
    incar_path = os.path.join(outdir, 'INCAR')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mode_str = "BULK（周期性子胞）" if is_bulk else "CLUSTER（真空簇）"
    
    # MDALGO: 2=Nosé-Hoover, 3=Langevin
    mdalgo = 3 if thermostat == 'langevin' else 2
    mdalgo_comment = "Langevin 恒温器" if mdalgo == 3 else "Nosé-Hoover 恒温器"
    
    element_str = " ".join(poscar_elements) if poscar_elements else f"NTYP={n_element_types}"
    
    with open(incar_path, 'w') as f:
        f.write(f"# INCAR for AIMD - Generated by setup_aimd_ase.py {VERSION}\n")
        f.write(f"# {timestamp}\n")
        f.write(f"# Mode: {mode_str}\n")
        f.write(f"# Thermostat: {thermostat} (MDALGO={mdalgo})\n")
        f.write(f"# Temperature: {temp} K, Steps: {steps}, POTIM: {potim} fs\n")
        
        if not is_bulk:
            f.write("#\n")
            f.write("# ⚠️ WARNING: CLUSTER 模式 - 有限簇模型！\n")
            f.write("# ⚠️ 扩散系数等输运性质不能与 bulk 结果比较！\n")
            f.write("#\n")
        
        f.write("\n# ============ 基础参数 ============\n")
        f.write("PREC = Normal\n")
        f.write(f"ENCUT = {encut}\n")
        f.write("ALGO = VeryFast\n")
        f.write("EDIFF = 1E-5\n")
        f.write("NELM = 200\n")
        f.write("LREAL = Auto\n\n")
        
        f.write("# ============ 展宽 ============\n")
        f.write("ISMEAR = 0\n")
        f.write("SIGMA = 0.05\n\n")
        
        f.write("# ============ 分子动力学 ============\n")
        f.write("IBRION = 0      # MD 模式\n")
        f.write(f"MDALGO = {mdalgo}      # {mdalgo_comment}\n")
        f.write("ISYM = 0        # AIMD 必须关闭对称性！\n")
        f.write(f"TEBEG = {temp}\n")
        f.write(f"TEEND = {temp}\n")
        f.write(f"POTIM = {potim}  # fs，含 H 建议 0.5-1.0\n")
        f.write(f"NSW = {steps}\n")
        
        if thermostat == 'langevin':
            f.write(f"\n# Langevin 恒温器摩擦系数 (1/ps)\n")
            f.write(f"# 元素顺序与 POSCAR 第 6 行一致: {element_str}\n")
            if gamma_1ps >= 10:
                f.write(f"# ⚠️ gamma={gamma_1ps} 较大，适合平衡段；生产段建议 1-5\n")
            f.write(f"LANGEVIN_GAMMA = {' '.join([str(gamma_1ps)] * n_element_types)}\n")
        else:
            f.write("SMASS = -3      # Nosé-Hoover\n")
        
        f.write("\n# ============ 输出控制 ============\n")
        f.write("LWAVE = .FALSE.\n")
        f.write("LCHARG = .FALSE.\n")
        
        if ncore:
            f.write(f"\nNCORE = {ncore}\n")


def _write_kpoints_manual(outdir: str, kpoints: Tuple[int, int, int]):
    """手动写入 KPOINTS"""
    with open(os.path.join(outdir, 'KPOINTS'), 'w') as f:
        f.write("Automatic mesh\n0\nGamma\n")
        f.write(f"{kpoints[0]} {kpoints[1]} {kpoints[2]}\n0 0 0\n")


def write_selected_indices(outdir: str, center_idx: int, selected_indices: np.ndarray):
    """写入选中索引"""
    with open(os.path.join(outdir, 'selected_indices.txt'), 'w') as f:
        f.write(f"# Center atom index (0-based): {center_idx}\n")
        f.write(f"# Total selected atoms: {len(selected_indices)}\n")
        for idx in selected_indices:
            f.write(f"{idx}\n")


def write_index_map(outdir: str, selected_indices: np.ndarray, center_idx: int) -> int:
    """
    写入原始索引与 cluster 索引映射，并返回中心原子的 cluster 索引
    """
    cluster_to_original = [int(i) for i in selected_indices.tolist()]
    original_to_cluster = {str(orig): int(i) for i, orig in enumerate(cluster_to_original)}

    center_key = str(int(center_idx))
    if center_key not in original_to_cluster:
        raise ValueError("中心原子不在最终选中集合中，无法写入 index_map.json")
    center_idx_cluster = int(original_to_cluster[center_key])

    index_map = {
        "cluster_to_original": cluster_to_original,
        "original_to_cluster": original_to_cluster,
        "center_index_original": int(center_idx),
        "center_index_cluster": center_idx_cluster
    }

    with open(os.path.join(outdir, 'index_map.json'), 'w', encoding='utf-8') as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)

    return center_idx_cluster


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"从大体系切割 AIMD 子体系 (setup_aimd_ase.py {VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # bulk 模式（默认，按原体系密度定盒子）
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --mode bulk

  # bulk 模式，指定目标密度
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --density_g_cm3 1.2

  # cluster 模式（真空簇）
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --mode cluster --vacuum 20

  # 显式控制 MIC：无效 cell 时强制报错
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --mic_mode on

  # 使用 bond_hops 避免切断聚合物链
  python3 setup_aimd_ase.py --src eq.pdb --center_atom Li --radius 8 --bond_hops 3

注意:
  - 默认 bulk 模式按原体系密度计算盒子体积（避免"低压气相"陷阱）
  - cluster 模式必须显式指定 --mode cluster
  - --mic_mode auto 仅在输入 cell/pbc 有效时启用 MIC
  - 切断化学键会生成警告报告
        """
    )
    
    # 必需参数
    parser.add_argument("--src", required=True, help="输入结构文件")
    parser.add_argument("--center_atom", required=True, help="中心原子索引或元素")
    
    # 模式
    parser.add_argument("--mode", choices=['bulk', 'cluster'], default='bulk',
                        help="bulk=周期性(默认), cluster=真空簇")
    
    # 切割参数
    parser.add_argument("--radius", type=float, default=8.0, help="切割半径 Å")
    parser.add_argument("--selection", choices=['sphere', 'molecule'], default='sphere',
                        help="选择模式")
    parser.add_argument("--bond_hops", type=int, default=0,
                        help="键跳扩展步数（避免切断聚合物链）")
    parser.add_argument("--one_based", action="store_true", help="1-based 索引")
    
    # 盒子/密度参数
    parser.add_argument("--vacuum", type=float, default=20.0,
                        help="真空层 Å（仅 cluster）")
    parser.add_argument("--density_g_cm3", type=float, default=None,
                        help="目标密度 g/cm³（bulk 模式）")
    parser.add_argument("--cell_shape", choices=['scale_parent', 'cubic'], default='scale_parent',
                        help="盒子形状")
    parser.add_argument("--bulk_span_padding", type=float, default=4.0,
                        help="bulk 模式最小边界留白 Å（防止 PBC 自交叠，默认 4.0）")
    parser.add_argument("--bulk_density_warn_pct", type=float, default=10.0,
                        help="bulk 模式密度偏差警告阈值 (%%)，默认 10")
    parser.add_argument("--bulk_clash_max_expand_iter", type=int, default=5,
                        help="bulk 自动扩胞消除重叠的最大迭代次数（默认 5）")
    
    # 电荷
    parser.add_argument("--neutralize", choices=['none', 'nearest_counterions'], default='none',
                        help="电荷中和")
    parser.add_argument("--target_charge", type=int, default=0, help="目标电荷")
    parser.add_argument("--charge_map_file", type=str, default=None,
                        help="自定义电荷映射文件")
    
    # 限制
    parser.add_argument("--max_atoms", type=int, default=400, help="最大原子数")
    parser.add_argument("--allow_exceed_max_atoms", action="store_true")
    
    # AIMD 参数
    parser.add_argument("--temp", type=float, default=350.0, help="温度 K")
    parser.add_argument("--steps", type=int, default=2000, help="步数")
    parser.add_argument("--potim", type=float, default=1.0, help="POTIM fs")
    parser.add_argument("--thermostat", choices=['langevin', 'nose'], default='langevin',
                        help="恒温器")
    parser.add_argument("--gamma_1ps", type=float, default=10.0,
                        help="Langevin gamma (1/ps)")
    parser.add_argument("--kpoints", default="1 1 1", help="K 点")
    parser.add_argument("--ncore", type=int, default=None)
    parser.add_argument("--encut", type=float, default=400.0)
    
    # 输出
    parser.add_argument("--outdir", default="aimd_sub", help="输出目录")
    parser.add_argument("--overwrite", action="store_true")
    
    # v2.3 新增: 碰撞检测与弛豫输入
    parser.add_argument("--write_relax_inputs", action="store_true",
                        help="生成弛豫辅助文件 (INCAR.relax, RELAX_GUIDE.txt)")
    parser.add_argument("--clash_threshold_scale", type=float, default=0.75,
                        help="碰撞检测阈值比例 (默认 0.75)")
    parser.add_argument("--force_density", action="store_true",
                        help="强制使用可疑密度（跳过警告）[已弃用，使用 --density_check]")
    
    # v2.4 新增: 切断键策略、密度检查策略、索引映射
    parser.add_argument("--cut_bond_policy", choices=['heal', 'warn', 'error'], default='heal',
                        help="切断键策略: heal=自动扩展选择修复(默认), warn=警告继续, error=报错退出")
    parser.add_argument("--density_check", choices=['strict', 'warn', 'skip'], default='warn',
                        help="密度健全性检查: strict=异常时报错, warn=警告继续(默认), skip=跳过")
    parser.add_argument("--mic_mode", choices=['auto', 'on', 'off'], default='auto',
                        help="MIC 使用模式: auto=输入 cell 有效时启用(默认), on=强制启用并校验, off=禁用")
    parser.add_argument("--write_index_map", action="store_true",
                        help="写入 index_map.json (原始↔cluster 索引映射)")
    
    args = parser.parse_args()
    
    # 验证
    if args.mode == 'bulk' and args.vacuum != 20.0:
        print("[ERROR] --vacuum 仅在 cluster 模式有效")
        sys.exit(1)
    
    if args.mode == 'cluster' and args.density_g_cm3 is not None:
        print("[ERROR] --density_g_cm3 仅在 bulk 模式有效")
        sys.exit(1)
    
    print("=" * 70)
    print(f"setup_aimd_ase.py {VERSION} - AIMD 子体系切割")
    print("=" * 70)
    
    mode_desc = "BULK（周期性子胞，按密度定盒子）" if args.mode == "bulk" else "CLUSTER（真空簇）"
    print(f"模式: {mode_desc}")
    
    if args.mode == "cluster":
        print("")
        print("!" * 70)
        print("!!! 警告: CLUSTER 模式 - 有限真空簇！")
        print("!!! 扩散系数等输运性质不能与 bulk 结果比较！")
        print("!" * 70)
    
    print(f"输入: {args.src}")
    print(f"中心: {args.center_atom}, 半径: {args.radius} Å")
    print(f"选择: {args.selection}, bond_hops: {args.bond_hops}")
    print(f"MIC 模式: {args.mic_mode}")
    if args.density_g_cm3:
        print(f"目标密度: {args.density_g_cm3} g/cm³")
    print("=" * 70)
    
    # 检查文件
    if not os.path.isfile(args.src):
        print(f"[ERROR] 文件不存在: {args.src}")
        sys.exit(1)
    
    # 检查输出目录
    if os.path.exists(args.outdir):
        if args.overwrite:
            backup = f"{args.outdir}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(args.outdir, backup)
        else:
            print(f"[ERROR] 目录已存在: {args.outdir}，使用 --overwrite")
            sys.exit(1)
    
    # 验证 K 点
    try:
        kpts = validate_kpoints(args.kpoints)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # 读取结构
    print("\n>>> 读取结构...")
    try:
        atoms = safe_read_structure(args.src)
        print(f"    原子数: {len(atoms)}")
        print(f"    元素: {set(atoms.get_chemical_symbols())}")

        original_density = None
        parent_cell = None
        original_volume = None
        density_input_is_3d_valid = False
        input_pbc = [bool(x) for x in atoms.pbc]
        input_cell_matrix = _cell_matrix_as_list(atoms)

        # 使用安全的晶胞检查
        if has_valid_cell(atoms):
            atoms.wrap()
            parent_cell = atoms.cell.array.copy()
            original_volume = abs(np.linalg.det(parent_cell))

            # bulk 且未显式指定目标密度时，必须是有效 3D 周期体系
            density_input_is_3d_valid = has_valid_cell_for_density(atoms)
            if args.mode == 'bulk' and args.density_g_cm3 is None and not density_input_is_3d_valid:
                print("[ERROR] bulk 模式在未提供 --density_g_cm3 时要求输入为有效 3D 周期体系")
                print("[ERROR] 请提供 --density_g_cm3，或修正输入结构的 cell/pbc（需 pbc=[T,T,T]）")
                sys.exit(1)

            if density_input_is_3d_valid:
                original_density = compute_density(atoms.get_chemical_symbols(), original_volume)
                print(f"    原体系密度: {original_density:.4f} g/cm³")

                # v2.4: 密度健全性检查 - 仅对真正的 3D 周期体系执行
                density_check = args.density_check if hasattr(args, 'density_check') else 'warn'
                # 兼容旧版 --force_density
                if args.force_density:
                    density_check = 'skip'

                if args.mode == 'bulk' and args.density_g_cm3 is None and density_check != 'skip':
                    density_anomaly = original_density < 0.5 or original_density > 3.0
                    if density_anomaly:
                        print("")
                        print("!" * 70)
                        print(f"!!! 警告: 原体系密度 {original_density:.4f} g/cm³ 异常！")
                        if original_density < 0.5:
                            print("!!! 密度过低，可能是气相或未正确设置晶胞")
                        else:
                            print("!!! 密度过高，可能是晶胞设置错误")
                        print("!!! 强烈建议使用 --density_g_cm3 显式指定目标密度")
                        print("!" * 70)
                        if density_check == 'strict':
                            print("[ERROR] 使用 --density_check warn 允许继续，或指定 --density_g_cm3")
                            sys.exit(1)
                        else:
                            print("[WARN] 继续执行（密度可能不正确）")
            else:
                print("    [WARN] 检测到非 3D 周期 cell/pbc，原体系密度将不用于 bulk 反推")
        else:
            print("    [WARN] 无 PBC/cell")
            if args.mode == 'bulk' and args.density_g_cm3 is None:
                print("[ERROR] bulk 模式需要 --density_g_cm3（无法从输入获取）")
                sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        sys.exit(1)

    if args.density_g_cm3 is not None:
        density_decision_reason = "explicit_density"
    elif density_input_is_3d_valid:
        density_decision_reason = "3d_valid"
    elif args.mode == 'bulk':
        density_decision_reason = "3d_invalid_error"
    else:
        density_decision_reason = "3d_invalid_relaxed"

    mic_cell_valid = has_valid_cell_for_mic(atoms)
    try:
        use_mic = resolve_use_mic(atoms, args.mic_mode)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.mic_mode == "off":
        mic_decision_reason = "user_off"
    elif args.mic_mode == "on":
        mic_decision_reason = "user_on"
    else:
        mic_decision_reason = "auto_cell_valid" if mic_cell_valid else "auto_cell_invalid"

    print(f"    MIC 解析: {args.mic_mode} -> {'on' if use_mic else 'off'} ({mic_decision_reason})")

    # 解析中心
    print("\n>>> 解析中心原子...")
    try:
        center_idx = parse_center_atom(args.center_atom, atoms, args.one_based)
        center_sym = atoms.get_chemical_symbols()[center_idx]
        print(f"    中心: {center_sym} (索引 {center_idx})")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # 选择原子（带 MIC 向量）
    print("\n>>> 选择原子...")
    try:
        selected_indices, all_distances, mic_vectors = select_indices_with_mic_vectors(
            atoms, center_idx, args.radius, use_mic
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(f"    初步选中: {len(selected_indices)} 原子")
    
    # 扩展选择
    selected_indices, truncated, n_cut_bonds = expand_selection_with_molecules(
        atoms, selected_indices, args.selection, args.bond_hops,
        args.max_atoms, args.allow_exceed_max_atoms, use_mic
    )
    
    if truncated:
        print("[WARN] 选择被截断到 sphere 模式")
    
    # v2.4: 切断键处理（基于策略）
    cut_bond_policy = args.cut_bond_policy if hasattr(args, 'cut_bond_policy') else 'warn'
    heal_log = []
    initial_cut_bonds = n_cut_bonds
    final_cut_bonds = n_cut_bonds
    healed = False
    
    if n_cut_bonds > 0:
        print(f"\n>>> 切断键处理 (策略: {cut_bond_policy})...")
        print(f"    检测到 {n_cut_bonds} 个切断键")
        
        if cut_bond_policy == 'heal':
            # 尝试修复切断键
            selected_indices, initial_cut_bonds, final_cut_bonds, heal_log = heal_cut_bonds(
                atoms, selected_indices, use_mic, args.max_atoms, args.allow_exceed_max_atoms
            )
            healed = (final_cut_bonds < initial_cut_bonds)
            
            for log_line in heal_log:
                print(f"    {log_line}")
            
            if final_cut_bonds > 0:
                print(f"    [WARN] 仍有 {final_cut_bonds} 个切断键无法修复")
                if not args.allow_exceed_max_atoms:
                    print(f"    [ERROR] 无法修复所有切断键，切断键可能引入自由基")
                    print(f"    [ERROR] 使用 --allow_exceed_max_atoms 允许扩展，或 --cut_bond_policy warn 继续")
                    sys.exit(1)
            else:
                print(f"    [OK] 所有切断键已修复 (新原子数: {len(selected_indices)})")
        
        elif cut_bond_policy == 'error':
            print(f"    [ERROR] 切断键策略为 'error'，检测到 {n_cut_bonds} 个切断键")
            print(f"    [ERROR] 使用 --cut_bond_policy heal 尝试修复，或 --cut_bond_policy warn 忽略")
            sys.exit(1)
        
        else:  # warn
            print(f"    [WARN] 继续执行，但切断键可能引入不物理的自由基")
            print(f"    [WARN] 建议增大 --radius 或使用 --bond_hops 或 --cut_bond_policy heal")
    
    # 更新切断键计数
    n_cut_bonds = final_cut_bonds
    
    # 电荷中和
    neutralized = False
    neutralized_count = 0
    neutralize_info = []
    neutralize_verified = True
    neutralize_charge_before: Optional[int] = None
    neutralize_reliable_before: Optional[bool] = None
    neutralize_charge_after: Optional[int] = None
    neutralize_reliable_after: Optional[bool] = None
    remaining_charge: Optional[int] = None
    
    if args.neutralize == 'nearest_counterions':
        print("\n>>> 电荷中和...")
        temp_cluster = atoms[selected_indices]
        current_charge, _, current_reliable, current_warnings = estimate_charge_comprehensive(
            temp_cluster, use_mic, args.charge_map_file
        )
        neutralize_charge_before = current_charge
        neutralize_reliable_before = current_reliable
        neutralize_charge_after = current_charge
        neutralize_reliable_after = current_reliable
        remaining_charge = current_charge - args.target_charge
        print(f"    当前电荷: {current_charge:+d}, 目标: {args.target_charge:+d}")

        if not current_reliable:
            print("    [ERROR] 当前电荷估计不可靠，拒绝执行中和。")
            for w in current_warnings:
                print(f"    [WARN] {w}")
            print("    [ERROR] Provide residue info in the input OR supply --charge_map_file OR disable --neutralize.")
            sys.exit(1)
        
        if current_charge != args.target_charge:
            selected_indices, neutralized_count, neutralize_info = neutralize_by_residue(
                atoms, selected_indices, center_idx,
                current_charge, args.target_charge,
                all_distances, use_mic
            )
            if neutralized_count > 0:
                neutralized = True
                print(f"    添加 {neutralized_count} 原子")
                for info in neutralize_info:
                    print(f"    {info}")
                
                # v2.3: 中和验证
                temp_cluster_verify = atoms[selected_indices]
                verify_charge, _, verify_reliable, verify_warnings = estimate_charge_comprehensive(
                    temp_cluster_verify, use_mic, args.charge_map_file
                )
                neutralize_charge_after = verify_charge
                neutralize_reliable_after = verify_reliable
                remaining_charge = verify_charge - args.target_charge
                if (not verify_reliable) or (verify_charge != args.target_charge):
                    neutralize_verified = False
                    print(f"    [WARN] 中和验证: 电荷 {verify_charge:+d}，目标 {args.target_charge:+d}")
                    if not verify_reliable:
                        print("    [WARN] 验证阶段电荷估计不可靠")
                        for w in verify_warnings:
                            print(f"    [WARN] {w}")
                        print(f"    [WARN] 可能需要更多反离子或调整选择半径")
                else:
                    print(f"    [OK] 中和验证通过: 电荷 = {verify_charge:+d}")
            else:
                neutralize_verified = False
                print("    [WARN] 未能添加反离子，目标电荷未满足")
        else:
            print("    [OK] 已满足目标电荷，无需添加反离子")
    
    print(f"\n>>> 最终: {len(selected_indices)} 原子")
    
    # 检查限制
    if len(selected_indices) > args.max_atoms and not args.allow_exceed_max_atoms:
        print(f"[ERROR] 超过 max_atoms ({args.max_atoms})")
        sys.exit(1)
    
    runtime_warnings: List[str] = []

    # 重新成像
    print("\n>>> 重新成像原子到中心附近...")
    try:
        cluster, reimage_warnings = reimage_atoms_around_center(
            atoms,
            selected_indices,
            center_idx,
            mic_vectors,
            use_mic=use_mic,
            selection_mode=args.selection,
            bond_hops=args.bond_hops,
            radius=args.radius
        )
        runtime_warnings.extend(reimage_warnings)
        for w in reimage_warnings:
            print(f"    {w}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # 估算电荷
    total_charge, elem_counts, charge_reliable, charge_warnings = estimate_charge_comprehensive(
        cluster, use_mic, args.charge_map_file
    )
    if args.neutralize != 'none':
        neutralize_charge_after = total_charge
        neutralize_reliable_after = charge_reliable
        remaining_charge = total_charge - args.target_charge
    runtime_warnings.extend(charge_warnings)
    print(f"    估算电荷: {total_charge:+d}")
    if not charge_reliable:
        print("    [WARN] 电荷估计可能不准确")
        for w in charge_warnings:
            print(f"    [WARN] {w}")
    
    # 创建盒子
    print("\n>>> 创建盒子...")
    if args.mode == 'bulk':
        try:
            cluster, target_density, achieved_density, box_warnings = create_density_based_bulk_box(
                cluster, original_density, args.density_g_cm3, use_mic,
                args.cell_shape, parent_cell,
                span_padding=args.bulk_span_padding,
                density_warn_pct=args.bulk_density_warn_pct,
                clash_scale=args.clash_threshold_scale,
                max_expand_iters=args.bulk_clash_max_expand_iter
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        runtime_warnings.extend(box_warnings)
        print(f"    目标密度: {target_density:.4f} g/cm³")
        print(f"    实际密度: {achieved_density:.4f} g/cm³")
    else:
        cluster = create_vacuum_box(cluster, args.vacuum)
        achieved_density = 0.0
        target_density = None
        print(f"    真空层: {args.vacuum} Å")
    
    cell = cluster.get_cell().lengths()
    print(f"    盒子: {cell[0]:.1f} x {cell[1]:.1f} x {cell[2]:.1f} Å")
    
    # 切断键检测
    if n_cut_bonds > 0:
        print(f"\n[WARN] 检测到 {n_cut_bonds} 个切断的化学键！")
        print("[INFO] 建议: 增大半径 / 使用 --bond_hops / molecule 模式")
        
        # 写入报告
        if HAS_UTILS:
            os.makedirs(args.outdir, exist_ok=True)
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            valid_cell = use_mic and has_valid_cell_for_mic(atoms)
            cell_arr = atoms.cell if valid_cell else None
            pbc_arr = atoms.pbc if valid_cell else None
            graph = build_bond_graph(positions, symbols, cell_arr, pbc_arr)
            cut_bonds = detect_cut_bonds(graph, set(selected_indices))
            write_cut_bonds_report(
                os.path.join(args.outdir, 'cut_bonds_report.txt'),
                cut_bonds, symbols, positions, cell_arr
            )
    
    # v2.3: 碰撞检测
    print("\n>>> 碰撞检测...")
    clash_info = detect_atomic_clashes(cluster, use_mic=use_mic, scale=args.clash_threshold_scale)
    
    if clash_info['d_min'] is not None:
        print(f"    最小原子间距: {clash_info['d_min']:.3f} Å")
    
    # 计算体积压缩比例
    volume_compression_ratio = 0.0
    if original_volume is not None and args.mode == 'bulk':
        current_volume = abs(np.linalg.det(cluster.get_cell()))
        # 根据原子数比例估算等效原始体积
        n_ratio = len(cluster) / len(atoms)
        equivalent_original_vol = original_volume * n_ratio
        if equivalent_original_vol > 0:
            volume_compression_ratio = max(0, (equivalent_original_vol - current_volume) / equivalent_original_vol)
    
    if clash_info['has_clashes']:
        print("")
        print("!" * 70)
        print(f"!!! 检测到 {clash_info['clash_count']} 对原子碰撞/重叠！")
        print("!!! 直接运行 AIMD 可能导致模拟崩溃！")
        print("!!! 必须先进行离子弛豫！")
        print("!" * 70)
        print(f"    最差的 5 对原子:")
        for i, cp in enumerate(clash_info['clash_pairs'][:5]):
            print(f"      {i+1}. 原子 {cp['i']} ({cp['sym_i']}) - "
                  f"原子 {cp['j']} ({cp['sym_j']}): {cp['distance']:.3f} Å")
    elif volume_compression_ratio > 0.3:
        print(f"    [WARN] 体积压缩 {volume_compression_ratio*100:.1f}%，建议弛豫")
    else:
        print(f"    [OK] 未检测到严重碰撞")
    
    # 生成 VASP 输入
    print("\n>>> 生成 VASP 输入...")
    potcar_ok, n_element_types = write_vasp_inputs(
        cluster, args.outdir, args.temp, args.steps, args.potim,
        kpts, args.ncore, args.encut, args.mode == 'bulk',
        args.thermostat, args.gamma_1ps
    )
    
    # v2.3: 弛豫指导与辅助输入
    has_h = 'H' in cluster.get_chemical_symbols()
    need_relax_guide = clash_info['has_clashes'] or volume_compression_ratio > 0.3
    dipol_suggestion = 0.5 * np.array(cluster.get_cell().lengths(), dtype=float)
    
    if need_relax_guide or args.write_relax_inputs:
        print("\n>>> 生成弛豫指导...")
        write_relax_guide(
            args.outdir, clash_info, has_h, args.gamma_1ps, volume_compression_ratio,
            is_cluster=(args.mode == 'cluster'),
            dipol_suggestion=dipol_suggestion if args.mode == 'cluster' else None
        )
        print(f"    [OK] RELAX_GUIDE.txt")

    if args.mode == 'cluster':
        write_cluster_dipole_hint(args.outdir, cluster)
        print(f"    [OK] INCAR.cluster_hint")
    
    if args.write_relax_inputs or clash_info['has_clashes']:
        write_incar_relax(args.outdir, args.encut, args.ncore, has_h, n_element_types)
        write_relax_script(args.outdir)
        print(f"    [OK] INCAR.relax")
        print(f"    [OK] run_relax_then_aimd.sh")
    
    # 索引映射（original ↔ cluster）
    try:
        center_idx_cluster = int(np.where(selected_indices == center_idx)[0][0])
    except IndexError:
        print("[ERROR] 中心原子未包含在最终选中集合中")
        sys.exit(1)

    if args.write_index_map:
        center_idx_cluster = write_index_map(args.outdir, selected_indices, center_idx)
        print("    [OK] index_map.json")

    # 元数据
    write_model_meta(
        args.outdir, args, len(cluster), center_idx, center_sym,
        total_charge, elem_counts, neutralized, neutralized_count,
        truncated, cell, [bool(x) for x in cluster.pbc],
        original_density, target_density, achieved_density,
        density_input_is_3d_valid, density_decision_reason,
        mic_cell_valid, use_mic, mic_decision_reason,
        input_pbc, input_cell_matrix,
        n_cut_bonds, runtime_warnings, clash_info, neutralize_verified,
        center_idx_cluster=center_idx_cluster,
        remaining_charge=remaining_charge,
        healed=healed, initial_cut_bonds=initial_cut_bonds,
        neutralize_charge_before=neutralize_charge_before,
        neutralize_reliable_before=neutralize_reliable_before,
        neutralize_charge_after=neutralize_charge_after,
        neutralize_reliable_after=neutralize_reliable_after
    )
    
    write_selected_indices(args.outdir, center_idx, selected_indices)
    
    # 摘要
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"模式: {mode_desc}")
    print(f"原子数: {len(cluster)}")
    print(f"电荷: {total_charge:+d}")
    if args.mode == 'bulk':
        print(f"密度: {achieved_density:.4f} g/cm³")
    if clash_info['d_min'] is not None:
        print(f"最小原子间距: {clash_info['d_min']:.3f} Å")
    if clash_info['has_clashes']:
        print(f"碰撞原子对: {clash_info['clash_count']} 对 ⚠️")
    if n_cut_bonds > 0:
        print(f"切断键: {n_cut_bonds} 个（见 cut_bonds_report.txt）")
    print(f"输出: {args.outdir}")
    print("=" * 70)
    
    if args.mode == "bulk":
        print("\n>>> 【周期性子胞】模型")
        print("    ✓ 按原体系密度定盒子，物理合理")
        print("    ✓ 可用于局域结构分析")
        print("    ⚠️ AIMD 时间有限（ps），长程扩散用经典 MD")
    else:
        print("\n>>> 【有限真空簇】模型")
        print("    ⚠️ 不能用于 bulk 输运性质！")
        print("    ⚠️ 表面效应显著")
    
    if clash_info['has_clashes']:
        print("")
        print("!" * 70)
        print("!!! 重要: 检测到原子碰撞，必须先弛豫！")
        print("!!! 请查看 RELAX_GUIDE.txt 和 INCAR.relax")
        print("!" * 70)
    
    if args.gamma_1ps >= 10:
        print(f"\n[WARN] Langevin gamma={args.gamma_1ps} 较大")
        print("[INFO] 适合平衡段；生产段建议 gamma=1-5")


if __name__ == "__main__":
    main()
