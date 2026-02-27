# -*- coding: utf-8 -*-
"""
charges.py - 残基/分子电荷映射

功能:
- 元素级电荷估计（粗略）
- 残基/分子级电荷估计（更准确）
- 支持用户自定义电荷映射文件

常见凝胶电解质组分:
- LiTFSI（双三氟甲磺酰亚胺锂）: Li+ + TFSI-
- LiFSI（双氟磺酰亚胺锂）: Li+ + FSI-
- LiPF6（六氟磷酸锂）: Li+ + PF6-

作者: STAR0418-ABC
"""

from typing import Dict, Tuple, Optional, List, Set
import os
import json

# ==============================================================================
# 元素级电荷表（粗略估计，仅用于回退）
# ==============================================================================

ELEMENT_CHARGE_MAP: Dict[str, int] = {
    # 碱金属阳离子
    'Li': +1, 'Na': +1, 'K': +1, 'Rb': +1, 'Cs': +1,
    # 碱土金属阳离子
    'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
    # 过渡金属（常见氧化态）
    'Zn': +2, 'Cu': +2, 'Fe': +2, 'Co': +2, 'Ni': +2,
    'Al': +3,
    # 卤素阴离子
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    # 有机元素（通常中性）
    'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'P': 0, 'Si': 0, 'B': 0,
}

# ==============================================================================
# 残基/分子级电荷表（更准确）
# ==============================================================================

# 常见阴离子残基名称 -> 电荷
# 注意: 残基名称可能因力场/建模工具不同而异
RESIDUE_CHARGE_MAP: Dict[str, int] = {
    # TFSI（双三氟甲磺酰亚胺）阴离子
    'TFSI': -1, 'TFS': -1, 'TFSA': -1, 'NTF': -1, 'NTF2': -1,
    'Tf2N': -1, 'NTSI': -1, 'OTF': -1,
    # FSI（双氟磺酰亚胺）阴离子
    'FSI': -1, 'FSA': -1,
    # PF6（六氟磷酸）阴离子
    'PF6': -1,
    # BF4（四氟硼酸）阴离子
    'BF4': -1,
    # ClO4（高氯酸）阴离子
    'CLO4': -1, 'PCLO4': -1,
    # NO3（硝酸）阴离子
    'NO3': -1,
    # 三氟甲磺酸（OTf/triflate）阴离子
    'OTF': -1, 'TFL': -1, 'CF3SO3': -1,
    # 草酸阴离子
    'OXA': -2,
    # 碳酸酯（中性）
    'EC': 0, 'PC': 0, 'DMC': 0, 'DEC': 0, 'EMC': 0,
    # 常见聚合物单体（中性）
    'EO': 0, 'PEO': 0, 'PEG': 0, 'PEGDA': 0, 'PEGMA': 0,
    'ACR': 0, 'MMA': 0, 'EA': 0,
    # 水
    'HOH': 0, 'WAT': 0, 'SOL': 0, 'TIP': 0, 'SPC': 0,
    # 锂离子（单原子残基）
    'LI': +1, 'LI+': +1, 'LIP': +1,
    # 钠离子
    'NA': +1, 'NA+': +1, 'NAP': +1,
    # 钾离子
    'K': +1, 'K+': +1, 'KP': +1,
    # 锌离子
    'ZN': +2, 'ZN2+': +2, 'ZNP': +2,
    # 氯离子
    'CL': -1, 'CL-': -1, 'CLM': -1,
}

# 离子残基（用于中和）
CATION_RESIDUES: Set[str] = {'LI', 'LI+', 'LIP', 'NA', 'NA+', 'NAP', 'K', 'K+', 'KP', 'ZN', 'ZN2+', 'ZNP'}
ANION_RESIDUES: Set[str] = {'TFSI', 'TFS', 'TFSA', 'NTF', 'NTF2', 'FSI', 'FSA', 'PF6', 'BF4', 'CLO4', 'NO3', 'OTF', 'CL', 'CL-', 'CLM'}


def estimate_charge_by_element(
    symbols: List[str],
    charge_map: Optional[Dict[str, int]] = None
) -> Tuple[int, Dict[str, int], bool]:
    """
    按元素估算电荷（粗略）
    
    Args:
        symbols: 元素符号列表
        charge_map: 自定义电荷表
    
    Returns:
        total_charge: 总电荷
        element_counts: 元素计数
        is_reliable: 是否可靠（False 表示使用了默认 0 电荷）
    """
    table = charge_map if charge_map else ELEMENT_CHARGE_MAP
    element_counts: Dict[str, int] = {}
    total_charge = 0
    is_reliable = True
    
    for sym in symbols:
        element_counts[sym] = element_counts.get(sym, 0) + 1
        charge = table.get(sym, None)
        if charge is None:
            charge = 0
            is_reliable = False  # 有未知元素
        total_charge += charge
    
    return total_charge, element_counts, is_reliable


def estimate_charge_by_residue(
    residue_names: List[str],
    residue_indices: List[int],
    selected_atom_indices: Set[int],
    residue_charge_map: Optional[Dict[str, int]] = None,
    element_symbols: Optional[List[str]] = None,
    element_charge_map: Optional[Dict[str, int]] = None
) -> Tuple[int, Dict[str, int], bool, List[str]]:
    """
    按残基估算电荷（更准确）
    
    Args:
        residue_names: 每个原子的残基名称
        residue_indices: 每个原子的残基编号
        selected_atom_indices: 选中的原子索引集合
        residue_charge_map: 残基电荷表
        element_symbols: 元素符号（用于 fallback）
        element_charge_map: 元素电荷表（用于 fallback）
    
    Returns:
        total_charge: 总电荷
        residue_counts: 残基计数
        is_reliable: 是否可靠
        warnings: 警告信息
    """
    res_map = residue_charge_map if residue_charge_map else RESIDUE_CHARGE_MAP
    
    # 统计选中的残基
    selected_residues: Dict[Tuple[str, int], Set[int]] = {}  # (resname, resid) -> atom indices
    
    for atom_idx in selected_atom_indices:
        if atom_idx < len(residue_names):
            resname = residue_names[atom_idx].upper().strip()
            resid = residue_indices[atom_idx] if atom_idx < len(residue_indices) else 0
            key = (resname, resid)
            
            if key not in selected_residues:
                selected_residues[key] = set()
            selected_residues[key].add(atom_idx)
    
    # 计算电荷
    total_charge = 0
    residue_counts: Dict[str, int] = {}
    warnings: List[str] = []
    is_reliable = True
    unknown_residues: Set[str] = set()
    
    for (resname, resid), atom_indices in selected_residues.items():
        residue_counts[resname] = residue_counts.get(resname, 0) + 1
        
        if resname in res_map:
            total_charge += res_map[resname]
        else:
            # 未知残基：尝试按元素 fallback
            unknown_residues.add(resname)
            is_reliable = False
            
            if element_symbols is not None:
                elem_charge = 0
                for atom_idx in atom_indices:
                    if atom_idx < len(element_symbols):
                        sym = element_symbols[atom_idx]
                        elem_map = element_charge_map if element_charge_map else ELEMENT_CHARGE_MAP
                        elem_charge += elem_map.get(sym, 0)
                total_charge += elem_charge
    
    if unknown_residues:
        warnings.append(f"未知残基 (按元素估计): {', '.join(sorted(unknown_residues))}")
    
    return total_charge, residue_counts, is_reliable, warnings


def load_charge_map_file(filepath: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    从文件加载自定义电荷映射
    
    支持 JSON 或 YAML 格式:
    {
        "residue_charges": {"CUSTOM_RES": -2, ...},
        "element_charges": {"X": +3, ...}
    }
    
    Returns:
        residue_charges: 残基电荷表
        element_charges: 元素电荷表
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"电荷映射文件不存在: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        if ext in ['.yaml', '.yml']:
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("需要 pyyaml 来读取 YAML 文件: pip install pyyaml")
        else:
            data = json.load(f)
    
    residue_charges = {k.upper(): v for k, v in data.get('residue_charges', {}).items()}
    element_charges = {k: v for k, v in data.get('element_charges', {}).items()}
    
    return residue_charges, element_charges


def find_counterion_residues(
    residue_names: List[str],
    residue_indices: List[int],
    selected_atom_indices: Set[int],
    current_charge: int,
    target_charge: int,
    positions,  # np.ndarray
    center_position,  # np.ndarray
    cell=None,  # Optional[np.ndarray]
    residue_charge_map: Optional[Dict[str, int]] = None
) -> Tuple[List[Set[int]], int]:
    """
    寻找可用于中和电荷的反离子残基
    
    按"整个残基"为单位加入，而不是单个原子
    
    Returns:
        counterion_atoms: 按距离排序的反离子原子组列表
        remaining_charge: 添加后的剩余电荷
    """
    import numpy as np
    
    res_map = residue_charge_map if residue_charge_map else RESIDUE_CHARGE_MAP
    
    # 确定需要的反离子类型
    needed_charge = current_charge - target_charge
    if needed_charge > 0:
        # 需要阴离子
        target_residues = ANION_RESIDUES
    elif needed_charge < 0:
        # 需要阳离子
        target_residues = CATION_RESIDUES
    else:
        return [], 0
    
    # 找出所有不在选中集合中的目标残基
    available: Dict[Tuple[str, int], Dict] = {}  # (resname, resid) -> {indices, charge, min_dist}
    
    for atom_idx in range(len(residue_names)):
        if atom_idx in selected_atom_indices:
            continue
        
        resname = residue_names[atom_idx].upper().strip()
        if resname not in target_residues:
            continue
        
        resid = residue_indices[atom_idx] if atom_idx < len(residue_indices) else 0
        key = (resname, resid)
        
        if key not in available:
            charge = res_map.get(resname, 0)
            if abs(charge) == 0:
                continue
            available[key] = {'indices': set(), 'charge': charge, 'min_dist': float('inf')}
        
        available[key]['indices'].add(atom_idx)
        
        # 计算到中心的距离
        d = positions[atom_idx] - center_position
        if cell is not None:
            cell_inv = np.linalg.inv(cell)
            d_frac = np.dot(d, cell_inv)
            d_frac -= np.round(d_frac)
            d = np.dot(d_frac, cell)
        dist = np.linalg.norm(d)
        available[key]['min_dist'] = min(available[key]['min_dist'], dist)
    
    # 按距离排序
    sorted_residues = sorted(available.items(), key=lambda x: x[1]['min_dist'])
    
    # 贪心选择
    counterion_atoms: List[Set[int]] = []
    remaining = abs(needed_charge)
    
    for (resname, resid), info in sorted_residues:
        if remaining <= 0:
            break
        counterion_atoms.append(info['indices'])
        remaining -= abs(info['charge'])
    
    return counterion_atoms, remaining

