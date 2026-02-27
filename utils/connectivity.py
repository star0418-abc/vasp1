# -*- coding: utf-8 -*-
"""
connectivity.py - 化学键图与分子识别

功能:
- 基于共价半径的键图构建
- 连通分量（分子）识别
- BFS 键跳扩展（避免切断聚合物链）
- 切断键检测

作者: STAR0418-ABC
"""

from typing import List, Set, Dict, Tuple, Optional
from collections import deque
import numpy as np

# 共价半径表 (Å) - 用于判断化学键
# 来源: Cordero et al., Dalton Trans., 2008
COVALENT_RADII: Dict[str, float] = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32,
    'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20,
    'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
    'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
}

# 默认共价半径（未知元素）
DEFAULT_COVALENT_RADIUS = 1.5

# 键判定容差因子
BOND_TOLERANCE = 1.3  # 距离 < (r1 + r2) * tolerance 视为成键


def get_covalent_radius(symbol: str) -> float:
    """获取元素的共价半径"""
    return COVALENT_RADII.get(symbol, DEFAULT_COVALENT_RADIUS)


def build_bond_graph(
    positions: np.ndarray,
    symbols: List[str],
    cell: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None,
    tolerance: float = BOND_TOLERANCE,
    max_bond_length: float = 3.5
) -> Dict[int, Set[int]]:
    """
    构建化学键图（邻接表）
    
    Args:
        positions: (N, 3) 原子坐标
        symbols: 元素符号列表
        cell: 晶格矩阵 (3, 3)，可选
        pbc: PBC 标志，可选
        tolerance: 键判定容差因子
        max_bond_length: 最大键长 (Å)，超过此值不考虑
    
    Returns:
        graph: {atom_idx: {neighbor_indices}}
    """
    n_atoms = len(positions)
    graph: Dict[int, Set[int]] = {i: set() for i in range(n_atoms)}
    
    # 预计算共价半径
    radii = np.array([get_covalent_radius(s) for s in symbols])
    
    # 计算距离矩阵
    use_mic = cell is not None and pbc is not None and np.any(pbc)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # 计算距离
            if use_mic:
                # 最小镜像距离
                d_cart = positions[j] - positions[i]
                # 转换到分数坐标
                cell_inv = np.linalg.inv(cell)
                d_frac = np.dot(d_cart, cell_inv)
                d_frac -= np.round(d_frac)
                d_cart = np.dot(d_frac, cell)
                dist = np.linalg.norm(d_cart)
            else:
                dist = np.linalg.norm(positions[j] - positions[i])
            
            # 键判定
            bond_cutoff = (radii[i] + radii[j]) * tolerance
            
            if dist < min(bond_cutoff, max_bond_length):
                graph[i].add(j)
                graph[j].add(i)
    
    return graph


def find_connected_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    寻找图的连通分量（即分子）
    
    Args:
        graph: 邻接表
    
    Returns:
        components: 连通分量列表，每个元素是一组原子索引
    """
    visited = set()
    components = []
    
    for start in graph:
        if start in visited:
            continue
        
        # BFS
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


def expand_by_bond_hops(
    graph: Dict[int, Set[int]],
    seed_indices: Set[int],
    max_hops: int = 3,
    max_component_size: int = 500
) -> Set[int]:
    """
    从种子原子出发，按键跳扩展
    
    用于避免切断聚合物链但不吞掉整个网络
    
    Args:
        graph: 邻接表
        seed_indices: 种子原子索引
        max_hops: 最大键跳数
        max_component_size: 扩展后最大原子数（超过则停止）
    
    Returns:
        expanded: 扩展后的原子索引集合
    """
    expanded = set(seed_indices)
    frontier = set(seed_indices)
    
    for hop in range(max_hops):
        if len(expanded) >= max_component_size:
            break
        
        new_frontier = set()
        for atom in frontier:
            for neighbor in graph.get(atom, set()):
                if neighbor not in expanded:
                    new_frontier.add(neighbor)
                    expanded.add(neighbor)
                    
                    if len(expanded) >= max_component_size:
                        return expanded
        
        frontier = new_frontier
        
        if not frontier:
            break
    
    return expanded


def detect_cut_bonds(
    graph: Dict[int, Set[int]],
    selected_indices: Set[int]
) -> List[Tuple[int, int]]:
    """
    检测被切断的化学键
    
    Args:
        graph: 邻接表（原体系）
        selected_indices: 选中的原子索引
    
    Returns:
        cut_bonds: [(atom_in, atom_out), ...] 被切断的键列表
    """
    cut_bonds = []
    
    for atom_in in selected_indices:
        for neighbor in graph.get(atom_in, set()):
            if neighbor not in selected_indices:
                # 避免重复记录
                bond = tuple(sorted([atom_in, neighbor]))
                if bond not in cut_bonds:
                    cut_bonds.append((atom_in, neighbor))
    
    return cut_bonds


def write_cut_bonds_report(
    filepath: str,
    cut_bonds: List[Tuple[int, int]],
    symbols: List[str],
    positions: np.ndarray,
    cell: Optional[np.ndarray] = None
) -> None:
    """写入切断键报告"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 切断键报告 (Cut Bonds Report)\n")
        f.write(f"# 检测到 {len(cut_bonds)} 个被切断的化学键\n")
        f.write("# ⚠️ 警告: 切断化学键可能引入不物理的自由基/断链！\n")
        f.write("# 建议: 增大切割半径 / 使用 --bond_hops / 使用 molecule 模式\n")
        f.write("#\n")
        f.write("# atom_in_idx  atom_in_elem  atom_out_idx  atom_out_elem  distance_A\n")
        
        for atom_in, atom_out in cut_bonds:
            # 计算距离
            d = positions[atom_out] - positions[atom_in]
            if cell is not None:
                cell_inv = np.linalg.inv(cell)
                d_frac = np.dot(d, cell_inv)
                d_frac -= np.round(d_frac)
                d = np.dot(d_frac, cell)
            dist = np.linalg.norm(d)
            
            f.write(f"{atom_in:6d}  {symbols[atom_in]:4s}  {atom_out:6d}  {symbols[atom_out]:4s}  {dist:.3f}\n")

