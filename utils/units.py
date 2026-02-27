# -*- coding: utf-8 -*-
"""
units.py - 单位转换与物理常量

功能:
- 原子质量表
- 质量/密度计算
- 扩散系数单位转换

作者: STAR0418-ABC
"""

from typing import List, Optional
import numpy as np

# ==============================================================================
# 物理常量
# ==============================================================================

# 阿伏伽德罗常数
AVOGADRO = 6.02214076e23  # mol^-1

# 1 Å³ = 1e-24 cm³
A3_TO_CM3 = 1e-24

# 1 Å²/ps = 1e-4 cm²/s
A2_PS_TO_CM2_S = 1e-4

# ==============================================================================
# 原子质量表 (g/mol)
# ==============================================================================

ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.01,
    'N': 14.01, 'O': 16.00, 'F': 19.00, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.31,
    'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07, 'Cl': 35.45, 'Ar': 39.95,
    'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00,
    'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38,
    'Ga': 69.72, 'Ge': 72.63, 'As': 74.92, 'Se': 78.97, 'Br': 79.90, 'Kr': 83.80,
    'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.95,
    'Tc': 98.00, 'Ru': 101.1, 'Rh': 102.9, 'Pd': 106.4, 'Ag': 107.9, 'Cd': 112.4,
    'In': 114.8, 'Sn': 118.7, 'Sb': 121.8, 'Te': 127.6, 'I': 126.9, 'Xe': 131.3,
    'Cs': 132.9, 'Ba': 137.3, 'La': 138.9, 'Ce': 140.1, 'Pr': 140.9, 'Nd': 144.2,
    'Pm': 145.0, 'Sm': 150.4, 'Eu': 152.0, 'Gd': 157.3, 'Tb': 158.9, 'Dy': 162.5,
    'Ho': 164.9, 'Er': 167.3, 'Tm': 168.9, 'Yb': 173.0, 'Lu': 175.0, 'Hf': 178.5,
    'Ta': 180.9, 'W': 183.8, 'Re': 186.2, 'Os': 190.2, 'Ir': 192.2, 'Pt': 195.1,
    'Au': 197.0, 'Hg': 200.6, 'Tl': 204.4, 'Pb': 207.2, 'Bi': 209.0, 'Po': 209.0,
    'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.0,
    'Pa': 231.0, 'U': 238.0,
}

DEFAULT_MASS = 10.0  # 未知元素的默认质量


def get_atomic_mass(symbol: str) -> float:
    """获取原子质量 (g/mol)"""
    return ATOMIC_MASSES.get(symbol, DEFAULT_MASS)


def compute_mass(symbols: List[str]) -> float:
    """
    计算原子集合的总质量 (g/mol)
    
    Args:
        symbols: 元素符号列表
    
    Returns:
        total_mass: 总分子量 (g/mol)
    """
    return sum(get_atomic_mass(s) for s in symbols)


def compute_mass_grams(symbols: List[str]) -> float:
    """
    计算原子集合的实际质量 (g)
    
    一个"分子"的质量 = M / N_A
    
    Args:
        symbols: 元素符号列表
    
    Returns:
        mass_g: 质量 (g)
    """
    molar_mass = compute_mass(symbols)  # g/mol
    return molar_mass / AVOGADRO


def compute_density(symbols: List[str], cell_volume_A3: float) -> float:
    """
    计算密度 (g/cm³)
    
    Args:
        symbols: 元素符号列表
        cell_volume_A3: 晶胞体积 (Å³)
    
    Returns:
        density: 密度 (g/cm³)
    """
    mass_g = compute_mass_grams(symbols)
    volume_cm3 = cell_volume_A3 * A3_TO_CM3
    return mass_g / volume_cm3


def volume_from_density(symbols: List[str], density_g_cm3: float) -> float:
    """
    从密度反算体积 (Å³)
    
    Args:
        symbols: 元素符号列表
        density_g_cm3: 目标密度 (g/cm³)
    
    Returns:
        volume_A3: 体积 (Å³)
    """
    mass_g = compute_mass_grams(symbols)
    volume_cm3 = mass_g / density_g_cm3
    return volume_cm3 / A3_TO_CM3


def scale_cell_to_volume(
    cell: np.ndarray,
    target_volume: float,
    mode: str = 'scale_proportional'
) -> np.ndarray:
    """
    缩放晶胞到目标体积
    
    Args:
        cell: (3, 3) 晶胞矩阵
        target_volume: 目标体积 (Å³)
        mode: 缩放模式
            - 'scale_proportional': 等比例缩放
            - 'cubic': 变为立方盒子
    
    Returns:
        new_cell: 缩放后的晶胞矩阵
    """
    if mode == 'cubic':
        L = target_volume ** (1/3)
        return np.diag([L, L, L])
    
    # scale_proportional
    current_volume = np.abs(np.linalg.det(cell))
    if current_volume < 1e-10:
        # 无效晶胞，返回立方盒子
        L = target_volume ** (1/3)
        return np.diag([L, L, L])
    
    scale = (target_volume / current_volume) ** (1/3)
    return cell * scale


def A2_ps_to_cm2_s(D_A2_ps: float) -> float:
    """
    扩散系数单位转换: Å²/ps -> cm²/s
    
    1 Å²/ps = 1e-4 cm²/s
    """
    return D_A2_ps * A2_PS_TO_CM2_S


def cm2_s_to_A2_ps(D_cm2_s: float) -> float:
    """
    扩散系数单位转换: cm²/s -> Å²/ps
    """
    return D_cm2_s / A2_PS_TO_CM2_S

