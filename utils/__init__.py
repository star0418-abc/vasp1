# -*- coding: utf-8 -*-
"""
VASP Scripts 工具模块

提供通用功能:
- connectivity: 化学键图与分子识别
- charges: 残基/分子电荷映射
- units: 单位转换与物理常量
"""

from .connectivity import (
    build_bond_graph,
    find_connected_components,
    expand_by_bond_hops,
    detect_cut_bonds,
)

from .charges import (
    ELEMENT_CHARGE_MAP,
    RESIDUE_CHARGE_MAP,
    estimate_charge_by_residue,
    estimate_charge_by_element,
    load_charge_map_file,
)

from .units import (
    ATOMIC_MASSES,
    compute_mass,
    compute_density,
    A2_ps_to_cm2_s,
)

__all__ = [
    # connectivity
    'build_bond_graph',
    'find_connected_components', 
    'expand_by_bond_hops',
    'detect_cut_bonds',
    # charges
    'ELEMENT_CHARGE_MAP',
    'RESIDUE_CHARGE_MAP',
    'estimate_charge_by_residue',
    'estimate_charge_by_element',
    'load_charge_map_file',
    # units
    'ATOMIC_MASSES',
    'compute_mass',
    'compute_density',
    'A2_ps_to_cm2_s',
]

