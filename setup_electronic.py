#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_electronic.py - 生成 VASP 电子性质计算输入文件 (v2.1)

功能：
  - 功函数 (wf): 生成 slab 静态计算输入，输出 LOCPOT
  - DOS/PDOS (dos): 生成两步法输入（SCF + NSCF）

关键改进 (v2.1)：
  - 新增 --functional_tag / --scissor_ev 泛函标签和剪刀修正
  - 新增 --sigma_scf / --sigma_nscf 展宽宽度控制
  - PBE 带隙低估警告（自动输出）

v2.0 改进：
  - 自动 k-点选择基于结构类型（bulk/slab/cluster/wire）
  - 功函数模式对 bulk 结构硬性拒绝（防止真空切割产生伪表面）
  - 真空轴自动检测，IDIPOL 与真空方向一致
  - ISMEAR 可分别配置 SCF/NSCF 步骤

⚠️ PBE 带隙低估警告：
  PBE (及类似 GGA) 系统性低估带隙，基于带边的电化学稳定窗口 (ESW)
  可能低于实验值。如需定量设计，推荐 HSE06 或剪刀修正。
  PBE 趋势仍适用于相对比较。

用法：
  python3 setup_electronic.py --src CONTCAR --mode wf --vacuum 20
  python3 setup_electronic.py --src CONTCAR --mode dos --two_step
  python3 setup_electronic.py --src CONTCAR --mode dos --functional_tag HSE06

依赖：
  pip install ase

作者：STAR0418-ABC
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# 检查 ASE
try:
    from ase.io import read, write
    from ase.calculators.vasp import Vasp
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("[ERROR] 需要 ASE 库: pip install ase")
    sys.exit(1)


# =============================================================================
# Utility Functions
# =============================================================================

def check_vasp_pp_path() -> Tuple[bool, str]:
    """
    检查 VASP_PP_PATH 环境变量
    
    返回: (是否存在, 路径)
    """
    pp_path = os.environ.get('VASP_PP_PATH', '')
    if pp_path and os.path.isdir(pp_path):
        return True, pp_path
    return False, pp_path


def parse_kpts(kpts_str: str) -> Tuple[int, int, int]:
    """解析 K 点字符串 '8 8 1' -> (8, 8, 1)"""
    parts = kpts_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"K 点格式错误: {kpts_str}，应为 'k1 k2 k3'")
    return tuple(int(x) for x in parts)


def vacuum_axis_to_int(axis_str: str) -> Optional[int]:
    """Convert vacuum axis string to integer (0=x, 1=y, 2=z)"""
    mapping = {'x': 0, 'y': 1, 'z': 2, 'auto': None}
    return mapping.get(axis_str.lower())


def axis_int_to_name(axis: int) -> str:
    """Convert axis integer to name"""
    return ['x', 'y', 'z'][axis]


# =============================================================================
# Structure Detection (v2.0 - Enhanced)
# =============================================================================

def detect_structure_type(atoms) -> Tuple[str, Optional[int], str]:
    """
    自动检测结构类型（v2.0 增强版）
    
    返回: (type, vacuum_axis, recommended_kpts)
        type: 'cluster' / 'wire' / 'slab' / 'bulk'
        vacuum_axis: 0/1/2 for slab (single vacuum), None for bulk/cluster
                     对于 wire (两个真空方向), 返回周期方向的轴
        recommended_kpts: 推荐的 K 点字符串
    
    检测原理:
        - 计算每个方向的原子跨度与cell长度的比值
        - ratio < 0.6 认为存在真空
        - 0 个真空方向 = bulk
        - 1 个真空方向 = slab
        - 2 个真空方向 = wire (1D)
        - 3 个真空方向 = cluster (0D)
    """
    if not atoms.pbc.any():
        # 无 PBC，一定是 cluster
        return 'cluster', None, '1 1 1'
    
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    
    # 计算每个方向的尺寸和原子分布
    cell_lengths = cell.lengths()
    
    # 检查每个方向的原子分布
    vacuum_directions: List[int] = []
    periodic_directions: List[int] = []
    
    for axis in range(3):
        coords = positions[:, axis]
        span = coords.max() - coords.min()
        cell_len = cell_lengths[axis]
        
        # 如果原子跨度远小于 cell 长度，可能有真空
        # 使用 0.6 作为阈值（可配置）
        if cell_len > 0 and span / cell_len < 0.6:
            vacuum_directions.append(axis)
        else:
            periodic_directions.append(axis)
    
    n_vacuum = len(vacuum_directions)
    
    if n_vacuum == 3:
        # 所有方向都有真空 -> cluster (0D)
        return 'cluster', None, '1 1 1'
    
    elif n_vacuum == 2:
        # 两个方向有真空 -> wire (1D)
        # 返回周期方向（非真空方向）
        periodic_axis = periodic_directions[0] if periodic_directions else 2
        # 推荐 k-mesh: 周期方向用密集网格
        kpts = ['1', '1', '1']
        kpts[periodic_axis] = '12'
        return 'wire', periodic_axis, ' '.join(kpts)
    
    elif n_vacuum == 1:
        # 一个方向有真空 -> slab (2D)
        vacuum_axis = vacuum_directions[0]
        # 推荐 k-mesh: 非真空方向用密集网格
        kpts = ['8', '8', '8']
        kpts[vacuum_axis] = '1'
        return 'slab', vacuum_axis, ' '.join(kpts)
    
    else:
        # 无明显真空 -> bulk (3D)
        return 'bulk', None, '12 12 12'


def get_recommended_kpts(struct_type: str, vacuum_axis: Optional[int], 
                         mode: str) -> str:
    """
    根据结构类型和计算模式返回推荐的 k-points
    
    Args:
        struct_type: 'bulk', 'slab', 'wire', 'cluster'
        vacuum_axis: 真空方向轴 (0/1/2) 或 None
        mode: 'wf' 或 'dos'
    
    Returns:
        推荐的 k-points 字符串
    """
    if struct_type == 'cluster':
        return '1 1 1'
    
    if struct_type == 'wire':
        # 1D: 只有一个周期方向
        # vacuum_axis 在这里实际上是周期方向
        periodic_axis = vacuum_axis if vacuum_axis is not None else 2
        kpts = ['1', '1', '1']
        kpts[periodic_axis] = '12' if mode == 'dos' else '8'
        return ' '.join(kpts)
    
    if struct_type == 'slab':
        # 2D: 两个周期方向
        kpts = ['12' if mode == 'dos' else '8'] * 3
        if vacuum_axis is not None:
            kpts[vacuum_axis] = '1'
        else:
            kpts[2] = '1'  # 默认 z 真空
        return ' '.join(kpts)
    
    # bulk: 3D
    if mode == 'dos':
        return '12 12 12'
    else:  # wf
        return '8 8 8'


def print_structure_detection(struct_type: str, vacuum_axis: Optional[int],
                               recommended_kpts: str) -> None:
    """打印结构检测结果（advisory）"""
    print("\n>>> 结构检测（advisory）:")
    print(f"    类型: {struct_type}")
    
    if vacuum_axis is not None:
        if struct_type == 'wire':
            print(f"    周期方向: {axis_int_to_name(vacuum_axis)} (axis={vacuum_axis})")
        else:
            print(f"    真空方向: {axis_int_to_name(vacuum_axis)} (axis={vacuum_axis})")
    else:
        if struct_type == 'bulk':
            print("    真空方向: 无 (3D 周期)")
        else:
            print("    真空方向: 全部 (0D cluster)")
    
    print(f"    推荐 K 点: {recommended_kpts}")
    print("    注意: 自动检测是启发式的，必要时用 --kpts_*/--vacuum_axis 覆盖")


# =============================================================================
# INCAR Generation
# =============================================================================

def get_common_incar_settings(encut: float, ediff: float, ncore: Optional[int]) -> Dict[str, Any]:
    """
    获取通用 INCAR 设置
    
    这些是电子性质计算的基本参数
    """
    settings = {
        # 精度控制
        'prec': 'Accurate',      # 高精度模式
        'encut': encut,          # 截断能 (eV)
        'ediff': ediff,          # 电子步收敛判据
        'nelm': 200,             # 最大电子步数
        'algo': 'Normal',        # 电子步算法
        
        # 输出控制
        'lwave': False,          # 不写 WAVECAR（节省空间）
    }
    
    if ncore is not None:
        settings['ncore'] = ncore
    
    return settings


# =============================================================================
# Work Function Setup (v2.0 - With Bulk Protection)
# =============================================================================

def setup_work_function(atoms, outdir: str, vacuum: float, kpts: Tuple[int, int, int],
                        gamma: bool, encut: float, ediff: float, ncore: Optional[int],
                        xc: str, vacuum_axis: int, struct_type: str,
                        ismear: int, force_slab: bool) -> str:
    """
    设置功函数计算 (v2.0 - 增强版)
    
    功函数 Φ = V_vac - E_F
    需要输出 LOCPOT（静电势）用于提取真空电势 V_vac
    
    v2.0 改进:
        - 对 bulk 结构强制拒绝（防止vacuum cleavage）
        - 真空轴由检测或用户指定，不硬编码 z
        - IDIPOL 与真空方向一致
        - 添加 ISYM=0（偶极修正推荐）
    
    参数:
        atoms: ASE Atoms 对象
        outdir: 输出目录
        vacuum: 真空层厚度 (Å)
        kpts: K 点网格
        gamma: 是否 Gamma-centered
        encut: 截断能
        ediff: 收敛判据
        ncore: 并行参数
        xc: 交换关联泛函
        vacuum_axis: 真空方向 (0=x, 1=y, 2=z)
        struct_type: 结构类型
        ismear: ISMEAR 值
        force_slab: 是否强制对 bulk 应用 WF（不推荐）
    
    返回:
        计算目录路径
    """
    # =========================================================================
    # v2.0: Bulk 保护 - 防止真空切割产生伪表面
    # =========================================================================
    if struct_type == 'bulk':
        if not force_slab:
            print("\n" + "=" * 70)
            print("[ERROR] 功函数计算需要 slab 结构（带真空）")
            print("=" * 70)
            print("""
您的结构看起来是 bulk（3D 周期性）。

问题：在 bulk 结构上添加真空层会：
  - 在周期边界处"切割"结构
  - 产生非物理的表面（可能切断聚合物链/溶剂分子）
  - 创建自由基样的悬挂键和伪表面态
  - 使功函数计算结果无意义

解决方案：
  1. 提供正确的 slab 模型（已有真空）
  2. 使用专门的 slab 构建工具创建表面模型

如果您确定要继续（**不推荐**），请使用 --force_slab 标志。
但请注意，这可能产生物理上无意义的结果。
""")
            sys.exit(1)
        else:
            print("\n[WARN] " + "=" * 60)
            print("[WARN] --force_slab: 在 bulk 结构上强制应用功函数设置")
            print("[WARN] 这可能产生非物理结果！请确保您了解后果。")
            print("[WARN] " + "=" * 60)
    
    calcdir = os.path.join(outdir, 'wf_static')
    os.makedirs(calcdir, exist_ok=True)
    
    # 结构分析（使用正确的真空轴）
    cell = atoms.get_cell()
    cell_lengths = cell.lengths()
    positions = atoms.get_positions()
    
    axis_coords = positions[:, vacuum_axis]
    coord_min, coord_max = axis_coords.min(), axis_coords.max()
    slab_thickness = coord_max - coord_min
    axis_length = cell_lengths[vacuum_axis]
    
    axis_name = axis_int_to_name(vacuum_axis)
    
    print(f"\n>>> 结构分析 (真空轴: {axis_name}):")
    print(f"    原始 cell {axis_name} 方向: {axis_length:.2f} Å")
    print(f"    原子 {axis_name} 范围: {coord_min:.2f} ~ {coord_max:.2f} Å (厚度 {slab_thickness:.2f} Å)")
    
    # 添加真空并居中（使用正确的轴）
    atoms.center(vacuum=vacuum, axis=vacuum_axis)
    
    new_cell = atoms.get_cell()
    new_length = new_cell.lengths()[vacuum_axis]
    print(f"    添加真空后 {axis_name} 方向: {new_length:.2f} Å (真空层 ~{vacuum:.1f} Å)")
    
    if struct_type == 'slab' and slab_thickness > axis_length * 0.85:
        print(f"[WARN] 原结构真空较小，建议增加 --vacuum 值")
    
    # 获取通用设置
    incar_settings = get_common_incar_settings(encut, ediff, ncore)
    
    # 功函数特定设置 (v2.0 - 正确的真空轴处理)
    # IDIPOL: 1=x, 2=y, 3=z (VASP 使用 1-indexed)
    idipol = vacuum_axis + 1
    
    incar_settings.update({
        # 静态计算
        'ibrion': -1,            # 不进行离子弛豫
        'nsw': 0,                # 离子步数为 0
        
        # 功函数关键参数
        'lvhar': True,           # LVHAR=.TRUE. 输出 LOCPOT
                                 # LOCPOT 包含局域静电势（Hartree势的平面平均）
                                 # 用于提取真空电势 V_vac 计算功函数
        
        'ldipol': True,          # LDIPOL=.TRUE. 启用偶极修正
                                 # 对于不对称 slab 消除周期性镜像的人工偶极
        
        'idipol': idipol,        # IDIPOL 偶极修正方向
                                 # 1=x, 2=y, 3=z，与真空方向一致
        
        # v2.0: 偶极修正时推荐关闭对称性
        'isym': 0,               # ISYM=0 关闭对称性
                                 # 使用偶极修正时推荐，避免对称性相关问题
        
        # 展宽设置
        'ismear': ismear,        # ISMEAR 展宽方法
        'sigma': 0.05,           # SIGMA 展宽宽度 (eV)
        
        # 输出
        'lcharg': False,         # 不需要 CHGCAR
    })
    
    # 写入 VASP 输入
    has_pp, pp_path = check_vasp_pp_path()
    
    try:
        calc = Vasp(
            directory=calcdir,
            xc=xc,
            kpts=kpts,
            gamma=gamma,
            **incar_settings
        )
        calc.write_input(atoms)
        print(f"\n>>> 已写入 VASP 输入到: {calcdir}")
        
    except Exception as e:
        # 如果 ASE 写入失败（可能是 POTCAR 问题），手动写入
        print(f"[WARN] ASE 写入出错: {e}")
        print("[INFO] 尝试手动写入 POSCAR/INCAR/KPOINTS...")
        
        write(os.path.join(calcdir, 'POSCAR'), atoms, format='vasp')
        _write_incar_manual(calcdir, incar_settings)
        _write_kpoints_manual(calcdir, kpts, gamma)
    
    # 检查 POTCAR
    if not has_pp:
        print(f"\n[WARN] VASP_PP_PATH 未设置或不存在: {pp_path}")
        print("[INFO] 请手动准备 POTCAR 文件")
        print("[INFO] 设置方法: export VASP_PP_PATH=/path/to/potentials")
    
    # 打印关键参数摘要
    print(f"\n>>> 功函数 INCAR 关键参数:")
    print(f"    LVHAR = .TRUE.   (输出 LOCPOT)")
    print(f"    LDIPOL = .TRUE.  (偶极修正)")
    print(f"    IDIPOL = {idipol}        ({axis_name} 方向)")
    print(f"    ISYM = 0         (关闭对称性)")
    print(f"    ISMEAR = {ismear}")
    
    return calcdir


# =============================================================================
# DOS Setup (v2.0 - With Dimensionality Validation)
# =============================================================================

def setup_dos(atoms, outdir: str, kpts: Tuple[int, int, int], gamma: bool,
              encut: float, ediff: float, ncore: Optional[int], xc: str,
              ismear_scf: int, ismear_nscf: int, sigma_scf: float, sigma_nscf: float,
              two_step: bool, struct_type: str) -> str:
    """
    设置 DOS 计算 (v2.1 - 增强版)
    
    两步法：
      1. SCF 自洽计算 → 产生 CHGCAR
      2. NSCF 非自洽计算 (ICHARG=11) → 产生 DOSCAR
    
    v2.1 改进:
        - 新增 sigma_scf / sigma_nscf 参数
    
    v2.0 改进:
        - SCF/NSCF 的 ISMEAR 可分别配置
        - 打印详细的 ISMEAR 选择说明
    
    参数:
        atoms: ASE Atoms 对象
        outdir: 输出目录
        kpts: K 点网格
        gamma: 是否 Gamma-centered
        encut: 截断能
        ediff: 收敛判据
        ncore: 并行参数
        xc: 交换关联泛函
        ismear_scf: SCF 步骤的 ISMEAR
        ismear_nscf: NSCF 步骤的 ISMEAR
        sigma_scf: SCF 步骤的 SIGMA
        sigma_nscf: NSCF 步骤的 SIGMA
        two_step: 是否生成两步目录
        struct_type: 结构类型
    
    返回:
        计算目录路径
    """
    # 获取通用设置
    base_settings = get_common_incar_settings(encut, ediff, ncore)
    
    # 打印 ISMEAR 选择说明
    print(f"\n>>> ISMEAR 设置:")
    print(f"    SCF:  ISMEAR = {ismear_scf}", end="")
    if ismear_scf == 0:
        print(" (Gaussian 展宽)")
    elif ismear_scf == -5:
        print(" (四面体法)")
    else:
        print()
    
    print(f"    NSCF: ISMEAR = {ismear_nscf}", end="")
    if ismear_nscf == -5:
        print(" (四面体法 - 推荐用于半导体/绝缘体 DOS)")
    elif ismear_nscf == 0:
        print(" (Gaussian 展宽 - 适用于金属)")
    else:
        print()
    
    if two_step:
        # ============ 步骤 1: SCF 自洽 ============
        scf_dir = os.path.join(outdir, 'dos_scf')
        os.makedirs(scf_dir, exist_ok=True)
        
        scf_settings = base_settings.copy()
        scf_settings.update({
            'ibrion': -1,
            'nsw': 0,
            'ismear': ismear_scf,
            'sigma': sigma_scf,
            'lcharg': True,          # LCHARG=.TRUE. 输出 CHGCAR
                                     # CHGCAR 包含自洽电荷密度，供 NSCF 使用
        })
        
        _write_vasp_input(atoms, scf_dir, scf_settings, kpts, gamma, xc)
        print(f"\n>>> 已写入 SCF 输入到: {scf_dir}")
        
        # ============ 步骤 2: NSCF DOS ============
        nscf_dir = os.path.join(outdir, 'dos_nscf')
        os.makedirs(nscf_dir, exist_ok=True)
        
        nscf_settings = base_settings.copy()
        nscf_settings.update({
            'ibrion': -1,
            'nsw': 0,
            
            'icharg': 11,            # ICHARG=11 从 CHGCAR 读取电荷密度
                                     # 不进行自洽，只计算本征值和 DOS
                                     # 需要将 SCF 的 CHGCAR 拷贝到此目录
            
            'lorbit': 11,            # LORBIT=11 输出投影 DOS (PDOS)
                                     # 将 DOS 分解到各原子和轨道 (s/p/d)
                                     # 输出 DOSCAR 和 PROCAR
            
            'nedos': 3000,           # NEDOS=3000 DOS 采样点数
                                     # 更多点 = 更平滑的 DOS 曲线
            
            # 展宽设置
            # ISMEAR=-5: 四面体法，适合半导体/绝缘体，DOS 更准确
            # ISMEAR=0: Gaussian 展宽，适合金属
            'ismear': ismear_nscf,
            'sigma': sigma_nscf,
            
            'lcharg': False,
        })
        
        _write_vasp_input(atoms, nscf_dir, nscf_settings, kpts, gamma, xc)
        print(f">>> 已写入 NSCF 输入到: {nscf_dir}")
        
        return nscf_dir
    
    else:
        # 单步 DOS（不推荐，但有时可用）
        dos_dir = os.path.join(outdir, 'dos_single')
        os.makedirs(dos_dir, exist_ok=True)
        
        settings = base_settings.copy()
        settings.update({
            'ibrion': -1,
            'nsw': 0,
            'lorbit': 11,
            'nedos': 3000,
            'ismear': ismear_nscf,
            'sigma': sigma_nscf,
            'lcharg': False,
        })
        
        _write_vasp_input(atoms, dos_dir, settings, kpts, gamma, xc)
        print(f"\n>>> 已写入 DOS 输入到: {dos_dir}")
        
        return dos_dir


# =============================================================================
# File Writing Helpers
# =============================================================================

def _write_vasp_input(atoms, calcdir: str, settings: Dict, kpts: Tuple[int, int, int],
                      gamma: bool, xc: str):
    """写入 VASP 输入文件"""
    has_pp, pp_path = check_vasp_pp_path()
    
    try:
        calc = Vasp(
            directory=calcdir,
            xc=xc,
            kpts=kpts,
            gamma=gamma,
            **settings
        )
        calc.write_input(atoms)
        
    except Exception as e:
        print(f"[WARN] ASE 写入出错: {e}")
        print("[INFO] 手动写入 POSCAR/INCAR/KPOINTS...")
        
        write(os.path.join(calcdir, 'POSCAR'), atoms, format='vasp')
        _write_incar_manual(calcdir, settings)
        _write_kpoints_manual(calcdir, kpts, gamma)
    
    if not has_pp:
        potcar_path = os.path.join(calcdir, 'POTCAR')
        if not os.path.exists(potcar_path):
            print(f"[WARN] POTCAR 未生成，请手动准备: {potcar_path}")


def _write_incar_manual(calcdir: str, settings: Dict):
    """手动写入 INCAR"""
    incar_path = os.path.join(calcdir, 'INCAR')
    
    with open(incar_path, 'w') as f:
        f.write("# INCAR generated by setup_electronic.py v2.0\n\n")
        
        for key, value in settings.items():
            if isinstance(value, bool):
                val_str = '.TRUE.' if value else '.FALSE.'
            elif isinstance(value, float):
                if abs(value) < 1e-4:
                    val_str = f"{value:.0E}"
                else:
                    val_str = str(value)
            else:
                val_str = str(value)
            
            f.write(f"{key.upper()} = {val_str}\n")


def _write_kpoints_manual(calcdir: str, kpts: Tuple[int, int, int], gamma: bool):
    """手动写入 KPOINTS"""
    kpoints_path = os.path.join(calcdir, 'KPOINTS')
    
    with open(kpoints_path, 'w') as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write("Gamma\n" if gamma else "Monkhorst-Pack\n")
        f.write(f"{kpts[0]} {kpts[1]} {kpts[2]}\n")
        f.write("0 0 0\n")


# =============================================================================
# Next Steps Printing
# =============================================================================

def print_next_steps(mode: str, calcdir: str, outdir: str, two_step: bool):
    """打印后续步骤提示"""
    print("\n" + "=" * 70)
    print("下一步操作")
    print("=" * 70)
    
    if mode == 'wf':
        print(f"""
1. 检查 POTCAR 是否存在:
   ls {calcdir}/POTCAR

2. 运行功函数计算:
   cd {calcdir}
   NP=16 EXE=vasp_std run_vasp.sh

3. 计算完成后进行后处理:
   python3 analyze_electronic.py --calcdir {calcdir} --mode wf

输出文件:
   - OUTCAR: 包含 E-fermi
   - LOCPOT: 静电势数据（用于计算 V_vac）
""")
    
    elif mode == 'dos':
        if two_step:
            scf_dir = os.path.join(outdir, 'dos_scf')
            nscf_dir = os.path.join(outdir, 'dos_nscf')
            print(f"""
=== 两步法 DOS 计算 ===

步骤 1: 运行 SCF 自洽计算
   cd {scf_dir}
   NP=16 EXE=vasp_std run_vasp.sh

步骤 2: 拷贝 CHGCAR 到 NSCF 目录
   cp {scf_dir}/CHGCAR {nscf_dir}/

步骤 3: 运行 NSCF DOS 计算
   cd {nscf_dir}
   NP=16 EXE=vasp_std run_vasp.sh

步骤 4: 后处理
   python3 analyze_electronic.py --calcdir {nscf_dir} --mode dos

输出文件:
   - DOSCAR: DOS 数据
   - PROCAR: 投影信息 (如果 LORBIT=11)
""")
        else:
            print(f"""
运行 DOS 计算:
   cd {calcdir}
   NP=16 EXE=vasp_std run_vasp.sh

后处理:
   python3 analyze_electronic.py --calcdir {calcdir} --mode dos
""")
    
    print("=" * 70)


def print_resolved_parameters(mode: str, struct_type: str, vacuum_axis: Optional[int],
                               kpts: Tuple[int, int, int], ismear_values: Dict[str, int]):
    """打印解析后的参数摘要"""
    print("\n>>> 解析后的参数:")
    print(f"    结构类型: {struct_type}")
    
    if vacuum_axis is not None:
        if struct_type == 'wire':
            print(f"    周期方向: {axis_int_to_name(vacuum_axis)} (1D)")
        else:
            print(f"    真空方向: {axis_int_to_name(vacuum_axis)}")
    else:
        if struct_type == 'bulk':
            print("    真空方向: 无 (3D 周期)")
        else:
            print("    真空方向: 全部 (0D)")
    
    print(f"    K 点: {kpts[0]} {kpts[1]} {kpts[2]}")
    
    if mode == 'wf':
        print(f"    ISMEAR: {ismear_values.get('wf', 0)}")
    else:
        print(f"    ISMEAR (SCF): {ismear_values.get('scf', 0)}")
        print(f"    ISMEAR (NSCF): {ismear_values.get('nscf', -5)}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 VASP 电子性质计算输入（功函数/DOS）v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v2.0 关键改进:
  - 自动 k-点选择基于结构类型
  - bulk 结构功函数模式硬性拒绝
  - 真空轴自动检测
  - ISMEAR 可分别配置

示例:
  # 功函数计算（slab）
  python3 setup_electronic.py --src CONTCAR --mode wf --vacuum 20 --ncore 8

  # DOS 计算（两步法，bulk）
  python3 setup_electronic.py --src CONTCAR --mode dos --two_step

  # DOS 计算（金属体系，使用 Gaussian 展宽）
  python3 setup_electronic.py --src POSCAR --mode dos --ismear_nscf 0

  # 覆盖自动 k-点
  python3 setup_electronic.py --src POSCAR --mode dos --no_auto_kpts --kpts_dos "16 16 16"

  # 指定真空轴
  python3 setup_electronic.py --src slab.vasp --mode wf --vacuum_axis x
        """
    )
    
    # 必需参数
    parser.add_argument("--src", required=True,
                        help="输入结构文件路径 (POSCAR/CONTCAR/cif/xyz)")
    parser.add_argument("--mode", required=True, choices=['wf', 'dos'],
                        help="计算模式: wf=功函数, dos=DOS")
    parser.add_argument("--outdir", default="calc_electronic",
                        help="输出目录 (默认: calc_electronic)")
    
    # 功函数参数
    parser.add_argument("--vacuum", type=float, default=20.0,
                        help="真空层厚度 Å (默认: 20, 仅 wf 模式)")
    parser.add_argument("--vacuum_axis", choices=['auto', 'x', 'y', 'z'], default='auto',
                        help="真空方向 (默认: auto = 自动检测)")
    parser.add_argument("--force_slab", action="store_true",
                        help="[危险] 强制对 bulk 结构应用功函数设置")
    
    # 通用计算参数
    parser.add_argument("--ncore", type=int, default=None,
                        help="NCORE 并行参数 (可选)")
    parser.add_argument("--encut", type=float, default=500.0,
                        help="截断能 eV (默认: 500)")
    parser.add_argument("--ediff", type=float, default=1e-6,
                        help="电子步收敛判据 (默认: 1e-6)")
    parser.add_argument("--xc", default="PBE",
                        help="交换关联泛函 (默认: PBE)")
    
    # K 点参数 (v2.0: 更改默认值)
    parser.add_argument("--kpts_wf", default=None,
                        help="功函数 K 点 (默认: 自动根据结构类型)")
    parser.add_argument("--kpts_dos", default=None,
                        help="DOS K 点 (默认: 自动根据结构类型)")
    parser.add_argument("--gamma", type=bool, default=True,
                        help="是否 Gamma-centered (默认: True)")
    
    # 自动 k-点控制 (v2.0 新增)
    parser.add_argument("--auto_kpts", action="store_true", default=True,
                        help="自动根据结构类型选择 k-点 (默认: True)")
    parser.add_argument("--no_auto_kpts", action="store_false", dest="auto_kpts",
                        help="禁用自动 k-点选择")
    parser.add_argument("--no_auto_fix", action="store_true",
                        help="允许 bulk 结构使用可能无效的 k-mesh (仅警告)")
    
    # ISMEAR 参数 (v2.0 新增)
    parser.add_argument("--ismear_wf", type=int, default=0,
                        help="功函数的 ISMEAR (默认: 0 Gaussian)")
    parser.add_argument("--ismear_scf", type=int, default=0,
                        help="DOS SCF 的 ISMEAR (默认: 0 Gaussian)")
    parser.add_argument("--ismear_nscf", type=int, default=-5,
                        help="DOS NSCF 的 ISMEAR (默认: -5 四面体法)")
    
    # SIGMA 参数 (v2.1 新增)
    parser.add_argument("--sigma_scf", type=float, default=None,
                        help="SCF 的 SIGMA (默认: ISMEAR=-5 时 0.01, 否则 0.05)")
    parser.add_argument("--sigma_nscf", type=float, default=None,
                        help="NSCF 的 SIGMA (默认: ISMEAR=-5 时 0.01, 否则 0.05)")
    
    # 泛函/带隙元数据 (v2.1 新增)
    parser.add_argument("--functional_tag", default="PBE",
                        help="泛函标签用于输出注释 (默认: PBE)")
    parser.add_argument("--scissor_ev", type=float, default=None,
                        help="剪刀修正值 eV (可选, 仅用于注释/元数据)")
    
    # DOS 参数
    parser.add_argument("--ismear_dos", type=int, default=None,
                        help="[已废弃] 使用 --ismear_nscf 代替")
    parser.add_argument("--two_step", action="store_true", default=True,
                        help="DOS 使用两步法 (默认: True)")
    parser.add_argument("--no_two_step", action="store_false", dest="two_step",
                        help="DOS 使用单步法")
    
    args = parser.parse_args()
    
    # 处理废弃参数
    if args.ismear_dos is not None:
        print("[WARN] --ismear_dos 已废弃，请使用 --ismear_nscf")
        args.ismear_nscf = args.ismear_dos
    
    print("=" * 70)
    print("setup_electronic.py v2.1 - VASP 电子性质输入生成器")
    print("=" * 70)
    print(f"结构文件: {args.src}")
    print(f"计算模式: {args.mode}")
    print(f"输出目录: {args.outdir}")
    
    # 检查输入文件
    if not os.path.isfile(args.src):
        print(f"[ERROR] 输入文件不存在: {args.src}")
        sys.exit(1)
    
    # 读取结构
    print(f"\n>>> 读取结构: {args.src}")
    try:
        atoms = read(args.src)
        print(f"    原子数: {len(atoms)}")
        print(f"    元素: {set(atoms.get_chemical_symbols())}")
    except Exception as e:
        print(f"[ERROR] 读取结构失败: {e}")
        sys.exit(1)
    
    # 自动检测结构类型
    struct_type, detected_vacuum_axis, recommended_kpts = detect_structure_type(atoms)
    print_structure_detection(struct_type, detected_vacuum_axis, recommended_kpts)
    
    # 确定真空轴
    if args.vacuum_axis == 'auto':
        if detected_vacuum_axis is not None:
            vacuum_axis = detected_vacuum_axis
        else:
            vacuum_axis = 2  # 默认 z
            if struct_type == 'bulk' and args.mode == 'wf':
                # bulk 没有真空轴，但如果强制执行，使用 z
                pass
    else:
        vacuum_axis = vacuum_axis_to_int(args.vacuum_axis)
        if vacuum_axis is None:
            vacuum_axis = 2
    
    # 确定 k-点
    if args.mode == 'wf':
        if args.kpts_wf is not None:
            # 用户显式指定
            kpts_str = args.kpts_wf
            user_specified_kpts = True
        elif args.auto_kpts:
            # 自动选择
            kpts_str = get_recommended_kpts(struct_type, detected_vacuum_axis, 'wf')
            user_specified_kpts = False
        else:
            # 禁用自动选择，使用旧默认值
            kpts_str = '8 8 1'
            user_specified_kpts = False
    else:  # dos
        if args.kpts_dos is not None:
            # 用户显式指定
            kpts_str = args.kpts_dos
            user_specified_kpts = True
        elif args.auto_kpts:
            # 自动选择
            kpts_str = get_recommended_kpts(struct_type, detected_vacuum_axis, 'dos')
            user_specified_kpts = False
        else:
            # 禁用自动选择，使用新默认值 (v2.0: 12 12 12)
            kpts_str = '12 12 12'
            user_specified_kpts = False
    
    kpts = parse_kpts(kpts_str)
    
    # =========================================================================
    # v2.0: K-点维度验证（防止 bulk + 1D k-mesh）
    # =========================================================================
    if struct_type == 'bulk' and not user_specified_kpts:
        collapsed_dims = [i for i, k in enumerate(kpts) if k == 1]
        if collapsed_dims:
            if args.no_auto_fix:
                print(f"\n[WARN] Bulk 结构使用可能无效的 k-mesh: {kpts_str}")
                print(f"[WARN] 维度 {collapsed_dims} 的 k 值为 1，可能导致 DOS 不准确")
                print("[WARN] 继续执行（--no_auto_fix 已启用）")
            else:
                print(f"\n[ERROR] K 点维度错误")
                print("=" * 60)
                print(f"""
检测到 bulk (3D) 结构，但 k-mesh 有维度 = 1: {kpts_str}

这会导致：
  - DOS k 空间采样维度坍缩（3D → 2D 或更低）
  - 物理上错误的 DOS 和能带
  - 对凝胶电解质等 3D bulk 体系无意义

解决方案：
  1. 使用 3D k-mesh，如 "12 12 12"（自动模式默认）
  2. 如果这确实是您想要的，使用 --no_auto_fix 标志

当前推荐: --kpts_dos "12 12 12" 或依赖自动选择
""")
                sys.exit(1)
    
    # 收集 ISMEAR 值
    ismear_values = {
        'wf': args.ismear_wf,
        'scf': args.ismear_scf,
        'nscf': args.ismear_nscf,
    }
    
    # 解析 SIGMA 值 (v2.1)
    sigma_scf = args.sigma_scf if args.sigma_scf is not None else (0.01 if args.ismear_scf == -5 else 0.05)
    sigma_nscf = args.sigma_nscf if args.sigma_nscf is not None else (0.01 if args.ismear_nscf == -5 else 0.05)
    
    # 打印解析后的参数
    print_resolved_parameters(args.mode, struct_type, 
                               vacuum_axis if args.mode == 'wf' else detected_vacuum_axis,
                               kpts, ismear_values)
    
    # v2.1: 打印泛函和剪刀信息
    print(f"\n>>> 泛函元数据:")
    print(f"    泛函标签: {args.functional_tag}")
    if args.scissor_ev is not None:
        print(f"    剪刀修正: {args.scissor_ev} eV")
    else:
        print(f"    剪刀修正: 未应用")
    
    # v2.1: PBE 带隙低估警告
    if args.functional_tag.upper() in ['PBE', 'GGA', 'LDA', 'PBESOL', 'RPBE', 'REVPBE']:
        print("\n" + "=" * 70)
        print("⚠️  PBE/GGA 带隙低估警告")
        print("=" * 70)
        print("""
    PBE （及类似 GGA）系统性低估带隙。
    基于带边的电化学稳定窗口 (ESW) 可能小于实验值。
    
    建议：
      - PBE 结果适用于相对比较（趋势预测）
      - 如需定量设计，推荐：
        * HSE06 杂化泛函，或
        * 剪刀修正 (--scissor_ev)
      - 始终明确标注所用泛函
""")
        print("=" * 70)
    
    # 检查 VASP_PP_PATH
    has_pp, pp_path = check_vasp_pp_path()
    if has_pp:
        print(f"\n>>> VASP_PP_PATH: {pp_path}")
    else:
        print(f"\n[WARN] VASP_PP_PATH 未设置或无效")
        print("[INFO] 将写入 POSCAR/INCAR/KPOINTS，但 POTCAR 需手动准备")
    
    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    # 执行设置
    if args.mode == 'wf':
        calcdir = setup_work_function(
            atoms, args.outdir, args.vacuum, kpts, args.gamma,
            args.encut, args.ediff, args.ncore, args.xc,
            vacuum_axis, struct_type, args.ismear_wf, args.force_slab
        )
        print_next_steps('wf', calcdir, args.outdir, False)
        
    elif args.mode == 'dos':
        calcdir = setup_dos(
            atoms, args.outdir, kpts, args.gamma,
            args.encut, args.ediff, args.ncore, args.xc,
            args.ismear_scf, args.ismear_nscf, sigma_scf, sigma_nscf,
            args.two_step, struct_type
        )
        print_next_steps('dos', calcdir, args.outdir, args.two_step)


if __name__ == "__main__":
    main()
