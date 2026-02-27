#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recipe_to_counts.py - 配方 wt% 转换为分子/原子数量 v3.1

功能：
  - 读取 recipe.yaml 配方文件
  - 将 wt% 换算为整数 entity 数量
  - 使用约束组分配方法保证离子配平（ion_group + stoich）
  - 支持低 wt% 组分跳过与重归一化
  - 输出目标/实际 wt% 对比与误差报告

核心改进 (v3.1)：
  - 默认 soft_atoms 软约束分配：优化原子数与 wt%，不再强制固定实体总数
  - 保留 legacy_total 兼容模式（用于复现实验）
  - 跳过组分后使用 effective wt% 作为误差基准并同时报告 original/effective
  - 分配前增加 min_count 可行性检查与中性硬约束预检查

用法：
  python3 recipe_to_counts.py --target_atoms 200
  python3 recipe_to_counts.py --recipe recipe.yaml  # 使用 YAML 中的 target_atoms
  python3 recipe_to_counts.py --total_mass_g 1.0 --scale_to_atoms 5000

作者：STAR0418-ABC
"""

import argparse
import sys
import os
import json
import csv
import math
from typing import Dict, List, Any, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# 固定的 8 类顺序
CATEGORY_ORDER = [
    ("salt_solution", "盐溶液"),
    ("polymer_matrix", "聚合物基质"),
    ("crosslinker", "交联剂"),
    ("photoinitiator", "引发剂"),
    ("plasticizer_solvent", "增塑剂/溶剂"),
    ("functional_monomer", "功能单体"),
    ("stabilizer", "稳定剂"),
    ("functional_filler", "功能填料"),
]


def load_yaml(filepath: str) -> Dict[str, Any]:
    """加载 YAML 文件"""
    if not os.path.isfile(filepath):
        print(f"[ERROR] 配方文件不存在: {filepath}")
        sys.exit(1)

    if not HAS_YAML:
        print("[ERROR] 需要 PyYAML 库: pip install pyyaml")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data if data else {}


def safe_float(value: Any, default: float = 0.0) -> Optional[float]:
    """安全地将值转换为 float"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> Optional[int]:
    """安全地将值转换为 int"""
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def flatten_recipe(data: Dict) -> List[Dict]:
    """将配方数据展平为条目列表"""
    entries = []

    for cat_key, cat_name in CATEGORY_ORDER:
        cat_entries = data.get(cat_key, []) or []

        for entry in cat_entries:
            if not isinstance(entry, dict):
                continue

            item = entry.copy()
            item['category'] = cat_key
            item['category_cn'] = cat_name
            entries.append(item)

    return entries


def constrained_largest_remainder_legacy(
    entries: List[Dict],
    target_total: int,
    count_field: str = 'float_count',
    min_count_field: str = 'min_count'
) -> List[Dict]:
    """
    约束组分配方法 (B1)
    
    对于有 ion_group 的条目：
    - 组内成员作为整体分配单一 k 值
    - 每个成员的 scaled_count = k * stoich
    - 保证组内电荷配平
    
    对于无 ion_group 的条目：
    - 传统 Largest Remainder 方法
    
    算法：
    1. 构建分配变量（组变量 + 独立变量）
    2. 确定 target_total
    3. 贪婪分配：按 benefit/multiplier 排序增减
    """
    # 构建分配变量
    variables = []  # {multiplier, float_k, min_k, entries: [(idx, entry, stoich)], k}
    groups: Dict[str, List[Tuple[int, Dict]]] = {}
    
    for i, entry in enumerate(entries):
        ion_group = entry.get('ion_group')
        if ion_group:
            groups.setdefault(ion_group, []).append((i, entry))
        else:
            # 独立变量：multiplier=1
            float_val = entry.get(count_field, 0) or 0
            min_val = entry.get(min_count_field, 0) or 0
            variables.append({
                'multiplier': 1,
                'float_k': float_val,
                'min_k': min_val,
                'entries': [(i, entry, 1)],  # (idx, entry, stoich)
                'group_name': None
            })
    
    # 添加组变量
    for group_name, members in groups.items():
        group_stoichs = [m[1].get('stoich', 1) or 1 for m in members]
        group_multiplier = sum(group_stoichs)
        
        # 组的 float_k = sum(float_count) / multiplier
        group_float_entities = sum(m[1].get(count_field, 0) or 0 for m in members)
        group_float_k = group_float_entities / group_multiplier if group_multiplier > 0 else 0
        
        # 组的 min_k = max(ceil(min_count_i / stoich_i))
        group_min_k = 0
        for idx, entry in members:
            stoich = entry.get('stoich', 1) or 1
            min_count = entry.get(min_count_field, 0) or 0
            required_k = math.ceil(min_count / stoich) if stoich > 0 else min_count
            group_min_k = max(group_min_k, required_k)
        
        variables.append({
            'multiplier': group_multiplier,
            'float_k': group_float_k,
            'min_k': group_min_k,
            'entries': [(m[0], m[1], m[1].get('stoich', 1) or 1) for m in members],
            'group_name': group_name
        })
    
    # 初始分配：k = max(floor(float_k), min_k)
    for var in variables:
        floor_k = int(math.floor(var['float_k']))
        var['k'] = max(floor_k, var['min_k'])
        var['remainder'] = var['float_k'] - floor_k
    
    # 计算当前总实体数
    def current_total():
        return sum(v['multiplier'] * v['k'] for v in variables)
    
    remaining = target_total - current_total()
    
    # 贪婪调整
    max_iterations = target_total * 2 + 100  # 防止无限循环
    iteration = 0
    
    while remaining != 0 and iteration < max_iterations:
        iteration += 1
        
        if remaining > 0:
            # 需要增加：找可以增加的变量
            candidates = []
            for i, var in enumerate(variables):
                if var['multiplier'] <= remaining:
                    # benefit = 接近 float_k 的改进 / multiplier
                    new_k = var['k'] + 1
                    old_dist = abs(var['k'] - var['float_k'])
                    new_dist = abs(new_k - var['float_k'])
                    benefit = (old_dist - new_dist) / var['multiplier']
                    # 确定性排序：benefit desc, remainder desc, group_name/idx
                    sort_key = (-benefit, -var['remainder'], var['group_name'] or '', i)
                    candidates.append((sort_key, i))
            
            if not candidates:
                # 无法继续：所有变量的 multiplier 都大于 remaining
                # 尝试找最小 multiplier
                min_mult_vars = [(v['multiplier'], i) for i, v in enumerate(variables)]
                min_mult_vars.sort()
                stuck = True
                for mult, idx in min_mult_vars:
                    if mult <= remaining:
                        variables[idx]['k'] += 1
                        remaining -= mult
                        stuck = False
                        break
                if stuck:
                    break
            else:
                candidates.sort()
                _, best_idx = candidates[0]
                variables[best_idx]['k'] += 1
                remaining -= variables[best_idx]['multiplier']
        
        else:  # remaining < 0
            # 需要减少：找可以减少的变量
            need_remove = -remaining
            candidates = []
            for i, var in enumerate(variables):
                can_reduce = var['k'] - var['min_k']
                if can_reduce > 0 and var['multiplier'] <= need_remove:
                    # cost = 远离 float_k 的代价 / multiplier
                    new_k = var['k'] - 1
                    old_dist = abs(var['k'] - var['float_k'])
                    new_dist = abs(new_k - var['float_k'])
                    cost = (new_dist - old_dist) / var['multiplier']
                    # 确定性排序：cost asc, remainder asc, group_name/idx
                    sort_key = (cost, var['remainder'], var['group_name'] or '', i)
                    candidates.append((sort_key, i))
            
            if not candidates:
                # 无法继续减少
                break
            else:
                candidates.sort()
                _, best_idx = candidates[0]
                variables[best_idx]['k'] -= 1
                remaining += variables[best_idx]['multiplier']
    
    # 检查是否达成目标
    final_total = current_total()
    if final_total != target_total:
        # 详细诊断信息
        stuck_info = []
        for var in variables:
            if var['k'] <= var['min_k']:
                for idx, entry, stoich in var['entries']:
                    stuck_info.append(
                        f"{entry.get('name', 'N/A')} (min={var['min_k']}, k={var['k']}, mult={var['multiplier']})"
                    )
        
        raise ValueError(
            f"约束分配失败：最终总数 {final_total} != target_total {target_total}。\n"
            f"差额: {target_total - final_total}\n"
            f"可能被 min_count 约束的条目: {', '.join(stuck_info) if stuck_info else 'none'}"
        )
    
    # 展开结果
    results = [None] * len(entries)
    for var in variables:
        k = var['k']
        for idx, entry, stoich in var['entries']:
            result = entry.copy()
            result['scaled_count'] = k * stoich
            result['floor_count'] = result['scaled_count']  # 兼容性
            result['_group_k'] = k
            result['_group_name'] = var['group_name']
            results[idx] = result
    
    return results


# 兼容旧接口名称（legacy 固定实体总数约束）
def constrained_largest_remainder(
    entries: List[Dict],
    target_total: int,
    count_field: str = 'float_count',
    min_count_field: str = 'min_count'
) -> List[Dict]:
    return constrained_largest_remainder_legacy(entries, target_total, count_field, min_count_field)


def _set_skipped_defaults(entry: Dict) -> None:
    """保证被跳过条目字段一致。"""
    entry['float_count'] = 0.0
    entry['scaled_count'] = 0
    entry['scaled_atoms'] = 0
    entry['mass_prop'] = 0.0
    entry['actual_wt_pct'] = 0.0
    entry['wt_error_abs'] = 0.0
    entry['wt_error_rel'] = 0.0


def _prepare_effective_targets(entries: List[Dict], renormalize_skipped: bool) -> Dict[str, Any]:
    """处理跳过组分与有效 wt% 目标。"""
    skip_info: Dict[str, Any] = {
        'skipped_entries': [],
        'skipped_wt_total': 0.0,
        'renormalized': False,
        'original_wt_total': 0.0,
        'effective_wt_total': 0.0
    }

    skipped_wt = 0.0
    original_wt = 0.0
    for entry in entries:
        wt_pct = float(entry.get('wt_pct', 0) or 0)
        original_wt += wt_pct
        entry['target_wt_pct_original'] = wt_pct
        if entry.get('skip_reason'):
            skipped_wt += wt_pct
            skip_info['skipped_entries'].append({
                'name': entry.get('name'),
                'wt_pct': wt_pct,
                'reason': entry.get('skip_reason')
            })

    skip_info['skipped_wt_total'] = skipped_wt
    skip_info['original_wt_total'] = original_wt

    if renormalize_skipped and skipped_wt > 0:
        effective_wt_total = original_wt - skipped_wt
        skip_info['renormalized'] = True
    else:
        effective_wt_total = original_wt
    skip_info['effective_wt_total'] = effective_wt_total

    if effective_wt_total <= 0:
        raise ValueError(
            "有效 wt% 总和 <= 0：所有条目都被跳过或 wt% 不合法，无法继续分配。"
        )

    for entry in entries:
        wt_pct = float(entry.get('wt_pct', 0) or 0)
        if entry.get('skip_reason'):
            entry['wt_pct_effective'] = 0.0
            entry['target_wt_pct_effective'] = 0.0
            _set_skipped_defaults(entry)
            continue

        if skip_info['renormalized']:
            effective_wt = wt_pct / effective_wt_total * 100.0
        else:
            effective_wt = wt_pct
        entry['wt_pct_effective'] = effective_wt
        entry['target_wt_pct_effective'] = effective_wt

    return skip_info


def _build_allocation_variables(
    entries: List[Dict],
    count_field: str = 'float_count',
    min_count_field: str = 'min_count'
) -> List[Dict[str, Any]]:
    """按 ion_group 构建分配变量。"""
    variables: List[Dict[str, Any]] = []
    groups: Dict[str, List[Tuple[int, Dict]]] = {}

    for idx, entry in enumerate(entries):
        group_name = entry.get('ion_group')
        if group_name:
            groups.setdefault(str(group_name), []).append((idx, entry))
            continue

        atoms = safe_int(entry.get('atoms_per_entity')) or safe_int(entry.get('atoms_per_molecule')) or 0
        mw = safe_float(entry.get('mw_g_mol')) or 0.0
        min_k = int(entry.get(min_count_field, 0) or 0)
        float_k = float(entry.get(count_field, 0) or 0.0)
        variables.append({
            'group_name': None,
            'entries': [(idx, entry, 1)],
            'float_k': float_k,
            'min_k': min_k,
            'multiplier': 1,
            'atoms_per_step': atoms,
            'mass_per_step': mw,
            'k': 0
        })

    for group_name, members in groups.items():
        stoichs = []
        group_min_k = 0
        group_float_entities = 0.0
        atoms_per_step = 0
        mass_per_step = 0.0
        packed_members = []

        for idx, entry in members:
            stoich = safe_int(entry.get('stoich', 1), 1) or 1
            min_count = int(entry.get(min_count_field, 0) or 0)
            group_min_k = max(group_min_k, math.ceil(min_count / stoich))

            atoms = safe_int(entry.get('atoms_per_entity')) or safe_int(entry.get('atoms_per_molecule')) or 0
            mw = safe_float(entry.get('mw_g_mol')) or 0.0

            stoichs.append(stoich)
            group_float_entities += float(entry.get(count_field, 0) or 0.0)
            atoms_per_step += stoich * atoms
            mass_per_step += stoich * mw
            packed_members.append((idx, entry, stoich))

        multiplier = sum(stoichs)
        float_k = group_float_entities / multiplier if multiplier > 0 else 0.0
        variables.append({
            'group_name': group_name,
            'entries': packed_members,
            'float_k': float_k,
            'min_k': group_min_k,
            'multiplier': multiplier,
            'atoms_per_step': atoms_per_step,
            'mass_per_step': mass_per_step,
            'k': 0
        })

    return variables


def _precheck_charge_constraints(entries: List[Dict], require_neutral: bool) -> None:
    """中性体系硬约束：组单位电荷必须为 0，带电条目必须成组。"""
    if not require_neutral:
        return

    groups: Dict[str, List[Dict[str, Any]]] = {}
    ungrouped_charged = []

    for entry in entries:
        charge = safe_int(entry.get('charge'), 0) or 0
        if charge == 0:
            continue

        group_name = entry.get('ion_group')
        if group_name:
            stoich = safe_int(entry.get('stoich', 1), 1) or 1
            groups.setdefault(str(group_name), []).append({
                'name': entry.get('name', 'N/A'),
                'charge': charge,
                'stoich': stoich
            })
        else:
            expected_count = max(
                float(entry.get('float_count', 0) or 0.0),
                float(entry.get('min_count', 0) or 0.0)
            )
            if expected_count > 0:
                ungrouped_charged.append({
                    'name': entry.get('name', 'N/A'),
                    'charge': charge,
                    'wt_pct_effective': entry.get('wt_pct_effective', 0) or 0,
                    'min_count': entry.get('min_count', 0) or 0
                })

    non_neutral_groups = []
    for group_name, members in groups.items():
        unit_charge = sum(m['charge'] * m['stoich'] for m in members)
        if unit_charge != 0:
            details = ", ".join(
                f"{m['name']} (charge={m['charge']}, stoich={m['stoich']})" for m in members
            )
            non_neutral_groups.append(
                f"{group_name}: unit_charge={unit_charge}; members=[{details}]"
            )

    if non_neutral_groups:
        raise ValueError(
            "检测到非中性的 ion_group 单位（sum(charge*stoich) != 0），无法保证中性：\n"
            + "\n".join(f"  - {line}" for line in non_neutral_groups)
            + "\n请修正 ion_group 成员的 stoich/charge。"
        )

    if ungrouped_charged:
        lines = [
            (
                f"  - {item['name']}: charge={item['charge']}, "
                f"wt_pct_effective={item['wt_pct_effective']:.4f}, min_count={item['min_count']}"
            )
            for item in ungrouped_charged
        ]
        raise ValueError(
            "require_neutral 模式下检测到未分组带电条目（缺少 ion_group）：\n"
            + "\n".join(lines)
            + "\n请为带电条目定义 ion_group + stoich，或使用 --allow_charged 显式允许带电体系。"
        )


def _precheck_min_atoms_feasibility(
    entries: List[Dict],
    target_atoms: int,
    small_tol: float = 0.05
) -> None:
    """在分配前检查 min_count 强制原子数是否超出目标。"""
    if target_atoms <= 0:
        raise ValueError("target_atoms 必须为正整数")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    details = []
    min_atoms_total = 0

    for entry in entries:
        atoms = safe_int(entry.get('atoms_per_entity')) or safe_int(entry.get('atoms_per_molecule')) or 0
        min_count = int(entry.get('min_count', 0) or 0)
        group_name = entry.get('ion_group')
        if group_name:
            groups.setdefault(str(group_name), []).append({
                'name': entry.get('name', 'N/A'),
                'atoms': atoms,
                'stoich': safe_int(entry.get('stoich', 1), 1) or 1,
                'min_count': min_count
            })
            continue

        min_atoms = min_count * atoms
        min_atoms_total += min_atoms
        if min_atoms > 0:
            details.append(
                f"entry:{entry.get('name', 'N/A')} min_count={min_count}, atoms_per_entity={atoms}, min_atoms={min_atoms}"
            )

    for group_name, members in groups.items():
        min_k = 0
        atoms_per_unit = 0
        for member in members:
            stoich = member['stoich']
            min_k = max(min_k, math.ceil(member['min_count'] / stoich))
            atoms_per_unit += stoich * member['atoms']
        group_min_atoms = min_k * atoms_per_unit
        min_atoms_total += group_min_atoms
        if group_min_atoms > 0:
            member_desc = ", ".join(
                f"{m['name']}[stoich={m['stoich']}, min_count={m['min_count']}]"
                for m in members
            )
            details.append(
                f"group:{group_name} min_k={min_k}, atoms_per_unit={atoms_per_unit}, min_atoms={group_min_atoms}; members={member_desc}"
            )

    limit = target_atoms * (1.0 + small_tol)
    if min_atoms_total > limit:
        raise ValueError(
            f"min_count 可行性检查失败：最小强制原子数 {min_atoms_total} > target_atoms*{1.0 + small_tol:.2f} = {limit:.1f}\n"
            "造成不可行的条目/组如下：\n"
            + "\n".join(f"  - {line}" for line in details)
            + "\n建议：增大 target_atoms / 降低 min_count / 用更小的聚合物表示（如低聚体单元）。"
        )


def _expand_variables_to_results(entries: List[Dict], variables: List[Dict[str, Any]]) -> List[Dict]:
    """将变量 k 展开回逐条目结果。"""
    results: List[Optional[Dict]] = [None] * len(entries)
    for var in variables:
        k = int(var['k'])
        for idx, entry, stoich in var['entries']:
            result = entry.copy()
            result['scaled_count'] = int(k * stoich)
            result['floor_count'] = result['scaled_count']  # 兼容字段
            result['_group_k'] = k
            result['_group_name'] = var['group_name']
            results[idx] = result

    return [r if r is not None else entries[i].copy() for i, r in enumerate(results)]


def allocate_counts_soft_constraints(
    entries: List[Dict],
    target_atoms: int,
    renormalize_skipped: bool = True,
    require_neutral: bool = True,
    count_field: str = 'float_count',
    min_count_field: str = 'min_count',
    count_weight: float = 0.1,
    max_iterations: Optional[int] = None
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    软约束分配器：
    - 硬约束：stoich/min_count/（可选）中性规则
    - 软目标：逼近 target_atoms + wt% + float_count
    """
    del renormalize_skipped  # 已在上游体现在 wt_pct_effective

    _precheck_charge_constraints(entries, require_neutral)
    _precheck_min_atoms_feasibility(entries, target_atoms)

    variables = _build_allocation_variables(entries, count_field=count_field, min_count_field=min_count_field)
    if not variables:
        return entries, {'iterations': 0, 'objective_initial': 0.0, 'objective_final': 0.0}

    for var in variables:
        var['k'] = max(int(round(var['float_k'])), int(var['min_k']))

    n_entries = len(entries)
    atoms_per_entity = [
        safe_int(e.get('atoms_per_entity')) or safe_int(e.get('atoms_per_molecule')) or 0
        for e in entries
    ]
    mw_per_entry = [safe_float(e.get('mw_g_mol')) or 0.0 for e in entries]
    float_counts = [float(e.get(count_field, 0) or 0.0) for e in entries]
    wt_targets = [float(e.get('target_wt_pct_effective', e.get('wt_pct', 0)) or 0.0) for e in entries]

    def evaluate() -> Tuple[float, int]:
        scaled_counts = [0] * n_entries
        for var in variables:
            k = int(var['k'])
            for idx, _, stoich in var['entries']:
                scaled_counts[idx] = k * stoich

        total_atoms = sum(scaled_counts[i] * atoms_per_entity[i] for i in range(n_entries))
        masses = [scaled_counts[i] * mw_per_entry[i] for i in range(n_entries)]
        total_mass = sum(masses)

        atoms_term = ((total_atoms - target_atoms) / max(target_atoms, 1)) ** 2

        wt_term_sum = 0.0
        count_term_sum = 0.0
        for i in range(n_entries):
            if total_mass > 0:
                actual_wt = masses[i] / total_mass * 100.0
            else:
                actual_wt = 0.0
            wt_den = max(wt_targets[i], 1e-6)
            wt_term_sum += ((actual_wt - wt_targets[i]) / wt_den) ** 2

            count_den = max(float_counts[i], 1e-6)
            count_term_sum += ((scaled_counts[i] - float_counts[i]) / count_den) ** 2

        wt_term = wt_term_sum / max(n_entries, 1)
        count_term = count_term_sum / max(n_entries, 1)
        return atoms_term + wt_term + count_weight * count_term, total_atoms

    objective_initial, _ = evaluate()
    current_objective = objective_initial

    num_vars = len(variables)
    if max_iterations is None:
        max_iterations = max(5000, 100 * num_vars)
    improve_tol = 1e-12
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        best_move = None

        for idx, var in enumerate(variables):
            for direction in (1, -1):
                if direction < 0 and var['k'] <= var['min_k']:
                    continue

                var['k'] += direction
                trial_objective, _ = evaluate()
                var['k'] -= direction

                delta = trial_objective - current_objective
                if delta >= -improve_tol:
                    continue

                move_key = (delta, var['mass_per_step'], idx, 0 if direction == 1 else 1)
                if best_move is None or move_key < best_move['key']:
                    best_move = {
                        'key': move_key,
                        'idx': idx,
                        'direction': direction,
                        'new_objective': trial_objective
                    }

        if best_move is None:
            break

        variables[best_move['idx']]['k'] += best_move['direction']
        current_objective = best_move['new_objective']

    objective_final, final_atoms = evaluate()
    results = _expand_variables_to_results(entries, variables)
    diagnostics = {
        'rounding_mode': 'soft_atoms',
        'iterations': iterations,
        'max_iterations': max_iterations,
        'objective_initial': objective_initial,
        'objective_final': objective_final,
        'target_atoms': target_atoms,
        'actual_atoms': final_atoms,
        'converged': iterations < max_iterations
    }
    return results, diagnostics


def _compute_float_counts(entries: List[Dict], target_atoms: int, mass_basis_g: float = 1.0) -> None:
    """计算连续解 float_count（保持原有 relaxed 思路）。"""
    total_weighted = 0.0
    for entry in entries:
        if entry.get('skip_reason'):
            _set_skipped_defaults(entry)
            continue

        wt_pct = float(entry.get('target_wt_pct_effective', entry.get('wt_pct', 0)) or 0.0)
        mw = safe_float(entry.get('mw_g_mol'))
        atoms = safe_int(entry.get('atoms_per_entity')) or safe_int(entry.get('atoms_per_molecule'))
        if not mw or mw <= 0 or not atoms or atoms <= 0:
            min_count = int(entry.get('min_count', 0) or 0)
            if wt_pct <= 0 and min_count <= 0:
                entry['moles'] = 0.0
                entry['weighted_atoms'] = 0.0
                entry['float_count'] = 0.0
                continue
            raise ValueError(f"无法计算 float_count，条目缺少有效 mw/atoms: {entry.get('name', 'N/A')}")

        mass_g = mass_basis_g * wt_pct / 100.0
        moles = mass_g / mw
        weighted = moles * atoms
        total_weighted += weighted
        entry['moles'] = moles
        entry['weighted_atoms'] = weighted

    if total_weighted <= 0:
        raise ValueError("无法计算加权原子数：检查 mw_g_mol 与 atoms_per_entity")

    scale_factor = target_atoms / total_weighted
    for entry in entries:
        if entry.get('skip_reason'):
            _set_skipped_defaults(entry)
            continue
        moles = entry.get('moles')
        entry['float_count'] = float(moles * scale_factor) if moles is not None else 0.0


def _merge_allocated_results(entries: List[Dict], allocated: List[Dict]) -> List[Dict]:
    """将分配结果按原顺序合并回包含 skip 条目的完整列表。"""
    alloc_idx = 0
    merged = []
    for entry in entries:
        if entry.get('skip_reason'):
            _set_skipped_defaults(entry)
            merged.append(entry)
            continue
        merged.append(allocated[alloc_idx])
        alloc_idx += 1
    return merged


def _allocate_with_mode(
    entries: List[Dict],
    target_atoms: int,
    rounding_mode: str,
    renormalize_skipped: bool,
    require_neutral: bool
) -> Tuple[List[Dict], Optional[int], Dict[str, Any]]:
    """按 rounding_mode 执行离散分配。"""
    allocatable = [e for e in entries if not e.get('skip_reason')]
    if not allocatable:
        raise ValueError("没有可分配条目：请检查配方是否全部被跳过。")

    _precheck_charge_constraints(allocatable, require_neutral)
    _precheck_min_atoms_feasibility(allocatable, target_atoms)

    if rounding_mode == 'legacy_total':
        target_total = int(round(sum(e.get('float_count', 0) or 0 for e in allocatable)))
        if target_total > 0:
            allocated = constrained_largest_remainder_legacy(allocatable, target_total)
        else:
            allocated = allocatable
        diagnostics = {
            'rounding_mode': 'legacy_total',
            'target_total': target_total
        }
        return _merge_allocated_results(entries, allocated), target_total, diagnostics

    allocated, diagnostics = allocate_counts_soft_constraints(
        allocatable,
        target_atoms=target_atoms,
        renormalize_skipped=renormalize_skipped,
        require_neutral=require_neutral
    )
    return _merge_allocated_results(entries, allocated), None, diagnostics


def compute_by_target_atoms(
    entries: List[Dict],
    target_atoms: int,
    renormalize_skipped: bool = True,
    rounding_mode: str = 'soft_atoms',
    require_neutral: bool = True
) -> Tuple[List[Dict], Optional[int], Dict]:
    """
    按目标原子数计算。

    返回: (results, target_total_or_none, skip_info)
    """
    skip_info = _prepare_effective_targets(entries, renormalize_skipped)
    _compute_float_counts(entries, target_atoms=target_atoms, mass_basis_g=1.0)
    results, target_total, alloc_diag = _allocate_with_mode(
        entries=entries,
        target_atoms=target_atoms,
        rounding_mode=rounding_mode,
        renormalize_skipped=renormalize_skipped,
        require_neutral=require_neutral
    )
    skip_info['allocation_diagnostics'] = alloc_diag
    results = _compute_mass_and_wt(results)
    return results, target_total, skip_info


def compute_by_total_mass(
    entries: List[Dict],
    total_mass_g: float,
    scale_to_atoms: int,
    renormalize_skipped: bool = True,
    rounding_mode: str = 'soft_atoms',
    require_neutral: bool = True
) -> Tuple[List[Dict], Optional[int], Dict]:
    """按总质量换算，并缩放到目标原子数规模。"""
    if total_mass_g <= 0:
        raise ValueError("total_mass_g 必须为正数")
    if scale_to_atoms <= 0:
        raise ValueError("scale_to_atoms 必须为正整数")

    skip_info = _prepare_effective_targets(entries, renormalize_skipped)
    _compute_float_counts(entries, target_atoms=scale_to_atoms, mass_basis_g=total_mass_g)
    results, target_total, alloc_diag = _allocate_with_mode(
        entries=entries,
        target_atoms=scale_to_atoms,
        rounding_mode=rounding_mode,
        renormalize_skipped=renormalize_skipped,
        require_neutral=require_neutral
    )
    skip_info['allocation_diagnostics'] = alloc_diag
    results = _compute_mass_and_wt(results)
    return results, target_total, skip_info


def _compute_mass_and_wt(results: List[Dict]) -> List[Dict]:
    """计算实际原子数与 wt%"""
    total_mass = 0.0
    for result in results:
        count = result.get('scaled_count', 0) or 0
        atoms = safe_int(result.get('atoms_per_entity')) or safe_int(result.get('atoms_per_molecule'))
        mw = safe_float(result.get('mw_g_mol'))

        if count > 0 and atoms:
            result['scaled_atoms'] = count * atoms
        else:
            result['scaled_atoms'] = 0

        if count > 0 and mw:
            result['mass_prop'] = count * mw
            total_mass += result['mass_prop']
        else:
            result['mass_prop'] = 0

    for result in results:
        target_wt_original = result.get('target_wt_pct_original', result.get('wt_pct', 0)) or 0
        target_wt_effective = result.get(
            'target_wt_pct_effective',
            result.get('wt_pct_effective', result.get('wt_pct', 0))
        ) or 0
        actual_mass = result.get('mass_prop', 0)
        if total_mass > 0:
            actual_wt = actual_mass / total_mass * 100
        else:
            actual_wt = 0
        result['actual_wt_pct'] = actual_wt
        result['target_wt_pct_original'] = target_wt_original
        result['target_wt_pct_effective'] = target_wt_effective
        result['wt_error_abs'] = abs(actual_wt - target_wt_effective)
        result['wt_error_rel'] = (
            result['wt_error_abs'] / target_wt_effective * 100 if target_wt_effective > 0 else 0
        )

    return results


def check_charge_neutrality(results: List[Dict], require_neutral: bool) -> Tuple[bool, int, str]:
    """
    检查电荷中性 (B2)
    
    返回: (是否通过, 总电荷, 诊断信息)
    """
    total_charge = 0
    group_charges: Dict[str, int] = {}
    ungrouped_charged = []
    
    for r in results:
        charge = safe_int(r.get('charge'), 0) or 0
        count = r.get('scaled_count', 0) or 0
        contribution = charge * count
        total_charge += contribution
        
        group_name = r.get('_group_name')
        if group_name:
            group_charges[group_name] = group_charges.get(group_name, 0) + contribution
        elif charge != 0 and count > 0:
            ungrouped_charged.append({
                'name': r.get('name'),
                'charge': charge,
                'count': count,
                'contribution': contribution
            })
    
    if require_neutral and total_charge != 0:
        # 构建诊断信息
        diag_lines = [f"总电荷 = {total_charge} (要求中性)"]
        
        if ungrouped_charged:
            diag_lines.append("\n未分组的带电条目 (缺少 ion_group):")
            for uc in ungrouped_charged:
                diag_lines.append(
                    f"  - {uc['name']}: charge={uc['charge']}, count={uc['count']}, "
                    f"contribution={uc['contribution']}"
                )
            diag_lines.append("\n提示: 拆分离子需要 ion_group + stoich 确保配平")
        
        if group_charges:
            diag_lines.append("\n离子组电荷贡献:")
            for gname, gcharge in group_charges.items():
                diag_lines.append(f"  - {gname}: {gcharge}")
        
        return False, total_charge, "\n".join(diag_lines)
    
    return True, total_charge, ""


def write_csv(results: List[Dict], filepath: str):
    """输出 CSV 文件"""
    fieldnames = [
        'category', 'kind', 'name', 'wt_pct',
        'target_wt_pct_original', 'target_wt_pct_effective', 'actual_wt_pct',
        'wt_error_abs', 'wt_error_rel',
        'mw_g_mol', 'scaled_count', 'min_count',
        'atoms_per_entity', 'scaled_atoms', 'structure_file', 
        'skip_reason', 'ion_group', 'stoich', 'charge'
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            # 格式化数值
            for key in ['target_wt_pct_original', 'target_wt_pct_effective', 'actual_wt_pct', 'wt_error_abs']:
                if row[key] and isinstance(row[key], float):
                    row[key] = f"{row[key]:.2f}"
            if row['wt_error_rel'] and isinstance(row['wt_error_rel'], float):
                row['wt_error_rel'] = f"{row['wt_error_rel']:.1f}%"
            writer.writerow(row)


def write_json(results: List[Dict], filepath: str):
    """输出 JSON 文件"""
    output = []
    for r in results:
        item = {
            'category': r.get('category', ''),
            'kind': r.get('kind', ''),
            'name': r.get('name', ''),
            'wt_pct_target': r.get('wt_pct', 0),
            'wt_pct_target_original': r.get('target_wt_pct_original', r.get('wt_pct', 0)),
            'wt_pct_target_effective': r.get('target_wt_pct_effective', r.get('wt_pct', 0)),
            'wt_pct_actual': round(r.get('actual_wt_pct', 0), 2),
            'wt_error_rel_pct': round(r.get('wt_error_rel', 0), 1),
            'mw_g_mol': r.get('mw_g_mol'),
            'scaled_count': r.get('scaled_count'),
            'min_count': r.get('min_count', 0),
            'atoms_per_entity': r.get('atoms_per_entity') or r.get('atoms_per_molecule'),
            'scaled_atoms': r.get('scaled_atoms'),
            'structure_file': r.get('structure_file'),
            'charge': r.get('charge', 0),
            'ion_group': r.get('ion_group'),
            'stoich': r.get('stoich'),
        }
        if 'skip_reason' in r:
            item['skip_reason'] = r['skip_reason']
        output.append(item)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def write_report(results: List[Dict], filepath: str, target_atoms: int,
                 target_total: Optional[int], total_charge: int, wt_total: float,
                 wt_tol: float, wt_ok: bool, require_neutral: bool,
                 polymer_warnings: List[str], skip_info: Dict,
                 rounding_mode: str):
    """输出误差报告"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("配方换算误差报告 (v3.1)\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"目标原子数: {target_atoms}\n")
        if target_total is None:
            f.write("目标实体总数: N/A (soft_atoms 模式不再强制固定实体总数)\n")
        else:
            f.write(f"目标实体总数: {target_total}\n")
        f.write(f"凑整模式: {rounding_mode}\n")
        f.write(f"wt% 总和: {wt_total:.4f} (容差 ±{wt_tol}) -> {'通过' if wt_ok else '失败'}\n")
        f.write(f"总电荷: {total_charge} ({'要求中性' if require_neutral else '允许带电'})\n")
        
        actual_atoms = sum(r.get('scaled_atoms', 0) or 0 for r in results)
        f.write(f"实际原子数: {actual_atoms}\n")
        if target_atoms > 0:
            diff_pct = abs(actual_atoms - target_atoms) / target_atoms * 100
            f.write(f"差异: {abs(actual_atoms - target_atoms)} ({diff_pct:.1f}%)\n")
        
        # 跳过信息 (B3)
        if skip_info.get('skipped_entries'):
            f.write(f"\n--- 跳过的组分 ---\n")
            f.write(f"跳过 wt% 总计: {skip_info['skipped_wt_total']:.2f}%\n")
            if skip_info.get('renormalized'):
                f.write(f"[INFO] 剩余组分 wt% 已重归一化至 100%\n")
            for se in skip_info['skipped_entries']:
                f.write(f"  - {se['name']}: {se['wt_pct']:.2f}% ({se['reason']})\n")

        alloc_diag = skip_info.get('allocation_diagnostics', {}) or {}
        if alloc_diag:
            f.write("\n--- 分配诊断 ---\n")
            for key in ['iterations', 'max_iterations', 'objective_initial', 'objective_final', 'converged']:
                if key in alloc_diag:
                    f.write(f"{key}: {alloc_diag[key]}\n")
        
        f.write("\n" + "-" * 90 + "\n")
        f.write(f"{'名称':<29} {'目标orig%':>10} {'目标eff%':>10} {'实际wt%':>10} {'误差':>10} {'count':>8}\n")
        f.write("-" * 90 + "\n")
        
        warnings = []
        for r in results:
            name = r.get('name', 'N/A')
            if len(name) > 27:
                name = name[:24] + "..."
            
            target_orig = r.get('target_wt_pct_original', r.get('wt_pct', 0)) or 0
            target_eff = r.get('target_wt_pct_effective', r.get('wt_pct', 0)) or 0
            actual = r.get('actual_wt_pct', 0) or 0
            error = r.get('wt_error_rel', 0) or 0
            count = r.get('scaled_count', 0) or 0
            
            marker = "⚠️ " if error > 50 else "   "
            f.write(
                f"{marker}{name:<26} {target_orig:>10.2f} {target_eff:>10.2f} "
                f"{actual:>10.2f} {error:>9.1f}% {count:>8}\n"
            )
            
            if error > 50:
                warnings.append((name, error))
        
        f.write("-" * 90 + "\n\n")
        
        if warnings:
            f.write("⚠️ 警告: 以下组分相对误差 > 50%:\n")
            for name, error in warnings:
                f.write(f"   - {name}: {error:.1f}%\n")
            f.write("\n建议: 增大 target_atoms 以减小凑整误差\n")
        else:
            f.write("✓ 所有组分相对误差 < 50%\n")

        if polymer_warnings:
            f.write("\n⚠️ 聚合物定义一致性检查:\n")
            for w in polymer_warnings:
                f.write(f"   - {w}\n")

        f.write("\n" + "=" * 90 + "\n")


def print_summary(results: List[Dict], target_atoms: int,
                  target_total: Optional[int], total_charge: int,
                  wt_total: float, wt_tol: float, wt_ok: bool,
                  require_neutral: bool, polymer_warnings: List[str],
                  skip_info: Dict, rounding_mode: str):
    """打印摘要"""
    print("\n" + "=" * 90)
    print("换算结果摘要（含误差分析）v3.1")
    print("=" * 90)
    
    # 跳过信息
    if skip_info.get('skipped_entries'):
        print(f"\n--- 跳过的组分 (wt% 总计: {skip_info['skipped_wt_total']:.2f}%) ---")
        if skip_info.get('renormalized'):
            print("[INFO] 剩余组分 wt% 已重归一化至 100%")
        for se in skip_info['skipped_entries']:
            print(f"  - {se['name']}: {se['wt_pct']:.2f}% ({se['reason']})")

    print(f"\n{'类别':<20} {'名称':<30} {'目标eff%':>10} {'实际%':>8} {'误差':>8} {'count':>8}")
    print("-" * 90)

    current_cat = None
    total_count = 0
    total_atoms = 0
    warnings = []

    for r in results:
        cat = r.get('category', '')
        if cat != current_cat:
            current_cat = cat
            cat_cn = r.get('category_cn', cat)
            print(f"\n>>> {cat} ({cat_cn})")

        name = r.get('name', 'N/A')
        if len(name) > 28:
            name = name[:25] + "..."

        target = r.get('target_wt_pct_effective', r.get('wt_pct', 0)) or 0
        actual = r.get('actual_wt_pct', 0) or 0
        error = r.get('wt_error_rel', 0) or 0
        count = r.get('scaled_count', None)
        atoms = r.get('scaled_atoms', None)

        count_str = str(count) if count is not None else "---"
        error_str = f"{error:.1f}%" if error > 0 else "---"
        
        marker = "⚠️" if error > 50 else "  "
        print(f"{marker}  {name:<28} {target:>10.2f} {actual:>8.2f} {error_str:>8} {count_str:>8}")

        if count is not None:
            total_count += count
        if atoms is not None:
            total_atoms += atoms
        
        if error > 50:
            warnings.append((r.get('name', 'N/A'), error))

    print("\n" + "-" * 90)
    print(f"总计: {total_count} 分子, {total_atoms} 原子 (目标: {target_atoms})")
    diff_atoms = abs(total_atoms - target_atoms)
    diff_pct = diff_atoms / target_atoms * 100 if target_atoms > 0 else 0
    print(f"差异: {diff_atoms} 原子 ({diff_pct:.1f}%)")
    
    # B4: 偏差警告
    threshold = max(5, 0.02 * target_atoms)
    if diff_atoms > threshold:
        print(f"[WARN] 实际原子数偏差 > {threshold:.0f}，建议增大 target_atoms 或调整 atoms_per_entity")
    
    if target_total is None:
        print("目标实体总数: N/A (soft_atoms)")
    else:
        print(f"目标实体总数: {target_total}")
    print(f"凑整模式: {rounding_mode}")
    print(f"wt% 总和: {wt_total:.4f} (容差 ±{wt_tol}) -> {'通过' if wt_ok else '失败'}")
    print(f"总电荷: {total_charge} ({'要求中性' if require_neutral else '允许带电'})")
    
    if warnings:
        print("\n⚠️ 警告: 以下组分相对误差 > 50%:")
        for name, error in warnings:
            name_short = name[:40] + "..." if len(name) > 40 else name
            print(f"   - {name_short}: {error:.1f}%")
        print("\n   建议: 增大 --target_atoms 以减小凑整误差")

    if polymer_warnings:
        print("\n⚠️ 聚合物定义一致性检查:")
        for w in polymer_warnings:
            print(f"   - {w}")
    
    print("=" * 90)


def _parse_int_field(
    value: Any,
    field_name: str,
    entry_name: str,
    min_value: Optional[int] = None,
    default_if_none: Optional[int] = None
) -> int:
    """严格解析整数字段。"""
    if value is None:
        if default_if_none is not None:
            return default_if_none
        raise ValueError(f"{field_name} 缺失 (entry={entry_name})")

    if isinstance(value, bool):
        raise ValueError(f"{field_name} 不能是布尔值 (entry={entry_name})")

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} 必须为整数 (entry={entry_name})") from None

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field_name} 必须为整数 (entry={entry_name})")

    parsed = int(numeric)
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} 必须 >= {min_value} (entry={entry_name})")
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="配方 wt% 转换为分子/原子数量 v3.1（软约束分配）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 recipe_to_counts.py --target_atoms 200
  python3 recipe_to_counts.py --recipe recipe.yaml  # 使用 YAML 中的 target_atoms
  python3 recipe_to_counts.py --target_atoms 500 --output counts_500
  python3 recipe_to_counts.py --total_mass_g 1.0 --scale_to_atoms 5000

v3.0 新特性:
  - ion_group + stoich 约束分配，保证离子配平
  - --allow_missing_low_wt 跳过组分后自动重归一化
  - 从 simulation.target_atoms 读取默认值
  - 默认 wt_tol=1e-3（更严格）

v3.1 新特性:
  - 默认 soft_atoms 软约束凑整（不再强制固定实体总数）
  - legacy_total 兼容模式可复现实旧行为
        """
    )

    parser.add_argument("--recipe", default="recipe.yaml",
                        help="配方文件路径 (默认: recipe.yaml)")
    parser.add_argument("--output", default="counts",
                        help="输出文件前缀 (默认: counts)")
    
    # B6: target_atoms 可选，可从 YAML 读取
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--target_atoms", type=int,
                       help="目标总原子数 (可从 simulation.target_atoms 读取)")
    group.add_argument("--total_mass_g", type=float,
                       help="总质量 (g)，与 --scale_to_atoms 配合使用")
    
    parser.add_argument("--scale_to_atoms", type=int, default=None,
                        help="将体系缩放到约该原子数规模（配合 --total_mass_g）")
    parser.add_argument("--wt_tol", type=float, default=1e-3,
                        help="wt%% 总和容差 (默认: 1e-3)")
    parser.add_argument("--allow_missing_low_wt", type=float, default=None,
                        help="允许极低 wt%% 成分跳过的阈值 (例如 0.5)")
    parser.add_argument("--no_renormalize_skipped", action="store_true",
                        help="禁用跳过组分后的 wt%% 重归一化")
    parser.add_argument("--rounding_mode", choices=['soft_atoms', 'legacy_total'],
                        default='soft_atoms',
                        help="凑整模式: soft_atoms(默认) / legacy_total(兼容旧版固定实体总数)")
    parser.add_argument("--require_neutral", dest="require_neutral",
                        action="store_true", default=True,
                        help="要求总电荷为 0 (默认开启)")
    parser.add_argument("--allow_charged", dest="require_neutral",
                        action="store_false",
                        help="允许非中性体系（仅用于明确带电体系）")

    args = parser.parse_args()

    print("=" * 90)
    print("recipe_to_counts.py - 配方换算工具 (v3.1)")
    print("=" * 90)
    print(f"配方文件: {args.recipe}")

    # 加载配方
    data = load_yaml(args.recipe)
    
    # B6: 从 YAML 读取 target_atoms
    yaml_target_atoms = None
    sim = data.get('simulation', {})
    if sim:
        yaml_target_atoms = safe_int(sim.get('target_atoms'))
    
    # 确定 target_atoms
    if args.target_atoms is not None:
        target_atoms = args.target_atoms
    elif args.total_mass_g is not None:
        if args.scale_to_atoms is None:
            print("[ERROR] 使用 --total_mass_g 时必须同时指定 --scale_to_atoms")
            sys.exit(1)
        target_atoms = args.scale_to_atoms
    elif yaml_target_atoms is not None:
        target_atoms = yaml_target_atoms
        print(f"[INFO] 使用 YAML 中的 simulation.target_atoms = {target_atoms}")
    else:
        print("[ERROR] 必须指定 --target_atoms 或在 YAML 中定义 simulation.target_atoms")
        sys.exit(1)

    if target_atoms <= 0:
        print("[ERROR] target_atoms 必须为正整数")
        sys.exit(1)
    
    print(f"目标原子数: {target_atoms}")
    print(f"凑整模式: {args.rounding_mode}")

    entries = flatten_recipe(data)

    if not entries:
        print("[ERROR] 配方为空")
        sys.exit(1)

    print(f"\n>>> 读取 {len(entries)} 个条目")

    # 预检查：wt% 总和
    wt_total = sum((e.get('wt_pct', 0) or 0) for e in entries)
    wt_ok = abs(wt_total - 100.0) <= args.wt_tol
    if not wt_ok:
        print(f"[ERROR] wt% 总和 {wt_total:.4f} 超出容差 ±{args.wt_tol}")
        sys.exit(1)

    # 字段校验
    polymer_warnings = []
    for entry in entries:
        name = entry.get('name', 'N/A')
        wt_pct = float(entry.get('wt_pct', 0) or 0)
        mw = safe_float(entry.get('mw_g_mol'), None)
        is_polymer_unit = entry.get('is_polymer_unit', False)

        try:
            min_count = _parse_int_field(
                entry.get('min_count', 0), 'min_count', name, min_value=0, default_if_none=0
            )
            charge_int = _parse_int_field(
                entry.get('charge', 0), 'charge', name, default_if_none=0
            )
            stoich_int = _parse_int_field(
                entry.get('stoich', 1), 'stoich', name, min_value=1, default_if_none=1
            )
        except ValueError as exc:
            print(f"[ERROR] 字段校验失败: name={name}, {exc}")
            sys.exit(1)

        entry['min_count'] = min_count
        entry['charge'] = charge_int
        entry['stoich'] = stoich_int

        if entry.get('skip_reason'):
            if min_count > 0:
                print(f"[ERROR] 条目已跳过但 min_count>0: name={name}")
                sys.exit(1)
            _set_skipped_defaults(entry)
            continue

        atoms_source = entry.get('atoms_per_entity')
        if atoms_source is None:
            atoms_source = entry.get('atoms_per_molecule')
        atoms = None
        if atoms_source is not None:
            try:
                atoms = _parse_int_field(
                    atoms_source, 'atoms_per_entity/atoms_per_molecule', name, min_value=1
                )
            except ValueError:
                atoms = None

        invalid_mw = (mw is None or mw <= 0)
        invalid_atoms = (atoms is None or atoms <= 0)
        low_wt_skip = (args.allow_missing_low_wt is not None and wt_pct <= args.allow_missing_low_wt and wt_pct > 0)

        if invalid_mw or invalid_atoms:
            if low_wt_skip:
                reason_parts = []
                if invalid_mw:
                    reason_parts.append("mw_g_mol<=0")
                if invalid_atoms:
                    reason_parts.append("atoms_per_entity<=0")
                entry['skip_reason'] = f"{' 和 '.join(reason_parts)} (低 wt% 允许跳过)"
                _set_skipped_defaults(entry)
                continue

            if wt_pct > 0 or min_count > 0:
                print(
                    f"[ERROR] 活跃条目 mw/atoms 非法: name={name}, wt%={wt_pct}, "
                    f"mw_g_mol={entry.get('mw_g_mol')}, atoms_per_entity={entry.get('atoms_per_entity') or entry.get('atoms_per_molecule')}"
                )
                print("        要求 mw_g_mol > 0 且 atoms_per_entity/atoms_per_molecule > 0")
                sys.exit(1)
        else:
            entry['mw_g_mol'] = float(mw)
            entry['atoms_per_entity'] = int(atoms)
            if 'atoms_per_molecule' in entry and entry.get('atoms_per_molecule') in (None, 0):
                entry['atoms_per_molecule'] = int(atoms)

        # B5: 聚合物一致性 sanity-check（仅警告）
        kind = str(entry.get('kind', '')).lower()
        cat = str(entry.get('category', '')).lower()
        if (kind == 'polymer' or cat == 'polymer_matrix') and mw and atoms and not is_polymer_unit:
            ratio = mw / atoms if atoms > 0 else None
            if ratio is not None and (ratio < 2.0 or ratio > 50.0):
                polymer_warnings.append(
                    f"聚合物条目可能定义不一致: {entry.get('name')} (mw={mw}, atoms={atoms})"
                )
            if mw < 200 and atoms > 200:
                polymer_warnings.append(
                    f"聚合物条目可能单位混用: {entry.get('name')} (mw={mw}, atoms={atoms})"
                )

    for w in polymer_warnings:
        print(f"[WARN] {w}")

    # 执行换算
    renormalize = not args.no_renormalize_skipped
    try:
        if args.total_mass_g is not None:
            results, target_total, skip_info = compute_by_total_mass(
                entries,
                args.total_mass_g,
                args.scale_to_atoms,
                renormalize,
                args.rounding_mode,
                args.require_neutral
            )
        else:
            results, target_total, skip_info = compute_by_target_atoms(
                entries,
                target_atoms,
                renormalize,
                args.rounding_mode,
                args.require_neutral
            )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # B2: 电荷中性检查
    neutral_ok, total_charge, charge_diag = check_charge_neutrality(results, args.require_neutral)
    if not neutral_ok:
        print(f"\n[ERROR] 电荷中性检查失败")
        print(charge_diag)
        print("\n[INFO] 若要允许带电体系，请使用 --allow_charged")
        sys.exit(1)

    # 输出文件
    csv_path = f"{args.output}.csv"
    json_path = f"{args.output}.json"
    report_path = f"{args.output}_report.txt"

    print(f"\n>>> 输出文件:")
    write_csv(results, csv_path)
    print(f"    [OK] {csv_path}")

    write_json(results, json_path)
    print(f"    [OK] {json_path}")

    write_report(results, report_path, target_atoms, target_total,
                 total_charge, wt_total, args.wt_tol, wt_ok,
                 args.require_neutral, polymer_warnings, skip_info,
                 args.rounding_mode)
    print(f"    [OK] {report_path}")

    # 打印摘要
    print_summary(results, target_atoms, target_total, total_charge,
                  wt_total, args.wt_tol, wt_ok, args.require_neutral,
                  polymer_warnings, skip_info, args.rounding_mode)

    # 实际原子数
    actual_atoms = sum(r.get('scaled_atoms', 0) or 0 for r in results)
    diff = abs(actual_atoms - target_atoms)
    print(f"\n[INFO] 实际总原子数: {actual_atoms}")
    print(f"[INFO] 与目标差异: {diff} 原子 ({diff/target_atoms*100:.1f}%)")


if __name__ == "__main__":
    main()
