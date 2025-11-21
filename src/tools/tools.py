import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist
import autoadsorbate.Surf 
# 确保先导入原模块，以便我们覆盖它

# 修复 Autoadsorbate 库中 get_shrinkwrap_grid 函数的死循环 Bug
# 该修复通过添加 Z 轴下限检查，防止网格点从表面空隙中无限掉落
def get_shrinkwrap_grid_fixed(
    slab,
    precision,
    drop_increment=0.1,
    touch_sphere_size=3,
    marker="He",
    raster_speed_boost=False,
):
    # 引入必要的依赖 (原函数内部引用的依赖)
    from autoadsorbate.Surf import _get_starting_grid, get_large_atoms
    
    # 处理 raster_speed_boost
    if raster_speed_boost:
        from autoadsorbate.raster_utilities import get_surface_from_rasterized_top_view
        raster_surf_index = get_surface_from_rasterized_top_view(
            slab, pixel_per_angstrom=10
        )
        slab = slab[raster_surf_index]

    # 获取初始网格
    starting_grid, faces = _get_starting_grid(slab, precision=precision)
    grid_positions = starting_grid.positions
    large_slab = get_large_atoms(slab)
    slab_positions = large_slab.positions

    distances_to_grid = cdist(grid_positions, slab_positions).min(axis=1)
    drop_vectors = np.array([[0, 0, drop_increment] for _ in grid_positions])

    # 原代码: while (distances_to_grid > touch_sphere_size).any():
    # 修改后: 增加 (grid_positions[:, 2] > -1.0) 条件
    # 只有当点离表面远 且 Z坐标大于 -1.0 时才继续移动。
    # 一旦掉到 -1.0 以下，就视为“穿透”并停止移动，防止死循环。
    while ((distances_to_grid > touch_sphere_size) & (grid_positions[:, 2] > -1.0)).any():
        
        # 计算需要移动的点的掩码
        mask_to_move = (distances_to_grid > touch_sphere_size) & (grid_positions[:, 2] > -1.0)
        
        # 只更新这些点的位置
        grid_positions -= (
            drop_vectors * mask_to_move[:, np.newaxis]
        )
        
        # 重新计算距离
        distances_to_grid = cdist(grid_positions, slab_positions).min(axis=1)

        # 保留原有的退出条件作为双重保险
        if (distances_to_grid > touch_sphere_size).all() and (
            grid_positions[:, 2] <= 0
        ).all():
            break

    grid = Atoms(
        [marker for _ in grid_positions],
        grid_positions,
        pbc=[True, True, True],
        cell=slab.cell,
    )
    # 过滤掉掉到 Z=0 以下的点（即穿透表面的点），只保留挂在表面上的点
    grid = grid[[atom.index for atom in grid if atom.position[2] > 0]]

    return grid, faces

def get_shrinkwrap_ads_sites_fixed(
    atoms: Atoms,
    precision: float = 0.25,  # 默认精度从 0.5 提升到 0.25
    touch_sphere_size: float = 3,
    return_trj: bool = False,
    return_geometry = False
):
    import numpy as np
    import itertools
    from ase import Atom
    # 引用原库中的辅助函数
    from autoadsorbate.Surf import (
        get_shrinkwrap_grid, # 注意：这会自动使用我们刚才Patch过的Fixed版本
        shrinkwrap_surface, 
        get_list_of_touching, 
        get_wrapped_site,
        get_shrinkwrap_site_n_vector,
        get_shrinkwrap_site_h_vector
    )

    # 1. 获取网格
    grid, faces = get_shrinkwrap_grid(
        atoms, precision=precision, touch_sphere_size=touch_sphere_size
    )
    
    # 2. 获取表面原子索引
    surf_ind = shrinkwrap_surface(
        atoms, precision=precision, touch_sphere_size=touch_sphere_size
    )
    
    # 3. 识别接触点时，将 epsilon 从 0.1 提升到 0.3
    # 这允许网格点即使稍微偏离中心，也能正确“抓”住周围的所有原子
    targets = get_list_of_touching(atoms, grid, surf_ind, touch_sphere_size=touch_sphere_size, epsilon=0.3)

    # 以下逻辑与原函数保持一致，用于计算向量和拓扑
    trj = []
    coordinates = []
    connectivity = []
    topology = []
    n_vector = []
    h_vector = []
    site_formula = []

    for target in targets:
        atoms_copy = atoms.copy()

        for index in target:
            atoms_copy.append(Atom("X", atoms_copy[index].position + [0, 0, 0]))

        extended_atoms = atoms_copy.copy() * [2, 2, 1]
        extended_grid = grid.copy() * [2, 2, 1]

        if len(target) == 1:
            site_atoms = atoms_copy[target]
            site_coord = site_atoms.positions[0]

        else:
            combs = []
            min_std_devs = []

            # 寻找几何中心
            for c in itertools.combinations(
                [atom.index for atom in extended_atoms if atom.symbol == "X"],
                len(target),
            ):
                c = list(c)
                min_std_devs.append(max(extended_atoms.positions[c].std(axis=0)))
                combs.append(c)

            min_std_devs = np.array(min_std_devs)
            min_comb_index = np.argmin(min_std_devs)

            site_atoms = extended_atoms[combs[min_comb_index]]
            site_coord = np.mean(site_atoms.positions, axis=0)
            site_coord = get_wrapped_site(site_coord, atoms_copy)
            site_coord = np.array(site_coord)

        n_vec = get_shrinkwrap_site_n_vector(
            extended_atoms, site_coord, extended_grid, touch_sphere_size
        )
        h_vec = get_shrinkwrap_site_h_vector(site_atoms, n_vec)
        site_form = atoms[target].symbols.formula.count()

        coordinates.append(site_coord)
        n_vector.append(n_vec)
        h_vector.append(h_vec)
        topology.append(target)
        connectivity.append(len(target))
        site_formula.append(site_form)

    sites_dict = {
        "coordinates": coordinates,
        "connectivity": connectivity,
        "topology": topology,
        "n_vector": n_vector,
        "h_vector": h_vector,
        "site_formula": site_formula,
    }

    if return_trj:
        extended_atoms = extended_atoms[
            [
                atom.index
                for atom in extended_atoms
                if np.linalg.norm(atom.position - site_coord) < 7
            ]
        ]
        for m in range(20):
            extended_atoms.append(Atom("H", site_coord + n_vec * m * 0.5))
        trj.append(extended_atoms)
        return sites_dict, trj
    
    if return_geometry:
        return grid.positions, faces, sites_dict

    return sites_dict

# 应用补丁：用我们的修复版函数替换掉库中的原函数
print("--- 🩹 应用 Autoadsorbate 热修复 (Monkey Patch) ... ---")

# 1. Patch 源头 (Surf.py) - 以防万一有其他地方用它
autoadsorbate.Surf.get_shrinkwrap_grid = get_shrinkwrap_grid_fixed
autoadsorbate.Surf.get_shrinkwrap_ads_sites = get_shrinkwrap_ads_sites_fixed

# 2. 关键修复：Patch 消费者 (autoadsorbate.py)
# 必须覆盖 autoadsorbate.autoadsorbate 命名空间里已经导入的旧函数引用
import autoadsorbate.autoadsorbate 
autoadsorbate.autoadsorbate.get_shrinkwrap_ads_sites = get_shrinkwrap_ads_sites_fixed

print("--- ✅ 修复已应用。Surf 模块及 Surface 类引用的函数已被安全替换。 ---")

from collections import Counter
import ase
from ase.io import read, write
from autoadsorbate import Surface, Fragment
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp
from ase.md.langevin import Langevin
from ase import units
import os
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from scipy.sparse.csgraph import connected_components
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union, Tuple

def get_atom_index_menu(original_smiles: str) -> str:
    print(f"--- 🛠️ 正在为 {original_smiles} 生成重原子索引列表 ---")
    try:
        mol = Chem.MolFromSmiles(original_smiles)
        if not mol:
            raise ValueError(f"RDKit 无法解析 SMILES: {original_smiles}")
        atom_list = []
        for atom in mol.GetAtoms():
            atom_list.append({
                "index": atom.GetIdx(),
                "symbol": atom.GetSymbol()
            })
        heavy_atom_menu = [atom for atom in atom_list if atom["symbol"] != 'H']
        print(f"--- 🛠️ 重原子索引列表已生成: {json.dumps(heavy_atom_menu)} ---")
        return json.dumps(heavy_atom_menu, indent=2)
    except Exception as e:
        print(f"--- 🛑 get_atom_index_menu 失败: {e} ---")
        return json.dumps({"error": f"无法生成重原子索引列表: {e}"})

def generate_surrogate_smiles(original_smiles: str, binding_atom_indices: list[int], site_type: str) -> str:
    print(f"--- 🔬 调用 SMILES 翻译器: {original_smiles} via indices {binding_atom_indices} (位点: {site_type}) ---")
    
    mol = Chem.MolFromSmiles(original_smiles)
    if not mol:
        raise ValueError(f"RDKit 无法解析原始 SMILES: {original_smiles}")
    
    num_binding_indices = len(binding_atom_indices)
    
    # --- end-on @ ontop ---
    if site_type == "ontop":
        if num_binding_indices != 1:
            raise ValueError(f"'ontop' 位点需要 1 个键合索引，但提供了 {num_binding_indices} 个。")
            
        target_idx = binding_atom_indices[0]
        
        if target_idx >= mol.GetNumAtoms():
             raise ValueError(f"索引 {target_idx} 超出范围 (分子原子数: {mol.GetNumAtoms()})。")

        new_mol = Chem.RWMol()

        # 1. 添加 Cl 标记 (索引 0)，并设置原子映射号为 1
        marker_atom = Chem.Atom("Cl")
        marker_atom.SetAtomMapNum(1) # [Cl:1]
        marker_idx = new_mol.AddAtom(marker_atom) # index 0
        
        # 2. 复制原始分子原子
        idx_map = {}
        for atom in mol.GetAtoms():
            new_idx = new_mol.AddAtom(atom)
            idx_map[atom.GetIdx()] = new_idx
        
        # 3. 复制所有原始键
        for bond in mol.GetBonds():
            new_mol.AddBond(idx_map[bond.GetBeginAtomIdx()], idx_map[bond.GetEndAtomIdx()], bond.GetBondType())
        
        # 4. 添加 Cl-Atom 键
        new_mol.AddBond(marker_idx, idx_map[target_idx], Chem.rdchem.BondType.SINGLE)
        
        # 5. 调整电荷 (基于价电子数，区分共价键和配位键)
        target_atom_obj = new_mol.GetAtomWithIdx(idx_map[target_idx])

        # FIX: RDKit 可能会在 AddHs 或 Embed 过程中吞掉显式的 [H] 原子。
        # 强制将其设为同位素 2 (氘)，RDKit 会将其视为重原子保留，
        # 而 ASE 转换时 symbol 依然是 'H'，物理上无影响。
        if target_atom_obj.GetSymbol() == 'H':
            print(f"--- 🔬 检测到 H 原子吸附，应用同位素标记 [2H] 以防止 RDKit 吞噬... ---")
            target_atom_obj.SetIsotope(2)

        # 从 RDKit 获取化学原理
        atomic_num = target_atom_obj.GetAtomicNum()
        charge = target_atom_obj.GetFormalCharge()
        pt = Chem.GetPeriodicTable()
        
        # 使用 *正确* 的 RDKit API: GetNOuterElecs (获取外层/价电子数)
        n_outer_elecs = pt.GetNOuterElecs(atomic_num)

        # 特例：一氧化碳 ([C-]#[O+])，C[0] (4价电子) 但 charge = -1
        is_carbon_monoxide_case = (n_outer_elecs == 4 and charge == -1)

        # “价电子数>4”逻辑：(N, O, S, Se...) 
        # 并且它们是中性或负电性的（即它们有孤对电子可以给出）
        has_lone_pair_to_donate = (n_outer_elecs > 4 and charge <= 0)

        if has_lone_pair_to_donate or is_carbon_monoxide_case:
            # --- 模拟配位键 (Dative Bond) ---
            # (N, O, S, Se... 或 N- 或 C-)
            # 增加电荷以释放孤对电子用于成键
            print(f"--- 🔬 (价电子: {n_outer_elecs}) 正在为配位原子 {target_atom_obj.GetSymbol()} (Charge={charge}) 应用 +1 电荷调整... ---")
            target_atom_obj.SetFormalCharge(charge + 1)
        else:
            # --- 模拟共价键 (Covalent Bond) ---
            # (C, B, Si... 或 [O+] 等已氧化的原子)
            # 不调整电荷，让 Chem.AddHs 自动少加一个H
            print(f"--- 🔬 (价电子: {n_outer_elecs}) 正在为共价原子 {target_atom_obj.GetSymbol()} (Charge={charge}) 保留原始电荷... ---")

        # 6. 为我们关心的*成键原子*添加唯一的跟踪器
        target_atom_obj.SetAtomMapNum(114514)

        out_smiles = Chem.MolToSmiles(new_mol.GetMol(), canonical=False, rootedAtAtom=marker_idx)
        # RDKit 现在会生成类似 "[Cl:1][C:114514]#O" 的SMILES
        print(f"--- 🔬 SMILES 翻译器输出: {out_smiles} ---")
        return out_smiles

    # --- 逻辑 2 & 3: end-on/side-on @ bridge/hollow ---
    elif site_type in ["bridge", "hollow"]:
        if num_binding_indices == 1:
            # --- end-on @ bridge/hollow ---
            target_idx = binding_atom_indices[0]
            if target_idx >= mol.GetNumAtoms():
                 raise ValueError(f"索引 {target_idx} 超出范围 (分子原子数: {mol.GetNumAtoms()})。")

            rw_mol = Chem.RWMol(mol)
            atom1 = rw_mol.GetAtomWithIdx(target_idx)
            atom1.SetAtomMapNum(114514)

            original_smiles_mapped = Chem.MolToSmiles(rw_mol.GetMol(), canonical=False)

            # 使用“点运算符”来欺骗 RDKit 加氢
            out_smiles = f"{original_smiles_mapped}.[S:1].[S:2]"
            print(f"--- 🔬 SMILES 翻译器输出: {out_smiles} ---")
            return out_smiles

        elif num_binding_indices == 2:
            # --- side-on @ bridge/hollow ---
            target_indices = sorted(binding_atom_indices)
            idx1, idx2 = target_indices[0], target_indices[1]

            if idx2 >= mol.GetNumAtoms():
                 raise ValueError(f"索引 {idx2} 超出范围 (分子原子数: {mol.GetNumAtoms()})。")

            rw_mol = Chem.RWMol(mol)
            atom1 = rw_mol.GetAtomWithIdx(idx1)
            atom2 = rw_mol.GetAtomWithIdx(idx2)

            atom1.SetAtomMapNum(114514) # 跟踪器 1
            atom2.SetAtomMapNum(1919810) # 跟踪器 2

            original_smiles_mapped = Chem.MolToSmiles(rw_mol.GetMol(), canonical=False)

            # 使用“点运算符”来欺骗 RDKit 加氢
            out_smiles = f"{original_smiles_mapped}.[S:1].[S:2]"
            print(f"--- 🔬 SMILES 翻译器输出: {out_smiles} ---")
            return out_smiles

        else:
            raise ValueError(f"'{site_type}' 位点不支持 {num_binding_indices} 个键合索引。")

    else:
        raise ValueError(f"未知的 site_type: {site_type}。必须是 'ontop', 'bridge' 或 'hollow'。")

def read_atoms_object(slab_path: str) -> ase.Atoms:
    try:
        atoms = read(slab_path)  # 从 .xyz 或 .cif 文件中读取 slab 结构。
        print(f"成功: 已从 {slab_path} 读取 slab 原子。")
        return atoms
    except Exception as e:
        print(f"错误: 无法读取 {slab_path}: {e}")
        raise

# --- 统一处理表面的扩胞和清理 ---
def prepare_slab(slab_atoms: ase.Atoms) -> Tuple[ase.Atoms, bool]:
    """
    清理 Slab 的元数据，并根据需要进行扩胞 (Supercell)，以确保物理模拟的准确性。
    返回: (处理后的 Slab, 是否进行了扩胞)
    """
    print("--- 🛠️ [Prepare] 正在清理 Slab 元数据并检查尺寸... ---")
    
    # 1. 清理元数据 (解决 autoadsorbate 解析 extxyz 额外列时的崩溃问题)
    symbols = slab_atoms.get_chemical_symbols()
    positions = slab_atoms.get_positions()
    cell = slab_atoms.get_cell()
    pbc = slab_atoms.get_pbc()
    
    clean_slab = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    
    # 2. 智能扩胞 (解决 1x1 晶胞找不到 Hollow 位点的问题)
    # 逻辑: 如果 XY 平面任意晶格矢量长度小于 6.0 Å，则扩胞为 2x2
    cell_vectors = clean_slab.get_cell()
    a_len = np.linalg.norm(cell_vectors[0])
    b_len = np.linalg.norm(cell_vectors[1])
    
    is_expanded = False
    if a_len < 6.0 or b_len < 6.0:
        print(f"--- 🛠️ [Prepare] 检测到微小晶胞 (a={a_len:.2f}Å, b={b_len:.2f}Å)。正在扩胞为 2x2x1... ---")
        clean_slab = clean_slab * (2, 2, 1)
        is_expanded = True
    else:
        print(f"--- 🛠️ [Prepare] 晶胞尺寸足够 (a={a_len:.2f}Å, b={b_len:.2f}Å)。保持原样。 ---")
        
    return clean_slab, is_expanded

def analyze_surface_sites(slab_path: str) -> dict:
    """ 预扫描表面，找出实际存在的位点类型，供 Planner 参考 """
    from collections import defaultdict, Counter
    atoms = read_atoms_object(slab_path)
    clean_slab, _ = prepare_slab(atoms)
    
    # 空跑 Autoadsorbate
    s = Surface(clean_slab, precision=1.0, touch_sphere_size=3.0, mode='slab')
    s.sym_reduce()
    
    site_inventory = defaultdict(set)
    for _, row in s.site_df.iterrows():
        conn = row['connectivity']
        # 将 {'Mo':2, 'Pd':1} 转为 "Mo-Mo-Pd"
        elements = []
        for el, count in row['site_formula'].items():
            elements.extend([el] * count)
        site_desc = "-".join(sorted(elements))
        site_inventory[conn].add(site_desc)

    desc_list = []
    conn_map = {1: "Ontop", 2: "Bridge", 3: "Hollow-3", 4: "Hollow-4"}
    for conn, sites in site_inventory.items():
        label = conn_map.get(conn, f"{conn}-fold")
        desc_list.append(f"[{label}]: {', '.join(sorted(list(sites)))}")
        
    return {
        "surface_composition": [item[0] for item in Counter(clean_slab.get_chemical_symbols()).most_common()],
        "available_sites_description": "; ".join(desc_list)
    }

def _get_fragment(SMILES: str, site_type: str, num_binding_indices: int, to_initialize: int = 1) -> Union[Fragment, ase.Atoms]:
    TRICK_SMILES = "Cl" if site_type == "ontop" else "S1S"
    print(f"--- 🛠️ _get_fragment: 正在为 {site_type} 位点准备 {TRICK_SMILES} 标记...")

    try:
        mol = Chem.MolFromSmiles(SMILES, sanitize=False)
        if not mol:
            raise ValueError(f"RDKit 无法解析映射的 SMILES: {SMILES}")
        mol.UpdatePropertyCache(strict=False)
        
        try:
            mol_with_hs = Chem.AddHs(mol)
        except Exception:
            mol_with_hs = mol
        
        # 清除电荷以安抚 UFF 力场
        mol_for_opt = Chem.Mol(mol_with_hs)
        for atom in mol_for_opt.GetAtoms():
            atom.SetFormalCharge(0)

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        params.pruneRmsThresh = 0.5
        params.numThreads = 0
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_for_opt, numConfs=to_initialize, params=params))
        
        if not conf_ids:
            AllChem.EmbedMolecule(mol_for_opt, AllChem.ETKDGv2())
            conf_ids = [0]

        # 检查是否有带电荷的原子。如果有，UFF 力场可能会崩溃/报错，因此跳过 UFF。
        has_charge = False
        for atom in mol_for_opt.GetAtoms():
            if atom.GetFormalCharge() != 0:
                has_charge = True
                break
        
        if has_charge:
            print(f"--- 🛠️ _get_fragment: 检测到带电原子，跳过 UFF 预优化。 ---")
        else:
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol_for_opt)
            except Exception as e:
                print(f"--- ⚠️ UFF 优化警告: {e} ---")
        
        mol_with_hs.RemoveAllConformers()
        for i, cid in enumerate(conf_ids):
            conf_src = mol_for_opt.GetConformer(cid)
            new_conf = Chem.Conformer(conf_src)
            mol_with_hs.AddConformer(new_conf, assignId=True)

        reordered_conformers = []
        all_rdkit_atoms = list(mol_with_hs.GetAtoms())

        for conf_id in conf_ids:
            conf = mol_with_hs.GetConformer(conf_id)
            positions = conf.GetPositions()
            
            # 1. 查找所有映射的原子
            map_num_to_idx = {}
            for atom in all_rdkit_atoms:
                map_num = atom.GetAtomMapNum()
                idx = atom.GetIdx()
                if map_num > 0:
                    map_num_to_idx[map_num] = idx
            
            # 2. 根据 TRICK_SMILES 和 num_binding_indices 构建索引列表
            proxy_indices = []
            binding_indices = []

            if TRICK_SMILES == "Cl":
                # --- end-on @ ontop ---
                if num_binding_indices != 1:
                     raise ValueError(f"代码逻辑错误: TRICK_SMILES='Cl' 但键合索引 != 1")

                if 1 not in map_num_to_idx or 114514 not in map_num_to_idx:
                    raise ValueError(f"SMILES {SMILES} 缺少映射号 1 (Cl) 或 114514 (成键原子)。")
                
                proxy_indices = [map_num_to_idx[1]]
                binding_indices = [map_num_to_idx[114514]]

                # 清理临时映射号
                all_rdkit_atoms[map_num_to_idx[114514]].SetAtomMapNum(0)
                
            elif TRICK_SMILES == "S1S":
                # --- end-on/side-on @ bridge/hollow ---
                if 1 not in map_num_to_idx or 2 not in map_num_to_idx:
                     raise ValueError(f"SMILES {SMILES} 缺少映射号 1 (S1), 2 (S2)。")
                
                proxy_indices = [map_num_to_idx[1], map_num_to_idx[2]]

                if num_binding_indices == 1:
                    # --- end-on @ bridge/hollow ---
                    if 114514 not in map_num_to_idx:
                         raise ValueError(f"SMILES {SMILES} 缺少映射号 114514 (成键原子1)。")

                    binding_indices = [map_num_to_idx[114514]]

                    # 手动对齐 S-S 向量，使其 *垂直* 于 Z 轴（模拟 end-on）
                    s1_idx, s2_idx = proxy_indices[0], proxy_indices[1]
                    t1_idx = binding_indices[0]

                    p1 = positions[t1_idx]

                    # --- 防止 autoadsorbate 除以零或生成零向量 ---
                    # 1. 垂直向量 (S1-S2)
                    v_perp = np.array([0.0, 0.5, 0.0])
                    # 2. 倾斜的中点，使 nvector (p1-midpoint) 既非零也不平行于 Z 轴
                    midpoint = p1 - np.array([0.1, 0.0, 1.0])

                    # 放置 S1 和 S2
                    positions[s1_idx] = midpoint + v_perp
                    positions[s2_idx] = midpoint - v_perp

                    print(f"--- 🛠️ _get_fragment: 已手动对齐 S-S 标记用于 End-on 模式 (倾斜修正)。 ---")
                    all_rdkit_atoms[t1_idx].SetAtomMapNum(0)
                elif num_binding_indices == 2:
                    # --- side-on @ bridge/hollow ---
                    if 114514 not in map_num_to_idx or 1919810 not in map_num_to_idx:
                         raise ValueError(f"SMILES {SMILES} 缺少映射号 114514 (成键原子1) 或 1919810 (成键原子2)。")

                    binding_indices = [map_num_to_idx[114514], map_num_to_idx[1919810]]

                    # 手动对齐 S-S 向量，使其垂直于成键原子之间的键
                    s1_idx, s2_idx = proxy_indices[0], proxy_indices[1]
                    t1_idx, t2_idx = binding_indices[0], binding_indices[1]

                    # 1. 获取目标原子的位置
                    p1 = positions[t1_idx]
                    p2 = positions[t2_idx]
                        
                    # 2. 计算它们的中点和键向量
                    midpoint = (p1 + p2) / 2.0
                    v_bond = p1 - p2
                        
                    # 3. 计算一个垂直于键向量的向量 (即我们的 S-S 向量)
                    v_temp = np.array([1.0, 0.0, 0.0]) # 任意的非平行向量
                    v_perp = np.cross(v_bond, v_temp)

                    # 处理 v_bond 与 v_temp 共线的情况
                    if np.linalg.norm(v_perp) < 1e-3:
                        v_temp = np.array([0.0, 1.0, 0.0])
                        v_perp = np.cross(v_bond, v_temp)
                        
                    v_perp_norm = v_perp / np.linalg.norm(v_perp)
                        
                    # 4. 手动移动 RDKit 坐标数组中的 S 原子
                    # (距离 0.5 是任意的，autoadsorbate 只关心方向)
                    positions[s1_idx] = midpoint + v_perp_norm * 0.5
                    positions[s2_idx] = midpoint - v_perp_norm * 0.5
                        
                    print(f"--- 🛠️ _get_fragment: 已手动对齐 S-S 向量，使其垂直于 {t1_idx}-{t2_idx} 键。 ---")
                        
                    # 5. 清理临时映射号
                    all_rdkit_atoms[t1_idx].SetAtomMapNum(0)
                    all_rdkit_atoms[t2_idx].SetAtomMapNum(0)

            # 3. 构建新的、*有保证*的原子顺序

            # 收集所有*既不是*代理原子*也不是*成键原子的原子
            special_indices_set = set(proxy_indices + binding_indices)
            other_indices = [atom.GetIdx() for atom in all_rdkit_atoms if atom.GetIdx() not in special_indices_set and atom.GetAtomMapNum() == 0]

            # 强制执行 autoadsorbate 期望的顺序
            new_order = proxy_indices + binding_indices + other_indices
            
            # 4. 根据新顺序提取符号和位置
            new_symbols = [all_rdkit_atoms[i].GetSymbol() for i in new_order]
            new_positions = [positions[i] for i in new_order]
            
            # 5. 创建 ASE Atoms 对象，并设置关键的 .info["smiles"]
            new_atoms = Atoms(symbols=new_symbols, positions=new_positions)
            # 这是 autoadsorbate 库唯一关心的东西：
            new_atoms.info = {"smiles": TRICK_SMILES} 
            reordered_conformers.append(new_atoms)

        if not reordered_conformers:
            raise ValueError(f"RDKit 构象生成成功，但原子映射追踪失败 (SMILES: {SMILES})")

        # 1. 创建一个 *虚拟的* Fragment 对象，使用一个已知有效的SMILES (例如 "C") 来安全地完成 __init__。
        print(f"--- 🛠️ _get_fragment: 正在安全创建空 Fragment 对象 ... ---")
        fragment = Fragment.__new__(Fragment)
        
        # 2. 手动 *覆盖* 库生成的虚拟构象
        print(f"--- 🛠️ _get_fragment: 正在用 {len(reordered_conformers)} 个已重排的构象覆盖 .conformers ... ---")
        fragment.conformers = reordered_conformers
        fragment.conformers_aligned = [False] * len(reordered_conformers)
        
        # 3. 手动 *覆盖* smile 属性，以便 autoadsorbate.Surface 知道要剥离哪个代理（"Cl" 或 "S1S"）
        print(f"--- 🛠️ _get_fragment: 正在覆盖 .smile 为 '{TRICK_SMILES}' ... ---")
        fragment.smile = TRICK_SMILES
        fragment.to_initialize = to_initialize

        print(f"--- 🛠️ _get_fragment: 成功从 *SMILES '{SMILES}' (to_initialize={to_initialize}) 创建了片段对象。 ---")
        return fragment

    except Exception as e:
        print(f"--- 🛠️ _get_fragment: 错误: 无法从 SMILES '{SMILES}' 创建 Fragment: {e} ---")
        raise e

def create_fragment_from_plan(
    original_smiles: str, 
    binding_atom_indices: list[int], 
    plan_dict: dict,
    to_initialize: int = 1
) -> Fragment:
    print(f"--- 🛠️ 正在执行 create_fragment_from_plan ... ---")

    # 从规划字典中提取所需信息
    plan_solution = plan_dict.get("solution", {})
    adsorbate_type = plan_dict.get("adsorbate_type")
    site_type = plan_solution.get("site_type")
    num_binding_indices = len(binding_atom_indices)

    if not site_type or not adsorbate_type:
        raise ValueError("plan_dict 缺少 'site_type' 或 'adsorbate_type'。")
    
    # 1. 内部调用 SMILES 生成器
    surrogate_smiles = generate_surrogate_smiles(
        original_smiles=original_smiles,
        binding_atom_indices=binding_atom_indices,
        site_type=site_type
    )

    # 2. 内部调用构象生成器 (包含所有补丁和技巧)
    fragment = _get_fragment(
        SMILES=surrogate_smiles,
        site_type=site_type,
        num_binding_indices=num_binding_indices,
        to_initialize=to_initialize
    )
    
    # 确保 fragment 对象有一个 .info 字典
    if not hasattr(fragment, "info"):
        print("--- 🛠️ 原生 Fragment 对象缺少 .info 字典，正在添加... ---")
        fragment.info = {}

    # 3. 关键：将原始规划信息附加到 Fragment 对象上
    fragment.info["plan_site_type"] = site_type
    fragment.info["plan_original_smiles"] = original_smiles
    fragment.info["plan_binding_atom_indices"] = binding_atom_indices
    fragment.info["plan_adsorbate_type"] = adsorbate_type
    
    print(f"--- 🛠️ create_fragment_from_plan: 成功创建并标记了 Fragment 对象。 ---")
    return fragment

def _bump_adsorbate_to_safe_distance(slab_atoms: ase.Atoms, full_atoms: ase.Atoms, min_dist_threshold: float = 1.5) -> ase.Atoms:
    """
    检查吸附物是否与表面发生碰撞。如果有，沿 Z 轴向上推，直到没有碰撞。
    """
    # 1. 区分表面和吸附物
    n_slab = len(slab_atoms)
    adsorbate_indices = list(range(n_slab, len(full_atoms)))
    
    if not adsorbate_indices:
        return full_atoms

    # 2. 提取位置
    slab_pos = full_atoms.positions[:n_slab]
    ads_pos = full_atoms.positions[n_slab:]
    
    # 3. 计算距离矩阵 (Adsorbate vs Slab)
    # 注意：对于非常大的体系，可以使用 NeighborList，但这里直接计算 cdist 够快且稳健
    dists = cdist(ads_pos, slab_pos)
    min_d = np.min(dists)
    
    # 4. 如果太近，计算需要抬升多少
    if min_d < min_dist_threshold:
        # 我们希望 min_d 至少是 min_dist_threshold
        # 简单的策略：逐步抬升，或者直接一次性抬升 (threshold - min_d) + buffer
        # 考虑到几何形状复杂，直接加 Z 是最安全的
        bump_height = (min_dist_threshold - min_d) + 0.2 # 额外加 0.2 A 缓冲
        
        print(f"--- 🛡️ 碰撞检测: 发现原子重叠 (min_dist={min_d:.2f} Å < {min_dist_threshold} Å)。正在抬升 {bump_height:.2f} Å... ---")
        
        # 修改吸附物坐标
        full_atoms.positions[adsorbate_indices, 2] += bump_height
    
    return full_atoms

def populate_surface_with_fragment(
    slab_atoms: ase.Atoms, 
    fragment_object: Fragment,
    plan_solution: dict,
    **kwargs
) -> str:
    # --- 1. 从 Fragment 对象中检索规划 ---
    if not hasattr(fragment_object, "info") or "plan_site_type" not in fragment_object.info:
        raise ValueError("Fragment 对象缺少 'plan_site_type' 信息。")

    # --- 从规划中读取参数 (或使用默认值) ---
    site_type = plan_solution.get("site_type", "all")
    conformers_per_site_cap = plan_solution.get("conformers_per_site_cap", 2)
    overlap_thr = plan_solution.get("overlap_thr", 0.1)
    touch_sphere_size = plan_solution.get("touch_sphere_size", 3)

    print(f"--- 🛠️ 正在初始化表面 (touch_sphere_size={touch_sphere_size})... ---")
    
    # 为了安全起见，这里再次清理元数据，确保 autoadsorbate 接收到纯净的 Atoms 对象
    symbols = slab_atoms.get_chemical_symbols()
    positions = slab_atoms.get_positions()
    cell = slab_atoms.get_cell()
    pbc = slab_atoms.get_pbc()
    clean_slab_atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    # 明确设置 mode='slab'
    s = Surface(
        clean_slab_atoms,
        precision=1.0, 
        touch_sphere_size=touch_sphere_size,
        mode='slab'  # 明确设置模式，防止默认为 'dummy'
    )

    original_site_count = len(s.site_df)
    s.sym_reduce()
    print(f"--- 🛠️ 表面位点：从 {original_site_count} 个减少到 {len(s.site_df)} 个不等价位点。 ---")

    # 检查是否找到了位点
    # 这可以防止在 `s.site_df.connectivity` 上失败
    if s.site_df.empty or len(s.site_df) == 0:
        raise ValueError(
            f"Autoadsorbate 未能在表面上找到任何吸附位点 (0 sites found)。"
            f"这可能是由于 `touch_sphere_size` ({touch_sphere_size}) 不合适（太大或太小）。"
        )

    # --- 2. 验证规划与位点的兼容性 (Connectivity 过滤) ---
    site_df_filtered = s.site_df
    if site_type == "ontop":
        site_df_filtered = s.site_df[s.site_df.connectivity == 1]
    elif site_type == "bridge":
        site_df_filtered = s.site_df[s.site_df.connectivity == 2]
    elif site_type == "hollow":
        site_df_filtered = s.site_df[s.site_df.connectivity >= 3]
    elif site_type == "all":
        site_df_filtered = s.site_df
    else:
        raise ValueError(f"未知的 site_type: '{site_type}'。")

    # --- 3. 可选的表面原子过滤 ---
    allowed_symbols = plan_solution.get("surface_binding_atoms")
    if allowed_symbols and len(allowed_symbols) > 0:
        # 使用排序后的字符串做日志，清晰明了
        print(f"--- 🛠️ 正在按表面符号过滤 (严格匹配): {sorted(allowed_symbols)} ---")
        
        # 预先计算目标的原子计数 (例如: {'Mo': 2, 'Pd': 1})
        target_counts = Counter(allowed_symbols)
        
        def check_symbols(site_formula_dict):
            if not site_formula_dict or not isinstance(site_formula_dict, dict):
                return False
            
            # 严格匹配逻辑：
            # 将 site_formula_dict (例如 {'Mo': 2, 'Pd': 1}) 展开并计数，必须与目标完全一致
            # 防止请求 ['Mo', 'Mo'] (纯桥位) 却返回 {'Mo': 2, 'Pd': 1} (混合空位) 的情况
            
            # 1. 展开位点成分 (dict -> list)
            site_atoms_list = []
            for sym, count in site_formula_dict.items():
                site_atoms_list.extend([sym] * count)
            
            # 2. 比较计数器
            return Counter(site_atoms_list) == target_counts

        initial_count = len(site_df_filtered)
        # 应用严格过滤器
        site_df_filtered = site_df_filtered[
            site_df_filtered['site_formula'].apply(check_symbols)
        ]
        print(f"--- 🛠️ 表面符号过滤：位点从 {initial_count} 个减少到 {len(site_df_filtered)} 个。 ---")

    # 将 s.site_df 替换为过滤后的 df
    s.site_df = site_df_filtered
    site_index_arg = list(s.site_df.index)
    
    print(f"--- 🛠️ 规划已验证：正在搜索 {len(site_index_arg)} 个 '{site_type}' (过滤后) 位点。 ---")

    if len(site_index_arg) == 0:
        raise ValueError(f"未找到 '{site_type}' 类型且包含 {allowed_symbols} 的位点。无法继续。")

    # --- 4. 决定 sample_rotation ---
    sample_rotation = True
    num_binding_indices = len(fragment_object.info["plan_binding_atom_indices"])
    if num_binding_indices == 2:
        print("--- 🛠️ 检测到 2-index (side-on) 模式。禁用 sample_rotation。---")
        sample_rotation = False

    # --- 5. 调用库 ---
    print(f"--- 🛠️ 正在调用 s.get_populated_sites (cap={conformers_per_site_cap}, overlap={overlap_thr})... ---")
    
    raw_out_trj = s.get_populated_sites(
      fragment=fragment_object,
      site_index=site_index_arg,
      sample_rotation=sample_rotation,
      mode='all',
      conformers_per_site_cap=conformers_per_site_cap,
      overlap_thr=overlap_thr,
      verbose=True
    )
    
    # 对生成的构型进行碰撞检测和抬升 (阈值 1.8 Å)
    safe_out_trj = []
    for idx, atoms in enumerate(raw_out_trj):
        safe_atoms = _bump_adsorbate_to_safe_distance(slab_atoms, atoms, min_dist_threshold=1.6)
        safe_out_trj.append(safe_atoms)
    
    out_trj = safe_out_trj

    print(f"--- 🛠️ 成功生成了 {len(out_trj)} 个初始构型。 ---")
    
    if not out_trj:
        raise ValueError(f"get_populated_sites 未能生成任何构型。可能是因为 overlap_thr ({overlap_thr}) 太严格。")
    
    # 将 ase.Atoms 列表保存到 Trajectory 对象中
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    traj_file = f"outputs/generated_conformers_{fragment_object.info['plan_original_smiles'].replace('=','_').replace('#','_')}.traj"
    traj = Trajectory(traj_file, 'w')
    for atoms in out_trj:
        traj.write(atoms)
    traj.close()

    print(f"--- 🛠️ 构型已保存到 {traj_file} ---")
    return traj_file

def relax_atoms(
    atoms_list: list, 
    slab_indices: list, 
    relax_top_n: int = 1,
    fmax: float = 0.05, 
    steps: int = 500,
    md_steps: int = 20,
    md_temp: float = 150.0,
    mace_model: str = "small",
    mace_device: str = "cpu"
) -> str:
    print(f"--- 🛠️ 正在初始化 MACE 计算器 (Model: {mace_model}, Device: {mace_device})... ---")
    try:
        calculator = mace_mp(model=mace_model, device=mace_device, default_dtype='float32', dispersion=True)
    except Exception as e:
        print(f"--- 🛑 MACE 初始化失败: {e} ---")
        raise

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    # 优化：我们只弛豫最好的 N 个构型
    N_RELAX_TOP_N = relax_top_n

    # 约束
    constraint = FixAtoms(indices=slab_indices)

    def _get_bond_change_count(initial, final):
        if len(initial) != len(final):
            return 0
        radii = np.array(natural_cutoffs(initial, mult=1.05))
        cutoff_mat = radii[:, None] + radii[None, :]
        d_initial = initial.get_all_distances()
        d_final = final.get_all_distances()

        # 忽略 H-H 键
        symbols = initial.get_chemical_symbols()
        is_H = np.array([s == 'H' for s in symbols])
        mask = is_H[:, None] & is_H[None, :]
        np.fill_diagonal(d_initial, 99.0)
        np.fill_diagonal(d_final, 99.0)

        bonds_initial = (d_initial < cutoff_mat) & (~mask)
        # 宽松阈值检测断键 (1.5倍)
        bonds_final_loose = (d_final < cutoff_mat * 1.5) & (~mask)
        bonds_final_strict = (d_final < cutoff_mat) & (~mask)

        broken = bonds_initial & (~bonds_final_loose)
        formed = (~bonds_initial) & bonds_final_strict
        return int(np.sum(np.triu(broken | formed)))
    
    # --- 1. 评估阶段 (预热 + 单点能量) ---
    print(f"--- 🛠️ 评估阶段：正在评估 {len(atoms_list)} 个构型 (MD 预热 + SP 能量)... ---")
    evaluated_configs = []
    for i, atoms in enumerate(atoms_list):
        atoms.calc = calculator
        atoms.set_constraint(constraint)
        
        max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
        if max_force > 500.0:
            print(f"--- ⚠️ 跳过结构 {i+1}: 初始力过大 (Max Force = {max_force:.2f} eV/A)... ---")
            continue

        if md_steps > 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
            dyn_md = Langevin(atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
            dyn_md.run(md_steps)

        energy = atoms.get_potential_energy()
        print(f"--- 评估结构 {i+1}/{len(atoms_list)}... 能量 (预热后): {energy:.4f} eV ---")
        evaluated_configs.append((energy, i, atoms.copy())) # 存储副本

    if not evaluated_configs:
        raise ValueError("评估阶段未能成功评估任何构型。")

    # --- 2. 选择最佳 ---
    evaluated_configs.sort(key=lambda x: x[0]) # 按能量排序
    
    if N_RELAX_TOP_N > len(evaluated_configs):
        print(f"--- 🛠️ 警告: 请求弛豫 top {N_RELAX_TOP_N}，但只有 {len(evaluated_configs)} 个可用。将弛豫所有。 ---")
        N_RELAX_TOP_N = len(evaluated_configs)
    
    configs_to_relax = evaluated_configs[:N_RELAX_TOP_N]
    
    print(f"--- 🛠️ 评估完成。将从 {len(atoms_list)} 个构型中弛豫最好的 {N_RELAX_TOP_N} 个。---")
    
    # --- 3. 弛豫阶段 (仅 N_RELAX_TOP_N) ---
    traj_file = f"outputs/relaxation_run.traj"
    traj = Trajectory(traj_file, 'w')
    final_structures = []

    for i, (initial_energy, original_index, atoms) in enumerate(configs_to_relax):
        print(f"--- 弛豫最佳结构 {i+1}/{N_RELAX_TOP_N} (原始 Index {original_index}, 初始能量: {initial_energy:.4f} eV) ---")
        
        atoms.calc = calculator
        atoms.set_constraint(constraint)

        # --- 捕获弛豫前的吸附物 ---
        adsorbate_indices = list(range(len(slab_indices), len(atoms)))
        initial_adsorbate = atoms.copy()[adsorbate_indices]
        
        print(f"--- 优化 (BFGS): fmax={fmax}, steps={steps} ---")
        dyn_opt = BFGS(atoms, trajectory=None, logfile=None) 
        dyn_opt.attach(lambda: traj.write(atoms), interval=1)
        dyn_opt.run(fmax=fmax, steps=steps)

        # --- 捕获弛豫后的吸附物状态并检查键变化 ---
        final_adsorbate = atoms.copy()[adsorbate_indices]
        bond_change_count = _get_bond_change_count(initial_adsorbate, final_adsorbate)
        atoms.info["bond_change_count"] = bond_change_count
        print(f"--- 键完整性检查: 检测到 {bond_change_count} 个键发生了变化。 ---")
        
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        print(f"--- 最佳结构 {i+1} 弛豫完成。最终能量: {final_energy:.4f} eV ---")

        atoms.results = {
            'energy': final_energy,
            'forces': final_forces
        }
        
        final_structures.append(atoms)

    traj.close()
    
    final_traj_file = f"outputs/final_relaxed_structures.xyz"
 
    try:
        write(final_traj_file, final_structures)
    except Exception as e:
        print(f"--- 🛑 写入 final_relaxed_structures.xyz 失败: {e} ---")
        raise
    
    print(f"--- 🛠️ 弛豫完成。完整轨迹: {traj_file} | 最终结构 ({len(final_structures)}): {final_traj_file} ---")
    return final_traj_file

def save_ase_atoms(atoms: ase.Atoms, filename: str) -> str:
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    if not filename.startswith("outputs/"):
        filename = f"outputs/{filename}"
    
    try:
        write(filename, atoms)
        print(f"--- 🛠️ 成功将结构保存到 {filename} ---")
        return f"已保存到 {filename}"
    except Exception as e:
        print(f"--- 🛠️ 错误: 无法保存 Atoms 到 {filename}: {e} ---")
        raise

def analyze_relaxation_results(
    relaxed_trajectory_file: str, 
    slab_atoms: ase.Atoms,
    original_smiles: str,
    plan_dict: dict,
    e_surface_ref: float = 0.0,
    e_adsorbate_ref: float = 0.0
) -> str:
    try:
        print(f"--- 🛠️ 正在分析弛豫结果: {relaxed_trajectory_file} ---")
        traj = read(relaxed_trajectory_file, index=":")
        if len(traj) == 0:
            return json.dumps({"status": "error", "message": "弛豫轨迹为空或无法读取。"})

        # 1. 找到最稳定的构型
        energies = []
        for atoms in traj:
            try:
                e = atoms.get_potential_energy()
                energies.append(e)
            except Exception:
                pass
        
        min_energy_total = min(energies)
        best_index = np.argmin(energies)
        relaxed_atoms = traj[best_index]

        E_ads = min_energy_total - e_surface_ref - e_adsorbate_ref
        print(f"--- Analysis: E_ads = {E_ads:.4f} eV (E_total = {min_energy_total:.4f} eV, E_surf={e_surface_ref:.4f}, E_ads_mol={e_adsorbate_ref:.4f}) ---")
        
        # 1. 提取吸附物原子
        adsorbate_atoms = relaxed_atoms[len(slab_atoms):]

        # 2. 复制并应用 PBC 信息 (关键！防止跨边界原子被误判为断裂)
        # 我们创建一个临时的 Atoms 对象来进行拓扑分析
        check_atoms = adsorbate_atoms.copy()
        check_atoms.set_cell(relaxed_atoms.get_cell())
        check_atoms.set_pbc(relaxed_atoms.get_pbc())

        # 3. 构建邻接矩阵 (考虑 PBC)
        # mult=1.2 给键长一点裕度 (C-H ~1.1A -> cutoff ~1.3A)
        # 如果距离超过这个范围，那就是真的断了
        check_cutoffs = natural_cutoffs(check_atoms, mult=1.2)
        nl = build_neighbor_list(check_atoms, cutoffs=check_cutoffs, self_interaction=False)
        adjacency_matrix = nl.get_connectivity_matrix()

        # 4. 计算连通分量 (数一数分子碎成了几块)
        n_components, labels = connected_components(adjacency_matrix, directed=False)

        # 5. 判定逻辑
        # 正常情况下，单分子吸附应该只有 1 个连通分量
        is_dissociated = n_components > 1

        # 6. 获取键变化计数作为辅助参考
        bond_change_count = relaxed_atoms.info.get("bond_change_count", 0)

        # 如果分子碎成了 n 块 (n > 1)，说明至少断了 (n-1) 个键。
        # 防止出现 "is_dissociated=True" 但 "bond_change_count=0" 的矛盾。
        if is_dissociated and bond_change_count == 0:
            print(f"--- 🛠️ 修正逻辑矛盾: 检测到解离 (n_components={n_components}) 但 bond_change_count=0。强制修正。 ---")
            bond_change_count = max(1, n_components - 1)

        # 7. 综合判定反应性
        if is_dissociated:
             # 只要碎了，就是反应/失败
             reaction_detected = True
        elif bond_change_count > 0:
             # 没碎，但是键变了 -> 这是“内反应/异构化”
             # 我们可以标记为 reaction_detected = True，
             # 但在 Agent 的 route_after_analysis 中，你可以选择是否“宽容”处理这种情况
             reaction_detected = True
        else:
             # 没碎，键也没变 -> 完美的分子吸附
             reaction_detected = False

        # --- 从 plan_dict 检索信息 ---
        plan_solution = plan_dict.get("solution", {})
        adsorbate_type = plan_dict.get("adsorbate_type")
        site_type = plan_solution.get("site_type")
        binding_atom_indices = plan_solution.get("adsorbate_binding_indices", [])
        num_binding_indices = len(binding_atom_indices)

        # 1.1. 从 .info 字典中获取规划的位点信息
        planned_info = relaxed_atoms.info.get("adsorbate_info", {}).get("site", {})
        planned_connectivity = planned_info.get("connectivity")
        planned_site_type = "unknown"
        if planned_connectivity == 1: planned_site_type = "ontop"
        elif planned_connectivity == 2: planned_site_type = "bridge"
        elif planned_connectivity and planned_connectivity >= 3: planned_site_type = "hollow"
        
        # 1.2. 识别表面和吸附物索引
        slab_indices_check = list(range(len(slab_atoms)))
        adsorbate_indices_check = list(range(len(slab_atoms), len(relaxed_atoms)))
        cov_cutoffs_check = natural_cutoffs(relaxed_atoms, mult=1)
        
        actual_bonded_slab_indices = set()
        anchor_atom_indices = []
        
        if num_binding_indices == 1:
            anchor_atom_indices = [adsorbate_indices_check[0]]
        elif num_binding_indices == 2:
            if len(adsorbate_indices_check) >= 2:
                anchor_atom_indices = [adsorbate_indices_check[0], adsorbate_indices_check[1]]
        
        # 1.3. 计算实际成键的表面原子数量
        for anchor_idx in anchor_atom_indices:
            anchor_cutoff = cov_cutoffs_check[anchor_idx]
            for slab_idx in slab_indices_check:
                slab_cutoff = cov_cutoffs_check[slab_idx]
                bonding_cutoff_check = (anchor_cutoff + slab_cutoff) * 1.1
                dist = relaxed_atoms.get_distance(anchor_idx, slab_idx, mic=True) # 确保使用 MIC
                if dist <= bonding_cutoff_check:
                    actual_bonded_slab_indices.add(slab_idx)
        
        actual_connectivity = len(actual_bonded_slab_indices)
        actual_site_type = "unknown"
        if actual_connectivity == 1: actual_site_type = "ontop"
        elif actual_connectivity == 2: actual_site_type = "bridge"
        elif actual_connectivity >= 3: actual_site_type = "hollow"
        else: actual_site_type = "desorbed"

        slab_indices = list(range(len(slab_atoms)))
        adsorbate_indices = list(range(len(slab_atoms), len(relaxed_atoms)))
        
        slab_atoms_relaxed = relaxed_atoms[slab_indices]
        adsorbate_atoms_relaxed = relaxed_atoms[adsorbate_indices]

        # FCC/HCP 晶体学辨识
        # 只有当确认为 hollow 位点时，才进行深层探测
        site_crystallography = ""
        if actual_site_type == "hollow":
            try:
                # 1. 定义表面层和次表面层
                # 假设 slab 在 Z 方向上是对齐的，且 z_max 是最上层
                z_coords = slab_atoms_relaxed.positions[:, 2]
                max_z = np.max(z_coords)
                # 简单的层切分：认为距离顶层 1.5A 到 4.0A 之间的是次表面层 (Subsurface)
                # 适用于大多数金属 (层间距 ~2.0-2.3A)
                subsurface_mask = (z_coords < (max_z - 1.2)) & (z_coords > (max_z - 4.0))
                subsurface_indices_list = np.where(subsurface_mask)[0]

                if len(subsurface_indices_list) > 0:
                    # 2. 获取目标吸附原子的 XY 坐标
                    target_pos_xy = relaxed_atoms[target_atom_global_index].position[:2]
                    
                    # 3. 计算吸附原子与所有次表面原子在 XY 平面上的投影距离
                    subsurface_positions_xy = slab_atoms_relaxed.positions[subsurface_indices_list][:, :2]
                    
                    # 考虑周期性边界条件 (PBC) 计算 XY 距离
                    # 这里为了简化，我们假设原子正好在正下方，直接用欧氏距离通常足够，
                    # 但更严谨的做法是使用 ase.geometry.get_distances 或者手动处理 cell
                    # 这里使用简化的投影距离判定：
                    # 如果次表面原子在 XY 上的距离 < 1.0 Å，说明正下方有原子 -> HCP
                    dists_xy = np.linalg.norm(subsurface_positions_xy - target_pos_xy, axis=1)
                    min_dist_xy = np.min(dists_xy)
                    
                    if min_dist_xy < 1.0:
                        site_crystallography = "(HCP/Subsurf-Atom)"
                    else:
                        site_crystallography = "(FCC/No-Subsurf)"
                else:
                    site_crystallography = "(Unknown Layer)"
            except Exception as e_cryst:
                print(f"--- ⚠️ 晶体学分析警告: {e_cryst} ---")
        
        # 将此后缀添加到 actual_site_type 中，以便 Agent 能看到区别
        if site_crystallography:
            actual_site_type += f" {site_crystallography}"
        
        print(f"--- 分析: 位点滑移检查：规划 {planned_site_type} (conn={planned_connectivity}), 实际 {actual_site_type} (conn={actual_connectivity}) ---")

        # 2. 识别吸附物原子和表面原子
        
        target_atom_global_index = -1
        target_atom_symbol = ""
        analysis_message = ""
        result = {}

        # 准备共价键检查
        cov_cutoffs = natural_cutoffs(relaxed_atoms, mult=1)

        if num_binding_indices == 1:
            # 目标原子 *总是* 吸附物列表中的第一个
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position

            print(f"--- 分析: (1-index 模式) 正在检查第一个吸附物原子, 符号: '{target_atom_symbol}', 全局索引: {target_atom_global_index}。---")

            # --- 寻找所有成键的表面原子，而不仅仅是最近的一个 ---
            bonded_surface_atoms = []
            min_distance = float('inf')
            nearest_slab_atom_symbol = ""
            nearest_slab_atom_global_index = -1
            
            # 遍历所有表面原子计算距离
            for s_idx in slab_indices:
                # 使用 MIC (最小镜像约定) 计算距离，确保周期性边界下距离正确
                d = relaxed_atoms.get_distance(target_atom_global_index, s_idx, mic=True)
                
                # 更新最近原子记录 (作为备用信息)
                if d < min_distance:
                    min_distance = d
                    nearest_slab_atom_global_index = s_idx
                    nearest_slab_atom_symbol = relaxed_atoms[s_idx].symbol
                
                # 检查是否成键
                r_ads = cov_cutoffs[target_atom_global_index]
                r_slab = cov_cutoffs[s_idx]
                bonding_cutoff = (r_ads + r_slab) * 1.1 
                
                if d <= bonding_cutoff:
                    bonded_surface_atoms.append({
                        "symbol": relaxed_atoms[s_idx].symbol,
                        "index": s_idx,
                        "distance": round(d, 3)
                    })
            
            # 按距离排序，让最近的排前面
            bonded_surface_atoms.sort(key=lambda x: x["distance"])
            
            is_bound = len(bonded_surface_atoms) > 0
            
            # 生成成键描述字符串 (例如: "Cu-2.01Å, Ga-2.15Å")
            if is_bound:
                bonded_desc = ", ".join([f"{item['symbol']}-{item['distance']}Å" for item in bonded_surface_atoms])
            else:
                bonded_desc = "无"
            
            # 估算最近原子的 cutoff 用于报告
            nearest_radius_sum = cov_cutoffs[target_atom_global_index] + cov_cutoffs[nearest_slab_atom_global_index]
            estimated_covalent_cutoff_A = nearest_radius_sum * 1.1

            # 化学滑移检测 (Chemical Slip Detection)
            # 1. 获取规划的表面原子符号 (排序以忽略顺序差异)
            planned_symbols = sorted(plan_solution.get("surface_binding_atoms", []))
            
            # 2. 获取实际成键的表面原子符号
            actual_symbols = sorted([atom['symbol'] for atom in bonded_surface_atoms])
            
            # 3. 判定是否发生化学滑移
            # 注意：如果规划是空的(如未指定)则跳过；如果没成键也跳过
            is_chemical_slip = False
            if planned_symbols and bonded_surface_atoms:
                if planned_symbols != actual_symbols:
                    is_chemical_slip = True
                    print(f"--- ⚠️ 警告: 检测到化学位点滑移! 规划: {planned_symbols} -> 实际: {actual_symbols} ---")

            analysis_message = (
                f"最稳定构型吸附能: {E_ads:.4f} eV。"
                f"目标原子: {target_atom_symbol} (来自规划索引 {binding_atom_indices[0]}，在弛豫结构中为全局索引 {target_atom_global_index})。"
                f"  -> 最近: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}), 距离: {round(min_distance, 3)} Å (阈值: {round(bonding_cutoff, 3)}), 成键: {is_bound}。"
                f"成键表面原子: {bonded_desc}。 "
                f"是否成键: {is_bound}。"
                f"是否发生反应性转变: {reaction_detected} (键变化数: {bond_change_count} )。"
                f"化学滑移: {is_chemical_slip} (规划 {planned_symbols} -> 实际 {actual_symbols})。"
            )

            result = {
                "status": "success",
                "message": analysis_message,
                "most_stable_energy_eV": E_ads,
                "target_adsorbate_atom": target_atom_symbol,
                "target_adsorbate_atom_index": int(target_atom_global_index),
                "bonded_surface_atoms": bonded_surface_atoms,
                "nearest_slab_atom": nearest_slab_atom_symbol,
                "nearest_slab_atom_index": int(nearest_slab_atom_global_index),
                "final_bond_distance_A": round(min_distance, 3),
                "estimated_covalent_cutoff_A": round(estimated_covalent_cutoff_A, 3),
                "is_covalently_bound": bool(is_bound),
                "reaction_detected": bool(reaction_detected),
                "is_dissociated": bool(is_dissociated),
                "n_components": int(n_components),
                "bond_change_count": int(bond_change_count),
                "site_analysis": {
                    "planned_site_type": planned_site_type,
                    "planned_connectivity": planned_connectivity,
                    "actual_site_type": actual_site_type,
                    "actual_connectivity": actual_connectivity,
                    "is_chemical_slip": is_chemical_slip,
                    "planned_symbols": planned_symbols,
                    "actual_symbols": actual_symbols
                }
            }
        
        elif num_binding_indices == 2:
            if len(adsorbate_indices) < 2:
                 return json.dumps({"status": "error", "message": f"Side-on 模式需要至少 2 个吸附物原子，但只找到 {len(adsorbate_indices)} 个。"})
            
            # 目标原子 *总是* 吸附物列表中的前两个
            
            # --- 分析第一个原子 (Atom 0) ---
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position
            print(f"--- 分析: (2-index 模式) 正在检查第一个吸附物原子, 符号: '{target_atom_symbol}', 全局索引: {target_atom_global_index}。---")

            distances = np.linalg.norm(slab_atoms_relaxed.positions - target_atom_pos, axis=1)
            min_distance = np.min(distances)
            nearest_slab_atom_global_index = slab_indices[np.argmin(distances)]
            nearest_slab_atom_symbol = relaxed_atoms[nearest_slab_atom_global_index].symbol
            radius_1 = cov_cutoffs[target_atom_global_index]
            radius_2 = cov_cutoffs[nearest_slab_atom_global_index]
            bonding_cutoff = (radius_1 + radius_2) * 1.1
            is_bound_1 = min_distance <= bonding_cutoff

            # --- 分析第二个原子 (Atom 1) ---
            second_atom_global_index = adsorbate_indices[1]
            second_atom_symbol = relaxed_atoms[second_atom_global_index].symbol
            second_atom_pos = relaxed_atoms[second_atom_global_index].position
            print(f"--- 分析: (side-on 模式) 正在检查第二个吸附物原子, 符号: '{second_atom_symbol}', 全局索引: {second_atom_global_index}。---")
            
            distances_2 = np.linalg.norm(slab_atoms_relaxed.positions - second_atom_pos, axis=1)
            min_distance_2 = np.min(distances_2)
            nearest_slab_atom_global_index_2 = slab_indices[np.argmin(distances_2)]
            nearest_slab_atom_symbol_2 = relaxed_atoms[nearest_slab_atom_global_index_2].symbol
            radius_3 = cov_cutoffs[second_atom_global_index]
            radius_4 = cov_cutoffs[nearest_slab_atom_global_index_2]
            bonding_cutoff_2 = (radius_3 + radius_4) * 1.1
            is_bound_2 = min_distance_2 <= bonding_cutoff_2

            # --- 组合结果 ---
            # 只有两个原子都成键时，才算成功
            is_bound = bool(is_bound_1 and is_bound_2) 
            
            # 生成统一的 bonded_surface_atoms 和 final_bond_distance_A ===
            bonded_surface_atoms = []

            # 定义辅助函数：查找某个吸附原子的所有成键对象
            def find_bonds(ads_idx, ads_symbol):
                bonds = []
                r_ads = cov_cutoffs[ads_idx]
                for s_idx in slab_indices:
                    # 使用 MIC (最小镜像约定) 计算距离
                    d = relaxed_atoms.get_distance(ads_idx, s_idx, mic=True)
                    r_slab = cov_cutoffs[s_idx]
                    # 判定成键
                    if d <= (r_ads + r_slab) * 1.1:
                        bonds.append({
                            "adsorbate_atom": f"{ads_symbol}({ads_idx})",
                            "adsorbate_atom_index": int(ads_idx),
                            "symbol": relaxed_atoms[s_idx].symbol,
                            "index": int(s_idx),
                            "distance": round(d, 3)
                        })
                return bonds

            # 收集两个原子的成键信息
            bonded_surface_atoms.extend(find_bonds(target_atom_global_index, target_atom_symbol))
            bonded_surface_atoms.extend(find_bonds(second_atom_global_index, second_atom_symbol))
            
            # 按距离排序
            bonded_surface_atoms.sort(key=lambda x: x["distance"])

            # 计算最终的最短键长 (用于报告)
            if bonded_surface_atoms:
                final_bond_distance_A = bonded_surface_atoms[0]["distance"]
            else:
                final_bond_distance_A = min(min_distance, min_distance_2)
            
            # 生成描述字符串
            if bonded_surface_atoms:
                bonded_desc = ", ".join([f"{b['adsorbate_atom']}-{b['symbol']}({b['distance']}Å)" for b in bonded_surface_atoms])
            else:
                bonded_desc = "无"

            # 化学滑移检测 (Chemical Slip Detection)
            # 1. 获取规划的表面原子符号 (排序以忽略顺序差异)
            planned_symbols = sorted(plan_solution.get("surface_binding_atoms", []))
            
            # 2. 获取实际成键的表面原子符号
            actual_symbols = sorted([atom['symbol'] for atom in bonded_surface_atoms])
            
            # 3. 判定是否发生化学滑移
            # 注意：如果规划是空的(如未指定)则跳过；如果没成键也跳过
            is_chemical_slip = False
            if planned_symbols and bonded_surface_atoms:
                if planned_symbols != actual_symbols:
                    is_chemical_slip = True
                    print(f"--- ⚠️ 警告: 检测到化学位点滑移! 规划: {planned_symbols} -> 实际: {actual_symbols} ---")
            # === 🩹 修复结束 ===

            analysis_message = (
                f"最稳定构型吸附能: {E_ads:.4f} eV。"
                f"目标原子 1: {target_atom_symbol} (来自规划索引 {binding_atom_indices[0]}，全局索引 {target_atom_global_index})。"
                f"  -> 最近: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}), 距离: {round(min_distance, 3)} Å (阈值: {round(bonding_cutoff, 3)}), 成键: {is_bound_1}。"
                f"目标原子 2: {second_atom_symbol} (来自规划索引 {binding_atom_indices[1]}，全局索引 {second_atom_global_index})。"
                f"  -> 最近: {nearest_slab_atom_symbol_2} (Index {nearest_slab_atom_global_index_2}), 距离: {round(min_distance_2, 3)} Å (阈值: {round(bonding_cutoff_2, 3)}), 成键: {is_bound_2}。"
                f"成键表面原子: {bonded_desc}。 "
                f"是否成键: {is_bound}。"
                f"是否发生反应性转变: {reaction_detected} (键变化数: {bond_change_count} )。"
                f"化学滑移: {is_chemical_slip} (规划 {planned_symbols} -> 实际 {actual_symbols})。"
            )

            result = {
                "status": "success",
                "message": analysis_message,
                "most_stable_energy_eV": E_ads,
                "bonded_surface_atoms": bonded_surface_atoms,
                "final_bond_distance_A": round(final_bond_distance_A, 3),
                "is_covalently_bound": is_bound,
                "atom_1": {
                    "symbol": target_atom_symbol,
                    "global_index": int(target_atom_global_index),
                    "distance_A": round(min_distance, 3),
                    "is_bound": bool(is_bound_1)
                },
                "atom_2": {
                    "symbol": second_atom_symbol,
                    "global_index": int(second_atom_global_index),
                    "distance_A": round(min_distance_2, 3),
                    "is_bound": bool(is_bound_2)
                },
                "reaction_detected": bool(reaction_detected),
                "bond_change_count": int(bond_change_count),
                "is_dissociated": bool(is_dissociated),
                "n_components": int(n_components),
                "site_analysis": {
                    "planned_site_type": planned_site_type,
                    "planned_connectivity": planned_connectivity,
                    "actual_site_type": actual_site_type,
                    "actual_connectivity": actual_connectivity,
                    "is_chemical_slip": is_chemical_slip,
                    "planned_symbols": planned_symbols,
                    "actual_symbols": actual_symbols
                }
            }

        else:
             return json.dumps({"status": "error", "message": f"分析失败：不支持的键合索引数量 {num_binding_indices}。"})

        # 6. 保存最终结构
        # 防止文件名冲突导致覆盖历史最优解。
        # 在文件名中加入：位点类型、表面原子组成、能量。
        
        # 命名逻辑
        site_label = actual_site_type if actual_site_type != "unknown" else planned_site_type
        if planned_site_type != "unknown" and site_label != planned_site_type:
            site_label = f"{planned_site_type}_to_{site_label}"
            
        if is_dissociated: site_label += "_DISS"
        elif bond_change_count > 0: site_label += "_ISO"
        
        clean_smiles = original_smiles.replace('=', '_').replace('#', '_').replace('[', '').replace(']', '')
        best_atoms_filename = f"outputs/BEST_{clean_smiles}_{site_label}_E{E_ads:.3f}.xyz"
        
        try:
            write(best_atoms_filename, relaxed_atoms)
            print(f"--- 🛠️ 成功将最佳结构保存到 {best_atoms_filename} ---")
            # 将具体的文件名返回给 Agent，方便它在报告中引用
            result["best_structure_file"] = best_atoms_filename
        except Exception as e:
            print(f"--- 🛠️ 错误: 无法保存最佳结构到 {best_atoms_filename}: {e} ---")

        return json.dumps(result)
    
    except Exception as e:
        import traceback
        print(f"--- 🛠️ 错误: 分析弛豫时发生意外异常: {e} ---")
        print(traceback.format_exc())
        return json.dumps({"status": "error", "message": f"分析弛豫时发生意外异常: {e}"})
