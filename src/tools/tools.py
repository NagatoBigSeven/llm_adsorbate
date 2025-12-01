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
    touch_sphere_size=2,
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
    touch_sphere_size: float = 2,
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

# Apply Patch: Replace original function in library with our fixed version
print("--- 🩹 Applying Autoadsorbate Monkey Patch ... ---")

# 1. Patch 源头 (Surf.py) - 以防万一有其他地方用它
autoadsorbate.Surf.get_shrinkwrap_grid = get_shrinkwrap_grid_fixed
autoadsorbate.Surf.get_shrinkwrap_ads_sites = get_shrinkwrap_ads_sites_fixed

# 2. 关键修复：Patch 消费者 (autoadsorbate.py)
# 必须覆盖 autoadsorbate.autoadsorbate 命名空间里已经导入的旧函数引用
import autoadsorbate.autoadsorbate 
autoadsorbate.autoadsorbate.get_shrinkwrap_ads_sites = get_shrinkwrap_ads_sites_fixed

print("--- ✅ Patch applied. Surf module and Surface class references safely replaced. ---")

from collections import Counter
import ase
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from ase.optimize import BFGS
from autoadsorbate import Surface, Fragment
from mace.calculators import mace_mp
import os
import platform
import json
from scipy.sparse.csgraph import connected_components
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union, Tuple

def get_atom_index_menu(original_smiles: str) -> str:
    print(f"--- 🛠️ Generating heavy atom index list for {original_smiles} ---")
    try:
        mol = Chem.MolFromSmiles(original_smiles)
        if not mol:
            raise ValueError(f"RDKit cannot parse SMILES: {original_smiles}")
        atom_list = []
        for atom in mol.GetAtoms():
            atom_info = {
                "index": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "hybridization": str(atom.GetHybridization()), 
                "degree": atom.GetDegree(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
                "formal_charge": atom.GetFormalCharge()
            }
            atom_list.append(atom_info)
            
        heavy_atom_menu = [atom for atom in atom_list if atom["symbol"] != 'H']
        print(f"--- 🛠️ Heavy atom index list generated: {json.dumps(heavy_atom_menu)} ---")
        return json.dumps(heavy_atom_menu, indent=2)
    except Exception as e:
        print(f"--- 🛑 get_atom_index_menu failed: {e} ---")
        return json.dumps({"error": f"Unable to generate heavy atom index list: {e}"})

def generate_surrogate_smiles(original_smiles: str, binding_atom_indices: list[int], site_type: str) -> str:
    print(f"--- 🔬 Calling SMILES Translator: {original_smiles} via indices {binding_atom_indices} (Site: {site_type}) ---")
    
    mol = Chem.MolFromSmiles(original_smiles)
    if not mol:
        raise ValueError(f"RDKit cannot parse original SMILES: {original_smiles}")
    
    num_binding_indices = len(binding_atom_indices)
    
    # --- 情况 A: end-on @ ontop (单点吸附) ---
    if site_type == "ontop":
        if num_binding_indices != 1:
            raise ValueError(f"'ontop' site requires 1 binding index, but got {num_binding_indices}.")
            
        target_idx = binding_atom_indices[0]
        
        if target_idx >= mol.GetNumAtoms():
             raise ValueError(f"Index {target_idx} out of range (Atom count: {mol.GetNumAtoms()}).")

        # 1. 捕获原始状态 (防止 RDKit 自动推导)
        target_atom_original = mol.GetAtomWithIdx(target_idx)
        original_h_count = target_atom_original.GetTotalNumHs()
        num_radicals = target_atom_original.GetNumRadicalElectrons()

        new_mol = Chem.RWMol(mol)

        # 2. 添加 Cl 标记
        marker_atom = Chem.Atom("Cl")
        marker_atom.SetAtomMapNum(1) 
        marker_atom.SetIsotope(37)
        marker_idx = new_mol.AddAtom(marker_atom)
        
        # 3. Determine bond type based on electronic state
        if num_radicals > 0:
            print(f"--- 🔬 Smart Bonding: Radical detected (N={num_radicals}) -> Using Covalent Single Bond (SINGLE) ---")
            # 策略：自由基形成共价键，物理意义明确，几何稳定
            new_mol.AddBond(marker_idx, target_idx, Chem.rdchem.BondType.SINGLE)
            
            # 修正：消除自由基标记，使其成为饱和原子
            target_atom_obj = new_mol.GetAtomWithIdx(target_idx)
            target_atom_obj.SetNumRadicalElectrons(0)
            
        else:
            print(f"--- 🔬 Smart Bonding: Lone pair detected (Saturated/Double Bond) -> Using Dative Bond (DATIVE: Target->Surf) ---")
            # 策略：使用配位键连接。
            # 关键点1：方向必须是 目标原子 -> 标记原子 (Target Donates to Marker)
            # 关键点2：不增加电荷，不改变价态。RDKit 不计算 Dative 键的价态贡献，因此 C=O 不会报错。
            new_mol.AddBond(target_idx, marker_idx, Chem.rdchem.BondType.DATIVE)
            
            target_atom_obj = new_mol.GetAtomWithIdx(target_idx)

        # 4. [安全锁] 绝对锁定氢原子
        # 无论哪种情况，都严禁 RDKit 自动添加或删除氢原子
        target_atom_obj.SetNumExplicitHs(original_h_count)
        target_atom_obj.SetNoImplicit(True)

        # 5. 标记追踪
        target_atom_obj.SetAtomMapNum(114514)
        if target_atom_obj.GetSymbol() != 'H':
            target_atom_obj.SetIsotope(14) 

        # 6. 强制刷新
        try:
            # Catch errors just in case, but DATIVE + Neutral usually passes
            Chem.SanitizeMol(new_mol)
        except Exception as e:
            print(f"--- ⚠️ Sanitize Warning: {e} ---")

        out_smiles = Chem.MolToSmiles(new_mol.GetMol(), canonical=False, rootedAtAtom=marker_idx)
        print(f"--- 🔬 SMILES Translator Final Output: {out_smiles} ---")
        return out_smiles

    # --- 情况 B & C: bridge/hollow (保持原样) ---
    elif site_type in ["bridge", "hollow"]:
        if num_binding_indices == 1:
            target_idx = binding_atom_indices[0]
            if target_idx >= mol.GetNumAtoms(): raise ValueError(f"Index {target_idx} out of range.")
            rw_mol = Chem.RWMol(mol)
            rw_mol.GetAtomWithIdx(target_idx).SetAtomMapNum(114514)
            original_smiles_mapped = Chem.MolToSmiles(rw_mol.GetMol(), canonical=False)
            out_smiles = f"{original_smiles_mapped}.[S:1].[S:2]"
            print(f"--- 🔬 SMILES Translator Output: {out_smiles} ---")
            return out_smiles

        elif num_binding_indices == 2:
            target_indices = sorted(binding_atom_indices)
            idx1, idx2 = target_indices[0], target_indices[1]
            if idx2 >= mol.GetNumAtoms(): raise ValueError(f"Index {idx2} out of range.")
            rw_mol = Chem.RWMol(mol)
            rw_mol.GetAtomWithIdx(idx1).SetAtomMapNum(114514)
            rw_mol.GetAtomWithIdx(idx2).SetAtomMapNum(1919810)
            original_smiles_mapped = Chem.MolToSmiles(rw_mol.GetMol(), canonical=False)
            out_smiles = f"{original_smiles_mapped}.[S:1].[S:2]"
            print(f"--- 🔬 SMILES Translator Output: {out_smiles} ---")
            return out_smiles
        else:
            raise ValueError(f"'{site_type}' site does not support {num_binding_indices} binding indices.")
    else:
        raise ValueError(f"Unknown site_type: {site_type}.")

def read_atoms_object(slab_path: str) -> ase.Atoms:
    try:
        atoms = read(slab_path)  # Read slab structure from .xyz or .cif file.
        print(f"Success: Read slab atoms from {slab_path}.")
        return atoms
    except Exception as e:
        print(f"Error: Unable to read {slab_path}: {e}")
        raise

# --- 统一处理表面的扩胞和清理 ---
def prepare_slab(slab_atoms: ase.Atoms) -> Tuple[ase.Atoms, bool]:
    """
    Clean Slab metadata and expand supercell if needed for physical accuracy.
    Returns: (Processed Slab, Is Expanded)
    """
    print("--- 🛠️ [Prepare] Cleaning Slab metadata and checking dimensions... ---")
    
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
        print(f"--- 🛠️ [Prepare] Small cell detected (a={a_len:.2f}Å, b={b_len:.2f}Å). Expanding to 2x2x1... ---")
        clean_slab = clean_slab * (2, 2, 1)
        is_expanded = True
    else:
        print(f"--- 🛠️ [Prepare] Cell size sufficient (a={a_len:.2f}Å, b={b_len:.2f}Å). Keeping as is. ---")
        
    return clean_slab, is_expanded

def analyze_surface_sites(slab_path: str) -> dict:
    """ 预扫描表面，找出实际存在的位点类型，供 Planner 参考 """
    from collections import defaultdict, Counter
    atoms = read_atoms_object(slab_path)
    clean_slab, _ = prepare_slab(atoms)
    
    # 空跑 Autoadsorbate
    s = Surface(clean_slab, precision=1.0, touch_sphere_size=2.0, mode='slab')
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
    
    # 修复 FCC(100) 等正方形晶格上的虚构 3-fold 位点
    # 逻辑：如果一个表面同时拥有 4-fold (connectivity=4) 和 3-fold (connectivity=3)，
    # 且没有极其复杂的低对称性特征，通常 3-fold 是三角剖分的伪影。
    if 4 in site_inventory and 3 in site_inventory:
        print("--- 🛠️ Crystallographic Correction: Hollow-4 detected, filtering geometric artifact Hollow-3 sites. ---")
        del site_inventory[3]

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
    print(f"--- 🛠️ _get_fragment: Preparing {TRICK_SMILES} marker for {site_type} site...")

    try:
        mol = Chem.MolFromSmiles(SMILES, sanitize=False)
        if not mol:
            raise ValueError(f"RDKit cannot parse mapped SMILES: {SMILES}")
        mol.UpdatePropertyCache(strict=False)
        
        try:
            mol_with_hs = Chem.AddHs(mol)
        except Exception:
            mol_with_hs = mol
        
        # 清除电荷以安抚 UFF 力场
        mol_for_opt = Chem.Mol(mol_with_hs)
        for atom in mol_for_opt.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0) 
            atom.SetIsotope(0)
            atom.SetHybridization(Chem.rdchem.HybridizationType.UNSPECIFIED)
        
        try:
            Chem.SanitizeMol(mol_for_opt)
        except Exception as e:
            print(f"--- ⚠️ Sanitize Warning: {e} ---")

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        params.pruneRmsThresh = 0.5
        params.numThreads = 0
        
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_for_opt, numConfs=to_initialize, params=params))
        
        if not conf_ids:
            print("--- ⚠️ ETKDGv3 failed, trying ETKDGv2 ... ---")
            AllChem.EmbedMolecule(mol_for_opt, AllChem.ETKDGv2())
            if mol_for_opt.GetNumConformers() > 0:
                conf_ids = [0]
        
        if not conf_ids:
            print("--- ⚠️ ETKDG series failed, trying Random Coords ... ---")
            # For forced coordination structures, random coords usually generate "at least one" usable geometry
            params_rand = AllChem.ETKDGv3()
            params_rand.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(mol_for_opt, numConfs=1, params=params_rand))

        # 检查是否有带电荷的原子。如果有，UFF 力场可能会崩溃/报错，因此跳过 UFF。
        has_charge = False
        for atom in mol_for_opt.GetAtoms():
            if atom.GetFormalCharge() != 0:
                has_charge = True
                break
        
        if has_charge:
            print(f"--- 🛠️ _get_fragment: Charged atoms detected, skipping UFF pre-optimization. ---")
        else:
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol_for_opt)
            except Exception as e:
                print(f"--- ⚠️ UFF Optimization Warning: {e} ---")
        
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
            
            # 1. 查找所有映射的原子 (增加同位素双重保险)
            map_num_to_idx = {}
            for atom in all_rdkit_atoms:
                map_num = atom.GetAtomMapNum()
                idx = atom.GetIdx()
                iso = atom.GetIsotope()
                
                # 优先使用 Map Number
                if map_num > 0:
                    map_num_to_idx[map_num] = idx
                
                # === [锚点生效] 如果 Map 丢了，用同位素找回 ===
                if iso == 37: 
                    # 37Cl 是我们的标记
                    map_num_to_idx[1] = idx
                if iso == 14: 
                    # 14C (或同位素14的原子) 是我们的目标
                    map_num_to_idx[114514] = idx
            
            # 2. 根据 TRICK_SMILES 和 num_binding_indices 构建索引列表
            proxy_indices = []
            binding_indices = []

            if TRICK_SMILES == "Cl":
                # --- end-on @ ontop ---
                if num_binding_indices != 1:
                     raise ValueError(f"Logic Error: TRICK_SMILES='Cl' but binding indices != 1")

                if 1 not in map_num_to_idx or 114514 not in map_num_to_idx:
                    raise ValueError(f"SMILES {SMILES} missing map number 1 (Cl) or 114514 (binding atom).")
                
                proxy_indices = [map_num_to_idx[1]]
                binding_indices = [map_num_to_idx[114514]]

                # 清理临时映射号
                all_rdkit_atoms[map_num_to_idx[114514]].SetAtomMapNum(0)
                
            elif TRICK_SMILES == "S1S":
                # --- end-on/side-on @ bridge/hollow ---
                if 1 not in map_num_to_idx or 2 not in map_num_to_idx:
                     raise ValueError(f"SMILES {SMILES} missing map number 1 (S1), 2 (S2).")
                
                proxy_indices = [map_num_to_idx[1], map_num_to_idx[2]]

                if num_binding_indices == 1:
                    # --- end-on @ bridge/hollow ---
                    if 114514 not in map_num_to_idx:
                         raise ValueError(f"SMILES {SMILES} missing map number 114514 (binding atom 1).")

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

                    print(f"--- 🛠️ _get_fragment: Manually aligned S-S marker for End-on mode (Tilt Correction). ---")
                    all_rdkit_atoms[t1_idx].SetAtomMapNum(0)

                elif num_binding_indices == 2:
                    # --- side-on @ bridge/hollow ---
                    if 114514 not in map_num_to_idx or 1919810 not in map_num_to_idx:
                         raise ValueError(f"SMILES {SMILES} missing map number 114514 (binding atom 1) or 1919810 (binding atom 2).")

                    binding_indices = [map_num_to_idx[114514], map_num_to_idx[1919810]]

                    # 改用 Parallel-Bridge 策略
                    # 使 S-S 向量 (Dummy Atoms) 平行于成键原子之间的键向量
                    # 这样当 Autoadsorbate 将 S-S 对齐到表面 Bridge 轴时，分子键也会平行于 Bridge 轴。
                    s1_idx, s2_idx = proxy_indices[0], proxy_indices[1]
                    t1_idx, t2_idx = binding_indices[0], binding_indices[1]

                    # 1. 获取目标原子的位置
                    p1 = positions[t1_idx]
                    p2 = positions[t2_idx]
                        
                    # 2. 计算它们的中点和键向量
                    midpoint = (p1 + p2) / 2.0
                    v_bond = p1 - p2
                        
                    # 3. 归一化键向量
                    norm = np.linalg.norm(v_bond)
                    if norm < 1e-3: 
                        v_bond_norm = np.array([1.0, 0.0, 0.0])
                    else:
                        v_bond_norm = v_bond / norm
                        
                    # 4. 将 S1 和 S2 放置在中点两侧，沿键向量方向延伸
                    # 距离 0.5 是任意的，只要定义了方向即可。
                    positions[s1_idx] = midpoint + v_bond_norm * 0.5
                    positions[s2_idx] = midpoint - v_bond_norm * 0.5
                        
                    print(f"--- 🛠️ _get_fragment: Aligned S-S vector parallel to bond axis (Parallel Alignment) to avoid Cross-Bridge issues. ---")
                        
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
            raise ValueError(f"RDKit conformer generation succeeded, but atom mapping trace failed (SMILES: {SMILES})")

        # 1. Create a *dummy* Fragment object using a known valid SMILES (e.g. "C") to safely complete __init__.
        print(f"--- 🛠️ _get_fragment: Safely creating empty Fragment object ... ---")
        fragment = Fragment.__new__(Fragment)
        
        # 2. Manually *overwrite* library generated dummy conformers
        print(f"--- 🛠️ _get_fragment: Overwriting .conformers with {len(reordered_conformers)} reordered conformers ... ---")
        fragment.conformers = reordered_conformers
        fragment.conformers_aligned = [False] * len(reordered_conformers)
        
        # 3. Manually *overwrite* smile attribute so autoadsorbate.Surface knows which proxy to strip ("Cl" or "S1S")
        print(f"--- 🛠️ _get_fragment: Overwriting .smile to '{TRICK_SMILES}' ... ---")
        fragment.smile = TRICK_SMILES
        fragment.to_initialize = to_initialize

        print(f"--- 🛠️ _get_fragment: Successfully created Fragment object from *SMILES '{SMILES}' (to_initialize={to_initialize}). ---")
        return fragment

    except Exception as e:
        print(f"--- 🛠️ _get_fragment: Error: Unable to create Fragment from SMILES '{SMILES}': {e} ---")
        raise e

def create_fragment_from_plan(
    original_smiles: str, 
    binding_atom_indices: list[int], 
    plan_dict: dict,
    to_initialize: int = 1
) -> Fragment:
    print(f"--- 🛠️ Executing create_fragment_from_plan ... ---")

    # 从规划字典中提取所需信息
    plan_solution = plan_dict.get("solution", {})
    adsorbate_type = plan_dict.get("adsorbate_type")
    site_type = plan_solution.get("site_type")
    num_binding_indices = len(binding_atom_indices)

    if not site_type or not adsorbate_type:
        raise ValueError("plan_dict missing 'site_type' or 'adsorbate_type'.")
    
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
    
    # Ensure fragment object has an .info dictionary
    if not hasattr(fragment, "info"):
        print("--- 🛠️ Native Fragment object missing .info dictionary, adding it... ---")
        fragment.info = {}

    # 3. 关键：将原始规划信息附加到 Fragment 对象上
    fragment.info["plan_site_type"] = site_type
    fragment.info["plan_original_smiles"] = original_smiles
    fragment.info["plan_binding_atom_indices"] = binding_atom_indices
    fragment.info["plan_adsorbate_type"] = adsorbate_type
    
    print(f"--- 🛠️ create_fragment_from_plan: Successfully created and tagged Fragment object. ---")
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
        bump_height = (min_dist_threshold - min_d) + 0.2 # Extra 0.2 A buffer
        
        print(f"--- 🛡️ Collision Detected: Atom overlap found (min_dist={min_d:.2f} Å < {min_dist_threshold} Å). Bumping up by {bump_height:.2f} Å... ---")
        
        # 修改吸附物坐标
        full_atoms.positions[adsorbate_indices, 2] += bump_height
    
    return full_atoms

def populate_surface_with_fragment(
    slab_atoms: ase.Atoms, 
    fragment_object: Fragment,
    plan_solution: dict,
    **kwargs
) -> str:
    # --- 1. Retrieve plan from Fragment object ---
    if not hasattr(fragment_object, "info") or "plan_site_type" not in fragment_object.info:
        raise ValueError("Fragment object missing 'plan_site_type' info.")

    # --- 从规划中读取参数 (或使用默认值) ---
    raw_site_type = plan_solution.get("site_type", "all")
    # 强制归一化：将 "hollow-3", "hollow-4" 统一修正为 "hollow"
    if raw_site_type.lower().startswith("hollow"):
        site_type = "hollow"
    else:
        site_type = raw_site_type
    conformers_per_site_cap = plan_solution.get("conformers_per_site_cap", 4)
    overlap_thr = plan_solution.get("overlap_thr", 0.1)
    touch_sphere_size = plan_solution.get("touch_sphere_size", 2)

    print(f"--- 🛠️ Initializing Surface (touch_sphere_size={touch_sphere_size})... ---")
    
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
    print(f"--- 🛠️ Surface Sites: Reduced from {original_site_count} to {len(s.site_df)} inequivalent sites. ---")

    # 检查是否找到了位点
    # 这可以防止在 `s.site_df.connectivity` 上失败
    if s.site_df.empty or len(s.site_df) == 0:
        raise ValueError(
            f"Autoadsorbate failed to find any adsorption sites on the surface (0 sites found). "
            f"This might be due to inappropriate `touch_sphere_size` ({touch_sphere_size}) (too large or too small)."
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
        raise ValueError(f"Unknown site_type: '{site_type}'.")

    # --- 3. 可选的表面原子过滤 ---
    allowed_symbols = plan_solution.get("surface_binding_atoms")
    if allowed_symbols and len(allowed_symbols) > 0:
        # Use sorted string for logging, clear and concise
        print(f"--- 🛠️ Filtering by surface symbols (strict match): {sorted(allowed_symbols)} ---")
        
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
        print(f"--- 🛠️ Surface Symbol Filter: Sites reduced from {initial_count} to {len(site_df_filtered)}. ---")

    # 将 s.site_df 替换为过滤后的 df
    s.site_df = site_df_filtered
    site_index_arg = list(s.site_df.index)
    
    print(f"--- 🛠️ Plan Verified: Searching {len(site_index_arg)} '{site_type}' (filtered) sites. ---")

    if len(site_index_arg) == 0:
        raise ValueError(f"No sites of type '{site_type}' containing {allowed_symbols} found. Cannot proceed.")

    # --- 4. 决定 sample_rotation ---
    sample_rotation = True
    num_binding_indices = len(fragment_object.info["plan_binding_atom_indices"])
    if num_binding_indices == 2:
        print("--- 🛠️ 2-index (side-on) mode detected. Disabling sample_rotation. ---")
        sample_rotation = False

    # --- 5. Call library ---
    print(f"--- 🛠️ Calling s.get_populated_sites (cap={conformers_per_site_cap}, overlap={overlap_thr})... ---")
    
    raw_out_trj = s.get_populated_sites(
      fragment=fragment_object,
      site_index=site_index_arg,
      sample_rotation=sample_rotation,
      mode='all',
      conformers_per_site_cap=conformers_per_site_cap,
      overlap_thr=overlap_thr,
      verbose=True
    )

    # 针对 Bridge 和 Hollow 位点，预先抬升 0.5 Å
    # 原因：autoadsorbate 默认生成的初始距离对于大分子或多位点吸附往往太近，导致频繁触发碰撞修正。
    if site_type in ["bridge", "hollow"]:
        print(f"--- 🛠️ Geometry Optimization: Pre-lifting adsorbate by 0.5 Å for {site_type} site to reduce collisions... ---")
        for atoms in raw_out_trj:
            # 找到吸附物原子的索引 (假设最后加入的是吸附物)
            n_slab = len(slab_atoms)
            atoms.positions[n_slab:, 2] += 0.5
    
    
    # 对生成的构型进行碰撞检测和抬升 (阈值 1.8 Å)
    safe_out_trj = []
    for idx, atoms in enumerate(raw_out_trj):
        safe_atoms = _bump_adsorbate_to_safe_distance(slab_atoms, atoms, min_dist_threshold=1.6)
        safe_out_trj.append(safe_atoms)
    
    out_trj = safe_out_trj

    print(f"--- 🛠️ Successfully generated {len(out_trj)} initial configurations. ---")
    
    if not out_trj:
        raise ValueError(f"get_populated_sites failed to generate any configurations. overlap_thr ({overlap_thr}) might be too strict.")
    
    # 将 ase.Atoms 列表保存到 Trajectory 对象中
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    traj_file = f"outputs/generated_conformers_{fragment_object.info['plan_original_smiles'].replace('=','_').replace('#','_')}.traj"
    traj = Trajectory(traj_file, 'w')
    for atoms in out_trj:
        traj.write(atoms)
    traj.close()

    print(f"--- 🛠️ Configurations saved to {traj_file} ---")
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
    mace_device: str = "cpu",
    mace_precision: str = "float32",
    use_dispersion: bool = False
) -> str:
    print(f"--- 🛠️ Initializing MACE Calculator (Model: {mace_model}, Device: {mace_device})... ---")
    try:
        calculator = mace_mp(model=mace_model, device=mace_device, default_dtype=mace_precision, dispersion=use_dispersion)
    except Exception as e:
        print(f"--- 🛑 MACE Initialization Failed: {e} ---")
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
        radii = np.array(natural_cutoffs(initial, mult=1.25))
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
    
    # --- 1. Evaluation Phase (Warmup + SP Energy) ---
    print(f"--- 🛠️ Evaluation Phase: Evaluating {len(atoms_list)} configurations (MD Warmup + SP Energy)... ---")
    evaluated_configs = []
    for i, atoms in enumerate(atoms_list):
        atoms.calc = calculator
        atoms.set_constraint(constraint)
        
        max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
        if max_force > 200.0:
            print(f"--- ⚠️ Skipping structure {i+1}: Initial force too high (Max Force = {max_force:.2f} eV/A)... ---")
            continue

        if md_steps > 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
            dyn_md = Langevin(atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
            dyn_md.run(md_steps)

        energy = atoms.get_potential_energy()

        # --- 能量 sanity check，屏蔽非物理爆炸结构 ---
        if (not np.isfinite(energy)) or energy < -2000.0:
            print(f"--- ⚠️ Skipping structure {i+1}: Abnormal energy (E = {energy:.2f} eV), suspected numerical collapse ---")
            continue

        print(f"--- Evaluating structure {i+1}/{len(atoms_list)}... Energy (after warmup): {energy:.4f} eV ---")
        evaluated_configs.append((energy, i, atoms.copy())) # Store copy

    if not evaluated_configs:
        raise ValueError("Evaluation phase failed to evaluate any configurations.")

    # --- 2. 选择最佳 ---
    evaluated_configs.sort(key=lambda x: x[0]) # 按能量排序
    
    if N_RELAX_TOP_N > len(evaluated_configs):
        print(f"--- 🛠️ Warning: Requested to relax top {N_RELAX_TOP_N}, but only {len(evaluated_configs)} available. Relaxing all. ---")
        N_RELAX_TOP_N = len(evaluated_configs)
    
    configs_to_relax = evaluated_configs[:N_RELAX_TOP_N]
    
    print(f"--- 🛠️ Evaluation complete. Relaxing best {N_RELAX_TOP_N} of {len(atoms_list)} configurations. ---")
    
    # --- 3. 弛豫阶段 (仅 N_RELAX_TOP_N) ---
    traj_file = f"outputs/relaxation_run.traj"
    traj = Trajectory(traj_file, 'w')
    final_structures = []

    for i, (initial_energy, original_index, atoms) in enumerate(configs_to_relax):
        print(f"--- Relaxing best structure {i+1}/{N_RELAX_TOP_N} (Original Index {original_index}, Initial Energy: {initial_energy:.4f} eV) ---")
        
        atoms.calc = calculator
        atoms.set_constraint(constraint)

        # --- 捕获弛豫前的吸附物 ---
        adsorbate_indices = list(range(len(slab_indices), len(atoms)))
        initial_adsorbate = atoms.copy()[adsorbate_indices]
        
        print(f"--- Optimization (BFGS): fmax={fmax}, steps={steps} ---")
        dyn_opt = BFGS(atoms, trajectory=None, logfile=None) 
        dyn_opt.attach(lambda: traj.write(atoms), interval=1)
        dyn_opt.run(fmax=fmax, steps=steps)

        # --- 捕获弛豫后的吸附物状态并检查键变化 ---
        final_adsorbate = atoms.copy()[adsorbate_indices]
        bond_change_count = _get_bond_change_count(initial_adsorbate, final_adsorbate)
        atoms.info["bond_change_count"] = bond_change_count
        print(f"--- Bond Integrity Check: Detected {bond_change_count} bond changes. ---")
        
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        print(f"--- Best structure {i+1} relaxation complete. Final Energy: {final_energy:.4f} eV ---")

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
        print(f"--- 🛑 Failed to write final_relaxed_structures.xyz: {e} ---")
        raise
    
    print(f"--- 🛠️ Relaxation complete. Full Trajectory: {traj_file} | Final Structures ({len(final_structures)}): {final_traj_file} ---")
    return final_traj_file

def save_ase_atoms(atoms: ase.Atoms, filename: str) -> str:
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    if not filename.startswith("outputs/"):
        filename = f"outputs/{filename}"
    
    try:
        write(filename, atoms)
        print(f"--- 🛠️ Successfully saved structure to {filename} ---")
        return f"Saved to {filename}"
    except Exception as e:
        print(f"--- 🛠️ Error: Unable to save Atoms to {filename}: {e} ---")
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
        print(f"--- 🛠️ Analyzing relaxation results: {relaxed_trajectory_file} ---")

        try:
            traj = read(relaxed_trajectory_file, index=":")
        except Exception as e_read:
            return json.dumps({"status": "error", "message": f"Unable to read trajectory file (possibly corrupted): {e_read}"})
        
        if len(traj) == 0:
            return json.dumps({"status": "error", "message": "Relaxation trajectory is empty or unreadable."})

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
        
        # 1. 定义智能判定函数 (移动到最前方，供全局复用)
        # 针对 Float32 精度和金属吸附特性，将基础容忍度从 1.25 提升至 1.3
        def check_bonding_smart(atom_idx_1, atom_idx_2, r1, r2, current_energy_eV, check_atoms_obj):
            base_mult = 1.30 # 基础键长容忍度
            
            # 能量辅助判定: 如果能量极低 (< -0.5 eV)，说明必然有强相互作用，放宽几何判定
            if current_energy_eV < -0.5:
                base_mult = 1.45 # 即使几何略微拉伸，只要能量很低，就算成键
            
            d = check_atoms_obj.get_distance(atom_idx_1, atom_idx_2, mic=True)
            threshold = (r1 + r2) * base_mult
            return d <= threshold, d, threshold

        # 1. 提取吸附物原子
        adsorbate_atoms = relaxed_atoms[len(slab_atoms):]

        # 2. 复制并应用 PBC 信息 (关键！防止跨边界原子被误判为断裂)
        # 我们创建一个临时的 Atoms 对象来进行拓扑分析
        check_atoms = adsorbate_atoms.copy()
        check_atoms.set_cell(relaxed_atoms.get_cell())
        check_atoms.set_pbc(relaxed_atoms.get_pbc())

        # 3. 构建邻接矩阵 (考虑 PBC)
        # mult=1.35 增加对键长拉伸的容忍度
        # 避免因为强吸附导致的键活化被误判为断键
        check_cutoffs = natural_cutoffs(check_atoms, mult=1.35)
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
            print(f"--- 🛠️ Logic Contradiction Fix: Dissociation detected (n_components={n_components}) but bond_change_count=0. Forcing fix. ---")
            bond_change_count = max(1, n_components - 1)

        # 7. 综合判定反应性
        reaction_detected = False
        if is_dissociated:
             # 保留真实的 bond_change_count > 0，这代表异构化
             reaction_detected = True
        elif bond_change_count > 0:
             # 键变了但没碎 -> 异构化 (Isomerization)
             # 我们标记 reaction_detected = True，让 Agent 决定这是否是坏事
             reaction_detected = True
        else:
             # 键没变，分子也没碎 -> 完美的分子吸附
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
        if num_binding_indices == 1 and len(adsorbate_indices_check) > 0:
            anchor_atom_indices = [adsorbate_indices_check[0]]
        elif num_binding_indices == 2 and len(adsorbate_indices_check) >= 2:
            anchor_atom_indices = [adsorbate_indices_check[0], adsorbate_indices_check[1]]
        
        # 1.3. 计算实际成键的表面原子数量
        for anchor_idx in anchor_atom_indices:
            r_ads = cov_cutoffs_check[anchor_idx]
            for slab_idx in slab_indices_check:
                r_slab = cov_cutoffs_check[slab_idx]
                is_connected, _, _ = check_bonding_smart(anchor_idx, slab_idx, r_ads, r_slab, E_ads, relaxed_atoms)
                if is_connected:
                    actual_bonded_slab_indices.add(slab_idx)
        
        actual_connectivity = len(actual_bonded_slab_indices)
        actual_site_type = "unknown"
        if actual_connectivity == 1: actual_site_type = "ontop"
        elif actual_connectivity == 2: actual_site_type = "bridge"
        elif actual_connectivity >= 3: actual_site_type = "hollow"
        else: actual_site_type = "desorbed"

        # 物理一致性强制修正 (Sanity Check)
        # 如果能量很低 (强吸附)，但几何判定为 desorbed，这一定是几何判据太严，强制修正为 chemisorbed
        if actual_site_type == "desorbed" and E_ads < -0.5:
            print(f"--- 🛠️ Physical Correction: Strong adsorption energy ({E_ads:.2f} eV) detected but geometrically desorbed. Forcing 'hollow/promiscuous'. ---")
            actual_site_type = "hollow (inferred)"
            # 保持 actual_connectivity 为 0 或手动设为 3，防止 Agent 困惑
            if actual_connectivity == 0: actual_connectivity = 3

        slab_indices = list(range(len(slab_atoms)))
        adsorbate_indices = list(range(len(slab_atoms), len(relaxed_atoms)))
        
        slab_atoms_relaxed = relaxed_atoms[slab_indices]
        adsorbate_atoms_relaxed = relaxed_atoms[adsorbate_indices]

        # 我们默认取吸附物列表中的第一个原子作为晶体学探测的锚点
        target_atom_global_index = adsorbate_indices[0] if len(adsorbate_indices) > 0 else -1

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
                print(f"--- ⚠️ Crystallographic Analysis Warning: {e_cryst} ---")
        
        # 将此后缀添加到 actual_site_type 中，以便 Agent 能看到区别
        if site_crystallography:
            actual_site_type += f" {site_crystallography}"
        
        print(f"--- Analysis: Site Slip Check: Planned {planned_site_type} (conn={planned_connectivity}), Actual {actual_site_type} (conn={actual_connectivity}) ---")

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

            print(f"--- Analysis: (1-index mode) Checking first adsorbate atom, Symbol: '{target_atom_symbol}', Global Index: {target_atom_global_index}. ---")

            # --- 寻找所有成键的表面原子，而不仅仅是最近的一个 ---
            bonded_surface_atoms = []
            min_distance = float('inf')
            nearest_slab_atom_symbol = ""
            nearest_slab_atom_global_index = -1
            
            # 遍历所有表面原子
            for s_idx in slab_indices:
                r_ads = cov_cutoffs_check[target_atom_global_index]
                r_slab = cov_cutoffs_check[s_idx]
                
                # 使用智能判定
                is_connected, d, threshold = check_bonding_smart(
                    target_atom_global_index, s_idx, r_ads, r_slab, E_ads, relaxed_atoms
                )
                
                if d < min_distance:
                    min_distance = d
                    nearest_slab_atom_global_index = s_idx
                    nearest_slab_atom_symbol = relaxed_atoms[s_idx].symbol
                    # 动态更新阈值用于报告
                    bonding_cutoff = threshold 

                if is_connected:
                    bonded_surface_atoms.append({
                        "symbol": relaxed_atoms[s_idx].symbol,
                        "index": s_idx,
                        "distance": round(d, 3)
                    })
            
            # 按距离排序，让最近的排前面
            bonded_surface_atoms.sort(key=lambda x: x["distance"])

            # 生成带原子索引的唯一位点指纹 (Site Fingerprint)
            # 这能区分 "Ru-Ru Bridge near Mo" 和 "Ru-Ru Bridge far from Mo"
            bonded_indices = sorted([item['index'] for item in bonded_surface_atoms])
            site_fingerprint = "-".join([f"{item['symbol']}{item['index']}" for item in bonded_surface_atoms])
            
            is_bound = len(bonded_surface_atoms) > 0
            
            # 生成成键描述字符串 (例如: "Cu-2.01Å, Ga-2.15Å")
            if is_bound:
                bonded_desc = ", ".join([f"{item['symbol']}-{item['distance']}Å" for item in bonded_surface_atoms])
            else:
                bonded_desc = "None"
            
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
                    print(f"--- ⚠️ Warning: Chemical Site Slip Detected! Planned: {planned_symbols} -> Actual: {actual_symbols} ---")

            analysis_message = (
                f"Most stable config adsorption energy: {E_ads:.4f} eV. "
                f"Target Atom: {target_atom_symbol} (from plan index {binding_atom_indices[0]}, global index {target_atom_global_index} in relaxed structure). "
                f"  -> Nearest: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}), Distance: {round(min_distance, 3)} Å (Threshold: {round(bonding_cutoff, 3)}), Bound: {is_bound}. "
                f"Bonded Surface Atoms: {bonded_desc}. "
                f"Is Bound: {is_bound}. "
                f"Reactivity Change Detected: {reaction_detected} (Bond Changes: {bond_change_count}). "
                f"Chemical Slip: {is_chemical_slip} (Planned {planned_symbols} -> Actual {actual_symbols})."
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
                    "actual_symbols": actual_symbols,
                    "site_fingerprint": site_fingerprint
                }
            }
        
        elif num_binding_indices == 2:
            if len(adsorbate_indices) < 2:
                 return json.dumps({"status": "error", "message": f"Side-on mode requires at least 2 adsorbate atoms, but found {len(adsorbate_indices)}."})
            
            # 目标原子 *总是* 吸附物列表中的前两个
            
            # --- 分析第一个原子 (Atom 0) ---
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position
            print(f"--- Analysis: (2-index mode) Checking first adsorbate atom, Symbol: '{target_atom_symbol}', Global Index: {target_atom_global_index}. ---")

            distances = np.linalg.norm(slab_atoms.positions - target_atom_pos, axis=1)
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
            print(f"--- Analysis: (side-on mode) Checking second adsorbate atom, Symbol: '{second_atom_symbol}', Global Index: {second_atom_global_index}. ---")
            
            distances_2 = np.linalg.norm(slab_atoms.positions - second_atom_pos, axis=1)
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
                r_ads = cov_cutoffs_check[ads_idx]
                for s_idx in slab_indices:
                    r_slab = cov_cutoffs_check[s_idx]
                    is_connected, d, _ = check_bonding_smart(
                        ads_idx, s_idx, r_ads, r_slab, E_ads, relaxed_atoms
                    )
                    # 判定成键
                    if is_connected:
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

            # 生成带原子索引的唯一位点指纹 (Site Fingerprint)
            # 这能区分 "Ru-Ru Bridge near Mo" 和 "Ru-Ru Bridge far from Mo"
            bonded_indices = sorted([item['index'] for item in bonded_surface_atoms])
            site_fingerprint = "-".join([f"{item['symbol']}{item['index']}" for item in bonded_surface_atoms])

            # 计算最终的最短键长 (用于报告)
            if bonded_surface_atoms:
                final_bond_distance_A = bonded_surface_atoms[0]["distance"]
            else:
                final_bond_distance_A = min(min_distance, min_distance_2)
            
            # 生成描述字符串
            if bonded_surface_atoms:
                bonded_desc = ", ".join([f"{b['adsorbate_atom']}-{b['symbol']}({b['distance']}Å)" for b in bonded_surface_atoms])
            else:
                bonded_desc = "None"

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
                    print(f"--- ⚠️ Warning: Chemical Site Slip Detected! Planned: {planned_symbols} -> Actual: {actual_symbols} ---")
            # === 🩹 修复结束 ===

            analysis_message = (
                f"Most stable config adsorption energy: {E_ads:.4f} eV. "
                f"Target Atom 1: {target_atom_symbol} (from plan index {binding_atom_indices[0]}, global index {target_atom_global_index}). "
                f"  -> Nearest: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}), Distance: {round(min_distance, 3)} Å (Threshold: {round(bonding_cutoff, 3)}), Bound: {is_bound_1}. "
                f"Target Atom 2: {second_atom_symbol} (from plan index {binding_atom_indices[1]}, global index {second_atom_global_index}). "
                f"  -> Nearest: {nearest_slab_atom_symbol_2} (Index {nearest_slab_atom_global_index_2}), Distance: {round(min_distance_2, 3)} Å (Threshold: {round(bonding_cutoff_2, 3)}), Bound: {is_bound_2}. "
                f"Bonded Surface Atoms: {bonded_desc}. "
                f"Is Bound: {is_bound}. "
                f"Reactivity Change Detected: {reaction_detected} (Bond Changes: {bond_change_count}). "
                f"Chemical Slip: {is_chemical_slip} (Planned {planned_symbols} -> Actual {actual_symbols})."
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
                    "actual_symbols": actual_symbols,
                    "site_fingerprint": site_fingerprint
                }
            }

        else:
             return json.dumps({"status": "error", "message": f"Analysis failed: Unsupported number of binding indices {num_binding_indices}."})

        # 6. 保存最终结构
        # 防止文件名冲突导致覆盖历史最优解。
        # 在文件名中加入：位点类型、表面原子组成、能量。
        
        # 命名逻辑
        site_label = actual_site_type if actual_site_type != "unknown" else planned_site_type
        if planned_site_type != "unknown" and site_label != planned_site_type:
            site_label = f"{planned_site_type}_to_{site_label}"
            
        if is_dissociated: site_label += "_DISS"
        elif bond_change_count > 0: site_label += "_ISO"
        
        site_label = site_label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")

        clean_smiles = original_smiles.replace('=', '_').replace('#', '_').replace('[', '').replace(']', '')
        best_atoms_filename = f"outputs/BEST_{clean_smiles}_{site_label}_E{E_ads:.3f}.xyz"
        
        try:
            write(best_atoms_filename, relaxed_atoms)
            print(f"--- 🛠️ Successfully saved best structure to {best_atoms_filename} ---")
            # Return specific filename to Agent for reference in report
            result["best_structure_file"] = best_atoms_filename
        except Exception as e:
            print(f"--- 🛠️ Error: Unable to save best structure to {best_atoms_filename}: {e} ---")

        return json.dumps(result)
    
    except Exception as e:
        import traceback
        print(f"--- 🛠️ Error: Unexpected exception during relaxation analysis: {e} ---")
        print(traceback.format_exc())
        return json.dumps({"status": "error", "message": f"Unexpected exception during relaxation analysis: {e}"})
