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
from ase.neighborlist import natural_cutoffs
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
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

def _get_fragment(SMILES: str, site_type: str, num_binding_indices: int, to_initialize: int = 1) -> Union[Fragment, ase.Atoms]:
    # 确定 TRICK_SMILES，以便稍后设置 .info["smiles"]
    TRICK_SMILES = ""
    if site_type == "ontop":
        TRICK_SMILES = "Cl"
    elif site_type in ["bridge", "hollow"]:
        TRICK_SMILES = "S1S"
    else:
        raise ValueError(f"未知的 site_type: {site_type}")

    print(f"--- 🛠️ _get_fragment: 正在为 {site_type} 位点准备 {TRICK_SMILES} 标记...")

    try:
        mol = Chem.MolFromSmiles(SMILES)
        if not mol:
            raise ValueError(f"RDKit 无法解析映射的 SMILES: {SMILES}")
        
        try:
            mol_with_hs = Chem.AddHs(mol)
        except Exception:
            print(f"--- 🛠️ _get_fragment: 警告: Chem.AddHs 失败，正在尝试在没有显式H的情况下继续... ---")
            mol_with_hs = mol
        
        # 使用 RDKit 生成构象 (与 autoadsorbate 内部逻辑类似)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D # 任意的种子
        params.pruneRmsThresh = 0.5 # 合理的剪枝阈值
        params.numThreads = 0 # 使用所有核心
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_with_hs, numConfs=to_initialize, params=params))
        
        if not conf_ids:
             # 回退到更简单的嵌入器
             if AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv2()) == -1:
                 # 再次尝试
                 if AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv2()) == -1:
                    raise ValueError(f"RDKit 未能为 {SMILES} 生成构象。")
             conf_ids = [0]
        
        # 优化生成的构象
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol_with_hs)
        except Exception as e:
            # UFFTYPER 警告会在这里被捕获。我们忽略它们并继续。
            print(f"--- 🛠️ _get_fragment: 警告: UFF 优化失败或发出警告 ({e})。使用未优化的构象。 ---")

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
        print(f"--- 🛠️ 正在按表面符号过滤: {allowed_symbols} ---")
        
        def check_symbols(site_formula_dict):
            if not site_formula_dict or not isinstance(site_formula_dict, dict):
                return False
            # 检查此位点的 *任何* 原子是否在允许列表中
            return any(symbol in allowed_symbols for symbol in site_formula_dict.keys())

        initial_count = len(site_df_filtered)
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
    
    out_trj = s.get_populated_sites(
      fragment=fragment_object,
      site_index=site_index_arg,
      sample_rotation=sample_rotation,
      mode='all',
      conformers_per_site_cap=conformers_per_site_cap,
      overlap_thr=overlap_thr,
      verbose=True
    )
    
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
    
    # --- 1. 评估阶段 (预热 + 单点能量) ---
    print(f"--- 🛠️ 评估阶段：正在评估 {len(atoms_list)} 个构型 (MD 预热 + SP 能量)... ---")
    evaluated_configs = [] # 列表将存储: (energy, original_index, atoms_object)
    
    for i, atoms in enumerate(atoms_list):
        try:
            atoms.calc = calculator
            atoms.set_constraint(constraint)
            
            if md_steps > 0:
                MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
                dyn_md = Langevin(atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
                dyn_md.run(md_steps)

            energy = atoms.get_potential_energy()
            print(f"--- 评估 结构 {i+1}/{len(atoms_list)}. 能量 (预热后): {energy:.4f} eV ---")
            evaluated_configs.append((energy, i, atoms.copy())) # 存储副本
        except Exception as e:
            print(f"--- 🛑 评估 结构 {i+1} 失败: {e} ---")

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

    # 定义一个通用的、不依赖 autoadsorbate 库的键完整性检查器
    def _get_bond_change_count(initial_adsorbate, final_adsorbate):
        try:
            if len(initial_adsorbate) != len(final_adsorbate):
                return -2 # 原子数不匹配

            # 1. 获取初始连接矩阵 (忽略 H-H 键)
            initial_distances = initial_adsorbate.get_all_distances()
            initial_cutoffs = natural_cutoffs(initial_adsorbate, mult=1.1)
            initial_bonds = initial_distances < (np.array([initial_cutoffs]).T + initial_cutoffs)
            np.fill_diagonal(initial_bonds, False)
            h_indices_initial = [a.index for a in initial_adsorbate if a.symbol == 'H']
            for i in h_indices_initial:
                for j in h_indices_initial:
                    initial_bonds[i, j] = False
            
            # 2. 获取最终连接矩阵 (忽略 H-H 键)
            final_distances = final_adsorbate.get_all_distances()
            final_cutoffs = natural_cutoffs(final_adsorbate, mult=1.1)
            final_bonds = final_distances < (np.array([final_cutoffs]).T + final_cutoffs)
            np.fill_diagonal(final_bonds, False)
            h_indices_final = [a.index for a in final_adsorbate if a.symbol == 'H']
            for i in h_indices_final:
                for j in h_indices_final:
                    final_bonds[i, j] = False

            # 3. 比较
            diff_matrix = initial_bonds.astype(int) - final_bonds.astype(int)
            diff_upper = np.triu(diff_matrix)
            bond_change_count = np.sum(np.abs(diff_upper))
            return int(bond_change_count)

        except Exception as e_bond:
            print(f"--- 🛠️ 警告: 内部键完整性检查失败: {e_bond} ---")
            return -1 # 标记为检查失败

    for i, (initial_energy, original_index, atoms) in enumerate(configs_to_relax):
        print(f"--- 弛豫最佳结构 {i+1}/{N_RELAX_TOP_N} (原始 Index {original_index}, E_pre={initial_energy:.4f} eV) ---")
        
        atoms.calc = calculator
        atoms.set_constraint(constraint)

        # --- 🛠️ 捕获弛豫前的吸附物状态 ---
        adsorbate_indices_for_copy = list(range(len(slab_indices), len(atoms)))
        initial_adsorbate = atoms.copy()[adsorbate_indices_for_copy]
        
        print(f"--- 优化 (BFGS): fmax={fmax}, steps={steps} ---")
        dyn_opt = BFGS(atoms, trajectory=None, logfile=None) 
        
        dyn_opt.attach(lambda: traj.write(atoms), interval=1)
        
        dyn_opt.run(fmax=fmax, steps=steps)

        # --- 🛠️ 捕获弛豫后的吸附物状态并检查键变化 ---
        final_adsorbate = atoms.copy()[adsorbate_indices_for_copy]
        
        # 调用我们刚刚定义的内部函数
        bond_change_count = _get_bond_change_count(initial_adsorbate, final_adsorbate)
        atoms.info["bond_change_count"] = bond_change_count # 存储结果
        print(f"--- 键完整性检查: {bond_change_count} 个键发生变化。 ---")
        
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        print(f"--- 结构 {i+1} 弛豫完成。最终能量: {final_energy:.4f} eV ---")

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
        
        # --- 提取新添加的键变化信息 ---
        bond_change_count = relaxed_atoms.info.get("bond_change_count", -1) # -1 表示检查失败
        reaction_detected = bond_change_count > 0

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
        
        print(f"--- 分析: 位点滑移检查：规划 {planned_site_type} (conn={planned_connectivity}), 实际 {actual_site_type} (conn={actual_connectivity}) ---")

        # 2. 识别吸附物原子和表面原子
        slab_indices = list(range(len(slab_atoms)))
        adsorbate_indices = list(range(len(slab_atoms), len(relaxed_atoms)))
        
        slab_atoms_relaxed = relaxed_atoms[slab_indices]
        adsorbate_atoms_relaxed = relaxed_atoms[adsorbate_indices]
        
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

            # 4. 计算该原子与表面的最近距离
            distances = np.linalg.norm(slab_atoms_relaxed.positions - target_atom_pos, axis=1)
            min_distance = np.min(distances)
            nearest_slab_atom_global_index = slab_indices[np.argmin(distances)]
            nearest_slab_atom_symbol = relaxed_atoms[nearest_slab_atom_global_index].symbol

            # 5. 估计键合
            radius_1 = cov_cutoffs[target_atom_global_index]
            radius_2 = cov_cutoffs[nearest_slab_atom_global_index]
            bonding_cutoff = (radius_1 + radius_2) * 1.1 # 1.1 的容差
            is_bound = min_distance <= bonding_cutoff

            analysis_message = (
                f"最稳定构型吸附能: {E_ads:.4f} eV。"
                f"目标吸附物原子: {target_atom_symbol} (来自规划索引 {binding_atom_indices[0]}，在弛豫结构中为全局索引 {target_atom_global_index})。 "
                f"最近的表面原子: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index})。 "
                f"最终距离: {round(min_distance, 3)} Å。"
                f"估计共价键阈值: {round(bonding_cutoff, 3)} Å。"
                f"是否成键: {is_bound}。"
                f"是否发生反应性转变: {reaction_detected} (键变化数: {bond_change_count} )。"
            )

            result = {
                "status": "success",
                "message": analysis_message,
                "most_stable_energy_eV": E_ads,
                "target_adsorbate_atom": target_atom_symbol,
                "target_adsorbate_atom_index": int(target_atom_global_index),
                "nearest_slab_atom": nearest_slab_atom_symbol,
                "nearest_slab_atom_index": int(nearest_slab_atom_global_index),
                "final_bond_distance_A": round(min_distance, 3),
                "estimated_covalent_cutoff_A": round(bonding_cutoff, 3),
                "is_covalently_bound": bool(is_bound),
                "reaction_detected": bool(reaction_detected),
                "bond_change_count": int(bond_change_count),
                "site_analysis": {
                    "planned_site_type": planned_site_type,
                    "planned_connectivity": planned_connectivity,
                    "actual_site_type": actual_site_type,
                    "actual_connectivity": actual_connectivity
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
            
            analysis_message = (
                f"最稳定构型吸附能: {E_ads:.4f} eV。"
                f"目标原子 1: {target_atom_symbol} (来自规划索引 {binding_atom_indices[0]}，全局索引 {target_atom_global_index})。"
                f"  -> 最近: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}), 距离: {round(min_distance, 3)} Å (阈值: {round(bonding_cutoff, 3)}), 成键: {is_bound_1}。"
                f"目标原子 2: {second_atom_symbol} (来自规划索引 {binding_atom_indices[1]}，全局索引 {second_atom_global_index})。"
                f"  -> 最近: {nearest_slab_atom_symbol_2} (Index {nearest_slab_atom_global_index_2}), 距离: {round(min_distance_2, 3)} Å (阈值: {round(bonding_cutoff_2, 3)}), 成键: {is_bound_2}。"
                f"是否成键: {is_bound}。"
                f"是否发生反应性转变: {reaction_detected} (键变化数: {bond_change_count} )。"
            )

            result = {
                "status": "success",
                "message": analysis_message,
                "most_stable_energy_eV": E_ads,
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
                "site_analysis": {
                    "planned_site_type": planned_site_type,
                    "planned_connectivity": planned_connectivity,
                    "actual_site_type": actual_site_type,
                    "actual_connectivity": actual_connectivity
                }
            }

        else:
             return json.dumps({"status": "error", "message": f"分析失败：不支持的键合索引数量 {num_binding_indices}。"})

        # 6. 保存最终结构
        best_atoms_filename = f"outputs/BEST_{original_smiles.replace('=','_').replace('#','_')}_on_surface.xyz"
        try:
            write(best_atoms_filename, relaxed_atoms)
            print(f"--- 🛠️ 成功将最佳结构保存到 {best_atoms_filename} ---")
            result["best_structure_file"] = best_atoms_filename
        except Exception as e:
            print(f"--- 🛠️ 错误: 无法保存最佳结构到 {best_atoms_filename}: {e} ---")

        return json.dumps(result)
    
    except Exception as e:
        import traceback
        print(f"--- 🛠️ 错误: 分析弛豫时发生意外异常: {e} ---")
        print(traceback.format_exc())
        return json.dumps({"status": "error", "message": f"分析弛豫时发生意外异常: {e}"})
