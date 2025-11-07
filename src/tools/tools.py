import ase
from ase.io import read, write
from autoadsorbate import Surface, Fragment
from ase.constraints import FixAtoms
from autoadsorbate.Surf import attach_fragment
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
import torch
from mace.calculators import mace_mp, MACECalculator
from ase.md.langevin import Langevin
from ase import units
import os
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import natural_cutoffs, NeighborList
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from typing import Union

def generate_surrogate_smiles(original_smiles: str, binding_atom_indices: list[int], orientation: str) -> str:
    """
    通过将“替代”原子（如 Cl 或 S）连接到 LLM 规划的键合位点，
    将标准 SMILES 转换为 *SMILES (Surrogate-SMILES)。
    
    此版本为 end-on (Cl) 和 side-on (S1S) 模式
    创建*新的* RWMol，以强制标记位于索引 0 (或 0,1)。
    """
    print(f"--- 🔬 调用 SMILES 翻译器: {original_smiles} via indices {binding_atom_indices} (朝向: {orientation}) ---")
    
    sanitized_smiles = original_smiles
    if original_smiles == "C#O":
        sanitized_smiles = "[C-]#[O+]"
        print(f"--- 🔬 检测到无效的 'C#O'。已将其清理为 '{sanitized_smiles}'。 ---")
    
    mol = Chem.MolFromSmiles(sanitized_smiles)
    if not mol:
        raise ValueError(f"RDKit 无法解析原始 SMILES: {original_smiles}")
    
    # --- end-on (单点连接) 逻辑 ---
    if orientation == "end-on":
        if not binding_atom_indices or len(binding_atom_indices) != 1:
            raise ValueError(f"'end-on' 朝向需要 *一个* 键合索引，但提供了 {len(binding_atom_indices)} 个。")
            
        target_idx = binding_atom_indices[0]
        
        if target_idx >= mol.GetNumAtoms():
             raise ValueError(f"索引 {target_idx} 超出范围 (分子原子数: {mol.GetNumAtoms()})。")

        # 必须创建新分子以强制 Cl 位于索引 0
        new_mol = Chem.RWMol()
        
        # 1. 添加 Cl 标记 (索引 0)
        marker_idx = new_mol.AddAtom(Chem.Atom("Cl")) # index 0
        
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
        
        # 5. 调整电荷
        target_atom_obj = new_mol.GetAtomWithIdx(idx_map[target_idx])
        target_atom_obj.SetFormalCharge(target_atom_obj.GetFormalCharge() + 1)
        
        out_smiles = Chem.MolToSmiles(new_mol.GetMol())
        print(f"--- 🔬 SMILES 翻译器输出 (Cl-first): {out_smiles} ---")
        return out_smiles

    # --- side-on (双点连接) 逻辑 ---
    elif orientation == "side-on":
        if not binding_atom_indices or len(binding_atom_indices) != 2:
            raise ValueError(f"'side-on' 朝向需要 *两个* 键合索引，但提供了 {len(binding_atom_indices)} 个。")
        
        target_indices = sorted(binding_atom_indices)
        idx1, idx2 = target_indices[0], target_indices[1]

        if idx2 >= mol.GetNumAtoms():
             raise ValueError(f"索引 {idx2} 超出范围 (分子原子数: {mol.GetNumAtoms()})。")

        # 创建 [S, S, ...rest] 顺序的新 RWMol
        new_mol = Chem.RWMol()
        marker1_idx = new_mol.AddAtom(Chem.Atom("S")) # index 0
        marker2_idx = new_mol.AddAtom(Chem.Atom("S")) # index 1
        new_mol.AddBond(marker1_idx, marker2_idx, Chem.rdchem.BondType.SINGLE)
        
        idx_map = {}
        for atom in mol.GetAtoms():
            new_idx = new_mol.AddAtom(atom)
            idx_map[atom.GetIdx()] = new_idx
        
        target_idx_set = set(target_indices)
        for bond in mol.GetBonds():
            idx_a = bond.GetBeginAtomIdx()
            idx_b = bond.GetEndAtomIdx()
            
            if idx_a in target_idx_set and idx_b in target_idx_set:
                print(f"--- 🔬 SMILES 翻译器: 为 'side-on' 模式断开了索引 {idx_a}-{idx_b} 之间的多重键。 ---")
                continue
            
            new_mol.AddBond(idx_map[idx_a], idx_map[idx_b], bond.GetBondType())

        if mol.GetBondBetweenAtoms(idx1, idx2):
            new_mol.AddBond(idx_map[idx1], idx_map[idx2], Chem.rdchem.BondType.SINGLE)
        
        new_mol.AddBond(marker1_idx, idx_map[idx1], Chem.rdchem.BondType.SINGLE)
        new_mol.AddBond(marker2_idx, idx_map[idx2], Chem.rdchem.BondType.SINGLE)
        
        out_smiles = Chem.MolToSmiles(new_mol.GetMol())
        print(f"--- 🔬 SMILES 翻译器输出 (S1S-first): {out_smiles} ---")
        return out_smiles

    else:
        raise ValueError(f"未知的朝向: {orientation}。必须是 'end-on' 或 'side-on'。")

# --- 其他工具 ---
def read_atoms_object(slab_path: str) -> ase.Atoms:
    """Reads a atomistic structure file 
    Args:
        path: string - location on system
    returns:
        ase.Atoms object
    """
    try:
        atoms = read(slab_path)  # 从 .xyz 或 .cif 文件中读取 slab 结构。
        print(f"成功: 已从 {slab_path} 读取 slab 原子。")
        return atoms
    except Exception as e:
        print(f"错误: 无法读取 {slab_path}: {e}")
        raise

def _get_fragment(SMILES: str, orientation: str, to_initialize: int = 1) -> Union[Fragment, ase.Atoms]:
    """
    [内部函数] 原始的 get_fragment 实现，包含所有必要的补丁
    以处理 autoadsorbate 库的限制。
    
    移除了所有手动的构象构建器 (manual_conformer)，
    完全依赖通用的 S1S 原子重排 (re-ordering) 逻辑。
    
    现在显式接受 'orientation' 参数，以确定
    (a) TRICK_SMILES 是 "Cl" 还是 "S1S"
    (b) 是否应用 S1S 原子重排（re-ordering）补丁。
    这现在可以正确处理含有真实 S 或 Cl 的分子。
    """

    # 我们不再猜测 TRICK_SMILES。我们根据规划器传入的朝向来 *确定* 它。
    TRICK_SMILES = None
    if orientation == "end-on":
        TRICK_SMILES = "Cl"
    elif orientation == "side-on":
        TRICK_SMILES = "S1S"
    else:
        # 如果朝向未知（不应发生），则回退到原始 SMILES
        print(f"--- 🛠️ _get_fragment: 警告: 未知的朝向 '{orientation}'。将使用原始SMILES。---")
        TRICK_SMILES = SMILES

    # --- 统一路径：使用 autoadsorbate.Fragment 正常初始化 ---
    try:
        # 无论 SMILES 是什么 (例如 '[C-]1[O+]SS1' 或 'CC[S+](Cl)')，
        # 我们都要求 RDKit/autoadsorbate 尽力生成构象。
        fragment = Fragment(smile=SMILES, to_initialize=to_initialize)
        
        # 仅当规划的朝向是 "side-on" 时，我们才应用 S1S 重排补丁。
        if TRICK_SMILES == "S1S":
            
            # 检查 RDKit 是否返回了任何构象
            if not fragment.conformers or len(fragment.conformers) == 0:
                 # RDKit 确实失败了，但我们不能硬编码答案。
                 # 我们必须向上传递这个错误。
                 print(f"--- 🛠️ _get_fragment: 错误: RDKit 未能为代理 SMILES '{SMILES}' 生成任何 3D 构象。---")
                 raise ValueError(f"RDKit failed to generate conformers for SMILES: {SMILES}")

            # 检查是否真的需要重排 (例如，RDKit 规范化导致 [C,O,S,S] 而不是 [S,S,C,O])
            if fragment.conformers[0].get_chemical_symbols()[0] != 'S':
                print(f"--- 🛠️ _get_fragment: S1S 模式检测到构象顺序错误 (例如 [C, O, S, S])。---")
                print(f"--- 🛠️ _get_fragment: 正在将所有构象重排 (Re-ordering) 为 [S, S, ...rest]... ---")
                
                reordered_conformers = []
                
                for conf_atoms in fragment.conformers:
                    symbols = conf_atoms.get_chemical_symbols()
                    positions = conf_atoms.get_positions()
                    
                    # 我们必须精确地识别*代理 S* 原子。
                    # 我们的 generate_surrogate_smiles *总是* 添加两个 S。
                    # 假设 SMILES 是 'CCSS1S1' (来自 C=S)，
                    # RDKit 可能会返回 [C, C, S(真实), S(代理), S(代理)]
                    # 我们如何区分它们？
                    # 让我们依赖一个更强的假设：代理 SMILES (如 [C-]1[O+]SS1) 这对吗？？？
                    # *只* 包含代理 S 原子，而*不*包含真实的 S 原子。
                    # 这是 generate_surrogate_smiles 的当前行为。
                    #
                    # 因此，s_indices 列表中的*所有* S 原子都*是*代理 S。

                    s_indices = [i for i, s in enumerate(symbols) if s == 'S']
                    other_indices = [i for i, s in enumerate(symbols) if s != 'S']

                    if len(s_indices) < 2:
                         print(f"--- 🛠️ _get_fragment: 错误: S1S 模式但未找到 2 个 S 原子。SMILES: {SMILES}。原子: {symbols}。---")
                         continue # 跳过这个坏掉的构象
                    
                    if len(s_indices) > 2:
                        # 这种情况现在不应该发生，除非 generate_surrogate_smiles 坏了
                        print(f"--- 🛠️ _get_fragment: 警告: S1S 模式找到 {len(s_indices)} 个 S 原子。只使用前 2 个作为代理。---")
                        s_indices = s_indices[:2]
                        s_indices_set = set(s_indices)
                        other_indices = [i for i, s in enumerate(symbols) if i not in s_indices_set]
                    
                    # new_order = [S, S, C, O]
                    new_order = s_indices + other_indices
                    
                    if len(new_order) != len(symbols):
                        print(f"--- 🛠️ _get_fragment: 错误: 重排 (Re-ordering) 长度不匹配! {len(new_order)} vs {len(symbols)} ---")
                        continue 

                    new_atoms = Atoms(
                        symbols=[symbols[i] for i in new_order],
                        positions=[positions[i] for i in new_order]
                    )
                    new_atoms.info = conf_atoms.info.copy() if hasattr(conf_atoms, "info") else {}
                    reordered_conformers.append(new_atoms)
                
                fragment.conformers = reordered_conformers # 替换为重排后的列表
                print(f"--- 🛠️ _get_fragment: 成功重排 (re-ordered) {len(fragment.conformers)} 个构象。 ---")
            
            elif fragment.conformers and fragment.conformers[0].get_chemical_symbols()[0] == 'S':
                print(f"--- 🛠️ _get_fragment: S1S 模式检测到。构象已具有正确的 [S, S, ...] 顺序。无需重排。 ---")
        
        # 应用 TRICK_SMILES 补丁
        # (这现在是安全的，TRICK_SMILES 是基于 'orientation' 确定的，而不是猜测)
        fragment.smile = TRICK_SMILES 
        for conf in fragment.conformers:
            if not hasattr(conf, "info"):
                conf.info = {}
            conf.info["smiles"] = TRICK_SMILES
            
        print(f"--- 🛠️ _get_fragment: 已将 Fragment.smile 和 conformer.info['smiles'] 覆盖为 '{TRICK_SMILES}' 以兼容库。 ---")

        if not fragment.conformers or len(fragment.conformers) == 0:
             # 这现在是一个真正的错误
             raise ValueError(f"RDKit 未能为 {SMILES} 生成任何 3D 构象（在重排后为空）。")

        print(f"--- 🛠️ _get_fragment: 成功从 *SMILES '{SMILES}' (to_initialize={to_initialize}) 初始化 autoadsorbate.Fragment 对象。 ---")
        return fragment

    except Exception as e:
        print(f"--- 🛠️ _get_fragment: 警告: 无法使用 autoadsorbate.Fragment 初始化 ('{e}')。回退到手动 RDKit 构建... ---")
        
        try:
            mol = Chem.MolFromSmiles(SMILES)
            mol_with_hs = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv2())
            
            if mol_with_hs.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv2())

            if mol_with_hs.GetNumConformers() == 0:
                raise ValueError(f"RDKit Fallback (EmbedMolecule) 也未能为 {SMILES} 生成构象。")

            conf = mol_with_hs.GetConformer()
            positions = conf.GetPositions()
            symbols = [atom.GetSymbol() for atom in mol_with_hs.GetAtoms()]
            
            atoms = Atoms(symbols=symbols, positions=positions)
            atoms.info = {"smiles": TRICK_SMILES}
            print("--- 🛠️ _get_fragment: 成功通过 RDKit 手动回退构建了 ase.Atoms。 ---")
            
            # 同样应用 S1S 重排修复
            if TRICK_SMILES == "S1S":
                if atoms.get_chemical_symbols()[0] != 'S':
                    print(f"--- 🛠️ _get_fragment (Fallback): S1S 模式检测到。正在重排 (Re-ordering)... ---")
                    symbols = atoms.get_chemical_symbols()
                    positions = atoms.get_positions()
                    s_indices = [i for i, s in enumerate(symbols) if s == 'S']
                    other_indices = [i for i, s in enumerate(symbols) if s != 'S']
                    if len(s_indices) >= 2:
                        if len(s_indices) > 2:
                            s_indices = s_indices[:2] # 只取前两个
                        s_indices_set = set(s_indices)
                        other_indices = [i for i, s in enumerate(symbols) if i not in s_indices_set]
                        new_order = s_indices + other_indices
                        atoms = Atoms(
                            symbols=[symbols[i] for i in new_order],
                            positions=[positions[i] for i in new_order]
                        )
                        atoms.info = {"smiles": TRICK_SMILES} # 重新设置 info

            fragment = Fragment(smile=TRICK_SMILES, to_initialize=0)
            fragment.conformers = [atoms]
            fragment.conformers_aligned = [False]
            return fragment

        except Exception as e_inner:
            print(f"--- 🛠️ _get_fragment: 错误: 无法从 SMILES '{SMILES}' 创建 Fragment: {e_inner} ---")
            raise e_inner

def create_fragment_from_plan(
    original_smiles: str, 
    binding_atom_indices: list[int], 
    orientation: str,
    to_initialize: int = 1
) -> Fragment:
    """
    [新工具] 从一个规划中创建 autoadsorbate Fragment 对象。
    
    这是 AI 代理创建吸附物的首选工具。
    它封装了以下步骤：
    1. original_smiles -> *SMILES (使用 generate_surrogate_smiles)
    2. *SMILES -> 3D 构象 (使用 _get_fragment)
    3. 自动处理 RDKit 失败的特殊情况 (如 N2, C2H4 side-on)。
    4. 将规划信息附加到 Fragment 对象上，供后续工具使用。
    """
    print(f"--- 🛠️ 正在执行 create_fragment_from_plan ... ---")
    
    # 1. 内部调用 SMILES 生成器
    surrogate_smiles = generate_surrogate_smiles(
        original_smiles=original_smiles,
        binding_atom_indices=binding_atom_indices,
        orientation=orientation
    )

    # 2. 内部调用构象生成器 (包含所有补丁和技巧)
    # 我们必须将 'orientation' 参数从这里传递下去
    fragment = _get_fragment(
        SMILES=surrogate_smiles,
        orientation=orientation,
        to_initialize=to_initialize
    )
    
    # 确保 fragment 对象有一个 .info 字典
    if not hasattr(fragment, "info"):
        print("--- 🛠️ 原生 Fragment 对象缺少 .info 字典，正在添加... ---")
        fragment.info = {}

    # 3. 关键：将原始规划信息附加到 Fragment 对象上
    fragment.info["plan_orientation"] = orientation
    fragment.info["plan_original_smiles"] = original_smiles
    fragment.info["plan_binding_atom_indices"] = binding_atom_indices
    
    print(f"--- 🛠️ create_fragment_from_plan: 成功创建并标记了 Fragment 对象。 ---")
    return fragment

def populate_surface_with_fragment(
    slab_atoms: ase.Atoms, 
    fragment_object: Fragment,
    site_type: str,
    allowed_surface_symbols: list = None,
    conformers_per_site_cap: int = 2,
    overlap_thr: float = 0.1,
    touch_sphere_size: float = 2.8,
    **kwargs
) -> str:
    """
    使用 autoadsorbate.Surface.get_populated_sites 自动在表面上放置片段。
    
    此版本从 Fragment 对象中读取规划信息 (plan_orientation)，
    并验证它是否与请求的 site_type 兼容，而不是依赖
    脆弱的 SMILES 技巧。
    """
    
    # --- 1. 从 Fragment 对象中检索规划 ---
    if not hasattr(fragment_object, "info") or "plan_orientation" not in fragment_object.info:
        raise ValueError("Fragment 对象缺少 'plan_orientation' 信息。请使用 'create_fragment_from_plan' 创建它。")
        
    plan_orientation = fragment_object.info["plan_orientation"]

    print(f"--- 🛠️ 正在初始化表面 (touch_sphere_size={touch_sphere_size})... ---")
    s = Surface(slab_atoms, precision=1.0, touch_sphere_size=touch_sphere_size)

    original_site_count = len(s.site_df)
    s.sym_reduce()
    print(f"--- 🛠️ 表面位点：从 {original_site_count} 个减少到 {len(s.site_df)} 个不等价位点。 ---")

    # --- 2. 验证规划与位点的兼容性 ---
    site_df_filtered = s.site_df
    
    if site_type == "ontop":
        if plan_orientation != "end-on":
            raise ValueError(f"规划不匹配：'ontop' 位点 (connectivity=1) 与 '{plan_orientation}' 朝向不兼容。")
        site_df_filtered = s.site_df[s.site_df.connectivity == 1]
        
    elif site_type == "bridge":
        if plan_orientation != "side-on":
            raise ValueError(f"规划不匹配：'bridge' 位点 (connectivity=2) 与 '{plan_orientation}' 朝向不兼容。")
        site_df_filtered = s.site_df[s.site_df.connectivity == 2]

    elif site_type == "hollow":
        site_df_filtered = s.site_df[s.site_df.connectivity >= 3]
        if plan_orientation not in ["end-on"]:
             print(f"--- 🛠️ 警告: 尝试将 '{plan_orientation}' 放置在 'hollow' 位点上。这可能不是一个稳定的构型。 ---")

    else:
        # 允许 "all"
        if site_type == "all":
             print(f"--- 🛠️ 正在搜索 'all' 位点... ---")
             site_df_filtered = s.site_df
        else:
            raise ValueError(f"未知的 site_type: '{site_type}'。必须是 'ontop', 'bridge', 'hollow', 或 'all'。")

    # [可选] ... (allowed_surface_symbols 过滤逻辑可以在此添加) ...
    
    # 将 s.site_df 替换为过滤后的 df
    s.site_df = site_df_filtered
    site_index_arg = list(s.site_df.index)
    
    print(f"--- 🛠️ 规划已验证：正在搜索 {len(site_index_arg)} 个 '{site_type}' 位点以用于 '{plan_orientation}' 吸附。 ---")

    if len(site_index_arg) == 0:
        raise ValueError(f"未找到 '{site_type}' 类型的位点。无法继续。")

    # --- 3. 决定 sample_rotation ---
    sample_rotation = True
    if plan_orientation == "side-on":
        print("--- 🛠️ 检测到 'side-on' 模式。禁用 sample_rotation。---")
        sample_rotation = False

    # --- 4. 调用库 ---
    # 我们仍然依赖 _get_fragment 中的“SMILES 技巧”来使
    # 底层的 s.get_populated_sites [autoadsorbate.py:331] 工作。
    
    print(f"--- 🛠️ 正在调用 s.get_populated_sites (cap={conformers_per_site_cap}, overlap={overlap_thr})... ---")
    
    out_trj = s.get_populated_sites(
      fragment=fragment_object, # 传递完整的 Fragment 对象
      site_index=site_index_arg,
      sample_rotation=sample_rotation,
      mode='all', # 我们已经自己完成了 'heuristic' 过滤
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
    md_temp: float = 150.0
) -> str:
    print(f"--- 🛠️ 正在初始化 MACE 计算器... ---")
    try:
        calculator = mace_mp(model="small", device='cpu', default_dtype='float32', dispersion=True)
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

    for i, (initial_energy, original_index, atoms) in enumerate(configs_to_relax):
        print(f"--- 弛豫最佳结构 {i+1}/{N_RELAX_TOP_N} (原始 Index {original_index}, E_pre={initial_energy:.4f} eV) ---")
        
        # 计算器、约束和 MD 已经在评估阶段设置过
        # 我们需要重新附加，因为我们存储的是副本
        atoms.calc = calculator
        atoms.set_constraint(constraint)
        
        print(f"--- 优化 (BFGS): fmax={fmax}, steps={steps} ---")
        dyn_opt = BFGS(atoms, trajectory=None, logfile=None) 
        
        dyn_opt.attach(lambda: traj.write(atoms), interval=1)
        
        dyn_opt.run(fmax=fmax, steps=steps)
        
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
    """
    将单个 ase.Atoms 对象保存到文件。
    """
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
    binding_atom_indices: list[int],
    orientation: str
) -> str:
    """
    分析弛豫轨迹，找到最稳定的结构，并检查其键合情况。
    此版本不再使用 RDKit 索引进行猜测，而是依赖
    autoadsorbate 库保证的原子顺序 (即 adsorbate_indices[0] 
    始终是 end-on 的目标，[0] 和 [1] 始终是 side-on 的目标)。
    """
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
        
        if not energies:
            if len(traj) > 0:
                 relaxed_atoms = traj[-1]
                 min_energy = -999.0
                 best_index = len(traj) - 1
                 print(f"--- 分析: 警告：无法从 .xyz 读取能量。回退到分析最后一个结构 (Index {best_index}) ---")
            else:
                 return json.dumps({"status": "error", "message": "弛豫轨迹为空。"})
        else:
            min_energy = min(energies)
            best_index = np.argmin(energies)
            relaxed_atoms = traj[best_index]
            print(f"--- 分析: 找到最稳定的构型 (Index {best_index})，能量: {min_energy:.4f} eV ---")

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
        cov_cutoffs = natural_cutoffs(relaxed_atoms, mult=1.0)

        if orientation == "end-on":
            # 目标原子 *总是* 吸附物列表中的第一个
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position

            print(f"--- 分析: (end-on 模式) 正在检查第一个吸附物原子, 符号: '{target_atom_symbol}', 全局索引: {target_atom_global_index}。---")

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
                f"最稳定的构型能量: {min_energy:.4f} eV。 "
                f"目标吸附物原子: {target_atom_symbol} (来自规划索引 {binding_atom_indices[0]}，在弛豫结构中为全局索引 {target_atom_global_index})。 "
                f"最近的表面原子: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index})。 "
                f"最终距离: {round(min_distance, 3)} Å. "
                f"估计共价键阈值: {round(bonding_cutoff, 3)} Å. "
                f"是否成键: {is_bound}."
            )

            result = {
                "status": "success",
                "message": analysis_message,
                "most_stable_energy_eV": min_energy,
                "target_adsorbate_atom": target_atom_symbol,
                "target_adsorbate_atom_index": int(target_atom_global_index),
                "nearest_slab_atom": nearest_slab_atom_symbol,
                "nearest_slab_atom_index": int(nearest_slab_atom_global_index),
                "final_bond_distance_A": round(min_distance, 3),
                "estimated_covalent_cutoff_A": round(bonding_cutoff, 3),
                "is_covalently_bound": bool(is_bound)
            }
        
        elif orientation == "side-on":
            if len(adsorbate_indices) < 2:
                 return json.dumps({"status": "error", "message": f"Side-on 模式需要至少 2 个吸附物原子，但只找到 {len(adsorbate_indices)} 个。"})
            
            # 目标原子 *总是* 吸附物列表中的前两个
            
            # --- 分析第一个原子 (Atom 0) ---
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position
            print(f"--- 分析: (side-on 模式) 正在检查第一个吸附物原子, 符号: '{target_atom_symbol}', 全局索引: {target_atom_global_index}。---")

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
                f"最稳定的构型能量: {min_energy:.4f} eV。 "
                f"目标原子 1: {target_atom_symbol} (来自规划索引 {binding_atom_indices[0]}，全局索引 {target_atom_global_index})。 "
                f"  -> 最近: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}), 距离: {round(min_distance, 3)} Å (阈值: {round(bonding_cutoff, 3)}), 成键: {is_bound_1}。 "
                f"目标原子 2: {second_atom_symbol} (来自规划索引 {binding_atom_indices[1]}，全局索引 {second_atom_global_index})。 "
                f"  -> 最近: {nearest_slab_atom_symbol_2} (Index {nearest_slab_atom_global_index_2}), 距离: {round(min_distance_2, 3)} Å (阈值: {round(bonding_cutoff_2, 3)}), 成键: {is_bound_2}。 "
                f"整体是否成键: {is_bound}."
            )
            
            result = {
                "status": "success",
                "message": analysis_message,
                "most_stable_energy_eV": min_energy,
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
                }
            }

        else:
             return json.dumps({"status": "error", "message": f"分析失败：未知的朝向 '{orientation}'。"})

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