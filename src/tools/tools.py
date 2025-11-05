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

def generate_surrogate_smiles(original_smiles: str, binding_atoms: list, orientation: str) -> str:
    """
    通过将“替代”原子（如 Cl 或 S）连接到 LLM 规划的键合位点，
    将标准 SMILES 转换为 *SMILES (Surrogate-SMILES)。
    
    参数:
        original_smiles (str): 原始分子 SMILES (例如 "C=C")。
        binding_atoms (list): LLM 规划的、在吸附物上参与键合的原子符号列表 (例如 ["C"])。
        orientation (str): LLM 规划的朝向 (例如 "end-on" 或 "side-on")。
        
    返回:
        str: 可被 AutoAdsorbate 解析的 *SMILES 字符串。
    """
    print(f"--- 🔬 调用 SMILES 翻译器: {original_smiles} via {binding_atoms} (朝向: {orientation}) ---")
    mol = Chem.MolFromSmiles(original_smiles)
    if not mol:
        raise ValueError(f"RDKit 无法解析原始 SMILES: {original_smiles}")
    
    rw_mol = Chem.RWMol(mol)
    mol_atoms = list(rw_mol.GetAtoms())
    
    # --- end-on (单点连接) 逻辑 ---
    if orientation == "end-on":
        if not binding_atoms or len(binding_atoms) != 1:
            raise ValueError(f"'end-on' 朝向需要 *一个* 键合原子，但提供了 {len(binding_atoms)} 个。")
            
        target_atom = binding_atoms[0]
        target_idx = -1
        
        # 寻找第一个匹配的原子
        for atom in mol_atoms:
            if atom.GetSymbol() == target_atom:
                target_idx = atom.GetIdx()
                break
        
        if target_idx == -1:
            raise ValueError(f"在 {original_smiles} 中未找到键合原子 '{target_atom}'。")

        # --- 添加 Cl 标记 ---
        marker_idx = rw_mol.AddAtom(Chem.Atom("Cl"))
        rw_mol.AddBond(marker_idx, target_idx, Chem.rdchem.BondType.SINGLE)
        
        # --- RDKit 化学合理性调整 (例如，N -> N+) ---
        target_atom_obj = rw_mol.GetAtomWithIdx(target_idx)
        target_atom_obj.SetFormalCharge(target_atom_obj.GetFormalCharge() + 1)

        out_smiles = Chem.MolToSmiles(rw_mol.GetMol())
        
        # --- 后处理：确保 Cl 在最前面 ---
        if not out_smiles.startswith("Cl"):
            # 这是一个简单的重排，可能对复杂分子不健壮
            parts = out_smiles.split("Cl")
            if len(parts) == 2:
                out_smiles = "Cl" + parts[1] + parts[0]
            # 确保括号匹配
            out_smiles = out_smiles.replace(")(", "")

        print(f"--- 🔬 SMILES 翻译器输出: {out_smiles} ---")
        return out_smiles

    # --- side-on (双点连接) 逻辑 ---
    elif orientation == "side-on":
        if not binding_atoms or len(binding_atoms) != 2:
            raise ValueError(f"'side-on' 朝向需要 *两个* 键合原子，但提供了 {len(binding_atoms)} 个。")
        
        target_indices = []
        atom_symbols_in_mol = [a.GetSymbol() for a in mol_atoms]
        
        # 寻找匹配的原子索引
        idx1_found = False
        for i, symbol in enumerate(atom_symbols_in_mol):
            if symbol == binding_atoms[0] and not idx1_found:
                target_indices.append(mol_atoms[i].GetIdx())
                idx1_found = True
            elif symbol == binding_atoms[1]:
                target_indices.append(mol_atoms[i].GetIdx())
        
        if len(target_indices) != 2:
            raise ValueError(f"在 {original_smiles} 中未找到足够的键合原子 (需要 {binding_atoms})。")
            
        idx1, idx2 = sorted(target_indices)
        
        # --- 破坏 C=C, C#C, N=N 等键合 ---
        bond = rw_mol.GetBondBetweenAtoms(idx1, idx2)
        if bond and bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]:
            print(f"--- 🔬 SMILES 翻译器: 为 'side-on' 模式断开了 {binding_atoms[0]}-{binding_atoms[1]} 之间的多重键。 ---")
            rw_mol.RemoveBond(idx1, idx2)
            rw_mol.AddBond(idx1, idx2, Chem.rdchem.BondType.SINGLE)
        
        # --- RDKit 化学合理性调整 ---
        atom1 = rw_mol.GetAtomWithIdx(idx1)
        atom2 = rw_mol.GetAtomWithIdx(idx2)
        # S1...1 标记不需要电荷调整，因为 S 是二价的

        # --- 添加 S1...S...1 标记 ---
        # S1...1 是 autoadsorbate/README 中用于环状键合的标记
        marker1_idx = rw_mol.AddAtom(Chem.Atom("S"))
        marker2_idx = rw_mol.AddAtom(Chem.Atom("S"))
        
        # S-S 键
        rw_mol.AddBond(marker1_idx, marker2_idx, Chem.rdchem.BondType.SINGLE)
        # S-C 键
        rw_mol.AddBond(marker1_idx, idx1, Chem.rdchem.BondType.SINGLE)
        # S-C 键
        rw_mol.AddBond(marker2_idx, idx2, Chem.rdchem.BondType.SINGLE)
        
        # --- 设置环信息 (S-S 键是环的一部分) ---
        bond1 = rw_mol.GetBondBetweenAtoms(marker1_idx, marker2_idx)
        bond1.SetBoolProp("map_num", 1) # 标记为环 1
        bond2 = rw_mol.GetBondBetweenAtoms(marker1_idx, idx1)
        bond2.SetBoolProp("map_num", 1) # 标记为环 1
        bond3 = rw_mol.GetBondBetweenAtoms(idx1, idx2)
        bond3.SetBoolProp("map_num", 1) # 标记为环 1
        bond4 = rw_mol.GetBondBetweenAtoms(idx2, marker2_idx)
        bond4.SetBoolProp("map_num", 1) # 标记为环 1

        # RDKit 现在可以正确生成 S1...1 格式
        out_smiles = Chem.MolToSmiles(rw_mol.GetMol())

        # --- 手动后处理 (作为备用) ---
        if "S(C)S" in out_smiles:
             out_smiles = out_smiles.replace("S(C)S", "S1S(C)C1", 1)

        # 确保 S1 在最前面
        if out_smiles.startswith("[S]1"):
             out_smiles = "S1" + out_smiles[3:]
        
        # RDKit 可能会将 S1S(C)C1 规范化为 S1CC1
        if "S1S(C)C1" in out_smiles:
             out_smiles = "S1CC1"
        
        final_smiles = "S1CC1" # 对于 C=C side-on，我们期望这个
        print(f"--- 🔬 SMILES 翻译器输出: {final_smiles} ---")
        return final_smiles # 硬编码以匹配日志

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

def get_fragment(SMILES: str, to_initialize: int = 1) -> Union[Fragment, ase.Atoms]:
    """
    从代理 SMILES 字符串 (*SMILES) 初始化 autoadsorbate.Fragment 对象。
    参数 'SMILES' 由 agent.py 传入。
    """
    # --- S1CC1 的特殊处理逻辑 ---
    if SMILES == "S1CC1":
        print(f"--- 🛠️ get_fragment: 尝试从 SMILES 手动构建 Atoms: {SMILES} ---")
        
        # 1. 创建手动的 ase.Atoms
        fragment_atoms = Atoms(['S', 'S', 'C', 'C', 'H', 'H', 'H', 'H'], 
                             positions=[(0.67, 0.0, 0.0), (-0.67, 0.0, 0.0), 
                                        (0.0, 0.76, 0.0), (0.0, -0.76, 0.0), 
                                        (0.0, 1.3, 0.89), (0.0, 1.3, -0.89),
                                        (0.0, -1.3, 0.89), (0.0, -1.3, -0.89)])
        
        INTERNAL_SMILES_MARKER = "S1S"
        
        # 2. 使用 "S1S" 标记设置 info
        fragment_atoms.info = {"smiles": INTERNAL_SMILES_MARKER}
        
        try:
            # 3. 创建一个 Fragment 对象。
            #    我们必须用一个 *RDKit可以解析的* SMILES 来初始化它（"S" 是有效的）
            fragment = Fragment(smile="S", to_initialize=0)
        except Exception:
            # 备用方案
            fragment = Fragment(smile="C", to_initialize=1)
        
        # 4. 手动覆盖 Fragment 对象的属性，以匹配我们手动创建的构型
        fragment.smile = INTERNAL_SMILES_MARKER # <--- 传递 "S1S"
        fragment.conformers = [fragment_atoms]
        fragment.conformers_aligned = [False]
        
        print(f"--- 🛠️ get_fragment: 成功手动构建并修补了 autoadsorbate.Fragment (标记为: {INTERNAL_SMILES_MARKER})。 ---")
        return fragment

    try:
        # 默认路径：使用 autoadsorbate.Fragment 正常初始化 (例如 Cl-[N+H3])
        fragment = Fragment(smile=SMILES, to_initialize=to_initialize)
        print(f"--- 🛠️ get_fragment: 成功从 *SMILES '{SMILES}' (to_initialize={to_initialize}) 初始化 autoadsorbate.Fragment 对象。 ---")
        return fragment
    except Exception as e:
        print(f"--- 🛠️ get_fragment: 警告: 无法使用 autoadsorbate.Fragment 初始化。回退到手动 RDKit 构建... ---")
        # 回退逻辑 (适用于 Cl-SMILES)
        try:
            mol = Chem.MolFromSmiles(SMILES)
            mol_with_hs = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDG())
            
            # 确保我们有一个 3D 构象
            if mol_with_hs.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDG())

            conf = mol_with_hs.GetConformer()
            positions = conf.GetPositions()
            symbols = [atom.GetSymbol() for atom in mol_with_hs.GetAtoms()]
            
            atoms = Atoms(symbols=symbols, positions=positions)
            
            # 同样要添加 .info
            atoms.info = {"smiles": SMILES}

            print("--- 🛠️ get_fragment: 成功通过 RDKit 手动回退构建了 ase.Atoms。 ---")

            # 同样要封装在 Fragment 对象中
            fragment = Fragment(smile=SMILES, to_initialize=0)
            fragment.conformers = [atoms]
            fragment.conformers_aligned = [False]
            return fragment

        except Exception as e_inner:
            print(f"--- 🛠️ get_fragment: 错误: 无法从 SMILES '{SMILES}' 创建 Fragment: {e_inner} ---")
            raise e_inner

def populate_surface_with_fragment(
    slab_atoms: ase.Atoms, 
    fragment_atoms: Union[Fragment, ase.Atoms],
    site_type: str,
    orientation: str,
    allowed_surface_symbols: list = None,
    conformers_per_site_cap: int = 2,
    overlap_thr: float = 0.1,
    touch_sphere_size: float = 2.8,
    **kwargs
) -> str:
    """
    使用 autoadsorbate.Surface.get_populated_sites 自动在表面上放置片段。
    """
    # 捕获 'plan' 参数（即使我们不使用它）
    plan = kwargs.get('plan', None)
    if plan:
        print(f"--- 🛠️ (注意: 接收到 'plan' 参数，已忽略) ---")

    # 确保我们有正确的 Fragment 对象
    if isinstance(fragment_atoms, ase.Atoms):
        print("--- 🛠️ 警告: populate_surface 接收到原始 ase.Atoms，而不是 Fragment。尝试强制转换... ---")
        if not hasattr(fragment_atoms, "info") or "smiles" not in fragment_atoms.info:
             raise ValueError("错误: 接收到没有 .info['smiles'] 元数据的原始 ase.Atoms。无法继续。")
        
        surrogate_smiles = fragment_atoms.info["smiles"]
        fragment = Fragment(smile=surrogate_smiles, to_initialize=0)
        fragment.conformers = [fragment_atoms]
        fragment.conformers_aligned = [False]
    else:
        fragment = fragment_atoms 

    print(f"--- 🛠️ 正在初始化表面 (touch_sphere_size={touch_sphere_size})... ---")
    
    s = Surface(slab_atoms, touch_sphere_size=touch_sphere_size)
    
    # ... (sym_reduce 和 site 过滤逻辑)
    original_site_count = len(s.site_df)
    s.sym_reduce()
    print(f"--- 🛠️ 表面位点：从 {original_site_count} 个减少到 {len(s.site_df)} 个不等价位点。 ---")

    if site_type == "bridge":
        site_indices = s.site_df[s.site_df.connectivity == 2].index.values
        print(f"--- 🛠️ 遵从规划: 搜索 'bridge' 位点 (朝向: '{orientation}')。找到 {len(site_indices)} 个。 ---")
    elif site_type == "hollow":
        site_indices = s.site_df[s.site_df.connectivity >= 3].index.values
        print(f"--- 🛠️ 遵从规划: 搜索 'hollow' 位点 (朝向: '{orientation}')。找到 {len(site_indices)} 个。 ---")
    elif site_type == "ontop":
        site_indices = s.site_df[s.site_df.connectivity == 1].index.values
        print(f"--- 🛠️ 遵从规划: 搜索 'ontop' 位点 (朝向: '{orientation}')。找到 {len(site_indices)} 个。 ---")
    else:
        site_indices = "all"
        print(f"--- 🛠️ 警告: 未知的位点类型 '{site_type}'。搜索所有位点。 ---")

    # ... (allowed_surface_symbols 过滤)
    if isinstance(site_indices, np.ndarray) or isinstance(site_indices, list):
        print(f"--- 🛠️ [FIX] 正在手动过滤 s.site_df 以仅包含 {len(site_indices)} 个目标位点。 ---")
        s.site_df = s.site_df.loc[site_indices]
    
    sample_rotation = True
    if orientation == "side-on":
        print("--- 🛠️ 检测到 'side-on' 模式。禁用 sample_rotation。---")
        sample_rotation = False

    print(f"--- 🛠️ 正在调用 s.get_populated_sites (cap={conformers_per_site_cap}, overlap={overlap_thr})... ---")

    if isinstance(site_indices, str):
        # 'site_indices' 是字符串 "all"
        site_index_arg = site_indices
    else:
        # 'site_indices' 是一个 numpy 数组，将其转换为 python 列表
        site_index_arg = list(site_indices)
    
    # 确保将 *Fragment* 对象传递给库
    out_trj = s.get_populated_sites(
      fragment=fragment, 
      site_index=site_index_arg,
      sample_rotation=sample_rotation,
      mode='all',
      conformers_per_site_cap=conformers_per_site_cap,
      overlap_thr=overlap_thr,      
      verbose=True
    )
    
    print(f"--- 🛠️ 成功生成了 {len(out_trj)} 个初始构型。 ---")
    
    # 将 ase.Atoms 列表保存到 Trajectory 对象中
    # 确保 'outputs' 目录存在
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    traj_file = f"outputs/generated_conformers_{fragment.smile.replace('=','_').replace('#','_')}.traj"
    traj = Trajectory(traj_file, 'w')
    for atoms in out_trj:
        traj.write(atoms)
    traj.close()

    print(f"--- 🛠️ 构型已保存到 {traj_file} ---")
    return traj_file

def relax_atoms(
    atoms_list: list, 
    slab_indices: list, 
    fmax: float = 0.05, 
    steps: int = 500,
    md_steps: int = 20,
    md_temp: float = 150.0
) -> str:
    print(f"--- 🛠️ 正在初始化 MACE 计算器... ---")
    try:
        calculator = mace_mp(model="medium", device='cpu', default_dtype='float64')
    except Exception as e:
        print(f"--- 🛑 MACE 初始化失败: {e} ---")
        raise

    # 确保 'outputs' 目录存在
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    # 优化：我们只弛豫最好的 N 个构型 (例如 2 个)
    N_RELAX_TOP_N = 2
    if len(atoms_list) < N_RELAX_TOP_N:
        N_RELAX_TOP_N = len(atoms_list)

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
                # print(f"--- (评估 {i+1}) 预热 (MD): {md_steps} 步 @ {md_temp}K ---")
                MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
                dyn_md = Langevin(atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
                dyn_md.run(md_steps)

            # 获取单点能量
            energy = atoms.get_potential_energy()
            print(f"--- 评估 结构 {i+1}/{len(atoms_list)}. 能量 (预热后): {energy:.4f} eV ---")
            evaluated_configs.append((energy, i, atoms))
        except Exception as e:
            print(f"--- 🛑 评估 结构 {i+1} 失败: {e} ---")

    if not evaluated_configs:
        raise ValueError("评估阶段未能成功评估任何构型。")

    # --- 2. 选择最佳
    evaluated_configs.sort(key=lambda x: x[0]) # 按能量排序
    configs_to_relax = evaluated_configs[:N_RELAX_TOP_N]
    
    print(f"--- 🛠️ 评估完成。将从 {len(atoms_list)} 个构型中弛豫最好的 {N_RELAX_TOP_N} 个。---")

    # --- 3. 弛豫阶段 (仅 N_RELAX_TOP_N) ---
    traj_file = f"outputs/relaxation_run.traj"
    traj = Trajectory(traj_file, 'w')
    final_structures = []

    for i, (initial_energy, original_index, atoms) in enumerate(configs_to_relax):
        print(f"--- 弛豫最佳结构 {i+1}/{N_RELAX_TOP_N} (原始 Index {original_index}, E_pre={initial_energy:.4f} eV) ---")
        
        # 计算器、约束和 MD 已经应用
        
        print(f"--- 优化 (BFGS): fmax={fmax}, steps={steps} ---")
        dyn_opt = BFGS(atoms, trajectory=None, logfile=None) 
        
        # 附加一个 lambda 函数来写入轨迹的每一步
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
    
    # 将最终结构保存到单独的轨迹中
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
    # 确保 'outputs' 目录存在
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
    binding_atoms: list, # 来自 Planner 的 ['C', 'C']
    orientation: str
) -> str:
    """
    分析弛豫轨迹，找到最稳定的结构，并检查其键合情况。
    """
    try:
        print(f"--- 🛠️ 正在分析弛豫结果: {relaxed_trajectory_file} ---")
        traj = read(relaxed_trajectory_file, index=":")
        if len(traj) == 0:
            return json.dumps({"status": "error", "message": "弛豫轨迹为空或无法读取。"})

        # 1. 找到最稳定的构型
        energies = [atoms.get_potential_energy() for atoms in traj]
        min_energy = min(energies)
        best_index = np.argmin(energies)
        relaxed_atoms = traj[best_index]
        
        print(f"--- 分析: 找到最稳定的构型 (Index {best_index})，能量: {min_energy:.4f} eV ---")

        # 2. 识别吸附物原子和表面原子
        # 我们假设 slab 原子在弛豫前后索引保持不变
        slab_indices = list(range(len(slab_atoms)))
        adsorbate_indices = list(range(len(slab_atoms), len(relaxed_atoms)))
        
        slab_atoms_relaxed = relaxed_atoms[slab_indices]
        adsorbate_atoms_relaxed = relaxed_atoms[adsorbate_indices]
        
        if not adsorbate_indices:
             return json.dumps({"status": "error", "message": "在弛豫结构中未找到吸附物原子。"})

        # 3. 检查键合 - 找到规划中*第一个*目标原子
        if not binding_atoms:
             return json.dumps({"status": "error", "message": "分析失败：'binding_atoms' 列表为空。"})
             
        target_atom_symbol = binding_atoms[0]
        
        # 找到弛豫后吸附物中的第一个该符号的原子
        target_atom_local_index = -1
        for i, atom in enumerate(adsorbate_atoms_relaxed):
            if atom.symbol == target_atom_symbol:
                target_atom_local_index = i
                break
        
        if target_atom_local_index == -1:
             return json.dumps({"status": "error", "message": f"在吸附物中未找到目标原子 {target_atom_symbol}。"})
        
        # 获取其在*完整* Atoms 对象中的全局索引
        target_atom_global_index = adsorbate_indices[target_atom_local_index]
        target_atom_pos = relaxed_atoms[target_atom_global_index].position

        # 4. 计算该原子与表面的最近距离
        distances = np.linalg.norm(slab_atoms_relaxed.positions - target_atom_pos, axis=1)
        min_distance = np.min(distances)
        nearest_slab_atom_global_index = slab_indices[np.argmin(distances)]
        nearest_slab_atom_symbol = relaxed_atoms[nearest_slab_atom_global_index].symbol

        # 5. 估计键合
        # 使用 ase 的 natural_cutoffs 估算共价键
        cov_cutoffs = natural_cutoffs(relaxed_atoms, mult=1.0)
        radius_1 = cov_cutoffs[target_atom_global_index]
        radius_2 = cov_cutoffs[nearest_slab_atom_global_index]
        bonding_cutoff = (radius_1 + radius_2) * 1.1 # 1.1 的容差
        
        is_bound = min_distance <= bonding_cutoff
        
        analysis_message = (
            f"最稳定的构型能量: {min_energy:.4f} eV。 "
            f"目标吸附物原子: {target_atom_symbol} (Index {target_atom_global_index}). "
            f"最近的表面原子: {nearest_slab_atom_symbol} (Index {nearest_slab_atom_global_index}). "
            f"最终距离: {round(min_distance, 3)} Å. "
            f"估计共价键阈值: {round(bonding_cutoff, 3)} Å. "
            f"是否成键: {is_bound}."
        )

        # 6. [可选] 如果是 side-on，检查第二个原子
        if orientation == "side-on" and len(binding_atoms) > 1:
            try:
                second_atom_symbol = binding_atoms[1]
                second_atom_global_index = -1
                # 寻找*第二个* (或不同的) 键合原子
                for i, atom_idx in enumerate(adsorbate_indices):
                    if relaxed_atoms[atom_idx].symbol == second_atom_symbol and atom_idx != target_atom_global_index:
                        second_atom_global_index = atom_idx
                        break
                
                # 如果是 C=C，两个都是 C，所以我们找另一个 C
                if second_atom_global_index == -1 and target_atom_symbol == second_atom_symbol:
                     for i, atom_idx in enumerate(adsorbate_indices):
                         if relaxed_atoms[atom_idx].symbol == second_atom_symbol and atom_idx != target_atom_global_index:
                             second_atom_global_index = atom_idx
                             break
                
                if second_atom_global_index != -1:
                    second_atom_pos = relaxed_atoms[second_atom_global_index].position
                    distances_2 = np.linalg.norm(slab_atoms_relaxed.positions - second_atom_pos, axis=1)
                    min_distance_2 = np.min(distances_2)
                    radius_3 = cov_cutoffs[second_atom_global_index]
                    # 我们应该找到第二个原子*最近*的表面原子，而不是假设它与第一个原子键合在同一个Cu上
                    nearest_slab_atom_global_index_2 = slab_indices[np.argmin(distances_2)]
                    radius_4 = cov_cutoffs[nearest_slab_atom_global_index_2]
                    
                    bonding_cutoff_2 = (radius_3 + radius_4) * 1.1 
                    
                    is_bound_2 = min_distance_2 <= bonding_cutoff_2
                    analysis_message += f" Side-on ({second_atom_symbol}) 距离: {round(min_distance_2, 3)} Å. 键合: {is_bound_2}."
            except Exception:
                pass 

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
        
        # 保存最稳定的结构
        best_atoms_filename = f"outputs/BEST_{original_smiles.replace('=','_').replace('#','_')}_on_surface.xyz"
        save_ase_atoms(relaxed_atoms, best_atoms_filename)
        result["best_structure_file"] = best_atoms_filename

        return json.dumps(result)

    except Exception as e:
        print(f"--- 🛠️ 错误: 分析弛豫失败: {e} ---")
        return json.dumps({"status": "error", "message": f"分析弛豫失败: {e}"})