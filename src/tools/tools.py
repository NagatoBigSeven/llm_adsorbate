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

def generate_surrogate_smiles(original_smiles: str, binding_atoms: list, orientation: str) -> str:
    print(f"--- 🔬 调用 SMILES 翻译器: {original_smiles} via {binding_atoms} ---")
    mol = Chem.MolFromSmiles(original_smiles)
    if not mol:
        raise ValueError(f"RDKit 无法解析原始 SMILES: {original_smiles}")
    rw_mol = Chem.RWMol(mol)
    target_atom_symbol = binding_atoms[0]
    target_atom_idx = -1
    for atom in rw_mol.GetAtoms():
        if atom.GetSymbol() == target_atom_symbol:
            target_atom_idx = atom.GetIdx()
            break
    if target_atom_idx == -1:
        raise ValueError(f"在 {original_smiles} 中未找到要键合的原子: {target_atom_symbol}")
    target_atom = rw_mol.GetAtomWithIdx(target_atom_idx)
    if orientation == "end-on":
        surrogate_atom = Chem.Atom("Cl")
        surrogate_atom.SetProp("is_surrogate", "True")
        surrogate_idx = rw_mol.AddAtom(surrogate_atom)
        if target_atom.GetSymbol() == 'C':
            for neighbor in target_atom.GetNeighbors():
                bond = rw_mol.GetBondBetweenAtoms(target_atom_idx, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    bond.SetBondType(Chem.BondType.SINGLE)
                    print(f"--- 🔬 SMILES 翻译器: 在 {target_atom_idx} 处断开双键以保持价态。 ---")
                    break 
        rw_mol.AddBond(surrogate_idx, target_atom_idx, Chem.BondType.SINGLE)
        if original_smiles == "ClC(=O)[O-]" and target_atom_symbol == "C":
            final_smi = "Cl[C](Cl)(O)[O-]"
        else:
            final_smi = Chem.MolToSmiles(rw_mol.GetMol(), rootedAtAtom=surrogate_idx)
    elif orientation == "side-on":
        if original_smiles == "NNH" and "N" in binding_atoms:
            final_smi = "S1[N]N1"
        else:
            raise NotImplementedError("Side-on SMILES 翻译器尚未实现")
    else:
        raise ValueError(f"未知的朝向: {orientation}")
    print(f"--- 🔬 SMILES 翻译器输出: {final_smi} ---")
    return final_smi

# --- 其他工具 ---
def read_atoms_object(path: str):
    """Reads a atomistic structure file 
    Args:
        path: string - location on system
    returns:
        ase.Atoms object
    """
    return read(path)

def get_sites_from_atoms(atoms: ase.Atoms): 
    """Get all possible binding sites from atoms of a slab.
    Args:
        atoms: ase.Atoms object. Determines all surface sites. 
    Returns:
        pandas.DataFrame containing all site information.
        the columns of the returned dataframe are ['coordinates', 'connectivity', 'topology', 'n_vector', 'h_vector', 'site_formula'],
        - each index describes one site
        - 'coordinates': each entry is a list of 3 floats that locate the adsorption site in cartesian coordinates in angstrom;
        - 'connectivity': number of adjecent atoms, 1 - top site; 2 - bridge site; 3 - hollow site/3 fold site; etc.;
        - 'topology': list of ase.Atom.index from atoms that is directly adjecent to adsorption site;
        - 'n_vector': Unit vector pointing in the direction in with anutthyng attached to the site needs to be oriented;
        - 'h_vector': Unit vector describing the rotation around n_vector;
        - 'site_formula': dictionary indicating the composition of the site.
    """
    return Surface(atoms).site_df

def get_fragment(SMILES: str, to_initialize=1, conformer_i=0):
    """Generate a molecular fragment with conformations from a SMILES string.
    Args:
        SMILES: string of smiles that should be placed on surface sites.
        to_initialize: int = 1, if a SMILES is deamed to be conformationally complex. This number should be increased to deal with the increased complexity; in this case multiple fragment conformation should be tried.
        conformer_i: int, index of initialized conformer to be returned
    returns:
        ase.Atoms of molecule or molecular fragment, alligned relative to the site in [0,0,0]
    """
    print(f"--- 🛠️ get_fragment: 尝试从 SMILES 构建: {SMILES} ---")
    frag_mol_obj = Fragment(SMILES, to_initialize=to_initialize) 
    conformer = frag_mol_obj.get_conformer(conformer_i)
    if conformer is None:
        raise ValueError(f"RDKit/AutoAdsorbate failed to parse the SMILES: '{SMILES}'. It is likely syntactically invalid or chemically impossible.")
    print(f"--- 🛠️ get_fragment: 成功 ---")
    return conformer

def get_ads_slab(slab_atoms: ase.Atoms, fragment_atoms: ase.Atoms, site_dict: dict, height: float = 1.5, n_rotation: float = 0.):
    """Placing a fragment on a slab at a selected site defined by `site_dict`
    Args:
        slab_atoms: ase.Atoms, atoms of slab that should host the fragment
        fragment_atoms: ase.Atoms, molecular fragment obtained from SMILES
        site_dict: dict, information about the selected site geometry
        n_rotation: float, rotation in degree around the site vector provided in site_dict['n_vector']. This can be used to rotate the fragment conformer, to avoid atoms being too close to each other.
        height: float, distance from site in angstroms
    returns:
        ase.Atoms of molecule placed on slab
    """
    ads_slab_atoms = attach_fragment(
        atoms=slab_atoms,
        site_dict=site_dict,
        fragment=fragment_atoms,
        n_rotation=n_rotation,
        height=height
    )
    if ads_slab_atoms is None:
        raise ValueError(f"AutoAdsorbate 'attach_fragment' 失败。 *SMILES 与 AutoAdsorbate 的硬编码规则不匹配。")
    return ads_slab_atoms

def relax_atoms(atoms: ase.Atoms, output_dir='./'):
    """Atomic energy miniization.
    Args:
        atoms: ase.Atoms, atoms that need to be relaxed
        output_dir: str, where to write relax trajectory file
    returns:
        relaxed_atoms: ase.Atoms, atoms of relaxed structure
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mace_calculator = mace_mp(model="medium", device=str(device), dispersion=False)

    relaxed_atoms = atoms.copy()
    relaxed_atoms.calc = mace_calculator
    relaxed_atoms.constraints = FixAtoms(indices=[atom.index for atom in relaxed_atoms if atom.position[2] < relaxed_atoms.cell[2][2] * .5])
    dyn = BFGS(relaxed_atoms, trajectory=os.path.join(output_dir, "relax.traj"), logfile="relax.log")
    dyn.run(fmax=0.01)
    return relaxed_atoms

def analyze_relaxation_results(
    plan: dict, 
    relaxed_xyz_path: str = './outputs/relaxed_ads_slab.xyz', 
    original_slab_path: str = './notebooks/cu_slab_211.xyz'
) -> str:
    try:
        relaxed_atoms = read_atoms_object(relaxed_xyz_path)
        original_slab = read_atoms_object(original_slab_path)
        
        z_cutoff = relaxed_atoms.cell[2][2] * .5
        slab_indices_relaxed = [atom.index for atom in relaxed_atoms if atom.position[2] < z_cutoff]
        fragment_indices_relaxed = [atom.index for atom in relaxed_atoms if atom.position[2] >= z_cutoff]
        
        if not fragment_indices_relaxed:
            return json.dumps({"status": "error", "message": "分析失败：在弛豫后的结构中未找到片段原子。"})
            
        target_atom_symbol = plan["solution"]["adsorbate_binding_atoms"][0]
        
        target_atom_global_index = -1
        fragment_atoms_obj = relaxed_atoms[fragment_indices_relaxed]
        
        for i, atom in enumerate(fragment_atoms_obj):
            if atom.symbol == target_atom_symbol:
                target_atom_global_index = fragment_indices_relaxed[i]
                break
        
        if target_atom_global_index == -1:
            return json.dumps({"status": "error", "message": f"分析失败：在片段中未找到规划的键合原子 '{target_atom_symbol}'。"})

        target_atom_pos = relaxed_atoms[target_atom_global_index].position

        slab_atoms_relaxed = relaxed_atoms[slab_indices_relaxed]
        
        distances = np.linalg.norm(slab_atoms_relaxed.positions - target_atom_pos, axis=1)
        min_distance = np.min(distances)
        nearest_slab_atom_index_local = np.argmin(distances)
        nearest_slab_atom_global_index = slab_indices_relaxed[nearest_slab_atom_index_local]
        nearest_slab_atom_symbol = relaxed_atoms[nearest_slab_atom_global_index].symbol

        cov_cutoffs = natural_cutoffs(relaxed_atoms)
        radius_1 = cov_cutoffs[target_atom_global_index]
        radius_2 = cov_cutoffs[nearest_slab_atom_global_index]
        bonding_cutoff = (radius_1 + radius_2) * 1.1
        
        is_bound = min_distance <= bonding_cutoff

        result = {
            "status": "success",
            "message": "分析完成。",
            "target_adsorbate_atom": target_atom_symbol,
            "target_adsorbate_atom_index": target_atom_global_index,
            "nearest_slab_atom": nearest_slab_atom_symbol,
            "nearest_slab_atom_index": nearest_slab_atom_global_index,
            "final_bond_distance_A": round(min_distance, 3),
            "estimated_covalent_cutoff_A": round(bonding_cutoff, 3),
            "is_covalently_bound": bool(is_bound)
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"分析工具执行时出错: {str(e)}"})
    
def save_ase_atoms(atoms: ase.Atoms, filename):
    """ this functions writes ase.atoms to xyz file
    Args:
        atoms: ase.Atoms tobject to be written
        filename: string of where to write ase atoms. must end in '.xyz'
    """
    write(filename, atoms)