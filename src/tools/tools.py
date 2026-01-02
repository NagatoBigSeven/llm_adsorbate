import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist
import autoadsorbate.Surf
from src.utils.logger import get_logger
from enum import Enum

# Initialize logger for this module
logger = get_logger(__name__)

# Ensure original module is imported first so we can override it

# ========================================
# Physical and Chemical Constants
# ========================================

# Grid and site detection parameters
Z_AXIS_PENETRATION_LIMIT = -1.0  # Angstroms - prevents infinite loop in shrinkwrap algorithm
EPSILON_TOLERANCE = 0.3  # Tolerance for identifying contact points
DEFAULT_PRECISION = 0.25  # Angstroms - grid precision (improved from 0.5)

# Collision detection and geometry optimization
COLLISION_SAFETY_BUFFER_ANGSTROM = 0.2  # Extra safety margin to prevent edge-case collisions
MIN_COLLISION_THRESHOLD_ANGSTROM = 1.6  # Minimum allowed distance between adsorbate and surface
PRE_LIFT_HEIGHT_ANGSTROM = 0.5  # Pre-lift height for bridge/hollow sites to reduce collisions

# Bonding criteria for chemical analysis
BASE_BOND_MULTIPLIER = 1.30  # Standard tolerance multiplier for covalent radii
STRONG_ADSORPTION_BOND_MULTIPLIER = 1.45  # Relaxed multiplier for strong chemisorption
STRONG_ADSORPTION_THRESHOLD_EV = -0.5  # Energy threshold for strong adsorption detection

# Subsurface layer detection for crystallographic analysis
SUBSURFACE_LOWER_BOUND_ANGSTROM = 1.2  # Minimum depth below surface for subsurface layer
SUBSURFACE_UPPER_BOUND_ANGSTROM = 4.0  # Maximum depth below surface for subsurface layer
HCP_DETECTION_RADIUS_ANGSTROM = 1.0  # XY distance threshold for HCP site detection

# Slab expansion criteria
MIN_CELL_SIZE_ANGSTROM = 6.0  # Minimum cell dimension before triggering 2x2 expansion

# Surface relaxation settings
FIXED_BOTTOM_FRACTION = 1.0 / 3.0  # Fraction of slab (by Z) to keep fixed in STANDARD mode


class RelaxationMode(Enum):
    """
    Surface relaxation mode for adsorption calculations.
    
    FAST: All surface atoms fixed. Fastest option, suitable for rapid screening 
          on personal computers (e.g., MacBook). Default mode.
    STANDARD: Bottom 1/3 of slab fixed (by Z coordinate), top 2/3 + adsorbate relaxed.
              Better accuracy for workstations/servers.
    """
    FAST = "fast"
    STANDARD = "standard"


def get_shrinkwrap_grid_fixed(
    slab,
    precision,
    drop_increment=0.1,
    touch_sphere_size=2,
    marker="He",
    raster_speed_boost=False,
):
    """
    Fixed version of autoadsorbate's get_shrinkwrap_grid function.
    
    This fixes an infinite loop bug where grid points could fall indefinitely
    through surface voids. The fix adds a Z-axis lower bound check to prevent
    grid points from penetrating below Z = -1.0 Angstroms.
    
    Args:
        slab: ASE Atoms object representing the surface slab
        precision: Grid spacing in Angstroms
        drop_increment: Step size for dropping grid points (default: 0.1)
        touch_sphere_size: Radius for determining surface contact (default: 2)
        marker: Element symbol for grid markers (default: "He")
        raster_speed_boost: Enable rasterization optimization (default: False)
        
    Returns:
        tuple: (grid, faces) where grid is an ASE Atoms object with marker atoms
               at adsorption site positions, and faces is the surface triangulation
    """
    from autoadsorbate.Surf import _get_starting_grid, get_large_atoms
    

    if raster_speed_boost:
        from autoadsorbate.raster_utilities import get_surface_from_rasterized_top_view
        raster_surf_index = get_surface_from_rasterized_top_view(
            slab, pixel_per_angstrom=10
        )
        slab = slab[raster_surf_index]


    starting_grid, faces = _get_starting_grid(slab, precision=precision)
    grid_positions = starting_grid.positions
    large_slab = get_large_atoms(slab)
    slab_positions = large_slab.positions

    distances_to_grid = cdist(grid_positions, slab_positions).min(axis=1)
    drop_vectors = np.array([[0, 0, drop_increment] for _ in grid_positions])

    # Critical fix: Add Z-axis lower bound to prevent infinite loop
    # Original buggy code: while (distances_to_grid > touch_sphere_size).any()
    # Fixed: Only move points if they are both (1) far from surface AND (2) above Z = -1.0
    while ((distances_to_grid > touch_sphere_size) & (grid_positions[:, 2] > Z_AXIS_PENETRATION_LIMIT)).any():
        mask_to_move = (distances_to_grid > touch_sphere_size) & (grid_positions[:, 2] > Z_AXIS_PENETRATION_LIMIT)
        grid_positions -= (drop_vectors * mask_to_move[:, np.newaxis])
        distances_to_grid = cdist(grid_positions, slab_positions).min(axis=1)

        # Keep original exit condition as additional safeguard
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
    # Filter out points that penetrated below the surface (Z < 0)
    grid = grid[[atom.index for atom in grid if atom.position[2] > 0]]

    return grid, faces

def get_shrinkwrap_ads_sites_fixed(
    atoms: Atoms,
    precision: float = DEFAULT_PRECISION,
    touch_sphere_size: float = 2,
    return_trj: bool = False,
    return_geometry = False
):
    """
    Fixed version of autoadsorbate's get_shrinkwrap_ads_sites function.
    
    This improves upon the original by:
    1. Increasing default precision from 0.5 to 0.25 Angstroms
    2. Using epsilon=0.3 for contact point detection (improved from 0.1)
    3. Utilizing the fixed get_shrinkwrap_grid function
    
    Args:
        atoms: ASE Atoms object representing the surface
        precision: Grid spacing in Angstroms (default: 0.25)
        touch_sphere_size: Radius for contact detection (default: 2)
        return_trj: If True, return trajectory visualization (default: False)
        return_geometry: If True, return grid geometry (default: False)
        
    Returns:
        dict: Site information including coordinates, connectivity, topology,
              normal vectors, horizontal vectors, and site formulas
    """
    import numpy as np
    import itertools
    from ase import Atom
    from autoadsorbate.Surf import (
        get_shrinkwrap_grid,  # Automatically uses our monkey-patched version
        shrinkwrap_surface, 
        get_list_of_touching, 
        get_wrapped_site,
        get_shrinkwrap_site_n_vector,
        get_shrinkwrap_site_h_vector
    )

    grid, faces = get_shrinkwrap_grid(
        atoms, precision=precision, touch_sphere_size=touch_sphere_size
    )
    
    surf_ind = shrinkwrap_surface(
        atoms, precision=precision, touch_sphere_size=touch_sphere_size
    )
    
    # Critical improvement: Increased epsilon from 0.1 to 0.3 for better contact detection
    # This allows grid points to capture surrounding atoms even when slightly off-center
    targets = get_list_of_touching(atoms, grid, surf_ind, touch_sphere_size=touch_sphere_size, epsilon=EPSILON_TOLERANCE)


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

# ========================================
# Apply Autoadsorbate Monkey Patches
# ========================================

logger.info("Applying Autoadsorbate monkey patches for bug fixes")

# Patch both the source module and consumer namespace
autoadsorbate.Surf.get_shrinkwrap_grid = get_shrinkwrap_grid_fixed
autoadsorbate.Surf.get_shrinkwrap_ads_sites = get_shrinkwrap_ads_sites_fixed

import autoadsorbate.autoadsorbate
autoadsorbate.autoadsorbate.get_shrinkwrap_ads_sites = get_shrinkwrap_ads_sites_fixed

logger.info("Autoadsorbate patches applied successfully")

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
        logger.error(f"get_atom_index_menu failed: {e}")
        return json.dumps({"error": f"Unable to generate heavy atom index list: {e}"})

def generate_surrogate_smiles(original_smiles: str, binding_atom_indices: list[int], site_type: str) -> str:
    logger.info(f"Calling SMILES Translator: {original_smiles} via indices {binding_atom_indices} (Site: {site_type})")
    
    mol = Chem.MolFromSmiles(original_smiles)
    if not mol:
        raise ValueError(f"RDKit cannot parse original SMILES: {original_smiles}")
    
    num_binding_indices = len(binding_atom_indices)
    
    # --- Case A: end-on @ ontop (Single Point Adsorption) ---
    if site_type == "ontop":
        if num_binding_indices != 1:
            raise ValueError(f"'ontop' site requires 1 binding index, but got {num_binding_indices}.")
            
        target_idx = binding_atom_indices[0]
        
        if target_idx >= mol.GetNumAtoms():
             raise ValueError(f"Index {target_idx} out of range (Atom count: {mol.GetNumAtoms()}).")

        # 1. Capture original state (Prevent RDKit automatic deduction)
        target_atom_original = mol.GetAtomWithIdx(target_idx)
        original_h_count = target_atom_original.GetTotalNumHs()
        num_radicals = target_atom_original.GetNumRadicalElectrons()

        new_mol = Chem.RWMol(mol)

        # 2. Add Cl marker
        marker_atom = Chem.Atom("Cl")
        marker_atom.SetAtomMapNum(1) 
        marker_atom.SetIsotope(37)
        marker_idx = new_mol.AddAtom(marker_atom)
        
        # 3. Determine bond type based on electronic state
        if num_radicals > 0:
            logger.info(f"Smart Bonding: Radical detected (N={num_radicals}) -> Using Covalent Single Bond (SINGLE)")
            # Strategy: Radicals form covalent bonds, physically clear, geometrically stable
            new_mol.AddBond(marker_idx, target_idx, Chem.rdchem.BondType.SINGLE)
            
            # Fix: Remove radical marker, making it a saturated atom
            target_atom_obj = new_mol.GetAtomWithIdx(target_idx)
            target_atom_obj.SetNumRadicalElectrons(0)
            
        else:
            logger.info("Smart Bonding: Lone pair detected (Saturated/Double Bond) -> Using Dative Bond (DATIVE: Target->Surf)")
            # Strategy: Use dative bond connection.
            # Key Point 1: Direction must be Target Atom -> Marker Atom (Target Donates to Marker)
            # Key Point 2: No charge increase, no valence change. RDKit doesn't count Dative bond valence contribution, so C=O won't error.
            new_mol.AddBond(target_idx, marker_idx, Chem.rdchem.BondType.DATIVE)
            
            target_atom_obj = new_mol.GetAtomWithIdx(target_idx)

        # 4. [Safety Lock] Absolutely lock Hydrogen atoms
        # In any case, strictly forbid RDKit from automatically adding or removing Hydrogen atoms
        target_atom_obj.SetNumExplicitHs(original_h_count)
        target_atom_obj.SetNoImplicit(True)

        # 5. Marker Tracking
        target_atom_obj.SetAtomMapNum(114514)
        if target_atom_obj.GetSymbol() != 'H':
            target_atom_obj.SetIsotope(14) 

        # 6. Force Refresh
        try:
            # Catch errors just in case, but DATIVE + Neutral usually passes
            Chem.SanitizeMol(new_mol)
        except Exception as e:
            logger.warning(f"Sanitize Warning: {e}")

        out_smiles = Chem.MolToSmiles(new_mol.GetMol(), canonical=False, rootedAtAtom=marker_idx)
        logger.info(f"SMILES Translator Final Output: {out_smiles}")
        return out_smiles

    # --- Case B & C: bridge/hollow (Keep as is) ---
    elif site_type in ["bridge", "hollow"]:
        if num_binding_indices == 1:
            target_idx = binding_atom_indices[0]
            if target_idx >= mol.GetNumAtoms(): raise ValueError(f"Index {target_idx} out of range.")
            rw_mol = Chem.RWMol(mol)
            rw_mol.GetAtomWithIdx(target_idx).SetAtomMapNum(114514)
            original_smiles_mapped = Chem.MolToSmiles(rw_mol.GetMol(), canonical=False)
            out_smiles = f"{original_smiles_mapped}.[S:1].[S:2]"
            logger.info(f"SMILES Translator Output: {out_smiles}")
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
            logger.info(f"SMILES Translator Output: {out_smiles}")
            return out_smiles
        else:
            raise ValueError(f"'{site_type}' site does not support {num_binding_indices} binding indices.")
    else:
        raise ValueError(f"Unknown site_type: {site_type}.")

def read_atoms_object(slab_path: str) -> ase.Atoms:
    try:
        atoms = read(slab_path)  # ASE auto-detects format (XYZ, CIF, PDB, SDF, MOL, POSCAR, etc.)
        logger.info(f"Read slab atoms from {slab_path}.")
        return atoms
    except Exception as e:
        logger.error(f"Unable to read {slab_path}: {e}")
        raise

# --- Unified handling of surface expansion and cleaning ---
def prepare_slab(slab_atoms: ase.Atoms) -> Tuple[ase.Atoms, bool]:
    """
    Clean Slab metadata and expand supercell if needed for physical accuracy.
    
    Validates periodicity for non-crystallographic inputs (PDB, MOL, SDF).
    Returns: (Processed Slab, Is Expanded)
    """
    print("--- 🛠️ [Prepare] Cleaning Slab metadata and checking dimensions... ---")
    
    # 0. Periodicity validation for non-crystallographic inputs (PDB, MOL, SDF, etc.)
    pbc = slab_atoms.get_pbc()
    if not any(pbc):
        logger.warning(
            "Input structure has no periodic boundary conditions (PBC). "
            "This typically indicates a molecular cluster (e.g., from PDB/MOL files), not a surface slab. "
            "AdsKRK is designed for periodic surfaces; results may be unreliable."
        )
        print("--- ⚠️ [Prepare] WARNING: Non-periodic structure detected. Setting fallback PBC... ---")
        
        # Force 2D periodic boundary conditions (typical for surface slabs)
        slab_atoms.set_pbc([True, True, False])
        
        # If no cell is defined, create one with vacuum padding
        if slab_atoms.cell.volume < 1e-6:
            positions = slab_atoms.get_positions()
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            extent = max_pos - min_pos
            
            # Create cell with 15Å padding in XY and 20Å vacuum in Z
            cell_a = extent[0] + 15.0
            cell_b = extent[1] + 15.0
            cell_c = extent[2] + 20.0
            
            slab_atoms.set_cell([cell_a, cell_b, cell_c])
            slab_atoms.center()
            logger.info(f"Created fallback cell: {cell_a:.1f} x {cell_b:.1f} x {cell_c:.1f} Å")
            print(f"--- 🛠️ [Prepare] Created fallback cell: {cell_a:.1f} x {cell_b:.1f} x {cell_c:.1f} Å ---")
    
    # 1. Clean metadata (Fix autoadsorbate crash when parsing extxyz extra columns)
    symbols = slab_atoms.get_chemical_symbols()
    positions = slab_atoms.get_positions()
    cell = slab_atoms.get_cell()
    pbc = slab_atoms.get_pbc()
    
    clean_slab = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    
    # 2. Smart Expansion (Fix issue where 1x1 cell cannot find Hollow sites)
    # Logic: If any lattice vector length in XY plane is < 6.0 Å, expand to 2x2
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
    
    # 3. Vacuum layer thickness check
    # Calculate vacuum as (cell height - slab thickness)
    MIN_VACUUM_THICKNESS = 15.0  # Angstroms
    cell_c = np.linalg.norm(clean_slab.get_cell()[2])
    z_coords = clean_slab.positions[:, 2]
    slab_thickness = z_coords.max() - z_coords.min()
    vacuum_thickness = cell_c - slab_thickness
    
    if vacuum_thickness < MIN_VACUUM_THICKNESS:
        logger.warning(
            f"Vacuum layer thickness ({vacuum_thickness:.1f} Å) is less than recommended {MIN_VACUUM_THICKNESS} Å. "
            f"This may cause spurious interactions between periodic images in Z direction."
        )
        print(f"--- ⚠️ [Prepare] WARNING: Thin vacuum layer ({vacuum_thickness:.1f} Å < {MIN_VACUUM_THICKNESS} Å recommended). ---")
    else:
        logger.info(f"Vacuum layer thickness: {vacuum_thickness:.1f} Å (OK)")
        
    return clean_slab, is_expanded

def analyze_surface_sites(slab_path: str) -> dict:
    """ Pre-scan surface to find actually existing site types for Planner reference """
    from collections import defaultdict, Counter
    atoms = read_atoms_object(slab_path)
    clean_slab, _ = prepare_slab(atoms)
    
    # Dry run Autoadsorbate
    logger.info("Starting surface site analysis...")
    print("--- 🔍 [Site Analysis] Scanning surface for adsorption sites... ---")
    s = Surface(clean_slab, precision=1.0, touch_sphere_size=2.0, mode='slab')
    # NOTE: sym_reduce() removed due to O(n²) performance issue on large surfaces.
    # Site types are still deduplicated via set in site_inventory below.
    # s.sym_reduce()
    print(f"--- ✅ [Site Analysis] Found {len(s.site_df)} sites. ---")
    
    site_inventory = defaultdict(set)
    for _, row in s.site_df.iterrows():
        conn = row['connectivity']
        # Convert {'Mo':2, 'Pd':1} to "Mo-Mo-Pd"
        elements = []
        for el, count in row['site_formula'].items():
            elements.extend([el] * count)
        site_desc = "-".join(sorted(elements))
        site_inventory[conn].add(site_desc)
    
    # Fix fictitious 3-fold sites on square lattices like FCC(100)
    # Logic: If a surface has both 4-fold (connectivity=4) and 3-fold (connectivity=3),
    # and no extremely complex low-symmetry features, usually 3-fold is a triangulation artifact.
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
        
        # Clear charges to appease UFF force field
        mol_for_opt = Chem.Mol(mol_with_hs)
        for atom in mol_for_opt.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0) 
            atom.SetIsotope(0)
            atom.SetHybridization(Chem.rdchem.HybridizationType.UNSPECIFIED)
        
        try:
            Chem.SanitizeMol(mol_for_opt)
        except Exception as e:
            logger.warning(f"Sanitize Warning: {e}")

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        params.pruneRmsThresh = 0.5
        params.numThreads = 0
        
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_for_opt, numConfs=to_initialize, params=params))
        
        if not conf_ids:
            logger.warning("ETKDGv3 failed, trying ETKDGv2 ...")
            AllChem.EmbedMolecule(mol_for_opt, AllChem.ETKDGv2())
            if mol_for_opt.GetNumConformers() > 0:
                conf_ids = [0]
        
        if not conf_ids:
            logger.warning("ETKDG series failed, trying Random Coords ...")
            # For forced coordination structures, random coords usually generate "at least one" usable geometry
            params_rand = AllChem.ETKDGv3()
            params_rand.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(mol_for_opt, numConfs=1, params=params_rand))

        # Check for charged atoms. If present, UFF force field might crash/error, so skip UFF.
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
                logger.warning(f"UFF Optimization Warning: {e}")
        
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
            
            # 1. Find all mapped atoms (Add isotope double insurance)
            map_num_to_idx = {}
            for atom in all_rdkit_atoms:
                map_num = atom.GetAtomMapNum()
                idx = atom.GetIdx()
                iso = atom.GetIsotope()
                
                # Prioritize Map Number
                if map_num > 0:
                    map_num_to_idx[map_num] = idx
                
                # === [Anchor Active] If Map is lost, recover using Isotope ===
                if iso == 37: 
                    # 37Cl is our marker
                    map_num_to_idx[1] = idx
                if iso == 14: 
                    # 14C (or atom with isotope 14) is our target
                    map_num_to_idx[114514] = idx
            
            # 2. Build index list based on TRICK_SMILES and num_binding_indices
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

                # Clear temporary map numbers
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

                    # Manually align S-S vector to be *perpendicular* to Z axis (Simulate end-on)
                    s1_idx, s2_idx = proxy_indices[0], proxy_indices[1]
                    t1_idx = binding_indices[0]

                    p1 = positions[t1_idx]

                    # --- Prevent autoadsorbate division by zero or generating zero vector ---
                    # 1. Perpendicular vector (S1-S2)
                    v_perp = np.array([0.0, 0.5, 0.0])
                    # 2. Tilted midpoint, so nvector (p1-midpoint) is neither zero nor parallel to Z axis
                    midpoint = p1 - np.array([0.1, 0.0, 1.0])

                    # Place S1 and S2
                    positions[s1_idx] = midpoint + v_perp
                    positions[s2_idx] = midpoint - v_perp

                    print(f"--- 🛠️ _get_fragment: Manually aligned S-S marker for End-on mode (Tilt Correction). ---")
                    all_rdkit_atoms[t1_idx].SetAtomMapNum(0)

                elif num_binding_indices == 2:
                    # --- side-on @ bridge/hollow ---
                    if 114514 not in map_num_to_idx or 1919810 not in map_num_to_idx:
                         raise ValueError(f"SMILES {SMILES} missing map number 114514 (binding atom 1) or 1919810 (binding atom 2).")

                    binding_indices = [map_num_to_idx[114514], map_num_to_idx[1919810]]

                    # Switch to Parallel-Bridge Strategy
                    # Make S-S vector (Dummy Atoms) parallel to the bond vector between bonding atoms
                    # This way when Autoadsorbate aligns S-S to surface Bridge axis, molecular bond will also parallel Bridge axis.
                    s1_idx, s2_idx = proxy_indices[0], proxy_indices[1]
                    t1_idx, t2_idx = binding_indices[0], binding_indices[1]

                    # 1. Get target atom positions
                    p1 = positions[t1_idx]
                    p2 = positions[t2_idx]
                        
                    # 2. Calculate their midpoint and bond vector
                    midpoint = (p1 + p2) / 2.0
                    v_bond = p1 - p2
                        
                    # 3. Normalize bond vector
                    norm = np.linalg.norm(v_bond)
                    if norm < 1e-3: 
                        v_bond_norm = np.array([1.0, 0.0, 0.0])
                    else:
                        v_bond_norm = v_bond / norm
                        
                    # 4. Place S1 and S2 on both sides of midpoint, extending along bond vector
                    # Distance 0.5 is arbitrary, as long as direction is defined.
                    positions[s1_idx] = midpoint + v_bond_norm * 0.5
                    positions[s2_idx] = midpoint - v_bond_norm * 0.5
                        
                    print(f"--- 🛠️ _get_fragment: Aligned S-S vector parallel to bond axis (Parallel Alignment) to avoid Cross-Bridge issues. ---")
                        
                    # 5. Clear temporary map numbers
                    all_rdkit_atoms[t1_idx].SetAtomMapNum(0)
                    all_rdkit_atoms[t2_idx].SetAtomMapNum(0)

            # 3. Build new, *guaranteed* atom order

            # Collect all atoms that are *neither* proxy atoms *nor* bonding atoms
            special_indices_set = set(proxy_indices + binding_indices)
            other_indices = [atom.GetIdx() for atom in all_rdkit_atoms if atom.GetIdx() not in special_indices_set and atom.GetAtomMapNum() == 0]

            # Enforce order expected by autoadsorbate
            new_order = proxy_indices + binding_indices + other_indices
            
            # 4. Extract symbols and positions based on new order
            new_symbols = [all_rdkit_atoms[i].GetSymbol() for i in new_order]
            new_positions = [positions[i] for i in new_order]
            
            # 5. Create ASE Atoms object and set critical .info["smiles"]
            new_atoms = Atoms(symbols=new_symbols, positions=new_positions)
            # This is the only thing autoadsorbate library cares about:
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

    # Extract required info from plan dictionary
    plan_solution = plan_dict.get("solution", {})
    adsorbate_type = plan_dict.get("adsorbate_type")
    site_type = plan_solution.get("site_type")
    num_binding_indices = len(binding_atom_indices)

    if not site_type or not adsorbate_type:
        raise ValueError("plan_dict missing 'site_type' or 'adsorbate_type'.")
    
    # 1. Internally call SMILES generator
    surrogate_smiles = generate_surrogate_smiles(
        original_smiles=original_smiles,
        binding_atom_indices=binding_atom_indices,
        site_type=site_type
    )

    # 2. Internally call conformer generator (includes all patches and tricks)
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

    # 3. Critical: Attach original plan info to Fragment object
    fragment.info["plan_site_type"] = site_type
    fragment.info["plan_original_smiles"] = original_smiles
    fragment.info["plan_binding_atom_indices"] = binding_atom_indices
    fragment.info["plan_adsorbate_type"] = adsorbate_type
    
    print(f"--- 🛠️ create_fragment_from_plan: Successfully created and tagged Fragment object. ---")
    return fragment

def _bump_adsorbate_to_safe_distance(slab_atoms: ase.Atoms, full_atoms: ase.Atoms, min_dist_threshold: float = 1.5) -> ase.Atoms:
    """
    Check if adsorbate collides with surface. If so, bump up along Z axis until no collision.
    """
    # 1. Distinguish surface and adsorbate
    n_slab = len(slab_atoms)
    adsorbate_indices = list(range(n_slab, len(full_atoms)))
    
    if not adsorbate_indices:
        return full_atoms

    # 2. Extract positions
    slab_pos = full_atoms.positions[:n_slab]
    ads_pos = full_atoms.positions[n_slab:]
    
    # 3. Calculate distance matrix (Adsorbate vs Slab)
    # Note: For very large systems, NeighborList can be used, but cdist is fast and robust enough here
    dists = cdist(ads_pos, slab_pos)
    min_d = np.min(dists)
    
    # 4. If too close, calculate how much to bump up
    if min_d < min_dist_threshold:
        # We want min_d to be at least min_dist_threshold
        # Simple strategy: Stepwise bump, or one-time bump (threshold - min_d) + buffer
        # Considering complex geometry, adding Z directly is safest
        bump_height = (min_dist_threshold - min_d) + 0.2 # Extra 0.2 A buffer
        
        print(f"--- 🛡️ Collision Detected: Atom overlap found (min_dist={min_d:.2f} Å < {min_dist_threshold} Å). Bumping up by {bump_height:.2f} Å... ---")
        
        # Modify adsorbate coordinates
        full_atoms.positions[adsorbate_indices, 2] += bump_height
    
    return full_atoms

def populate_surface_with_fragment(
    slab_atoms: ase.Atoms, 
    fragment_object: Fragment,
    plan_solution: dict,
    session_id: str,  # UUID for session-isolated file paths
    **kwargs
) -> str:
    # --- 1. Retrieve plan from Fragment object ---
    if not hasattr(fragment_object, "info") or "plan_site_type" not in fragment_object.info:
        raise ValueError("Fragment object missing 'plan_site_type' info.")

    # --- Read parameters from plan (or use defaults) ---
    raw_site_type = plan_solution.get("site_type", "all")
    # Force normalization: Correct "hollow-3", "hollow-4" to "hollow"
    if raw_site_type.lower().startswith("hollow"):
        site_type = "hollow"
    else:
        site_type = raw_site_type
    conformers_per_site_cap = plan_solution.get("conformers_per_site_cap", 4)
    overlap_thr = plan_solution.get("overlap_thr", 0.1)
    touch_sphere_size = plan_solution.get("touch_sphere_size", 2)

    print(f"--- 🛠️ Initializing Surface (touch_sphere_size={touch_sphere_size})... ---")
    
    # For safety, clean metadata again here to ensure autoadsorbate receives clean Atoms object
    symbols = slab_atoms.get_chemical_symbols()
    positions = slab_atoms.get_positions()
    cell = slab_atoms.get_cell()
    pbc = slab_atoms.get_pbc()
    clean_slab_atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    # Explicitly set mode='slab'
    s = Surface(
        clean_slab_atoms,
        precision=1.0, 
        touch_sphere_size=touch_sphere_size,
        mode='slab'  # Explicitly set mode to prevent default 'dummy'
    )

    original_site_count = len(s.site_df)
    # NOTE: sym_reduce() removed due to O(n²) performance issue on large surfaces.
    # s.sym_reduce()
    print(f"--- 🛠️ Surface Sites: {original_site_count} sites found. ---")

    # Check if sites were found
    # This prevents failure on s.site_df.connectivity
    if s.site_df.empty or len(s.site_df) == 0:
        raise ValueError(
            f"Autoadsorbate failed to find any adsorption sites on the surface (0 sites found). "
            f"This might be due to inappropriate `touch_sphere_size` ({touch_sphere_size}) (too large or too small)."
        )

    # --- 2. Verify plan compatibility with sites (Connectivity filtering) ---
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

    # --- 3. Optional Surface Atom Filtering ---
    allowed_symbols = plan_solution.get("surface_binding_atoms")
    if allowed_symbols and len(allowed_symbols) > 0:
        # Use sorted string for logging, clear and concise
        print(f"--- 🛠️ Filtering by surface symbols (strict match): {sorted(allowed_symbols)} ---")
        
        # Pre-calculate target atom counts (e.g., {'Mo': 2, 'Pd': 1})
        target_counts = Counter(allowed_symbols)
        
        def check_symbols(site_formula_dict):
            if not site_formula_dict or not isinstance(site_formula_dict, dict):
                return False
            
            # Strict matching logic:
            # Expand and count site_formula_dict (e.g., {'Mo': 2, 'Pd': 1}), must match target exactly
            # Prevent requesting ['Mo', 'Mo'] (pure bridge) but returning {'Mo': 2, 'Pd': 1} (mixed hollow)
            
            # 1. Expand site composition (dict -> list)
            site_atoms_list = []
            for sym, count in site_formula_dict.items():
                site_atoms_list.extend([sym] * count)
            
            # 2. Compare counters
            return Counter(site_atoms_list) == target_counts

        initial_count = len(site_df_filtered)
        # Apply strict filter
        site_df_filtered = site_df_filtered[
            site_df_filtered['site_formula'].apply(check_symbols)
        ]
        print(f"--- 🛠️ Surface Symbol Filter: Sites reduced from {initial_count} to {len(site_df_filtered)}. ---")

    # Replace s.site_df with filtered df
    s.site_df = site_df_filtered
    site_index_arg = list(s.site_df.index)
    
    # --- Intelligent Site Sampling ---
    # Without sym_reduce, we may have many equivalent sites. Sample to limit computation.
    MAX_SITES_PER_TYPE = 8  # Balance between coverage and computation time
    if len(site_index_arg) > MAX_SITES_PER_TYPE:
        import random
        print(f"--- 🎲 Site Sampling: {len(site_index_arg)} sites found, sampling {MAX_SITES_PER_TYPE} for efficiency... ---")
        site_index_arg = random.sample(site_index_arg, MAX_SITES_PER_TYPE)
        # CRITICAL: Also update s.site_df to match - autoadsorbate uses this internally
        s.site_df = s.site_df.loc[site_index_arg]
    
    print(f"--- 🛠️ Plan Verified: Searching {len(site_index_arg)} '{site_type}' sites. ---")

    if len(site_index_arg) == 0:
        raise ValueError(f"No sites of type '{site_type}' containing {allowed_symbols} found. Cannot proceed.")

    # --- 4. Determine sample_rotation ---
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

    # For Bridge and Hollow sites, pre-lift by 0.5 Å
    # Reason: autoadsorbate default initial distance is often too close for large molecules or multi-site adsorption, causing frequent collision corrections.
    if site_type in ["bridge", "hollow"]:
        print(f"--- 🛠️ Geometry Optimization: Pre-lifting adsorbate by 0.5 Å for {site_type} site to reduce collisions... ---")
        for atoms in raw_out_trj:
            # Find adsorbate atom indices (assuming adsorbate is added last)
            n_slab = len(slab_atoms)
            atoms.positions[n_slab:, 2] += 0.5
    
    
    # Perform collision detection and lifting for generated configurations (Threshold 1.8 Å)
    safe_out_trj = []
    for idx, atoms in enumerate(raw_out_trj):
        safe_atoms = _bump_adsorbate_to_safe_distance(slab_atoms, atoms, min_dist_threshold=1.6)
        safe_out_trj.append(safe_atoms)
    
    out_trj = safe_out_trj

    print(f"--- 🛠️ Successfully generated {len(out_trj)} initial configurations. ---")
    
    if not out_trj:
        raise ValueError(f"get_populated_sites failed to generate any configurations. overlap_thr ({overlap_thr}) might be too strict.")
    
    # Save ase.Atoms list to Trajectory object
    # Use session-isolated directory to prevent concurrent access conflicts
    output_dir = f"outputs/{session_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    clean_smiles = fragment_object.info['plan_original_smiles'].replace('=','_').replace('#','_')
    traj_file = f"{output_dir}/conformers_{clean_smiles}.traj"
    traj = Trajectory(traj_file, 'w')
    for atoms in out_trj:
        traj.write(atoms)
    traj.close()

    print(f"--- 🛠️ Configurations saved to {traj_file} ---")
    return traj_file

# Note: Calculator caching is now handled by the backend module (src/calculators/)

def relax_atoms(
    atoms_list: list,
    slab_indices: list,
    calculator,  # ASE-compatible calculator (any backend)
    session_id: str,  # UUID for session-isolated file paths
    relax_top_n: int = 1,
    fmax: float = 0.05,
    steps: int = 500,
    md_steps: int = 20,
    md_temp: float = 150.0,
    relaxation_mode: str = "fast",  # "fast" or "standard"
) -> str:
    """
    Relax a list of atomic structures using the provided calculator.

    This function is backend-agnostic and works with any ASE-compatible calculator
    (MACE, OpenMD, DeePMD, etc.).

    Args:
        atoms_list: List of ASE Atoms objects to relax
        slab_indices: Indices of slab atoms (constrained based on relaxation_mode)
        calculator: ASE-compatible calculator instance
        session_id: UUID for session-isolated file paths
        relax_top_n: Number of top configurations to relax (by energy)
        fmax: Maximum force tolerance for optimization (eV/Å)
        steps: Maximum optimization steps
        md_steps: Number of MD warmup steps (0 to disable)
        md_temp: MD temperature in Kelvin
        relaxation_mode: "fast" (all slab fixed) or "standard" (bottom 1/3 fixed)

    Returns:
        Path to the output trajectory file
    """

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    # Optimization: We only relax the best N configurations
    N_RELAX_TOP_N = relax_top_n

    # Determine which atoms to fix based on relaxation_mode
    if relaxation_mode == "standard" and len(atoms_list) > 0 and len(slab_indices) > 0:
        # STANDARD mode: Fix bottom 1/3 of slab atoms by Z coordinate
        reference_atoms = atoms_list[0]
        slab_z_coords = reference_atoms.positions[slab_indices, 2]
        z_min, z_max = slab_z_coords.min(), slab_z_coords.max()
        z_threshold = z_min + (z_max - z_min) * FIXED_BOTTOM_FRACTION
        
        fixed_indices = [idx for idx in slab_indices if reference_atoms.positions[idx, 2] < z_threshold]
        logger.info(
            f"STANDARD relaxation mode: Fixing {len(fixed_indices)}/{len(slab_indices)} "
            f"bottom slab atoms (Z < {z_threshold:.2f} Å)"
        )
        print(f"--- 🛠️ STANDARD mode: Fixing bottom {len(fixed_indices)}/{len(slab_indices)} slab atoms ---")
        constraint = FixAtoms(indices=fixed_indices)
    else:
        # FAST mode (default): Fix all slab atoms
        if relaxation_mode != "fast":
            logger.warning(f"Unknown relaxation_mode '{relaxation_mode}', defaulting to 'fast'")
        constraint = FixAtoms(indices=slab_indices)

    def _get_bond_change_count(initial, final):
        if len(initial) != len(final):
            return 0
        radii = np.array(natural_cutoffs(initial, mult=1.25))
        cutoff_mat = radii[:, None] + radii[None, :]
        d_initial = initial.get_all_distances()
        d_final = final.get_all_distances()

        # Ignore H-H bonds
        symbols = initial.get_chemical_symbols()
        is_H = np.array([s == 'H' for s in symbols])
        mask = is_H[:, None] & is_H[None, :]
        np.fill_diagonal(d_initial, 99.0)
        np.fill_diagonal(d_final, 99.0)

        bonds_initial = (d_initial < cutoff_mat) & (~mask)
        # Loose threshold for bond breaking detection (1.5x)
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
            logger.warning(f"Skipping structure {i+1}: Initial force too high (Max Force = {max_force:.2f} eV/A)...")
            continue

        if md_steps > 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_temp)
            dyn_md = Langevin(atoms, 1 * units.fs, temperature_K=md_temp, friction=0.01)
            dyn_md.run(md_steps)

        energy = atoms.get_potential_energy()

        # --- Energy sanity check, mask non-physical explosive structures ---
        if (not np.isfinite(energy)) or energy < -2000.0:
            logger.warning(f"Skipping structure {i+1}: Abnormal energy (E = {energy:.2f} eV), suspected numerical collapse")
            continue

        logger.info(f"Evaluating structure {i+1}/{len(atoms_list)}... Energy (after warmup): {energy:.4f} eV ")
        evaluated_configs.append((energy, i, atoms.copy())) # Store copy

    if not evaluated_configs:
        raise ValueError("Evaluation phase failed to evaluate any configurations.")

    # --- 2. Select Best ---
    evaluated_configs.sort(key=lambda x: x[0]) # Sort by energy
    
    if N_RELAX_TOP_N > len(evaluated_configs):
        print(f"--- 🛠️ Warning: Requested to relax top {N_RELAX_TOP_N}, but only {len(evaluated_configs)} available. Relaxing all. ---")
        N_RELAX_TOP_N = len(evaluated_configs)
    
    configs_to_relax = evaluated_configs[:N_RELAX_TOP_N]
    
    print(f"--- 🛠️ Evaluation complete. Relaxing best {N_RELAX_TOP_N} of {len(atoms_list)} configurations. ---")
    
    # --- 3. Relaxation Phase (Only N_RELAX_TOP_N) ---
    output_dir = f"outputs/{session_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    traj_file = f"{output_dir}/relaxation.traj"
    traj = Trajectory(traj_file, 'w')
    final_structures = []

    for i, (initial_energy, original_index, atoms) in enumerate(configs_to_relax):
        logger.info(f"Relaxing best structure {i+1}/{N_RELAX_TOP_N} (Original Index {original_index}, Initial Energy: {initial_energy:.4f} eV) ")
        
        atoms.calc = calculator
        atoms.set_constraint(constraint)

        # --- Capture adsorbate before relaxation ---
        adsorbate_indices = list(range(len(slab_indices), len(atoms)))
        initial_adsorbate = atoms.copy()[adsorbate_indices]
        
        logger.info(f"Optimization (BFGS): fmax={fmax}, steps={steps}")
        dyn_opt = BFGS(atoms, trajectory=None, logfile=None) 
        dyn_opt.attach(lambda: traj.write(atoms), interval=1)
        dyn_opt.run(fmax=fmax, steps=steps)

        # --- Capture adsorbate state after relaxation and check bond changes ---
        final_adsorbate = atoms.copy()[adsorbate_indices]
        bond_change_count = _get_bond_change_count(initial_adsorbate, final_adsorbate)
        atoms.info["bond_change_count"] = bond_change_count
        logger.info(f"Bond Integrity Check: Detected {bond_change_count} bond changes. ")
        
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        logger.info(f"Best structure {i+1} relaxation complete. Final Energy: {final_energy:.4f} eV ")

        atoms.results = {
            'energy': final_energy,
            'forces': final_forces
        }
        
        final_structures.append(atoms)

    traj.close()
    
    final_traj_file = f"{output_dir}/final.xyz"
 
    try:
        write(final_traj_file, final_structures)
    except Exception as e:
        logger.error(f"Failed to write final_relaxed_structures.xyz: {e}")
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
    session_id: str,  # UUID for session-isolated file paths
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

        # 1. Find the most stable configuration
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
        logger.info(f"Analysis: E_ads = {E_ads:.4f} eV (E_total = {min_energy_total:.4f} eV, E_surf={e_surface_ref:.4f}, E_ads_mol={e_adsorbate_ref:.4f}) ")
        
        # 1. Define smart judgment function (Moved to front for global reuse)
        # For Float32 precision and metal adsorption characteristics, increased base tolerance from 1.25 to 1.3
        def check_bonding_smart(atom_idx_1, atom_idx_2, r1, r2, current_energy_eV, check_atoms_obj):
            base_mult = 1.30 # Base bond length tolerance
            
            # Energy-assisted judgment: If energy is very low (< -0.5 eV), strong interaction is certain, relax geometric judgment
            if current_energy_eV < -0.5:
                base_mult = 1.45 # Even if geometry is slightly stretched, count as bonded if energy is low
            
            d = check_atoms_obj.get_distance(atom_idx_1, atom_idx_2, mic=True)
            threshold = (r1 + r2) * base_mult
            return d <= threshold, d, threshold

        # 1. Extract adsorbate atoms
        adsorbate_atoms = relaxed_atoms[len(slab_atoms):]

        # 2. Copy and apply PBC info (Critical! Prevent cross-boundary atoms from being misjudged as broken)
        # Create a temporary Atoms object for topology analysis
        check_atoms = adsorbate_atoms.copy()
        check_atoms.set_cell(relaxed_atoms.get_cell())
        check_atoms.set_pbc(relaxed_atoms.get_pbc())

        # 3. Build adjacency matrix (Consider PBC)
        # mult=1.35 Increase tolerance for bond stretching
        # Avoid misjudging bond activation due to strong adsorption as bond breaking
        check_cutoffs = natural_cutoffs(check_atoms, mult=1.35)
        nl = build_neighbor_list(check_atoms, cutoffs=check_cutoffs, self_interaction=False)
        adjacency_matrix = nl.get_connectivity_matrix()

        # 4. Calculate connected components (Count how many pieces the molecule broke into)
        n_components, labels = connected_components(adjacency_matrix, directed=False)

        # 5. Judgment Logic
        # Normally, single molecule adsorption should have only 1 connected component
        is_dissociated = n_components > 1

        # 6. Get bond change count as auxiliary reference
        bond_change_count = relaxed_atoms.info.get("bond_change_count", 0)

        # If molecule broke into n pieces (n > 1), at least (n-1) bonds are broken.
        # Prevent contradiction of "is_dissociated=True" but "bond_change_count=0".
        if is_dissociated and bond_change_count == 0:
            print(f"--- 🛠️ Logic Contradiction Fix: Dissociation detected (n_components={n_components}) but bond_change_count=0. Forcing fix. ---")
            bond_change_count = max(1, n_components - 1)

        # 7. Comprehensive Reactivity Judgment
        reaction_detected = False
        if is_dissociated:
             # Keep real bond_change_count > 0, representing isomerization
             reaction_detected = True
        elif bond_change_count > 0:
             # Bonds changed but not broken -> Isomerization
             # Mark reaction_detected = True, let Agent decide if this is bad
             reaction_detected = True
        else:
             # Bonds unchanged, molecule unbroken -> Perfect molecular adsorption
             reaction_detected = False

        # --- Retrieve info from plan_dict ---
        plan_solution = plan_dict.get("solution", {})
        adsorbate_type = plan_dict.get("adsorbate_type")
        site_type = plan_solution.get("site_type")
        binding_atom_indices = plan_solution.get("adsorbate_binding_indices", [])
        num_binding_indices = len(binding_atom_indices)

        # 1.1. Get planned site info from .info dictionary
        planned_info = relaxed_atoms.info.get("adsorbate_info", {}).get("site", {})
        planned_connectivity = planned_info.get("connectivity")
        planned_site_type = "unknown"
        if planned_connectivity == 1: planned_site_type = "ontop"
        elif planned_connectivity == 2: planned_site_type = "bridge"
        elif planned_connectivity and planned_connectivity >= 3: planned_site_type = "hollow"
        
        # 1.2. Identify surface and adsorbate indices
        slab_indices_check = list(range(len(slab_atoms)))
        adsorbate_indices_check = list(range(len(slab_atoms), len(relaxed_atoms)))
        cov_cutoffs_check = natural_cutoffs(relaxed_atoms, mult=1)
        
        actual_bonded_slab_indices = set()
        anchor_atom_indices = []
        if num_binding_indices == 1 and len(adsorbate_indices_check) > 0:
            anchor_atom_indices = [adsorbate_indices_check[0]]
        elif num_binding_indices == 2 and len(adsorbate_indices_check) >= 2:
            anchor_atom_indices = [adsorbate_indices_check[0], adsorbate_indices_check[1]]
        
        # 1.3. Calculate number of actually bonded surface atoms
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

        # Physical Consistency Forced Correction (Sanity Check)
        # If energy is very low (strong adsorption) but geometrically desorbed, geometric criteria are too strict, force fix to chemisorbed
        if actual_site_type == "desorbed" and E_ads < -0.5:
            print(f"--- 🛠️ Physical Correction: Strong adsorption energy ({E_ads:.2f} eV) detected but geometrically desorbed. Forcing 'hollow/promiscuous'. ---")
            actual_site_type = "hollow (inferred)"
            # Keep actual_connectivity as 0 or manually set to 3 to prevent Agent confusion
            if actual_connectivity == 0: actual_connectivity = 3

        slab_indices = list(range(len(slab_atoms)))
        adsorbate_indices = list(range(len(slab_atoms), len(relaxed_atoms)))
        
        slab_atoms_relaxed = relaxed_atoms[slab_indices]
        adsorbate_atoms_relaxed = relaxed_atoms[adsorbate_indices]

        # We default to taking the first atom in the adsorbate list as the anchor for crystallographic probing
        target_atom_global_index = adsorbate_indices[0] if len(adsorbate_indices) > 0 else -1

        # FCC/HCP Crystallographic Identification
        # Only perform deep probing when confirmed as hollow site
        site_crystallography = ""
        if actual_site_type == "hollow":
            try:
                # 1. Define surface layer and subsurface layer
                # Assume slab is aligned in Z direction, and z_max is the top layer
                # Simple layer slicing: Consider 1.5A to 4.0A from top layer as Subsurface
                # Suitable for most metals (interlayer spacing ~2.0-2.3A)
                z_coords = slab_atoms_relaxed.positions[:, 2]
                max_z = np.max(z_coords)
                # Simple layer slicing: Consider 1.5A to 4.0A from top layer as Subsurface
                # Suitable for most metals (interlayer spacing ~2.0-2.3A)
                subsurface_mask = (z_coords < (max_z - 1.2)) & (z_coords > (max_z - 4.0))
                subsurface_indices_list = np.where(subsurface_mask)[0]

                if len(subsurface_indices_list) > 0:
                    # 2. Get XY coordinates of target adsorbate atom
                    target_pos_xy = relaxed_atoms[target_atom_global_index].position[:2]
                    
                    # 3. Calculate projected distance in XY plane between adsorbate atom and all subsurface atoms
                    subsurface_positions_xy = slab_atoms_relaxed.positions[subsurface_indices_list][:, :2]
                    
                    # Calculate XY distance considering Periodic Boundary Conditions (PBC)
                    # For simplicity, assuming atom is directly below, Euclidean distance is usually enough,
                    # but a more rigorous approach is using ase.geometry.get_distances or manually handling cell
                    # Using simplified projected distance judgment here:
                    # If subsurface atom XY distance < 1.0 Å, atom exists directly below -> HCP
                    dists_xy = np.linalg.norm(subsurface_positions_xy - target_pos_xy, axis=1)
                    min_dist_xy = np.min(dists_xy)
                    
                    if min_dist_xy < 1.0:
                        site_crystallography = "(HCP/Subsurf-Atom)"
                    else:
                        site_crystallography = "(FCC/No-Subsurf)"
                else:
                    site_crystallography = "(Unknown Layer)"
            except Exception as e_cryst:
                logger.warning(f"Crystallographic Analysis Warning: {e_cryst}")
        
        # Append this suffix to actual_site_type so Agent can see the difference
        if site_crystallography:
            actual_site_type += f" {site_crystallography}"
        
        logger.info(f"Analysis: Site Slip Check: Planned {planned_site_type} (conn={planned_connectivity}), Actual {actual_site_type} (conn={actual_connectivity}) ")

        # 2. Identify adsorbate atoms and surface atoms
        
        target_atom_global_index = -1
        target_atom_symbol = ""
        analysis_message = ""
        result = {}

        # Prepare covalent bond check
        cov_cutoffs = natural_cutoffs(relaxed_atoms, mult=1)

        if num_binding_indices == 1:
            # Target atom is *always* the first in adsorbate list
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position

            logger.info(f"Analysis: (1-index mode) Checking first adsorbate atom, Symbol: '{target_atom_symbol}', Global Index: {target_atom_global_index}.")

            # --- Find all bonded surface atoms, not just the nearest one ---
            bonded_surface_atoms = []
            min_distance = float('inf')
            nearest_slab_atom_symbol = ""
            nearest_slab_atom_global_index = -1
            
            # Iterate through all surface atoms
            for s_idx in slab_indices:
                r_ads = cov_cutoffs_check[target_atom_global_index]
                r_slab = cov_cutoffs_check[s_idx]
                
                # Use smart judgment
                is_connected, d, threshold = check_bonding_smart(
                    target_atom_global_index, s_idx, r_ads, r_slab, E_ads, relaxed_atoms
                )
                
                if d < min_distance:
                    min_distance = d
                    nearest_slab_atom_global_index = s_idx
                    nearest_slab_atom_symbol = relaxed_atoms[s_idx].symbol
                    # Dynamically update threshold for reporting
                    bonding_cutoff = threshold 

                if is_connected:
                    bonded_surface_atoms.append({
                        "symbol": relaxed_atoms[s_idx].symbol,
                        "index": s_idx,
                        "distance": round(d, 3)
                    })
            
            # Sort by distance, nearest first
            bonded_surface_atoms.sort(key=lambda x: x["distance"])

            # Generate unique Site Fingerprint with atom indices
            # This distinguishes "Ru-Ru Bridge near Mo" from "Ru-Ru Bridge far from Mo"
            bonded_indices = sorted([item['index'] for item in bonded_surface_atoms])
            site_fingerprint = "-".join([f"{item['symbol']}{item['index']}" for item in bonded_surface_atoms])
            
            is_bound = len(bonded_surface_atoms) > 0
            
            # Generate bonding description string (e.g., "Cu-2.01Å, Ga-2.15Å")
            if is_bound:
                bonded_desc = ", ".join([f"{item['symbol']}-{item['distance']}Å" for item in bonded_surface_atoms])
            else:
                bonded_desc = "None"
            
            # Estimate nearest atom cutoff for reporting
            nearest_radius_sum = cov_cutoffs[target_atom_global_index] + cov_cutoffs[nearest_slab_atom_global_index]
            estimated_covalent_cutoff_A = nearest_radius_sum * 1.1

            # Chemical Slip Detection
            # 1. Get planned surface atom symbols (Sorted to ignore order differences)
            planned_symbols = sorted(plan_solution.get("surface_binding_atoms", []))
            
            # 2. Get actually bonded surface atom symbols
            actual_symbols = sorted([atom['symbol'] for atom in bonded_surface_atoms])
            
            # 3. Determine if chemical slip occurred
            # Note: Skip if plan is empty (unspecified); skip if no bonding
            is_chemical_slip = False
            if planned_symbols and bonded_surface_atoms:
                if planned_symbols != actual_symbols:
                    is_chemical_slip = True
                    logger.warning(f"Warning: Chemical Site Slip Detected! Planned: {planned_symbols} -> Actual: {actual_symbols}")

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
            
            # Target atoms are *always* the first two in adsorbate list
            
            # --- Analyze first atom (Atom 0) ---
            target_atom_global_index = adsorbate_indices[0]
            target_atom_symbol = relaxed_atoms[target_atom_global_index].symbol
            target_atom_pos = relaxed_atoms[target_atom_global_index].position
            logger.info(f"Analysis: (2-index mode) Checking first adsorbate atom, Symbol: '{target_atom_symbol}', Global Index: {target_atom_global_index}.")

            distances = np.linalg.norm(slab_atoms.positions - target_atom_pos, axis=1)
            min_distance = np.min(distances)
            nearest_slab_atom_global_index = slab_indices[np.argmin(distances)]
            nearest_slab_atom_symbol = relaxed_atoms[nearest_slab_atom_global_index].symbol
            radius_1 = cov_cutoffs[target_atom_global_index]
            radius_2 = cov_cutoffs[nearest_slab_atom_global_index]
            bonding_cutoff = (radius_1 + radius_2) * 1.1
            is_bound_1 = min_distance <= bonding_cutoff

            # --- Analyze second atom (Atom 1) ---
            second_atom_global_index = adsorbate_indices[1]
            second_atom_symbol = relaxed_atoms[second_atom_global_index].symbol
            second_atom_pos = relaxed_atoms[second_atom_global_index].position
            logger.info(f"Analysis: (side-on mode) Checking second adsorbate atom, Symbol: '{second_atom_symbol}', Global Index: {second_atom_global_index}.")
            
            distances_2 = np.linalg.norm(slab_atoms.positions - second_atom_pos, axis=1)
            min_distance_2 = np.min(distances_2)
            nearest_slab_atom_global_index_2 = slab_indices[np.argmin(distances_2)]
            nearest_slab_atom_symbol_2 = relaxed_atoms[nearest_slab_atom_global_index_2].symbol
            radius_3 = cov_cutoffs[second_atom_global_index]
            radius_4 = cov_cutoffs[nearest_slab_atom_global_index_2]
            bonding_cutoff_2 = (radius_3 + radius_4) * 1.1
            is_bound_2 = min_distance_2 <= bonding_cutoff_2

            # --- Combine Results ---
            # Only successful if both atoms are bonded
            is_bound = bool(is_bound_1 and is_bound_2) 
            
            # Generate unified bonded_surface_atoms and final_bond_distance_A ===
            bonded_surface_atoms = []

            # Helper function: Find all bonding targets for an adsorbate atom
            def find_bonds(ads_idx, ads_symbol):
                bonds = []
                r_ads = cov_cutoffs_check[ads_idx]
                for s_idx in slab_indices:
                    r_slab = cov_cutoffs_check[s_idx]
                    is_connected, d, _ = check_bonding_smart(
                        ads_idx, s_idx, r_ads, r_slab, E_ads, relaxed_atoms
                    )
                    # Determine bonding
                    if is_connected:
                        bonds.append({
                            "adsorbate_atom": f"{ads_symbol}({ads_idx})",
                            "adsorbate_atom_index": int(ads_idx),
                            "symbol": relaxed_atoms[s_idx].symbol,
                            "index": int(s_idx),
                            "distance": round(d, 3)
                        })
                return bonds

            # Collect bonding info for both atoms
            bonded_surface_atoms.extend(find_bonds(target_atom_global_index, target_atom_symbol))
            bonded_surface_atoms.extend(find_bonds(second_atom_global_index, second_atom_symbol))
            
            # Sort by distance
            bonded_surface_atoms.sort(key=lambda x: x["distance"])

            # Generate unique Site Fingerprint with atom indices
            # This distinguishes "Ru-Ru Bridge near Mo" from "Ru-Ru Bridge far from Mo"
            bonded_indices = sorted([item['index'] for item in bonded_surface_atoms])
            site_fingerprint = "-".join([f"{item['symbol']}{item['index']}" for item in bonded_surface_atoms])

            # Calculate final shortest bond length (for reporting)
            if bonded_surface_atoms:
                final_bond_distance_A = bonded_surface_atoms[0]["distance"]
            else:
                final_bond_distance_A = min(min_distance, min_distance_2)
            
            # Generate description string
            if bonded_surface_atoms:
                bonded_desc = ", ".join([f"{b['adsorbate_atom']}-{b['symbol']}({b['distance']}Å)" for b in bonded_surface_atoms])
            else:
                bonded_desc = "None"

            # Chemical Slip Detection
            # 1. Get planned surface atom symbols (Sorted to ignore order differences)
            planned_symbols = sorted(plan_solution.get("surface_binding_atoms", []))
            
            # 2. Get actually bonded surface atom symbols
            actual_symbols = sorted([atom['symbol'] for atom in bonded_surface_atoms])
            
            # 3. Determine if chemical slip occurred
            # Note: Skip if plan is empty (unspecified); skip if no bonding
            is_chemical_slip = False
            if planned_symbols and bonded_surface_atoms:
                if planned_symbols != actual_symbols:
                    is_chemical_slip = True
                    logger.warning(f"Warning: Chemical Site Slip Detected! Planned: {planned_symbols} -> Actual: {actual_symbols}")
            # === 🩹 Fix End ===

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

        # 6. Save final structure
        # Prevent filename conflict overwriting history best.
        # Add to filename: Site type, surface composition, energy.
        
        # Naming Logic
        site_label = actual_site_type if actual_site_type != "unknown" else planned_site_type
        if planned_site_type != "unknown" and site_label != planned_site_type:
            site_label = f"{planned_site_type}_to_{site_label}"
            
        if is_dissociated: site_label += "_DISS"
        elif bond_change_count > 0: site_label += "_ISO"
        
        site_label = site_label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")

        clean_smiles = original_smiles.replace('=', '_').replace('#', '_').replace('[', '').replace(']', '')
        output_dir = f"outputs/{session_id}"
        best_atoms_filename = f"{output_dir}/BEST_{clean_smiles}_{site_label}_E{E_ads:.3f}.xyz"
        
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
