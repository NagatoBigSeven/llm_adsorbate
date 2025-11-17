import os
import ase.io
import numpy as np
from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# --- 1. Define the 20 Benchmark Systems from the Adsorb-Agent Paper ---
# Each tuple contains: (System_ID, Adsorbate_SMILES, Catalyst_Name, MP_ID, Target_Shift, Miller_Index)
SYSTEMS_TO_GENERATE = [
    # No, Adsorbate, Catalyst, mpid, Target_Shift, Miller
    (1, "H", "Mo3Pd_111", "mp-1186014", 0.167, (1, 1, 1)),
    (2, "NNH", "Mo3Pd_111", "mp-1186014", 0.167, (1, 1, 1)),
    (3, "H", "Pd3Cu_111", "mp-1184119", 0.063, (1, 1, 1)),
    (4, "NNH", "Pd3Cu_111", "mp-1184119", 0.063, (1, 1, 1)),
    (5, "H", "Cu3Ag_111", "mp-1184011", 0.063, (1, 1, 1)),
    (6, "NNH", "Cu3Ag_111", "mp-1184011", 0.063, (1, 1, 1)),
    (7, "H", "Ru3Mo_111", "mp-975834", 0.167, (1, 1, 1)),
    (8, "NNH", "Ru3Mo_111", "mp-975834", 0.167, (1, 1, 1)),
    (9, "OH", "Pt_111", "mp-126", 0.167, (1, 1, 1)),      # Expect mismatch
    (10, "OH", "Pt_100", "mp-126", 0.250, (1, 0, 0)),
    (11, "OH", "Pd_111", "mp-2", 0.167, (1, 1, 1)),      # Expect mismatch
    (12, "OH", "Au_111", "mp-81", 0.167, (1, 1, 1)),      # Expect mismatch
    (13, "OH", "Ag_100", "mp-124", 0.250, (1, 0, 0)),
    (14, "OH", "CoPt_111", "mp-1225998", 0.042, (1, 1, 1)),
    (15, "CH2CH2OH", "Cu6Ga2_100", "mp-865798", 0.248, (1, 0, 0)),
    (16, "CH2CH2OH", "Au2Hf_102", "mp-1018153", 0.028, (1, 0, 2)),
    (17, "OCHCH3", "Rh2Ti2_111", "mp-2583", 0.083, (1, 1, 1)),
    (18, "OCHCH3", "Al3Zr_101", "mp-1065309", 0.375, (1, 0, 1)),
    (19, "OCHCH3", "Hf2Zn6_110", "mp-866108", 0.120, (1, 1, 0)),
    (20, "ONN(CH3)2", "Bi2Ti6_211", "mp-866201", 0.000, (2, 1, 1)),
]

# --- 2. Standard Pymatgen & ASE Parameters ---
# These are computational "best-practice" values, as they were not specified in the Adsorb-Agent paper.
MIN_SLAB_THICKNESS_A = 10.0  # Minimum slab thickness in Angstroms
MIN_VACUUM_THICKNESS_A = 15.0 # Minimum vacuum padding in Angstroms

# --- 3. Set up Materials Project API Client ---
# Requires the PMG_MAPI_KEY environment variable to be set.
try:
    mpr = MPRester()
    print("✓ MPRester initialized successfully.")
except Exception as e:
    print(f"✗ ERROR: Could not initialize MPRester. Ensure PMG_MAPI_KEY environment variable is set.")
    print(f"  Details: {e}")
    exit(1)

# --- 4. Create Output Directory ---
output_dir = "benchmark_slabs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"✓ All slab files will be saved in '{output_dir}/' directory.\n")

# --- 5. Main Loop: Iterate Through and Generate All 20 Slabs ---
for system_data in SYSTEMS_TO_GENERATE:
    idx, adsorbate, name, mpid, target_shift, miller = system_data
    
    # Define the output path for the .xyz file
    slab_filename = f"{idx:02d}_{name}.xyz"
    slab_output_path = os.path.join(output_dir, slab_filename)

    print(f"--- [ {idx}/20 ] Processing System: {name} (mpid: {mpid}) ---")

    # Check if the file already exists to save time
    if os.path.exists(slab_output_path):
        print(f"  ✓ File already exists: {slab_output_path}. Skipping generation.")
        print("---")
        continue

    try:
        # Step 1: Download the primitive bulk structure from Materials Project
        print(f"  → Downloading bulk structure (mpid='{mpid}')...")
        primitive_structure = mpr.get_structure_by_material_id(mpid)

        # Step 2: Convert the downloaded primitive cell to a conventional standard cell.
        # Pymatgen's SlabGenerator works best with conventional cells.
        # This is likely the (unstated) step the Adsorb-Agent authors took.
        print(f"  → Converting primitive cell to conventional cell...")
        sga = SpacegroupAnalyzer(primitive_structure)
        conventional_structure = sga.get_conventional_standard_structure()

        # Step 3: Initialize the SlabGenerator
        # We now pass the CONVENTIONAL structure.
        # We set 'primitive=False' to ensure the generator *keeps* the conventional cell and doesn't re-reduce it, which is crucial for generating multiple terminations.
        slab_gen = SlabGenerator(
            initial_structure=conventional_structure,
            miller_index=miller,
            min_slab_size=MIN_SLAB_THICKNESS_A,
            min_vacuum_size=MIN_VACUUM_THICKNESS_A,
            primitive=False,
            center_slab=True,
            reorient_lattice=True
        )

        # Step 4: Generate a list of ALL unique slab terminations
        all_possible_slabs = slab_gen.get_slabs()

        if not all_possible_slabs:
            print(f"✗ ERROR: Pymatgen failed to generate any slabs for {name} (Miller: {miller}).")
            print("---")
            continue
            
        # Step 5: We must find which of the generated slabs (terminations) matches the 'shift' value reported in the paper.
        print(f"  → Pymatgen generated {len(all_possible_slabs)} unique slab(s). Matching paper's shift={target_shift}...")

        # Get the shift value for every slab Pymatgen generated
        shifts = np.array([s.shift for s in all_possible_slabs])
        
        # Find the index of the slab whose shift is mathematically closest to the one reported in the paper.
        best_match_index = np.argmin(np.abs(shifts - target_shift))
        
        # This is the slab we will use for our benchmark
        slab_pmg = all_possible_slabs[best_match_index]
        found_shift = slab_pmg.shift

        # Step 6: Verification and Warning
        # This is where we check for the mismatch you discovered.
        # A small tolerance (0.005) is used for floating-point comparison.
        if abs(found_shift - target_shift) > 0.005:
            print(f"  ! WARNING: Failed to precisely match paper's shift={target_shift}.")
            print(f"    → Pymatgen's best found shift is: {found_shift:.3f}. Using this approximate surface.")
            print(f"    → All available shifts from Pymatgen: {np.round(shifts, 3)}")
        else:
            print(f"  ✓ Successfully matched: Paper shift={target_shift}, Pymatgen found {found_shift:.3f}")

        # Step 7: Convert the chosen Pymatgen slab to an ASE Atoms object
        # This makes it compatible with your agent's tools (MACE, AutoAdsorbate)
        slab_ase = AseAtomsAdaptor.get_atoms(slab_pmg)
        
        # Ensure standard Periodic Boundary Conditions (PBC) are set (True, True, True)
        # The vacuum is part of the lattice vector 'c', so Z-axis periodicity is correct.
        slab_ase.set_pbc([True, True, True])

        # Step 8: Save the final structure to an .xyz file
        # We use 'extxyz' format to correctly save lattice and properties.
        ase.io.write(slab_output_path, slab_ase, format="extxyz")
        print(f"  ✓ Successfully generated and saved to: {slab_output_path}")

    except Exception as e:
        print(f"✗ ERROR: Failed during processing of {name} (mpid: {mpid}).")
        print(f"  Details: {e}")
    
    print("---")

print("\n--- ✓ Slab generation complete. ---")