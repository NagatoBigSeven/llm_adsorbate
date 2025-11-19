import asyncio
import os
import shutil
import numpy as np
from ase.io import write
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_slabs_with_miller_indices

SYSTEMS_TO_FETCH = [
    (1, "H", "mp-1186014", 0.167, (1, 1, 1)),
    (2, "NNH", "mp-1186014", 0.167, (1, 1, 1)),
    (3, "H", "mp-1184119", 0.063, (1, 1, 1)),
    (4, "NNH", "mp-1184119", 0.063, (1, 1, 1)),
    (5, "H", "mp-1184011", 0.063, (1, 1, 1)),
    (6, "NNH", "mp-1184011", 0.063, (1, 1, 1)),
    (7, "H", "mp-975834", 0.167, (1, 1, 1)),
    (8, "NNH", "mp-975834", 0.167, (1, 1, 1)),
    (9, "OH", "mp-126", 0.167, (1, 1, 1)),
    (10, "OH", "mp-126", 0.250, (1, 0, 0)),
    (11, "OH", "mp-2", 0.167, (1, 1, 1)),
    (12, "OH", "mp-81", 0.167, (1, 1, 1)),
    (13, "OH", "mp-124", 0.250, (1, 0, 0)),
    (14, "OH", "mp-1225998", 0.042, (1, 1, 1)),
    (15, "CH2CH2OH", "mp-865798", 0.248, (1, 0, 0)),
    (16, "CH2CH2OH", "mp-1018153", 0.028, (1, 0, 2)),
    (17, "OCHCH3", "mp-2583", 0.083, (1, 1, 1)),
    (18, "OCHCH3", "mp-1065309", 0.375, (1, 0, 1)),
    (19, "OCHCH3", "mp-866108", 0.120, (1, 1, 0)),
    (20, "ONN(CH3)2", "mp-866201", 0.000, (2, 1, 1)),
]

downloaded_slabs = {}

async def fetch_slab_smart(idx, original_ads, mpid, target_shift, miller):
    filename = f"benchmark_slabs/{idx:02d}_{mpid}_{original_ads}.xyz"
    
    if os.path.exists(filename):
        print(f"[{idx:02d}] File exists, skipping.")
        return

    cache_key = (mpid, miller, round(target_shift, 3))
    if cache_key in downloaded_slabs:
        existing_file = downloaded_slabs[cache_key]
        print(f"[{idx:02d}] Surface matches System {existing_file}. Copying file...")
        shutil.copy(existing_file, filename)
        return

    print(f"[{idx:02d}] Fetching from OCP API (using *H proxy)...")
    
    try:
        proxy_ads = "*H" 
        
        results = await find_adsorbate_binding_sites(
            adsorbate=proxy_ads,
            bulk=mpid,
            adslab_filter=keep_slabs_with_miller_indices([miller])
        )
        
        matched_slab = None
        for slab_container in results.slabs:
            slab_shift = slab_container.slab.metadata.shift
            if np.isclose(slab_shift, target_shift, atol=0.01):
                matched_slab = slab_container
                break
        
        if matched_slab is None:
            print(f"  ✗ No match for shift {target_shift}. Avail: {[s.slab.metadata.shift for s in results.slabs]}")
            return

        adslab_atoms = matched_slab.configs[0].to_ase_atoms()
        
        tags = adslab_atoms.get_tags()
        clean_slab = adslab_atoms[tags != 2]
        
        write(filename, clean_slab, format='extxyz')
        print(f"  ✓ Downloaded and cleaned: {filename}")
        
        downloaded_slabs[cache_key] = filename

    except Exception as e:
        print(f"  ✗ Error: {e}")

async def main():
    if not os.path.exists("benchmark_slabs"):
        os.makedirs("benchmark_slabs")
        
    for system in SYSTEMS_TO_FETCH:
        idx, ads, mpid, shift, miller = system
        await fetch_slab_smart(idx, ads, mpid, shift, miller)

if __name__ == "__main__":
    asyncio.run(main())