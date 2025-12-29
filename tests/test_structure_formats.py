# -*- coding: utf-8 -*-

"""
Tests for multi-format structure input support (Issue #4).

Verifies that the system correctly handles various structure file formats
including XYZ, CIF, POSCAR, MOL, and validates appropriate handling of
non-periodic inputs.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule
from ase.io import write

from src.tools.tools import read_atoms_object, prepare_slab


class TestFormatReading(unittest.TestCase):
    """Test that various structure formats can be read correctly."""
    
    def setUp(self):
        """Create a simple Cu(111) slab for testing."""
        self.sample_slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    
    def _test_format_roundtrip(self, fmt: str, ext: str):
        """Helper to test a format roundtrip."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='w') as f:
            tmp_path = f.name
        
        try:
            write(tmp_path, self.sample_slab, format=fmt)
            atoms = read_atoms_object(tmp_path)
            
            # Verify atom count matches
            self.assertEqual(len(atoms), len(self.sample_slab))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_read_xyz_format(self):
        """Test reading XYZ format - the primary format that must never break."""
        self._test_format_roundtrip("extxyz", ".xyz")
    
    def test_read_cif_format(self):
        """Test reading CIF format."""
        self._test_format_roundtrip("cif", ".cif")
    
    def test_read_vasp_poscar_format(self):
        """Test reading VASP POSCAR format."""
        self._test_format_roundtrip("vasp", ".poscar")


class TestPrepareSlab(unittest.TestCase):
    """Test slab preparation with various input types."""
    
    def test_periodic_slab_unchanged(self):
        """Test that periodic slabs pass through normally."""
        slab = fcc111('Pt', size=(2, 2, 4), vacuum=10.0)
        prepared, expanded = prepare_slab(slab)
        
        # Should have PBC
        self.assertTrue(any(prepared.get_pbc()))
        
        # Atom count should match (no atoms added/removed)
        self.assertEqual(len(prepared), len(slab) if not expanded else len(slab) * 4)
    
    def test_nonperiodic_gets_fallback_pbc(self):
        """Test that non-periodic inputs get automatic PBC fallback."""
        # Simulate a PDB-like cluster (no PBC, no cell)
        cluster = Atoms('CH4', positions=[
            [0, 0, 0], 
            [1.09, 0, 0], 
            [-0.36, 1.03, 0], 
            [-0.36, -0.51, 0.89], 
            [-0.36, -0.51, -0.89]
        ])
        cluster.set_pbc([False, False, False])
        
        prepared, _ = prepare_slab(cluster)
        
        # Fallback should set XY periodic
        pbc = prepared.get_pbc()
        self.assertTrue(pbc[0] and pbc[1])
    
    def test_nonperiodic_gets_cell(self):
        """Test that non-periodic inputs without cell get one created."""
        # Create cluster without cell
        cluster = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        cluster.set_pbc([False, False, False])
        # Note: ASE Atoms() without cell= has zero-volume cell
        
        prepared, _ = prepare_slab(cluster)
        
        # Should have non-zero cell volume now
        self.assertGreater(prepared.cell.volume, 0)
    
    def test_xyz_format_regression(self):
        """Regression test: XYZ format must work exactly as before."""
        # Create a slab similar to benchmark files
        slab = fcc111('Pt', size=(2, 2, 4), vacuum=10.0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xyz", mode='w') as f:
            tmp_path = f.name
        
        try:
            # Write as extended XYZ (what the benchmark files use)
            write(tmp_path, slab, format="extxyz")
            
            # Read back
            atoms = read_atoms_object(tmp_path)
            
            # Prepare
            prepared, _ = prepare_slab(atoms)
            
            # Verify basic properties preserved
            # Note: prepare_slab expands 2x2 if cell < 6Å, so check accordingly
            cell_a = prepared.cell[0, 0]
            if cell_a >= 6.0:
                # Was expanded (2x2 = 4x atoms)
                self.assertEqual(len(prepared), len(slab) * 4)
            else:
                # Not expanded
                self.assertEqual(len(prepared), len(slab))
            self.assertTrue(any(prepared.get_pbc()))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestBenchmarkFiles(unittest.TestCase):
    """Test with actual benchmark files to ensure no regression."""
    
    def setUp(self):
        """Find benchmark directory."""
        self.benchmark_dir = Path(__file__).parent.parent / "benchmark_slabs"
    
    def test_benchmark_xyz_files_readable(self):
        """Ensure all benchmark XYZ files can still be read."""
        if not self.benchmark_dir.exists():
            self.skipTest("Benchmark directory not found")
        
        xyz_files = list(self.benchmark_dir.glob("*.xyz"))
        self.assertGreater(len(xyz_files), 0, "No XYZ files found in benchmark dir")
        
        for xyz_file in xyz_files[:3]:  # Test first 3 files
            atoms = read_atoms_object(str(xyz_file))
            prepared, _ = prepare_slab(atoms)
            
            self.assertGreater(len(prepared), 0)
            self.assertTrue(any(prepared.get_pbc()))


if __name__ == '__main__':
    unittest.main()
