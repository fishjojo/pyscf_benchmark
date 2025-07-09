#!/usr/bin/env python3

import json
import sys
from pymatgen.ext.matproj import MPRester
from pyscf.pbc import gto, dft, tools

def load_structure_from_mp(material_id, conventional_unit_cell, api_key):
    # Get structure from Materials Project
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(material_id, conventional_unit_cell)
    return structure

def build_cell(structure, 
               basis="gth-szv",
               pseudo="gth-pade",
               super_cell=None,
               spin=0,
               max_memory=8000,
               verbose=4):
    """
    Build Cell object.
    """
    a = structure.lattice.matrix
    atom = list(zip(structure.labels, structure.frac_coords))
    charge = structure.charge
    magmom = None
    if "magmom" in structure.site_properties:
        magmom = structure.site_properties["magmom"]

    cell = gto.Cell()
    cell.a = a
    cell.atom = atom
    cell.fractional = True
    cell.basis = basis
    cell.pseudo = pseudo
    cell.charge = charge
    cell.spin = spin
    cell.magmom = magmom
    cell.max_memory = max_memory
    cell.verbose = verbose
    cell.build()
    if super_cell is not None:
        cell = tools.super_cell(cell, super_cell)
    return cell

def run_dft(cell, kpts, xc):
    mf = dft.KRKS(cell, xc=xc)
    mf.kernel()
    return mf

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pyscf.py input.json")
        sys.exit(1)

    # Read JSON
    with open(sys.argv[1]) as f:
        jobs = json.load(f)

    mp_api_key = "ZzfBvIX2MB03axdpfZCc60pHhcYlanHg"

    for data in jobs:
        material_ids = data["material_ids"]
        conventional_unit_cell = data.get("conventional_unit_cell", False)
        kmesh = data.get("kmesh", [1, 1, 1])
        super_cell = data.get("super_cell", None)
        spin = data.get("spin", 0)
        basis = data.get("basis", "gth-szv")
        pseudo = data.get("pseudo", "gth-pade")
        xc = data.get("xc", "lda,vwn")

        for material_id in material_ids:
            structure = load_structure_from_mp("mp-"+str(material_id), conventional_unit_cell, mp_api_key)
            cell = build_cell(structure, basis=basis, pseudo=pseudo, super_cell=super_cell, spin=spin)
            kpts = cell.make_kpts(kmesh)
            mf = run_dft(cell, kpts, xc)

if __name__ == "__main__":
    main()

