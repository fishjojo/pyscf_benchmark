import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import dft
from pyscf.pbc import tools
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as rks_grad

cell = gto.Cell()
cell.a = np.eye(3) * 5.4437023729394527
cell.atom = """
Si  0.75  0.75  0.25
Si  0.00  0.50  0.50
Si  0.75  0.25  0.75
Si  0.00  0.00  0.00
Si  0.25  0.75  0.75
Si  0.50  0.50  0.00
Si  0.25  0.25  0.25
Si  0.50  0.00  0.50
"""
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 140  # in a.u.
cell.max_memory = 32000 # in MB
cell.precision = 1e-8
cell.verbose = 4
cell.use_particle_mesh_ewald = True
cell.fractional = True
cell.build()
cell = tools.super_cell(cell, [4]*3)

mf = dft.RKS(cell, exxdiv=None)
mf.xc = "PBE, PBE"
mf.init_guess = "atom"
mf.with_df = multigrid.MultiGridFFTDF2(cell)
mf.kernel()

grad = rks_grad.Gradients(mf)
g = grad.kernel()
