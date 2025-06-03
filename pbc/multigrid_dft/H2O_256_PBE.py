import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import dft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as rks_grad

cell = gto.Cell()
cell.a = np.eye(3) * 19.7340
cell.atom = "H2O_256.xyz"
cell.basis = "gth-tzv2p"
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 200  # in a.u.
cell.max_memory = 64000 # in MB
cell.precision = 1e-6
cell.verbose = 4
cell.use_particle_mesh_ewald = True
cell.build()

mf = dft.RKS(cell, exxdiv=None)
mf.xc = "PBE, PBE"
mf.init_guess = "atom"
mf.with_df = multigrid.MultiGridFFTDF2(cell)
mf.kernel()

grad = rks_grad.Gradients(mf)
g = grad.kernel()
