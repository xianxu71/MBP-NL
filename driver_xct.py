import numpy as np
import quasiparticle
import exciton
import dipole
import optical_responses
from constants import Ry2eV
from mpi import MPI, comm, size, rank
import time

if __name__ == '__main__':

   nvband = 6
   ncband = 6
   nxct = 300
   
   wfn_name = '../input/wfn.h5'
   seed_name = '../input/GeS'
   eqp_fname = '../input/eqp.dat'
   dmat_fname = None
   xct_fname = '../input/eigenvectors.h5'
   
   QP = quasiparticle.quasiparticle(\
                nvband, ncband, wfn_name, seed_name, eqp_fname, dmat_fname)

   Xct = exciton.exciton(QP, nb=nxct, fname=xct_fname)

   wmin = 0.0
   wmax = 8.0
   dw = 0.01  
   omega = np.arange(wmin,wmax+dw,dw)

   eta = 0.05
   tetra = False
   brdfun = 'Lorentzian'
   
   op = optical_responses.optical_responses( QP, omega, eta, tetra=tetra,\
                                             brdfun=brdfun, exciton=Xct)

   start = time.time()

   #op.calc_absorption_len()
   #op.calc_conductivity()
   #op.calc_shift_current()
   #op.calc_nlo_conductivity(type='T-LPL')
   #op.calc_Jdos()

   #op.calc_absorption_with_eh()
   #op.calc_shift_current_with_eh()
   op.calc_SHG_with_eh()
   
   end = time.time()
   
   if rank == 0:
      print('time elapsed', end-start)
   
