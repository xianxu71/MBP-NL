import numpy as np
import quasiparticle
import dipole
import optical_responses
from constants import Ry2eV
from mpi import MPI, comm, size, rank
import time

if __name__ == '__main__':

   nvband = 6
   ncband = 6
   
   wfn_name = '../input/wfn.h5'
   seed_name = '../input/GeS'
   eqp_fname = '../input/eqp.dat'
   dmat_fname = None
   
   QP = quasiparticle.quasiparticle(\
                nvband, ncband, wfn_name, seed_name, eqp_fname, dmat_fname)

   wmin = 0.0
   wmax = 8.0
   dw = 0.01  
   omega = np.arange(wmin,wmax+dw,dw)

   eta = 0.1
   tetra = False
   brdfun = 'Gaussian'
   
   op = optical_responses.optical_responses( QP, omega, eta, tetra=tetra,\
                                             brdfun=brdfun)

   start = time.time()

   op.calc_absorption_len()
   #op.calc_conductivity()
   #op.calc_shift_current()
   #op.calc_nlo_conductivity(type='T-LPL')
   #op.calc_Jdos()
   
   end = time.time()
   
   if rank == 0:
      print('time elapsed', end-start)
   
