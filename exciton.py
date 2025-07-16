import h5py as h5
import numpy as np
from mpi import size, rank, comm
from constants import Ry2eV, tol12


class exciton:
    """
    Exciton properties:

    energy, envelop functions,
    exciton-phonon coupling matrix elements

    """
    def __init__(self, QP, nb=-1, loadevecs=True, fname='eigenvectors.h5'):
       """
       Initialize exciton properties

       """
       self.nb = nb
       self.loadevecs = loadevecs
       self.init_from_file(fname)

       self.reorder_kpts(QP.kpoint.rk)

       # Compute exciton dipole, exciton velocity
       # xeh  : dipole in exciton picture
       #        xeh_n = Psi^(n)_cvk * r_vck
       # peh  : momentum in exciton picture
       inds = np.ix_(range(1),range(QP.kpoint.nrk),range(self.nvb),\
                     range(QP.nvb,QP.nvb+self.ncb),range(3))

       if self.loadevecs:
         self.reh = np.einsum('skcv,kvcn->sn', self.evecs,\
                                               QP.dp.int_dipole[inds][0])
         self.veh = np.einsum('skcv,kvcn->sn', self.evecs,\
                                               QP.dp.int_velocity[inds][0])

       self.distribute_workload()

       return

    def init_from_file(self, fname):
       """

       """
       if rank == 0:
         print('\n  Initializing exciton from file:', fname)

       with h5.File(fname, 'r') as f:
         # !! Be careful that Q-shift is NOT exciton COM
         self.Qpts = f.get('exciton_header/kpoints/exciton_Q_shifts')[()]
         #self.Qpts = f.get('exciton_header/kpoints/Qpts')[()]
         self.kpts = f.get('exciton_header/kpoints/kpts')[()]
         self.nfk = f.get('exciton_header/kpoints/nk')[()]
         self.nevecs = f.get('exciton_header/params/nevecs')[()]
         if self.nb < 0:
           self.nb = self.nevecs
         else:
           self.nevecs = self.nb
           
         self.nvb = f.get('exciton_header/params/nv')[()]
         self.ncb = f.get('exciton_header/params/nc')[()]
         self.xdim = f.get('exciton_header/params/bse_hamiltonian_size')[()]
            
         self.blat = f.get('mf_header/crystal/blat')[()]
         self.bvec = f.get('mf_header/crystal/bvec')[()]

         # eigenvalues in eV 
         self.evals = f.get('exciton_data/eigenvalues')[:self.nb]/Ry2eV

         if rank == 0:
           mem = self.xdim*self.nevecs*16*2/1e9
           print('\n  Estimate memory needed for loading excitons {0:6.3e} (GB)'.format(mem))

         # FIX!! ignore spin, assume nQ=1
         # leave eigenvectors in the matrix form
         # evecs[nQ, nevecs, nk, nc, nv, ns, :]
         if self.loadevecs:
            tmp = f.get('exciton_data/eigenvectors')[0,:self.nb]
            self.evecs = tmp[...,0,0] + 1j*tmp[...,0,1]

       if not self.loadevecs:
         f = h5.File(fname,'r')
         self.evecs = f['exciton_data/eigenvectors']

       return

    def init_from_file_bcast(self, fname):
       """

       """
       if rank == 0:
         print('\n  Initializing exciton from file:', fname)
         with h5.File(fname, 'r') as f:
           # !! Be careful that Q-shift is NOT exciton COM
           self.Qpts = f.get('exciton_header/kpoints/exciton_Q_shifts')[()]
           #self.Qpts = f.get('exciton_header/kpoints/Qpts')[()]
           self.kpts = f.get('exciton_header/kpoints/kpts')[()]
           self.nevecs = f.get('exciton_header/params/nevecs')[()]
           if self.nb < 0:
             self.nb = self.nevecs
           else:
             self.nevecs = self.nb
             
           self.nvb = f.get('exciton_header/params/nv')[()]
           self.ncb = f.get('exciton_header/params/nc')[()]
           self.xdim = f.get('exciton_header/params/bse_hamiltonian_size')[()]
              
           self.blat = f.get('mf_header/crystal/blat')[()]
           self.bvec = f.get('mf_header/crystal/bvec')[()]

           # eigenvalues in eV 
           self.evals = f.get('exciton_data/eigenvalues')[:self.nb]/Ry2eV
           tmp = f.get('exciton_data/eigenvectors')[()]
           # FIX!! ignore spin, assume nQ=1
           # leave eigenvectors in the matrix form
           # evecs[nQ, nevecs, nk, nc, nv, ns, :]
           self.evecs = tmp[0,:self.nb,:,:,:,0,0] + 1j*tmp[0,:self.nb,:,:,:,0,1]
       else:
         self.Qpts = None
         self.kpts = None
         self.nevecs = None
         self.evals = None 
         self.evecs = None 
         self.nvb = None 
         self.ncb = None 
         self.xdim = None 
         self.blat = None
         self.bvec = None

       self.Qpts = comm.bcast(self.Qpts) 
       self.kpts = comm.bcast(self.kpts) 
       self.nevecs = comm.bcast(self.nevecs) 
       self.evals = comm.bcast(self.evals) 
       self.evecs = comm.bcast(self.evecs) 
       self.nvb = comm.bcast(self.nvb) 
       self.ncb = comm.bcast(self.ncb) 
       self.xdim = comm.bcast(self.xdim) 
       self.blat = comm.bcast(self.blat) 
       self.bvec = comm.bcast(self.bvec) 

       return

    def reorder_kpts(self, kref):
       """
       Reorder kpoints to match the kpt order in QP

       """
       from scipy.spatial import cKDTree
 
       ktmp = np.zeros_like(kref)
       for ii, kpt in enumerate(kref):
         ktmp[ii] = self.k_range(kpt)

       tree = cKDTree(ktmp)

       self.mapkreftok = np.zeros([len(kref)], dtype=int)
       for ii, kpt in enumerate(self.kpts):
         ds, ind = tree.query(self.k_range(kpt), 1)
         self.mapkreftok[ind] = ii

       # reorder kpts in eigenvectors to match those in QP
       if self.loadevecs:
         self.evecs = self.evecs[:,self.mapkreftok[:],...]

       return

    def distribute_workload(self):

       max_nkpt_per_worker = int( np.ceil(self.nevecs *1.0 / size) )

       self.my_xcts = list()

       for i in range(max_nkpt_per_worker):

          ikpt = rank * max_nkpt_per_worker + i

          if ikpt >= self.nevecs:
              break

          self.my_xcts.append(ikpt)

       self.my_nxct = len(self.my_xcts)

       return         

    def k_range(self, kin):
       """
       put kpoint in [0,1)

       """
       for i in range(3):
         while kin[i] < -tol12:
            kin[i] = kin[i] + 1.0
         while kin[i] >= 1-tol12:
            kin[i] = kin[i] - 1.0

       return kin
