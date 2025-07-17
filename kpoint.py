import numpy as np
import h5py as h5
from mpi import MPI, comm, size, rank
from constants import tol5

class kpoint:
    """
    kpoint information

    """
    def __init__(self, fname):

       """
       Initialize from h5 file or from inputs

       """
       self.init_from_file(fname)

       self.distribute_workload()

       return

    def init_from_file(self,fname):
       """
       load kpoint information from mf_header

       """
       if rank == 0:

         with h5.File(fname, 'r') as root:
           # nkpt, 3
           self.rk = np.array(root.get('mf_header/kpoints/rk'))
           self.nrk = root.get('mf_header/kpoints/nrk')[()] 
           self.bdot = np.array(root.get('mf_header/crystal/bdot'))
           self.bvec = np.array(root.get('mf_header/crystal/bvec'))
           self.blat = np.array(root.get('mf_header/crystal/blat'))
           self.kgrid = root.get('mf_header/kpoints/kgrid')[()]
           self.w = root.get('mf_header/kpoints/w')[()]
        
           self.symmat = root.get('mf_header/symmetry/mtrx')[()]
           self.ntran = root.get('mf_header/symmetry/ntran')[()]
           self.tnp = root.get('mf_header/symmetry/tnp')[()]
        
           dim = np.array([self.nrk, self.ntran], dtype=np.int32)

       else: # rank != 0

          dim = np.empty(2, dtype=np.int32)
          self.blat = np.empty(1, dtype=np.float64)
          self.bvec = np.empty([3,3], dtype=np.float64)
          self.bdot = np.empty([3,3], dtype=np.float64)
          self.kgrid = np.empty([3], dtype=np.int32)
         
       comm.Bcast([dim, MPI.INT])
       comm.Bcast([self.blat, MPI.DOUBLE])
       comm.Bcast([self.bvec, MPI.DOUBLE])
       comm.Bcast([self.bdot, MPI.DOUBLE])
       comm.Bcast([self.kgrid, MPI.INT])
       
       # Send arrays to workers
       if rank != 0:
          self.nrk, self.ntran = dim
          
          self.rk   = np.empty([self.nrk,3], dtype=float)
          self.w    = np.empty([self.nrk], dtype=float)
          self.symmat = np.empty([48,3,3], dtype=np.int32)
          self.tnp = np.empty([48,3], dtype=float)

       comm.Bcast([self.rk, MPI.DOUBLE])
       comm.Bcast([self.symmat, MPI.INT])
       comm.Bcast([self.tnp, MPI.DOUBLE])
       comm.Bcast([self.w, MPI.DOUBLE])

       if rank == 0:
          self.report_variables(fname)

       return  

    def load_bse_kpts(self, fname):
       """
       Load k-grid information in bse_header

       """
       if rank == 0:
         with h5.File(fname, 'r') as root:
            self.nbk = root.get('bse_header/kpoints/nk')[()]
            self.bkpts = np.array(root.get('bse_header/kpoints/kpts'))

            dim = np.array([self.nbk], dtype=int)
       else: # rank != 0
          dim = np.empty(1, dtype=np.int32)

       comm.Bcast([dim, MPI.INT])

       if rank != 0:
          self.nbk = dim
          self.bkpts = np.empty([self.nbk,3], dtype=float)

       comm.Bcast([self.bkpts, MPI.DOUBLE])

       self.map_full_bz()

       return

    def map_full_bz(self):
       """
       Find the map between kpoints in bsemat.h5 and mean-field
       vmtxel's kpts is ordered as those in kernel

       """
       self.fktork = []
       self.rktofk = [[] for i in range(self.nrk)]
       for i in range(self.nbk):
         kpt = self.bkpts[i]

         for j in range(self.ntran):
           qpt = np.dot(kpt, self.symmat[j])
           qpt = self.k_range(qpt)

           found = False
           for k in range(self.nrk):
              if np.linalg.norm(qpt - self.rk[k]) < tol5:
                 self.fktork.append(k)
                 self.rktofk[k].append(i)
                 found = True
                 break

           if found:
              break
          
         if not found:
             print('Can not find matching point in reduced grid, kpt= ', kpt)

       return

    def construct_kneighbor(self):
  
       # This function is used in tight-binding model only 
       # all workers save the same copy
       # kpt is identified from 'my_ikpts' later

       self.knbind = []
       for i in range(self.nfk):
          # FIX: this is for hexagonal cell now
          # need to be generalized for arbitrary cell
          klist = self.ind_kneighbor(i, self.dimK)
          self.knbind.append(klist)

       # overlap is stored in the order
       # ( b1, b2, -b1, -b2, b1+b2, -(b1+b2) )

       b1 = self.blat*self.bvec[0] / self.dimK
       b2 = self.blat*self.bvec[1] / self.dimK
       self.knbvec = [b1, b2, -b1, -b2, b1+b2, -b1-b2]

       return

    def k_range(self, kin):
       """
       put kpoint in [0,1)

       """
       for i in range(3):
         while kin[i] < -tol5:
            kin[i] = kin[i] + 1.0
         while kin[i] >= 1-tol5:
            kin[i] = kin[i] - 1.0

       return kin

    def report_variables(self, fname):
       """
       Report information in the file

       """
       print( '\n  Parameters read from {0:s}:\n'.format(fname))
       print( '  Reciprocal lattice vectors')
       print( '  blat =', self.blat )
       print( '    b1 = ', self.bvec[0] )
       print( '    b2 = ', self.bvec[1] )
       print( '    b3 = ', self.bvec[2], '\n')
       print( '  Number of kpts: {0:d}'.format(self.nrk))

       #if True:
       #   print '\n  List of kpts in the k-grid:\n'
       #   for i in range(self.nrk):
       #      print '   ', self.rk[i]

       #print '\n  Mapping from full k-grid to reduced grid:\n'
       #for i in range(self.nfk):
       #   print '   {0:d} -> {1:d} '.format(i, self.fktork[i]) 
     
       return   
 
    def ind_kneighbor(self, indk, dimk):
       """
       Given a kpt index find its 6 neighbors
       with the order ( b1, b2, -b1, -b2, b1+b2, -(b1+b2) )

       """
       # grid coodinate of indk
       ix = indk%dimk
       iy = indk/dimk
   
       ixp = (ix+1)%dimk
       iyp = (iy+1)%dimk
   
       ixm = (ix+dimk-1)%dimk
       iym = (iy+dimk-1)%dimk
   
       kneighbors = [ iy*dimk+ixp, iyp*dimk+ix, iy*dimk+ixm,\
                   iym*dimk+ix, iyp*dimk+ixp, iym*dimk+ixm]

       return kneighbors

    def active_worker(self):
       return bool(self.my_ikpts)

    def distribute_workload(self):

       max_nkpt_per_worker = int( np.ceil(self.nrk *1.0 / size) )
       n_active_workers = self.nrk // max_nkpt_per_worker +\
                           min(self.nrk % max_nkpt_per_worker, 1)
       self.active_ranks = np.arange(n_active_workers)

       self.my_ikpts = list()

       for i in range(max_nkpt_per_worker):

          ikpt = rank * max_nkpt_per_worker + i

          if ikpt >= self.nrk:
              break

          self.my_ikpts.append(ikpt)

       self.my_nrk = len(self.my_ikpts)

       ## Store size of density matrix for every local workers
       ## then compute the offsets
       #self.gsizes = np.zeros(size)
       #for j in range(size):
       #  for i in range(max_nkpt_per_worker):
       #     ikpt = j * max_nkpt_per_worker + i

       #     if ikpt >= self.nrk:
       #         break

       #     self.gsizes[j] = self.gsizes[j] + 1
  
       ## FIX: assume always use complex
       #self.gsizes = self.gsizes* (self.nvb_u+self.ncb_u)**2 * 2
       #self.offsets = np.zeros(size)
       #self.offsets[1:] = np.cumsum(self.gsizes)[:-1]

       #for i in range(size):
       #   if rank == i:
       #      print( " Rank", i, "gsize", self.gsizes)
       #      print( " offsets", self.offsets)
       #   comm.Barrier()

       return         
