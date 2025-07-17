import h5py as h5
import numpy as np
from constants import eV2Ry, Ry2eV, tol5, t2nsec, a2bohr
from mpi import MPI, comm, size, rank
import dipole as dp
import kpoint as kptinfo
import math_functions as mathfunc
import timer

class quasiparticle:
    """
    Quasiparticle information

    kpoint, energy, occupations, eqp corrections, dipole

    """
    def __init__(self, nvband, ncband, fname, seed_name,\
                 eqp_fname=None, dmat_fname=None):
       """
       Intialize quasiparticle from wfn.h5 or bsemat.h5
     
       """   
       self.nvb = nvband
       self.ncb = ncband
       self.nb = self.ncb + self.nvb

       # Initialize from files
       self.kpoint = kptinfo.kpoint(fname)
       self.read_system(fname)
       self.read_eqpcorr(eqp_fname)
       self.reorder_bands() 

       self.dp = dp.dipole(self, seed_name, dmat_fname)

       return

    def reorder_bands(self):
       """
       Pick out band of interests (nvb, ncb) for relevant quantity 
       {fk, ek, n1b, n2b}
       
       vc, w, and dipole matrix have a valence band order counting from Fermi level
       set fk, and ek to the same order here

       nvb, ncb is the number of valence band read from bsemat.h5

       """
       nrk = self.kpoint.nrk
       # fk and ek has all the valence band from DFT
       # Reorder to BerkeleyGW band order 
       fk = np.zeros([self.nspin, nrk, self.nvb+self.ncb])
       ek = np.zeros([self.nspin, nrk, self.nvb+self.ncb])
       for i in range(nrk):
         for js in range(self.nspin):
           n1m = self.ifmax[js,i] - self.nvb       
           n2m = self.ifmax[js,i] + self.ncb       
           if n1m < 0:
              raise Exception('Requested nvb exceeds number of valence bands!')

           idx = np.append(np.arange(self.ifmax[js,i]-1, n1m-1,-1),\
                           np.arange(self.ifmax[js,i], n2m))
           fk[js,i,:] = self.fk[js,i,idx]
           ek[js,i,:] = self.ek[js,i,idx]

       self.fk = fk[:]
       self.ek = ek[:]
       self.eqpk = ek[:] + self.eqpcorr[:]

       if rank == 0:
          print( '\n  Band reordered to BerkeleyGW orders')
          print( '  with ncb={0:d}, and nvb={1:d}'.format(self.ncb, self.nvb))
          #for i in range(nrk):
          #  print( '\n kpt: {0:5d}  fk'.format(i), self.ek[0,i,:]
          #  print( '         mfe (eV) :', self.mfe[0,i,:]*Ry2eV
          #  print( '         eqp_corr :', self.eqpcorr[i,:]*Ry2eV
          #  print( '         Gap      :', (self.mfe[0,i,self.nvb] + self.eqpcorr[i,self.nvb]\
          #                                     -self.mfe[0,i,0] - self.eqpcorr[i,0]) * Ry2eV

       return


    def read_system(self, fname):
       """
       Read system parameter.

       """
       nrk = self.kpoint.nrk

       if rank == 0:

         with h5.File(fname, 'r') as root:
            self.nspin = root.get('mf_header/kpoints/nspin')[()]
            self.nspinor = root.get('mf_header/kpoints/nspinor')[()]
            self.celvol = root.get('mf_header/crystal/celvol')[()]
            self.alat = root.get('mf_header/crystal/alat')[()]
            self.avec = np.array(root.get('mf_header/crystal/avec'))

            # spin, nkpt, nband
            self.fk = root.get('mf_header/kpoints/occ')[()]
            # spin, nkpt (maxium filling number for eack kpt)
            self.ifmax = root.get('mf_header/kpoints/ifmax')[()]
            # spin, nkpt, nband
            self.ek = root.get('mf_header/kpoints/el')[()]
            self.mnband = root.get('mf_header/kpoints/mnband')[()]
        
            # check if requested bands are there
            if self.nvb > np.amax(self.ifmax) or self.ncb+self.nvb > self.mnband:
             print(np.amax(self.ifmax), self.mnband)
             raise Exception('Number of requested bands exceed that in the file.')

            dim = np.array([self.nspin, self.nspinor, self.mnband,\
                            self.nvb, self.ncb], dtype=np.int32)

       else: # rank != 0

          self.celvol = None
          dim = np.empty(5, dtype=np.int32)
          self.alat = np.empty(1, dtype=np.float64)
          self.avec = np.empty([3,3], dtype=np.float64)
         
       self.celvol = comm.bcast(self.celvol)
       comm.Bcast([dim, MPI.INT])
       comm.Bcast([self.alat, MPI.DOUBLE])
       comm.Bcast([self.avec, MPI.DOUBLE])
       
       # initialize arrays in workers
       if rank != 0:

          self.nspin, self.nspinor, self.mnband, self.nvb, self.ncb = dim
          
          self.fk = np.empty([self.nspin, nrk, self.mnband], dtype=float)
          self.ek = np.empty([self.nspin, nrk, self.mnband], dtype=float)
          self.ifmax = np.empty([self.nspin, nrk], dtype=np.int32)
          
       comm.Bcast([self.fk, MPI.DOUBLE])
       comm.Bcast([self.ek, MPI.DOUBLE])
       comm.Bcast([self.ifmax, MPI.INT])

       # Print system parameters 
       if rank == 0:
          print( '\n  Number of spins:', self.nspin)
          print( '  Number of spinors:', self.nspinor)
          print( '  Maxium occupation number: ', np.amax(self.ifmax))
          print( '  Number of bands in the wfn:', self.mnband)
          print( '\n  Will compute with nvb: {0:d} ncb: {1:d}'.format(self.nvb, self.ncb) )

       return

    def read_eqpcorr(self, eqpfname):
       """
       Read quasi-particle energy correction.
       Also reorder them here 

       eqp1.dat format

       k1, k2, k3,  nb*ns
       spin  band   eMF   eQP
       ...

       """
       nrk = self.kpoint.nrk

       if eqpfname:
         if rank == 0:
            print('\n  Loading quasi-particle energy from {0:s}'.format(eqpfname))
            data = np.loadtxt(eqpfname)
            nbs = int(data[0][3])
            nk = int(data.shape[0]/(nbs+1))
            eqb_min = np.zeros([self.nspin])
            eqb_max = np.zeros([self.nspin])

            if self.nspin == 2:
               eqb_min[0] = int(data[1][1])
               eqb_max[1] = int(data[nbs][1])
               for i in range(nbs):
                 if int(data[i+1][0]) == 2:
                   eqb_max[0] = int(data[i][1])
                   eqb_min[1] = int(data[i+1][1])
                   break
            else:
               eqb_min[0] = int(data[1][1])
               eqb_max[0] = eqb_min[0] + nbs - 1
         
            if nk < nrk:
                raise Exception('  eqp_co.dat does not have enough kpts' )

            # read kpoints
            eqpkpt = [] 
            for i in range(nk):
               eqpkpt.append(data[i*(nbs+1), 0:3])
  
            # check spin
            if self.nspin == 2:
              count1 = 0
              count2 = 0
              for i in range(nbs):
                if data[1+i,0] == 1:
                  count1 = count1 + 1
                elif data[1+i,0] == 2:
                  count2 = count2 + 1
              if count1 != count2:
                raise Exception('  Numbers of spin up and down band differ.')

            # find eqp on reduced grid
            rktoeqpk = -1*np.ones(nrk, dtype=np.int32)
            for i in range(nrk):
              for j in range(nk):
                if np.linalg.norm( self.kpoint.k_range(eqpkpt[j]) \
                                   - self.kpoint.rk[i]) <= 1e-6:
                    rktoeqpk[i] = j
                    break

            if any(rktoeqpk) == -1:
                raise Exception('  Can not find match reduced kpt in eqp_co.dat' )

            eqpcorr = []
            for i in rktoeqpk:
               eqpcorr.append(data[i*(nbs+1)+1:(i+1)*(nbs+1), 3]\
                      -data[i*(nbs+1)+1:(i+1)*(nbs+1), 2])
          
            eqpcorr = np.array(eqpcorr)*eV2Ry
            self.eqpcorr =\
                 np.zeros([self.nspin, nrk, self.ncb+self.nvb])

            # Reorder, since BerkeleyGW counts bands from Fermi level 
            for js in range(self.nspin):
              nsshift = js*(eqb_max[js] - eqb_min[js] + 1)
              for i in range(nrk):
                nvbtop = self.ifmax[js,i]
                nvb_q = int(nvbtop - eqb_min[js] + 1 + nsshift)
                ncb_q = int(eqb_max[js] - nvbtop + nsshift)

                if nvb_q < self.nvb or ncb_q < self.ncb:
                   raise Exception('  eqp file does not have enough bands' )
       
                nbot = nvb_q - self.nvb - 1
                idx = list(range(nvb_q-1,nbot,-1)) + list(range(nvb_q, self.ncb+nvb_q))
                self.eqpcorr[js,i,:] = eqpcorr[i,idx]

            dim = np.array([nrk, self.ncb + self.nvb], dtype=np.int32)
         else:
            dim = np.empty(2, dtype=np.int32)

         comm.Bcast([dim, MPI.INT])
        
         if rank != 0:
            self.eqpcorr = np.empty([self.nspin,dim[0],dim[1]], dtype=np.float64)
     
         comm.Bcast([self.eqpcorr, MPI.DOUBLE])
       else:
          if rank == 0:
             print( '\n No quasi-particle energy given, ignore energy correction.')
          self.eqpcorr = np.zeros([self.nspin, nrk, self.nvb+self.ncb])

       return
   
