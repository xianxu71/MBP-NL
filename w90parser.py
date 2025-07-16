import numpy as np
from constants import a2bohr
from mpi import rank, comm, size

class w90parser:

    """
    Read w90 related files  

    """
    def __init__(self, seed_name, ifmax, mnband): 
    
      """
      Read basic variables in a W90 calculations

      Variables: nk, kpts, kgrid, nz, knbind, knbvec
                 exbands, ibmin, ibmax, nvblist 

      """
      self.seed_name = seed_name
      self.ifmax  = ifmax
      self.mnband = mnband
      self.spn = np.zeros([0])

      self.read_win()
      self.read_nnkp()
      self.read_wout()
      self.check_filling()

      self.report()

      return

    def read_win(self):
      """
      Input file of W90
    
      """
      if rank == 0:
        fname = self.seed_name + '.win'

        with open(fname, 'r') as f:
          for line in f:
            if 'mp_grid' in line:
              self.kgrid = list(map(int, line.split()[-3::]))
      else:
        self.kgrid = None

      self.kgrid = comm.bcast(self.kgrid)

      return


    def read_nnkp(self):
      """
      Read seed_name.nnkp

      ibmin : index of the starting band
      ibmax : index of the last band included
      bands : index of every band used

      kpts   : k-grid used in W90
      knbind : neighboring kpoints

      """
      fname = self.seed_name + '.nnkp'

      if rank == 0:
        with open(fname, 'r') as f:
          kpt_read = False
          for line in f:
            if 'begin kpoints' in line:
              for il, line in enumerate(f):
                if 'end kpoints' in line:
                  kpt_read = True
                  break
                if il == 0:
                  self.nk = int(line.split()[0])
                  self.kpts = np.zeros([self.nk,3])
                else:
                  self.kpts[il-1] = list(map(float, line.split()))

            if kpt_read:
              break      

        with open(fname, 'r') as f:
          nnk_read = False
          for line in f:
            if 'begin nnkpts' in line:
              for il, line in enumerate(f):
                if 'end nnkpts' in line:
                  nnk_read = True
                  break
                if il == 0:
                  self.nz = int(line.split()[0])
                  self.knbind = np.zeros([self.nk,self.nz],dtype=int)
                else:
                  iz = (il-1)%self.nz
                  tmp = list(map(int, line.split()[:2]))
                  self.knbind[tmp[0]-1,iz] = tmp[1]-1

            if nnk_read:
              break      

        # Determine band range
        exbands = []
        with open(fname, 'r') as f:
          nb_read = False
          for line in f:
            if 'begin exclude_bands' in line:
              for il, line in enumerate(f):
                if 'end exclude_bands' in line:
                  nb_read = True
                  break
                if il == 0:
                  nexb = int(line.split()[0])
                  if nexb == 0:
                    nb_read = True
                    break
                else:
                  exbands.append(int(line.split()[0]))

            if nb_read:
              break      

        if nexb != len(exbands):
          raise Exception('Number of excluded bands dose not match')
  
        self.exbands = exbands
      else:
        self.nk = None
        self.kpts = None
        self.nz = None
        self.knbind = None
        self.exbands = None

      self.nk = comm.bcast(self.nk)
      self.kpts = comm.bcast(self.kpts)
      self.nz = comm.bcast(self.nz)
      self.knbind = comm.bcast(self.knbind)
      self.exbands = comm.bcast(self.exbands)

      return

    def check_filling(self):
      """
      Determine the band index relative to ifmax

      """
      self.nvblist = np.zeros([self.nk],dtype=int)

      # band counts from 1
      bands = []
      for i in range(1, self.mnband+1):
        if i not in self.exbands:
          bands.append(i)

      self.nb = len(bands)
      self.ibmin = np.min(bands)
      self.ibmax = np.max(bands)
     
      for i in range(self.nk):
        self.nvblist[i] = self.ifmax[i] - self.ibmin + 1

      return
 
    def read_overlap_old(self):
      """
      Read overlap from .mmn file. 

      overlap : [ns, nk, nz, nb, nb]
     
      """
      fname = self.seed_name + '.mmn'
      
      with open(fname, 'r') as f:
         if rank == 0:
           print( '\n  Reading overlap from', fname)

         # skip first line
         dum = f.readline()
         # header
         nband, nk, nz = list(map(int, f.readline().split()))

         # Consistency check 
         if self.nk != nk:
           print(self.nk, nk)
           raise Exception("nk from header does not match")

         if self.nz != nz:
           print(self.nz, nz)
           raise Exception("nz from header does not match")

         if self.nb != nband:
           print(self.nb, nband)
           raise Exception("nband from header does not match")
    
         overlap = np.zeros([nk, nz, nband, nband], dtype=np.complex)
         knbind = np.zeros([nk,nz], dtype=int)
      
         lines = f.readlines()
         # Read index of neighbor k points
         for i in range(nk):
           ind = i*nz*(nband**2 + 1)
           for j in range(nz):
              ind = (i*nz + j)*(nband**2 + 1)
              knbind[i,j] = int(lines[ind].split()[1]) - 1
    
      if not np.allclose(self.knbind, knbind):
        print('Warning: knbind from .mmn and .nnkp differs!')
        #raise Exception('knbind from .mmn and .nnkp differs! This could happen when nk > 100')

      # Read one block of overlap matrix at a time
      tot = (nband**2 +1) * nk * nz + 2
      for i in range(nk):
        for j in range(nz):
           istart =  (i*nz + j)*(nband**2 + 1) + 1
           iend  =  istart + nband**2
           tmp = np.array([ line.strip().split() for line in lines[istart:iend]], dtype=float)
           overlap[i,j] = (tmp[:,0]+1j*tmp[:,1]).reshape(nband,nband)
    
      # .mmn print in column first order. M(m,n) m first
      overlap = overlap.transpose([0,1,3,2])

      return overlap
  
    def read_overlap(self):
      """
      Read overlap from .mmn file. 
    
      overlap : [ns, nk, nz, nb, nb]
     
      """
      fname = self.seed_name + '.mmn'
      
      with open(fname, 'r') as f:
        if rank == 0:
          print( '\n  Reading overlap from', fname)
    
        # skip first line
        dum = f.readline()
        # header
        nband, nk, nz = list(map(int, f.readline().split()))
    
        # Consistency check 
        if self.nk != nk:
          print(self.nk, nk)
          raise Exception("nk from header does not match")

        if self.nz != nz:
          print(self.nz, nz)
          raise Exception("nz from header does not match")

        if self.nb != nband:
          print(self.nb, nband)
          raise Exception("nband from header does not match")

        overlap = np.zeros([nk, nz, nband, nband], dtype=np.complex)
        knbind = np.zeros([nk,nz], dtype=int)
      
        for i in range(nk):
          ind = i*nz*(nband**2 + 1)
          for j in range(nz):
           ind = (i*nz + j)*(nband**2 + 1)
           # Read index of neighbor k points
           knbind[i,j] = int(f.readline().split()[1]) - 1
           for mb in range(nband):
            for nb in range(nband):
              tmp = list(map(float, f.readline().split()))
              overlap[i,j,mb,nb] = tmp[0] + 1j*tmp[1]
       
      if not np.allclose(self.knbind, knbind):
        #raise Exception('knbind from .mmn and .nnkp differs!')
        print('Warning: knbind from .mmn and .nnkp differs!')

      # .mmn print in column first order. M(m,n) m first
      overlap = overlap.transpose([0,1,3,2])
    
      return overlap
  
    def read_wout(self):
      """
      Read weight of neighboring kpts and b_k vectors from .wout file
      .wout gives neighboring order of first kpt only 

      nnkpt information can also be read from .nnkp file

      """
      fname = self.seed_name + '.wout'
      nk, nz = self.nk, self.nz

      wblist = np.zeros(nz, dtype=float)
      knbvec = np.zeros([nk,nz,3], dtype=float)

      if rank == 0:
        with open(fname, 'r') as f:
           for line in f:
             if 'b_k(x)' in line:
               for iz, line in enumerate(f):
                  if iz <= nz and iz > 0:
                     wblist[iz-1] = float( line.split()[5] )
                     knbvec[0,iz-1] = list(map(float, line.split()[2:5] ))
                  elif iz > nz:
                     break
    
        # Absorb weight into knbvec
        knbvec[0] = np.einsum('ij,i->ij', knbvec[0], wblist * a2bohr )
    
        neighbor_of_G = dict()
        # Record order of first kpt's neighbors
        for iz in range(nz):
           ik = self.which_neighbor(self.kpts[0], self.knbind[0,iz])
           neighbor_of_G[ik] = iz
    
        # Find knbvec for the rest of kpts
        for ik in range(1,nk):
           for iz in range(nz):
    
              # Find which neighbor is this          
              ineighbor = self.which_neighbor(self.kpts[ik], self.knbind[ik,iz])
              
              # get knbvec of other kpts through record of Gamma point 
              ind = neighbor_of_G[ineighbor]
              knbvec[ik,iz] = knbvec[0,ind]

        self.knbvec = knbvec
      else:
        self.knbvec = None

      self.knbvec = comm.bcast(self.knbvec)
    
      return

    def read_spn(self):
      """
      Read matrix elements of spin operators seedname.spn

      """
      fname = self.seed_name + '.spn'

      #if rank == 0:
      with open(fname, 'r') as f:
        print('\n  Reading spin matrix elements from', fname)
        # skip first line
        dum = f.readline()
        # header
        nband, nk = list(map(int, f.readline().split()))

        # Consistency check 
        if self.nk != nk:
          print(self.nk, nk)
          raise Exception("nk from header does not match")

        if self.nb != nband:
          print(self.nb, nband)
          raise Exception("nband from header does not match")

        spn = np.zeros([nk,nband,nband,3], dtype=complex)
        for ik in range(nk):
         for ib in range(nband):
          for jb in range(ib+1):
           for im in range(3):  
              line = f.readline().split()
              spn[ik,jb,ib,im] = float(line[0]) + float(line[1])*1j
              if ib != jb:
                spn[ik,ib,jb,im] = spn[ik,jb,ib,im].conj()
      #else:
      #  spn = None

      #spn = comm.bcast(spn)

      return spn

    def read_AA(self):
      """
      Berry connection matrix elements

      """
      dirs = ['x','y','z']
      d_fname = self.seed_name + '-AA_x.dat'

      with open(d_fname, 'r') as f:
         l1 = f.readline().split()[1:]
         nk, nband, _ = list(map(int, l1))

         print( '\n  Reading dipole matrix from', d_fname)
         print( '    Header info: nk={0:d} nband={1:d}'.format(nk, nband))
    
      dipole = np.zeros([nk, nband, nband, 3], dtype=np.complex)
    
      for i, idir in enumerate(dirs):
        d_fname = seed_name + '-AA_'+ idir+ '.dat'
        mat = np.loadtxt(d_fname)
        mat = mat[...,0] + 1j*mat[...,1]
        dipole[...,i] = mat.reshape(nk, nband, nband)

      # Convert angstrom to bohr
      dipole = dipole * a2bohr

      return dipole

    def which_neighbor(self, k0, iknb):
      """
      Given knb, find its relative position to k0 in kgrid
      FIX: assume kpt in [0,1) 
    
      """
      fk = self.kpts
      kgrid = self.kgrid
    
      kdiff = list(map(int, list(map(round,\
                        np.multiply(fk[iknb] - k0, kgrid)) )))
    
      # check if cross boundary
      for i in range(3):
         if kdiff[i] > kgrid[i]/2:
            kdiff[i] = kdiff[i] - kgrid[i]
         elif kdiff[i] < -kgrid[i]/2:
            kdiff[i] = kdiff[i] + kgrid[i]
    
      return tuple(kdiff)

    def read_gdiv_dipole(self, seed_name, ibot, itop, ifmax):
      """
      Read general derivative from W90
    
      """
      dirs=['x','y','z']
      d_fname = seed_name + '-dAA_xx.dat'
      with open(d_fname, 'r') as f:
         l1 = f.readline().split()[1:]
         nk, nband, _ = list(map(int, l1))
         print( '\n Reading general derivative of dipole matrix from', d_fname)
         print( '  Header info: nk={0:d} nband={1:d}'.format(nk, nband))
    
      gdiv_dipole = np.zeros([nk, nband, nband, 3, 3], dtype=np.complex)
    
      for i, idir in enumerate(dirs):
       for j, jdir in enumerate(dirs):
         d_fname = seed_name + '-dAA_'+ idir+jdir+ '.dat'
         mat = np.loadtxt(d_fname)
         mat = mat[...,0] + 1j*mat[...,1]
         gdiv_dipole[...,i,j] = mat.reshape(nk, nband, nband)
    
      # Select requested bands and reorder matrix to BerkeleyGW order
      # dipole(nk, ncb_u+nvb_u, ncb_u+nvb_u)
      ifmax_wan = ifmax - ibot + 1
      ibot_u = ifmax_wan - nvb_u
      itop_u = ifmax_wan + ncb_u
      if ibot_u < 0 :
         raise Exception('Request more nvb than that in the file')
    
      band_range = list(range(ifmax_wan-1, ibot_u-1, -1))\
                   + list(range(ifmax_wan, itop_u))
      tmp = gdiv_dipole[np.ix_(range(nk),band_range,\
                               band_range,range(3),range(3))]
    
      # Convert angstrom to bohr
      gdiv_dipole = np.array(tmp) * a2bohr**2
    
      return gdiv_dipole

    def report(self):
    
      if rank == 0: 
        print("\n  W90 setup from files:") 
        print("    Number of kpoints:", np.prod(self.kgrid)) 
        print("    Number of neighbors for eack kpoint:", self.nz) 
        print("    Number of bands:", self.nb)

      return
