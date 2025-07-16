import numpy as np
from mpi import MPI, comm, size, rank, get_gsize_offset
from constants import tol5
import w90parser
import bgwparser
#import psutil
import time

class dipole:

    """
    Compute dipole, velocity, and general derivative
    
    """

    def __init__(self, QP, seed_name, bse_fname=None, dmat_fname=None,\
                 read_dpW90=False):
       """
       Initialize from W90 or BerkeleyGW vmtxel.dat

       """
       self.nspin, self.nrk = QP.nspin, QP.kpoint.nrk
       self.my_ikpts, self.my_nrk = QP.kpoint.my_ikpts, QP.kpoint.my_nrk
       self.nvb, self.ncb, self.nb = QP.nvb, QP.ncb, QP.nb
       self.mnband, self.ifmax = QP.mnband, QP.ifmax

       self.w90parser = []

       if dmat_fname and bse_fname:
          self.init_from_BGW(bse_fname, dmat_fname) 
       else:
          self.init_from_W90(seed_name, read_dpW90)

       self.get_inter_dipole(QP.eqpk)

       if not bse_fname or not dmat_fname:
          self.compute_velocity_from_dipole(QP, eqp_correction=True)

       return

    def init_from_BGW(self, bse_fname, dmat_fname):
       """
       Read vmtxel.dat,
       k-grid information is stored in bsemat.h5

       """
       if rank == 0:
         print( '\n  Read dipole from BerkeleyGW vmtxel.dat')
         print( '  Polarziation of Efield is fixed in the calculations.')

       QP.kpoints.load_bse_kpts(bse_fname)

       ns, nk, nb = self.nspin, self.nrk, self.nb 

       if rank == 0:
         self.read_dipole_BGW(QP, dmat_fname)
       else:
         self.dipole = np.zeros([ns,nk,nb,nb], dtype=complex)

       comm.Bcast(self.dipole)

       return

    def init_from_W90(self,seed_name, read_dpW90):
       """
       Read basic information of a W90 calculation
       then compute dipole

       """
       if self.nspin == 2:
         tmp1 = w90parser.w90parser(seed_name+'_up',self.ifmax[0],self.mnband)
         self.w90parser.append(tmp1)
         tmp2 = w90parser.w90parser(seed_name+'_dw',self.ifmax[1],self.mnband)
         self.w90parser.append(tmp2)
       else:
         tmp1 = w90parser.w90parser(seed_name,self.ifmax[0],self.mnband)
         self.w90parser.append(tmp1)

       # Consistency check
       if self.w90parser[0].nk != self.nrk:
         raise Exception('Number of kpoints inconsistent')

       if self.nspin == 2:
         if not (np.allclose(self.w90parser[0].knbind,\
                             self.w90parser[1].knbind) and \
                 np.allclose(self.w90parser[0].knbvec,\
                             self.w90parser[1].knbvec)):
          raise Exception(' Kpoints in _up and _dw files are not consistent')
 
       if rank == 0:
         print( '\n  Compute dipole from W90 overlap .mmn files')
       self.compute_dipole_W90()

       return

    def compute_dipole_W90(self):
       """
       Read overlap matrix elements from .mmn
       then compute dipole matrix elements

       overlap : [ns, nk, nz, nb, nb]
       knbind  : [nk, nz]
       knbvec  : [nk, nz, 3] -> [nk, nz] after knbvec*pol

       """
       nk = self.w90parser[0].nk
       nz = self.w90parser[0].nz
       nb = self.w90parser[0].nb
       ns = self.nspin
       knbind  = self.w90parser[0].knbind
       knbvec  = self.w90parser[0].knbvec
       nvblist = [self.w90parser[i].nvblist for i in range(self.nspin)]

       nbs = self.nb

       overlap = np.zeros([ns, nk, nz, nb, nb], dtype=complex)

       # TODO: parallelize this
       # !! Everyone reads is more efficient   
       for i in range(self.nspin):
         overlap[i] = self.w90parser[i].read_overlap() 

       #comm.Bcast(overlap)

       overlap, invS = self.get_covariant_overlap_check(overlap)
    
       overlap = \
          self.reorder_overlap(overlap, self.nvb, self.ncb, nvblist)

       invS = \
          self.reorder_overlap(invS, self.nvb, self.ncb, nvblist)
       self.invS = invS
 
       self.dipole = self.compute_MV_dipole_from_overlap(overlap, knbvec)
  
       overlap = None

       return
    
    def reorder_overlap(self, overlap, nvb, ncb, nvblist):
        """
        Reorder overlap to be consistent with BerkeleyGW's order
        and pick requested bands
    
        """
        ns, nk, nz = overlap.shape[:3]
        tol = 1e-10
    
        nb = nvb + ncb
        overlap_bgw = np.zeros([ns,nk,nz,nb,nb], dtype=complex)
    
        for js in range(ns):
          for i in range(nk):
             nfmax = nvblist[js][i]
             n1m = nfmax - nvb 
             n2m = nfmax + ncb       
      
             if n1m < 0:
                raise Exception('Assigned nvb is larger than that from file.')
    
             idx = list(range(nfmax-1, n1m-1,-1)) + list(range(nfmax, n2m))
             inds = np.ix_(range(js,js+1),range(i,i+1),range(nz),idx,idx)
             overlap_bgw[js,i,:] = overlap[inds]
    
        ## Check overlap order
        #for js in range(ns):
        # for i in range(nk):
        #  nfmax = nvblist[js][i]
        #  for j in range(nz):
        #   for k1 in range(nb):
        #    for k2 in range(nb):
        #     if k1 < nvb:
        #        k1p = nfmax - k1 - 1
        #     else:
        #        k1p = nfmax - nvb + k1
        #     if k2 < nvb:
        #        k2p = nfmax - k2 - 1
        #     else:
        #        k2p = nfmax - nvb + k2
    
        #     if np.abs(overlap_bgw[js,i,j,k1,k2] - overlap[js,i,j,k1p,k2p]) > tol:
        #        raise Exception('Something wrong in the overlap order')
    
        return overlap_bgw

    
    def read_dipole_BGW(self, QP, fname):
        """
        Read dipole from vmtxel of ascii format
        reduced the number of bands to nvb and ncb

        dipole[nspin, nk, nvb+ncb, nvb+ncb)

        Note that kpoint order of vmtxel follows that in kernel
    
        By default BerkeleyGW compute cv elements so padded with zero
        """
        nvb, ncb, nb, nspin = self.nvb, self.ncb, self.nb, self.nspin

        dipole = bgwparser.read_vmtxel(fname)        
        nk, ncband, nvband = dipole.shape[1:]
        ntot = ncband + nvband

        # Compatibility check
        if self.nrk != nk:
           raise Exception('Number of kpoint do not match')
        
        print( '  Padding zeros to dipole matrix', ncband, nvband, nvb, ncb)

        tmp = np.zeros([nk, nspin, ntot, ntot], dtype=complex)
        tmp[np.ix_(range(nspin),range(nk),range(nvband,nctot),range(nvband))]\
            = dipole[:]

        for js in range(nspin):
          for i in range(nk):
            tmp[js,i] = tmp[js,i] + tmp[js,i].conjugate().T
    
        # Select requested bands
        if nvband > nvb or ncband > ncb:
           # dipole(nk, ncb+nvb, ncb+nvb)
           band_range = list(range(nvb)) + list(range(nvb, nb))
           tmp = tmp[np.ix_(range(nspin),range(nk),band_range,band_range)]
    
        dipole = np.zeros([nspin, nk, ncvb, ncvb], dtype=complex)
        for i in range(nk):
           dipole[:,QP.kpoint.fktork[i],...] = tmp[:,i,...]
    
        return dipole
    
    def compute_MV_dipole_from_overlap(self, overlap, knbvec):
       """
       Compute dipole from overlap
    
       overlap: [nspin, nk, nz, nb, nb]
       knbvec : [nk, nz, 3]
    
       derivative schemes: MV_auto
    
       """
       nspin, nk, nz, nb, _ = overlap.shape
       dipole = np.zeros([nspin, nk, nb, nb, 3], dtype=complex)
       gbuffer = np.zeros([nk, nb, nb, 3], dtype=complex)
       gsizes, offsets = get_gsize_offset(nk, 3*nb**2)

       # M-V derivative
       for js in range(nspin):
         pd = np.zeros([self.my_nrk, nb, nb, 3], dtype=complex)
         for ik,ikf in enumerate(self.my_ikpts):
           pd[ik] = 1j*np.einsum('zij,zm->ijm', overlap[js,ikf], knbvec[ikf])

         gbuffer[:] = 0.
         comm.Allgatherv( pd,\
                 [gbuffer, gsizes, offsets, MPI.COMPLEX])
         dipole[js] = np.array(gbuffer[:])
    
       # set hermiticity
       for js in range(nspin):
         for ik in range(nk):
            dipole[js,ik] = 0.5*(dipole[js,ik]\
                            + dipole[js,ik].transpose([1,0,2]).conj())
    
       return dipole
    
    def compute_IV_dipole_from_overlap(overlap, dimK, nzlist,\
                                      alat, avec, pol_vec):
       """
       Compute dipole from overlap.
    
       derivative schemes: IV first order
    
       """
       nk, nz, nb_coarse, _ = overlap.shape
       dipole = np.zeros([nk, nb_coarse, nb_coarse, 3], dtype=np.complex)
    
       # I-V derivative
       fac = 1j/(4*np.pi)
    
       for ik in range(nk):
         for iz, indz in enumerate(nzlist[ik]):
            aE = alat * avec[iz/2] * dimK[iz/2]
            dipole[ik] = dipole[ik] + fac * (-1)**(iz) * \
                               np.einsum('m,ij->ijm', aE, overlap[ik,indz])              
       # Assign direction
       dipole = np.array(np.einsum('kmni,i->kmn', dipole, pol_vec))
    
       # set hermiticity
       for ik in range(nk):
          dipole[ik] = 0.5*(dipole[ik] + dipole[ik].transpose([1,0]).conj())
    
       return dipole
    
 
    def get_covariant_overlap(self, overlap, knbind, nvblist):
       """
       Compute < u_n,k | \hat{u}_m,k+b>
    
       \hat{u}_m,k+b is the "dual" of \hat{u}_m 
       in the space of k+b
    
       Assume nvb and ncb are two separately connected space
       Follow PRB 69, 085106   
    
       """
       ns, nk, nz, nband, _ = overlap.shape
    
       Sinv = np.zeros([ns,nk,nz,nband,nband], dtype=np.complex)
       cov_ovlap = np.zeros([ns,nk,nz,nband,nband], dtype=np.complex)
    
       for js in range(ns):
         for ik in range(nk):
           for iz in range(nz):
             nvb = nvblist[js,knbind[ik,iz]]
             # If using full inverse S, 
             # by construction diagonal term are 1
             #cov_ovlap[ik,iz] = np.eye(nband,nband)
    
             #invS = np.linalg.inv(overlap[ik,iz,:nvb,:nvb])
             # c-v
             #cov_ovlap[ik,iz, nvb::, :nvb] = \
             #     np.dot(overlap[ik,iz, nvb::, :nvb], invS)
    
             #invS = np.linalg.inv(overlap[ik,iz, nvb::, nvb::])
             # v-c
             #cov_ovlap[ik,iz, :nvb, nvb::] = \
             #     np.dot(overlap[ik,iz, :nvb, nvb::], invS)
    
             tmp = overlap[js,ik,iz,:nvb,:nvb]
             U, s, V = np.linalg.svd(tmp)
             invS = np.dot(V.T.conj(), U.T.conj())
             cov_ovlap[js,ik,iz, :, :nvb] = \
                  np.dot(overlap[js,ik,iz, :, :nvb], invS)
             Sinv[js,ik,iz, :nvb, :nvb] = invS
    
             tmp = overlap[js,ik,iz, nvb::, nvb::]
             U, s, V = np.linalg.svd(tmp)
             invS = np.dot(V.T.conj(), U.T.conj())
             cov_ovlap[js,ik,iz, :, nvb::] = \
                  np.dot(overlap[js,ik,iz, :, nvb::], invS)
             Sinv[js,ik,iz, nvb::, nvb::] = invS
    
       return cov_ovlap, Sinv
    
    def get_covariant_overlap_check(self, overlap):
       """
       Compute < u_n,k | \hat{u}_m,k+b>
       Follow PRB 77, 045102 (2008). Haven't treated degeneracy and band order   
     
       \hat{u}_m,k+b is the "dual" of \hat{u}_m 
       Using band-by-band procedure, in contrast to SVD in the above
     
       """
       ns, nk, nz, nband, _ = overlap.shape
       S = np.zeros([ns,nk,nz,nband,nband], dtype=complex)
       cov_ovlap = np.zeros([ns,nk,nz,nband,nband], dtype=complex)
    
       gsizes, offsets = get_gsize_offset(nk, nz*nband*nband)
    
       for js in range(ns):
         for ik,ikf in enumerate(self.my_ikpts):
           for iz in range(nz):
             overlap_tmp = np.array(overlap[js,ikf,iz])
             # check connectedness
             for ib in range(nband):
               for jb in range(nband):
                  if abs(overlap_tmp[ib,jb]) <= 0.5:
                     overlap_tmp[ib,jb] = 0.0
    
             U, s, V = np.linalg.svd(overlap_tmp)
             invS = np.dot(V.T.conj(), U.T.conj())
    
             cov_ovlap[js,ikf,iz, :, :] = \
                   np.dot(overlap[js,ikf,iz, :, :], invS)
    
             S[js,ikf,iz] = np.array(invS[:])

         comm.Allgatherv(cov_ovlap[js,self.my_ikpts],\
                 [cov_ovlap[js], gsizes, offsets, MPI.COMPLEX])

         comm.Allgatherv(S[js,self.my_ikpts],\
                 [S[js], gsizes, offsets, MPI.COMPLEX])
    
       return cov_ovlap, S
    
    def get_covariant_overlap_bbb(self, overlap):
       """
       Compute < u_n,k | \hat{u}_m,k+b>
       Follow PRB 77, 045102 (2008). Haven't treated degeneracy and band order   
     
       \hat{u}_m,k+b is the "dual" of \hat{u}_m 
       Using band-by-band procedure, in contrast to SVD in the above
     
       """
       nk, nz, nband, _ = overlap.shape
       S = np.zeros([nk,nz,nband,nband], dtype=np.complex)
       cov_ovlap = np.zeros([nk,nz,nband,nband], dtype=np.complex)
    
       # U = |  <u1k | u1kq>  <u1k | u2kq> |  S = |  1 / <u1k|u1kq>  1 / <u2k|u2kq> |
       #     |  <u2k | u1kq>  <u2k | u2kq> |      |  1 / <u1k|u1kq>  1 / <u2k|u2kq> |
       for ik in range(nk):
         for iz in range(nz):
           for ib in range(nband):
    
            #  divide |u1kq> with <u1k | u1kq>
            #  divide |u2kq> with <u2k | u2kq>
            tmp = overlap[ik,iz,ib,ib] / abs(overlap[ik,iz,ib,ib])
            if abs(tmp) > 1e-5:
              cov_ovlap[ik,iz, :, ib] = overlap[ik,iz,:,ib] / tmp
              S[ik,iz, ib, ib] = 1./tmp
    
       return cov_ovlap, S
 
    def compute_Ddipole(self):
       """
       Compute covariant derivative

       Derivative direction in the last index:  I^{b,a} = D^a \Xi^b

       gdiv^a f_k^b = dk^a f^b_nmk - 1j* f^b_nmk * ( r^a_nn -r^a_mm )
     
       """
       dpintra = self.dipole - self.int_dipole

       # U(2) covariant derivative, Eq. B1
       ddp = np.zeros([self.nspin,self.my_nrk,self.nb,self.nb,3,3], dtype=complex)

       for js in range(self.nspin):
         for ik,ikg in enumerate((self.my_ikpts)):
           ddp[js,ik] = self.derivative_Mat( self.int_dipole[js], ikg, self.invS[js])

         ddp[js] = ddp[js] - 1j*(np.einsum('kacn,kcbm->kabnm', dpintra[js,self.my_ikpts], self.dipole[js,self.my_ikpts])\
               - np.einsum('kacm,kcbn->kabnm', self.dipole[js,self.my_ikpts], dpintra[js,self.my_ikpts]))

       self.Ddipole = np.array(ddp)
    
       return self.Ddipole


    def derivative(self, fq, ik):
       """
       Given a scalar in the full k-mesh
       compute Dk^i f(k) with finite difference

       kpoint weights are included in knbvec

       """
       dFq = np.zeros(3, dtype=complex)

       # knbind stores index neighbors in the closest shell

       for j,indkq in enumerate(self.w90parser[0].knbind[ik]):
          dFq = dFq + fq[indkq]*self.w90parser[0].knbvec[j]

       return dFq


    def derivative_Mat(self, fq, ik, invS=np.zeros([])):
       """
       Given a matrix or a vector in the full k-mesh
       compute Dk^i f(k) with finite difference

       kpoint weights are included in knbvec
       knbind stores index neighbors in the closest shell

       fk: [nk, nb, nb], quantities on full k grid
       knbind: [nk, nz] index of neighboring kpts
       knbvec: [nk, nz, 3] vec to neighboring kpts

       Direction of derivative in the LAST index

       """
       knbind = self.w90parser[0].knbind
       knbvec = self.w90parser[0].knbvec
         
       if not invS.any():

          for j,indkq in enumerate(knbind[ik]):
            FqM = fq[indkq]
            if len(FqM.shape) == 3: # FqM is a matrix (n,m,dir)

              if j == 0:
                 dFq = np.einsum('abc,m->abcm',FqM,knbvec[ik,j])
              else:
                 dFq = dFq + np.einsum('abc,m->abcm',FqM,knbvec[ik,j])

            elif len(FqM.shape) == 2: # vector (n,dir)

              if j == 0:
                 dFq = np.einsum('ab,m->abm',FqM,knbvec[ik,j])
              else:
                 dFq = dFq + np.einsum('ab,m->abm',FqM,knbvec[ik,j])
       else:

          for j,indkq in enumerate(knbind[ik]):
            if len(fq.shape) == 4: # FqM is a matrix (n,m,dir)
              FqM = np.einsum('nm,mlc->nlc', invS[ik,j].T.conj(),\
                    np.einsum('nmc,ml->nlc', fq[indkq], invS[ik,j]))
              if j == 0:
                 dFq = np.einsum('abc,m->abcm',FqM,knbvec[ik,j])
              else:
                 dFq = dFq + np.einsum('abc,m->abcm',FqM,knbvec[ik,j])

            elif len(fq.shape) == 3: # FqM is a vector
              FqM = np.einsum('nm,ml->nl',\
                      invS[ik,j].T.conj()[self.nvb:,self.nvb:],\
                    np.einsum('nm,ml->nl',\
                      fq[indkq], invS[ik,j,:self.nvb,:self.nvb]))
              if j == 0:
                 dFq = np.einsum('ab,m->abm',FqM,knbvec[ik,j])
              else:
                 dFq = dFq + np.einsum('ab,m->abm',FqM,knbvec[ik,j])

       return dFq


    def general_deriv_sum(self, b, a, ibn, ibm, r, vel, ene):
       """
       Summation form of general derivative r^b_{ibn,ibm};k^a
       Notice that the derivative is taken on the second input direction
    
       """
       nbtot = r.shape[1]
    
       enm = ene[ibn] - ene[ibm]
       Vmnb = vel[ibm,ibm,b] - vel[ibn,ibn,b]
       Vmna = vel[ibm,ibm,a] - vel[ibn,ibn,a]
       dr = (r[ibn,ibm,a]*Vmnb + r[ibn,ibm,b]*Vmna) / enm
    
       # Approximate p by vel
       for il in range(nbtot):
         if il != ibn and il != ibm:
           dr = dr + 1j / enm * ( r[ibn,il,a]*(ene[il]-ene[ibm])*r[il,ibm,b]\
                         - (ene[ibn]-ene[il])*r[ibn,il,b]*r[il,ibm,a] )
    
       return dr

   
    def get_inter_dipole(self, eqpk): 
       """
       Interband dipole, zero-out diagonal and energy-degenerate elements

       """
       self.int_dipole = np.array(self.dipole)

       # zero out diagnoal term and degenerate bands
       for js in range(self.nspin):
        for i in range(self.nrk):
         for j in range(self.nvb+self.ncb):
           self.int_dipole[js,i,j,j] = 0.0
           for k in range(self.nvb+self.ncb):
             if abs(eqpk[js,i,j] - eqpk[js,i,k]) < tol5:
               self.int_dipole[js,i,j,k] = 0.0
 
       return 
    
    def compute_velocity_from_dipole(self, QP, eqp_correction=True):
       """
       Given mean-field energy and dipole matrix
       compute velocity matrix using,
    
       v_nm = 1j * (En - Em) * r_nm
    
       ek, eqpk: [nrk, nb]

       knbvec: vector connects to neighboring kpts
       knbind: index of neighboring kpts
    
       v_nn is computed from derivative of energy
    
       """
       ns, nk, nb = self.nspin, self.nrk, self.nb
       self.int_velocity = np.zeros([ns, nk, nb, nb, 3], dtype=complex)
       self.velocity = np.zeros([ns, nk, nb, nb, 3], dtype=complex)
   
       if rank == 0: 
         if eqp_correction == True:
           ene = QP.eqpk[:]
         else:
           ene = QP.ek[:]
            
         for js in range(ns):
           for ik in range(nk):
             #tmp1 = 1j * np.einsum('i,ijm->ijm', ene[js,ik], self.int_dipole[js,ik])
             #tmp2 = 1j * np.einsum('j,ijm->ijm', ene[js,ik], self.int_dipole[js,ik])
             tmp1 = 1j * np.einsum('i,ijm->ijm', ene[js,ik], self.dipole[js,ik])
             tmp2 = 1j * np.einsum('j,ijm->ijm', ene[js,ik], self.dipole[js,ik])
    
             self.int_velocity[js,ik] = tmp1 - tmp2
    
         # Energy derivative
         self.velocity = np.array(self.int_velocity[:])
         for js in range(ns):
           for ik in range(nk):
             tmp = np.zeros((nb, 3), dtype=complex)
             for iz, iknb in enumerate(self.w90parser[0].knbind[ik]):
                tmp = tmp +  np.einsum( 'i,m->im',\
                                       ene[js, iknb, :], self.w90parser[0].knbvec[ik,iz])
    
             self.velocity[js,ik] = self.velocity[js,ik]\
                                    + np.einsum('im,ij->ijm', tmp, np.eye(nb)) 

       comm.Bcast(self.int_velocity)
       comm.Bcast(self.velocity)
    
       return 

    def computeQ(self, evec1, evec2):
       """
       Derivatives of exciton envelop functions  

       Compute Qmn = i \sum_k psi^(m)*_{cvk} psi^(n)_cvk;k

       derivative along all three direction

       invS[ns,nk,nz,nb,nb]

       # FIX: nspin in evecs?
    
       """
       nk, nc, nv = evec1.shape
       dpintra = self.dipole - self.int_dipole
       inds = np.ix_(range(1),range(nk),range(nv,nv+nc),range(nv,nv+nc))

       devec = np.zeros([nk,nc,nv,3], dtype=complex)

       for ik in range(nk):
         devec[ik] = self.derivative_Mat(evec2, ik, self.invS[0])

       devec = devec - 1j*(np.einsum('kacn,kcb->kabn', dpintra[inds], evec2)\
               - np.einsum('kcv,kvbn->kcbn', evec2, dpintra[0,:,:nv,:nv]))
    
       Qmn = 1j* np.einsum('kij,kijn->n', evec1.conjugate(), devec)
    
       return Qmn
    
    def computeR(self, evec1, evec2):
       """
       Intraband exciton dipole terms

       Compute Rmn = \sum_k psi^(m)*_{cvk} {
                            r_clk psi^(n)_lvk - r_lvk psi^(n)_clk }

       no diagonal term in the dipole

       # FIX: nspin in evecs?
    
       """
       nvb = evec1.shape[2]
    
       tmp1 = np.einsum('kcla,klv->kcva', self.int_dipole[0,:,nvb::,nvb::], evec2)
       tmp1 = tmp1 - np.einsum('klva,kcl->kcva', self.int_dipole[0,:,:nvb,:nvb], evec2)
    
       Rmn = np.einsum('kcv,kcva->a', evec1.conjugate(), tmp1)
    
       return Rmn

    def computeP(self, evec1, evec2, eqpk, eta, dephase=True, diag=False):
       """
       Intraband exciton velocity terms

       Compute Pnm = \sum_k psi^(n)_{cvk}  V_c'ck psi^(m)*_c'vk
                   - \sum_k psi^(n)_{cv'k} V_v'vk psi^(m)*_cvk

       no diagonal term in the velocity

       If dephasing

       Pnm = \sum_k psi^(n)_{cvk}  V_c'ck psi^(m)*_c'vk * (e_cc' - 2i*eta)/(e_cc'-i*eta)
           - \sum_k psi^(n)_{cv'k} V_v'vk psi^(m)*_cvk * (e_vv' - 2i*eta)/(e_vv'-i*eta)

       # FIX: nspin in evecs?
    
       """
       ncb, nvb = evec1.shape[1:3]

       if diag:
         vel_tmp = self.velocity
       else:
         vel_tmp = self.int_velocity

       if dephase:
         ecc = np.einsum('kc,p->kcp', eqpk[0,:,nvb:], np.ones(ncb))\
               - np.einsum('c,kp->kcp', np.ones(ncb), eqpk[0,:,nvb:])
         ecc = (ecc-2*1j*eta)/(ecc-1j*eta)

         evv = np.einsum('kv,p->kvp', eqpk[0,:,:nvb], np.ones(nvb))\
               - np.einsum('v,kp->kvp', np.ones(nvb), eqpk[0,:,:nvb])
         evv = (evv-2*1j*eta)/(evv-1j*eta)

         gcc = np.einsum('kpca,kcp->kpca', vel_tmp[0,:,nvb:,nvb:], ecc)
         tmp1 = np.einsum('klca,klv->kcva', gcc, evec2.conj())

         gvv = np.einsum('kpva,kvp->kpva', vel_tmp[0,:,:nvb,:nvb], evv)
         tmp1 = tmp1 - np.einsum('kpva,kcv->kcpa', gvv, evec2.conj())
       else:

         tmp1 = np.einsum('kpca,kpv->kcva', vel_tmp[0,:,nvb:,nvb:], evec2.conj())
         tmp1 = tmp1 -\
                np.einsum('kpva,kcv->kcpa', vel_tmp[0,:,:nvb,:nvb], evec2.conj())
    
       Pmn = np.einsum('kcv,kcva->a', evec1, tmp1)
    
       return Pmn

# Utilities-----------------------------------------------------------

    def compare_dipole(self, dipole1, dipole2, kvec,\
                      name1='dipole1', name2='dipole2'):
       """
       Compare two dipoles. For test purpose
    
       """
       nk = dipole1.shape[0]
       maxd = 0.
       kmax = -1
    
       for ik in range(nk):
    
          d12 = np.linalg.norm(np.abs(dipole1[ik])-np.abs(dipole2[ik]))
    
          if d12 > maxd:
             maxd = d12
             kmax = ik
    
          #if ik < 5:
          #  print( '\n\n  difference at kpt: ', ik, kvec[ik])
          #  print( '\n  '+name1, np.abs(dipole1[ik]))
          #  print( '\n  '+name2, np.abs(dipole2[ik]))
          #  print( '\n' )
       print( '\n\n  Max difference at kpt: ', kmax, kvec[kmax])
       print( '\n  '+name1, np.abs(dipole1[kmax]))
       print( '\n  '+name2, np.abs(dipole2[kmax]))
       print( '\n' )
    
       return   

