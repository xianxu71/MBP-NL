import numpy as np
from mpi import MPI, comm, size, rank
import quasiparticle
import IO
from math_functions import delta_lorentzian, delta_gaussian
from constants import Ry2eV, eRy, eps0, tol5, au2muAdivV2, au2pmdivV

class optical_responses:

   """
   Compute optical resposnses from first-principles inputs

   Using the sum-over-state method from the perturbation theory

   """

   def __init__(self, QP, w, eta, brdfun='Lorentzian',\
                tetra=False, exciton=None):
      """
      Initialize the calculation

      """
      self.QP = QP
      self.exciton = exciton

      self.w = np.array(w[:])
      self.nw = len(self.w)
      self.eta = eta
      
      self.tetra = tetra

      # Note: tetrahedron supress broadening options
      if brdfun == 'Lorentzian':
        self.brdfunc = delta_lorentzian
      else:
        self.brdfunc = delta_gaussian

      if self.QP.nspin == 2 or self.QP.nspinor == 2:
        self.sdeg = 1
      else:
        self.sdeg = 2

      return

   def calc_Jdos(self, fname='Jdos.txt'):
      """
      Compute joint-density of states \delta(wc-wv)

      Final unit is 1/Energy(eV)

      """
      eta = self.eta
      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      nvb = self.QP.nvb
      ncb = self.QP.ncb
      nb  = self.QP.nb

      pref = ns/(nfk)/Ry2eV
      eta = eta / Ry2eV
      wrange = self.w / Ry2eV

      jdos = np.zeros([len(wrange)], dtype=float)

      for js in range(self.QP.nspin): 
        for ic in range(nvb,nb):
          for iv in range(nvb):
            ecv = self.QP.eqpk[js,:,ic] - self.QP.eqpk[js,:,iv]
            for ikf in self.QP.kpoint.my_ikpts:
              jdos = jdos + self.brdfunc(wrange, ecv[ikf], eta)

      jdos = pref*jdos
      jdos = comm.allreduce(jdos)

      if rank == 0:
         IO.write_Jdos(fname, wrange*Ry2eV, jdos)

      return 
   
   def calc_conductivity(self, fname='sigma'):
      """
      Compute conductivity, using velocity matrix
   
      """
      ns = self.sdeg
      celvol = self.QP.celvol
      nfk = self.QP.kpoint.nrk
      nvb = self.QP.nvb
      ncb = self.QP.ncb
      nb = self.QP.nb

      brdfunc = self.brdfunc

      pref = 1j* eRy * ns / (nfk * celvol)
      eta = self.eta / Ry2eV
      wrange = self.w / Ry2eV
   
      sigma = np.zeros([3,3,len(wrange)], dtype=complex)

      for js in range(self.QP.nspin):
       for ikf in self.QP.kpoint.my_ikpts:
         ene = self.QP.eqpk[js,ikf]
         vel = self.QP.dp.velocity[js,ikf]
   
         for iv in range(nvb):
          for ic in range(nvb,ncb+nvb):
            wcv = ene[ic] - ene[iv]
            for ia in range(3):
             for ib in range(3):
            
               sigma[ia,ib] = sigma[ia,ib] \
                    - vel[iv,ic,ia]*vel[ic,iv,ib]/(wrange*(wcv-wrange-1j*eta))
 
      sigma = pref*sigma 
      sigma = comm.allreduce(sigma) 
   
      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + '.txt'
           IO.write_conductivity(tmp, wrange*Ry2eV, sigma[i,j])

      return 

   def calc_absorption_len(self, fname='abs_len'):
      """
      Compute absorption spectrum in length gauge

      """
      dim = 3
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      nvb = self.QP.nvb
      ncb = self.QP.ncb
      nb = self.QP.nb
      eqpk = self.QP.eqpk
      my_ikpts = self.QP.kpoint.my_ikpts
      dipole = self.QP.dp.dipole
      celvol = self.QP.celvol

      pref = 1j*16*np.pi**2/(nfk*celvol*self.QP.nspin*self.QP.nspinor)
      eta = self.eta / Ry2eV
      wrange = self.w / Ry2eV

      chi = np.zeros([dim,dim,len(wrange)], dtype=complex)

      #  b=c a=v, \nu=z
      for js in range(self.QP.nspin):
       for ic in range(nvb,nb):
        for iv in range(nvb):
         for ik in my_ikpts:
          ecv = eqpk[js,ik,ic] - eqpk[js,ik,iv]
          for im in range(dim):
            ovcm = dipole[js,ik,iv,ic,im]

            for iu in range(dim):
              ocvu = dipole[js,ik,ic,iv,iu]

              tmp = ovcm*ocvu
              # zero T, f_vc = 1
              chi[im,iu] = chi[im,iu] + tmp*self.brdfunc(wrange, ecv, eta)

      chi = pref * chi
      chi = comm.allreduce(chi)

      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + '.txt'
           IO.write_absorption(tmp, wrange*Ry2eV, chi[i,j])

      return 

   def calc_absorption_with_eh(self, fname='abs_eh'):
      """
      Compute absorption spectrum in length gauge

      """
      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      nevecs = self.exciton.nevecs
      evals = self.exciton.evals
      xeh = self.exciton.reh

      pref = 16*np.pi**2/(nfk*celvol*self.QP.nspin*self.QP.nspinor)
      eta = self.eta / Ry2eV
      wrange = self.w / Ry2eV

      chi = np.zeros([3,3,len(wrange)], dtype=complex)

      for i in self.exciton.my_xcts :

        num1 = np.einsum('a,b->ab', xeh[i].conjugate(), xeh[i]) 
        chi = chi + np.einsum('ab,w->abw', num1,\
                              self.brdfunc(wrange, evals[i], eta))
      
      chi = pref * chi

      chi = comm.allreduce(chi)

      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + '.txt'
           IO.write_absorption(tmp, wrange*Ry2eV, chi[i,j])

      return 

   def calc_SHG(self, fname='SHG', analysis=False):
      """
      Compute SHG \chi^2(-2omega; omega, omega)
   
      Use length gauge formula in J. L. Cabellos, et al., PRB 80 155205, 2009
      Compute imaginary part then do Hilbert transformation
   
      i1,i2,i3 is the Cartesian direction in chi2, refer to the rotated system

      TODO: Rewrite the code
      
      """
      dim = 3
      tol = 1e-4

      raise Exception('Not fully compatible to new codes')

      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      nvb = self.QP.nvb
      ncb = self.QP.ncb
      nb = self.QP.nb
      my_ikpts = self.QP.kpoint.my_ikpts
      my_nfk = self.QP.my_nfk

      echarge = eRy
      pref = echarge**3/(2*nfk*celvol)

      brdfunc = self.brdfunc
      eta = self.eta / Ry2eV
      nw = self.nw
      wrange = self.w / Ry2eV
      
      eqpk = np.array(self.QP.eqpk[:])

      dipole = np.array(self.QP.dp.dipole[:]) 
      velocity = np.array(self.QP.dp.velocity[:]) 
      int_velocity = np.array(self.QP.dp.int_velocity[:]) 

      # zero out diagnoal term and degenerate bands
      for js in range(self.QP.nspin):
       for i in range(nfk):
         ene = eqpk[js,i,:]
         for j in range(nvb+ncb):
           dipole[js,i,j,j] = 0.0
           for k in range(nvb+ncb):
             if abs(ene[j] - ene[k]) < tol:
                dipole[js,i,j,k] = 0.0
  
 
      chi_inter = np.zeros(nw, dtype=complex)
      chi_intra = np.zeros(nw, dtype=complex)

      # interband
      for js in range(self.QP.nspin):
       for ikf in my_ikpts:
         ene = eqpk[js, ikf, : ]
         r = dipole[js,ikf]
   
         for iv in range(nvb):
          for ic in range(nvb_top,ncb+nvb_top):
   
           wcv = ene[ic] - ene[iv]
           d2w = brd_func(2*wrange, wcv, eta)
           dw = brd_func(wrange, wcv, eta)
   
           term1 = 0
           term2 = 0
           term3 = 0
   
           for il in range(nvb) + range(nvb_top, ncb+nvb_top):
   
            wcl = ene[ic] - ene[il]
            wlc = -wcl
            wlv = ene[il] - ene[iv]
            wvl = -wlv
            if il != iv and il != ic and abs(wcl-wlv)>1e-3\
              and abs(wcv-wlc)>1e-3 and abs(wvl-wcv)>1e-3:
   
              term1 = term1 + 2*(r[iv,ic,i1]*( r[ic,il,i2]*r[il,iv,i3]+r[ic,il,i3]*r[il,iv,i2] )/2).real/(wcl - wlv)
              term2 = term2 + (r[iv,il,i1]*( r[il,ic,i2]*r[ic,iv,i3]+r[il,ic,i3]*r[ic,iv,i2] )/2).real/(wcv - wlc)
              term3 = term3 + (r[il,ic,i1]*( r[ic,iv,i2]*r[iv,il,i3]+r[ic,iv,i3]*r[iv,il,i2] )/2).real/(wvl - wcv)
   
           chi_inter = chi_inter + term1*d2w + (term2+term3)*dw
   
      chi_inter = comm.allreduce(chi_inter)
      chi_inter = pref * chi_inter
  
      if analysis: 
        tmp1 = np.zeros(nw, dtype=complex)
        tmp2 = np.zeros(nw, dtype=complex)
        tmp3 = np.zeros(nw, dtype=complex)
        tmp4 = np.zeros(nw, dtype=complex)
        tmp5 = np.zeros(nw, dtype=complex)
   
      # intraband
      for js in range(self.QP.nspin):
       for ikf in my_ikpts:
         ene = eqpk[js,ikf,:]
         r   = dipole[js,ikf]
         vel = velocity[js,ikf]
   
         for iv in range(nvb):
          for ic in range(nvb,nb):
   
           wcv = ene[ic] - ene[iv]
           Vcv3 = vel[ic,ic,i3] - vel[iv,iv,i3]
           Vcv2 = vel[ic,ic,i2] - vel[iv,iv,i2]
   
           d2w = brd_func(2*wrange, wcv, eta)
           dw = brd_func(wrange, wcv, eta)
   
           drcv23 = general_deriv(i2, i3, ic, iv, r, vel, ene)
           drcv32 = general_deriv(i3, i2, ic, iv, r, vel, ene)
           drvc13 = general_deriv(i1, i3, iv, ic, r, vel, ene)
           drvc12 = general_deriv(i1, i2, iv, ic, r, vel, ene)
           drvc21 = general_deriv(i2, i1, iv, ic, r, vel, ene)
           drvc31 = general_deriv(i3, i1, iv, ic, r, vel, ene)
   
           term1 =  2*(r[iv,ic,i1]*( drcv23 + drcv32 )/2).imag / wcv
           term2 = -4*(r[iv,ic,i1]*( r[ic,iv,i2]*Vcv3 + r[ic,iv,i3]*Vcv2 )/2).imag / wcv**2
           term3 =  (( drvc13*r[ic,iv,i2] + drvc12*r[ic,iv,i3] )/2).imag / wcv
           term4 =  (r[iv,ic,i1]*( r[ic,iv,i2]*Vcv3 + r[ic,iv,i3]*Vcv2 )/2).imag / wcv**2
           term5 = -(( drvc21*r[ic,iv,i3] + drvc31*r[ic,iv,i2] )/2).imag / (2*wcv)
   
           chi_intra = chi_intra + (term1+term2)*d2w + (term3+term4+term5)*dw

           if analysis:
             tmp1 = tmp1 + pref*term1*d2w
             tmp2 = tmp2 + pref*term2*d2w
             tmp3 = tmp3 + pref*term3*dw
             tmp4 = tmp4 + pref*term4*dw
             tmp5 = tmp5 + pref*term5*dw
   
      chi_intra = comm.allreduce(chi_intra)
      chi_intra = pref * chi_intra

      if analysis:
        tmp1 = comm.reduce(tmp1)
        tmp2 = comm.reduce(tmp2)
        tmp3 = comm.reduce(tmp3)
        tmp4 = comm.reduce(tmp4)
        tmp5 = comm.reduce(tmp5)
   
      if rank == 0:
        chi_tot = chi_inter + chi_intra
        chi_real = hilbert_transform(chi_tot.real, wrange,\
                                     wrange, eta=0.02/Ry2eV)
        IO.write_SHG(fname, wrange*Ry2eV, chi_inter, chi_intra, chi_real)
  
        if analysis:
          IO.write_SHG_terms('SHG_terms.txt', wrange*Ry2eV, tmp1, tmp2,\
                             tmp3, tmp4, tmp5) 

      return
   
   def calc_shift_current(self, fname='shift_current'):
      """
      Compute photocurrent \chi^0(0; omega, -omega)
   
      eq. 2 in PRL 119, 067402 (2017), eq. 57 in Sipe and Shkrebtii
      eq. 2 in PRB 96, 115147 (2017) and eq. 1 in PRL 109, 116601 (2012)

      factor of 2 difference ? 

      Use PRB 97, 245143 (2018)
      
      """
      if self.QP.dp.w90parser == None:
        raise Exception('Nonlinear responses require W90 calculations.')

      eta = self.eta
      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      nvb = self.QP.nvb
      ncb = self.QP.ncb
      nb = self.QP.nb

      eta = eta / Ry2eV
      wrange = self.w / Ry2eV
      pref = -1j * eRy**3 * np.pi / (4*nfk * celvol)
  
      int_dipole = self.QP.dp.int_dipole
      # D^a r^b = Ddipole[b,a]
      Ddipole = self.QP.dp.compute_Ddipole()
   
      sigma0 = np.zeros([3,3,3,len(wrange)], dtype=complex)
  
      for js in range(self.QP.nspin): 
       for ik,ikf in enumerate(self.QP.kpoint.my_ikpts):
        for ij in range(nvb):
         for im in range(nvb,nb):
          wcv = self.QP.eqpk[js,ikf,im] - self.QP.eqpk[js,ikf,ij]
          dw = self.brdfunc(wrange, wcv, eta)
          ndw = self.brdfunc(wrange, -wcv, eta)
          for ia in range(3):
           for ib in range(3): 
            for ic in range(3):
  
             # I_mj^{abc} + I_mj^{acb}
             sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                + (int_dipole[js,ikf,im,ij,ib]*Ddipole[js,ik,ij,im,ic,ia] \
                   + int_dipole[js,ikf,im,ij,ic]*Ddipole[js,ik,ij,im,ib,ia])\
                * (dw + ndw)             
             # I_jm^{abc} + I_jm^{acb}
             sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                - (int_dipole[js,ikf,ij,im,ib]*Ddipole[js,ik,im,ij,ic,ia] \
                   + int_dipole[js,ikf,ij,im,ic]*Ddipole[js,ik,im,ij,ib,ia])\
                * (dw + ndw)             
     
      sigma0 = pref * sigma0
      sigma0 = comm.allreduce(sigma0)
    
      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
          for l,d3 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + d3 + '.txt'
           IO.write_shiftcurrent(tmp, wrange*Ry2eV, sigma0[i,j,l]*au2muAdivV2)

      return

   def calc_nlo_conductivity(self, fname='nlo_sigma', type=None):
      """
      General nonlinear charge conductivity.

      Eq. S22 in Nat. Comm. 12, 4330 (2021), velocity gauge

      """
      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      nvb = self.QP.nvb
      ncb = self.QP.ncb
      nb = self.QP.nb

      eta = self.eta / Ry2eV
      wrange = self.w / Ry2eV

      sigma0 = np.zeros([3,3,3,len(wrange)], dtype=complex)

      if type == 'T-LPL':
        # T-symmetry, linearly polarized light (shift current)
        pref = -np.pi*eRy**3 / (nfk * celvol)
        for js in range(self.QP.nspin): 
         for ikf in self.QP.kpoint.my_ikpts:
          occ = self.QP.fk[js,ikf]
          mfe = self.QP.eqpk[js,ikf]
          for il in range(nb):
           for im in range(nb):
             wml = mfe[im] - mfe[il]
             vlm = self.QP.dp.velocity[js,ikf,il,im]
             flm = occ[il] - occ[im]
             
             for it in range(nb):
               wmn = mfe[im] - mfe[it]
               wnl = mfe[it] - mfe[il]
               vmn = self.QP.dp.velocity[js,ikf,im,it]
               vnl = self.QP.dp.velocity[js,ikf,it,il]

               if np.abs(wml*wmn*wnl) > 1e-8:
                for ia in range(3): 
                 for ib in range(3):
                  for ic in range(3):
                    sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                                      + flm*(vlm[ib]/wml**2*(vmn[ia]*vnl[ic]/wmn\
                                        - vmn[ic]*vnl[ia]/wnl)).imag\
                                      *self.brdfunc(wrange, wml, eta)
      elif type == 'PT-CPL':
        # PT-symmetry, circularly polarized light (gyration current, shift mechanism)
        pref = 1j*np.pi*eRy**3 / (nfk * celvol)
        for js in range(self.QP.nspin): 
         for ikf in self.QP.kpoint.my_ikpts:
          occ = self.QP.fk[js,ikf]
          mfe = self.QP.eqpk[js,ikf]
          for il in range(nb):
           for im in range(nb):
             wml = mfe[im] - mfe[il]
             vlm = self.QP.dp.velocity[js,ikf,il,im]
             flm = occ[il] - occ[im]
             
             for it in range(nb):
               wmn = mfe[im] - mfe[it]
               wnl = mfe[it] - mfe[il]
               vmn = self.QP.dp.velocity[js,ikf,im,it]
               vnl = self.QP.dp.velocity[js,ikf,it,il]

               if np.abs(wml*wmn*wnl) > 1e-8:
                for ia in range(3): 
                 for ib in range(3):
                  for ic in range(3):
                    sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                                      + flm*(vlm[ib]/wml**2 *( vmn[ia]*vnl[ic]/wmn\
                                        - vmn[ic]*vnl[ia]/wnl )).real\
                                      *self.brdfunc(wrange, wml, eta)

      elif type == 'T-CPL':
        # T-symmetry, circularly polarized light (injection current)
        pref = 1j*np.pi*(eRy**3) / (nfk * celvol * eta)
        for js in range(self.QP.nspin): 
         for ikf in self.QP.kpoint.my_ikpts:
          occ = self.QP.fk[js,ikf]
          mfe = self.QP.eqpk[js,ikf]
          for il in range(nb):
           vl = self.QP.dp.velocity[js,ikf,il,il]
           for im in range(nb):
             if im != il:
               wml = mfe[im] - mfe[il]
               vm = self.QP.dp.velocity[js,ikf,im,im]
               flm = occ[il] - occ[im]
               rlm = self.QP.dp.int_dipole[js,ikf,il,im]
               rml = self.QP.dp.int_dipole[js,ikf,im,il]
               
               for ia in range(3): 
                for ib in range(3):
                 for ic in range(3):
                   sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                                     + flm*self.brdfunc(wrange, wml, eta)\
                                       *(rlm[ib]*rml[ic]*(vm[ia]-vl[ia])).imag

      elif type == 'PT-LPL':
        # PT-symmetry, linearly polarized light (magnetic injection current)
        pref = np.pi*(eRy**3) / (nfk * celvol * eta)
        for js in range(self.QP.nspin): 
         for ikf in self.QP.kpoint.my_ikpts:
          occ = self.QP.fk[js,ikf]
          mfe = self.QP.eqpk[js,ikf]
          for il in range(nb):
           vl = self.QP.dp.velocity[js,ikf,il,il]
           for im in range(nb):
             if im != il:
               wml = mfe[im] - mfe[il]
               vm = self.QP.dp.velocity[js,ikf,im,im]
               flm = occ[il] - occ[im]
               rlm = self.QP.dp.int_dipole[js,ikf,il,im]
               rml = self.QP.dp.int_dipole[js,ikf,im,il]
               
               for ia in range(3): 
                for ib in range(3):
                 for ic in range(3):
                   sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                                     + flm*self.brdfunc(wrange, wml, eta)\
                                       *(rlm[ib]*rml[ic]*(vm[ia]-vl[ia])).real

      else:
        # General form. Tend to diverge at low frequency
        pref = - (eRy**3) / (nfk * celvol)
        for js in range(self.QP.nspin): 
         for ikf in self.QP.kpoint.my_ikpts:
          occ = self.QP.fk[js,ikf]
          mfe = self.QP.eqpk[js,ikf]
          for il in range(nb):
           for im in range(nb):
             wml = mfe[im] - mfe[il]
             vlm = self.QP.dp.velocity[js,ikf,il,im]
             flm = occ[il] - occ[im]
             
             for it in range(nb):
               wmn = mfe[im] - mfe[it]
               wnl = mfe[it] - mfe[il]
               vmn = self.QP.dp.velocity[js,ikf,im,it]
               vnl = self.QP.dp.velocity[js,ikf,it,il]

               for ia in range(3): 
                for ib in range(3):
                 for ic in range(3):
                   sigma0[ia,ib,ic] = sigma0[ia,ib,ic]\
                                     + flm*vlm[ib]/(wml-wrange+1j*eta)\
                                     *( vmn[ia]*vnl[ic]/(wmn+1j*eta)\
                                       - vmn[ic]*vnl[ia]/(wnl+1j*eta) )

      sigma0 = pref*sigma0
      sigma0 = comm.allreduce(sigma0)

      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
          for l,d3 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + d3 + '.txt'
           IO.write_gyration(tmp, wrange*Ry2eV, sigma0[i,j,l]*au2muAdivV2)

      return

   def calc_shift_current_with_eh(self, fname='xct-shift_current'):
      """
      Ref. : Pedersen, PRB 92, 235432
             Taghizadeh and Pedersen, PRB 97, 205432   
   
      Shift current with excitonic effects
   
      evals : exciton energy, index evals[n]
      evecs : exciton wf Psi^(n)_cvk, index evecs[kcv,n] 
      xeh  : dipole in exciton picture
      veh  : velocity in exciton picture
             xeh_n = Psi^(n)_cvk * r_vck

      # Implementation 1 
      !! Mix circular and linear polarized components
      sigma0_i = sigma0_i\
                  + np.einsum('abc,w->abcw', num1,\
                                 1./(evals[i]*(evals[j]-(wrange+1j*eta))))\
                  + np.einsum('abc,w->abcw', num2,\
                                 1./(evals[i]*(evals[j]+(wrange+1j*eta))))
   
      Pij = 1j*(evals[i]-evals[j])*Xij
      num1 = np.einsum('a,bc->abc', xeh[i],\
                 np.einsum('b,c->bc', xeh[j].conjugate(), Pij))
   
      sigma0_e = sigma0_e\
                  - np.einsum('abc,w->abcw', num1,\
                  1./((evals[i]+(wrange+1j*eta)*(evals[j]+(wrange+1j*eta)))))

      sigma0_e = comm.allreduce(sigma0_e)
      sigma0_i = comm.allreduce(sigma0_i)
      sigma0 = -sigma0_i + sigma0_e
   
      """
      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      xctdim = self.exciton.xdim
      nevecs = self.exciton.nevecs
      evecs = self.exciton.evecs
      evals = self.exciton.evals
      peh = self.exciton.veh
      xeh = self.exciton.reh

      eta = self.eta / Ry2eV
      wrange = self.w / Ry2eV
      brdf = self.brdfunc

      pref = -eRy**3 / (2*nfk * celvol)
      pref2 = eRy**3 / (2*nfk * celvol)

      # use PRB 97, 205432
      #sigma0_e = np.zeros([3, 3, 3, len(wrange)], dtype=complex)
      #sigma0_i = np.zeros([3, 3, 3, len(wrange)], dtype=complex)
      sigma0 = np.zeros([3, 3, 3, len(wrange)], dtype=complex)

      Dpsi = self.compute_dpsi()

      for j_loc,j in enumerate(self.exciton.my_xcts):

        if rank == 0 and j_loc%int(self.exciton.my_nxct/1) == 0:
          print('  Progress: {0:4.1f}%'.format(j_loc/self.exciton.my_nxct*100))

        for i in range(self.exciton.nevecs):

          Qij = 1j*np.dot(evecs[i].conjugate().flatten(), Dpsi[j_loc])
          #Qij = self.QP.dp.computeQ(evecs[i], evecs[j])
          Yij = self.QP.dp.computeR(evecs[i], evecs[j])
          Xij = Qij + Yij

          # Intraband velocity
          Pij = self.QP.dp.computeP(evecs[i], evecs[j], self.QP.eqpk, eta, dephase=True)
  
          num1 = np.einsum('a,bc->abc', peh[i],\
                     np.einsum('b,c->bc', Xij, xeh[j].conjugate()))
          num2 = num1.conjugate()

          num3 = np.einsum('b,ac->abc', xeh[i].conjugate(),\
                     np.einsum('a,c->ac', Pij, xeh[j]))
          num4 = num3.conjugate()


          # Implementation 2
          sigma0 = sigma0 + pref*1j*np.pi/evals[i]*(\
                    -np.einsum('abc,w->abcw',num1,brdf(wrange, evals[j], eta))
                    +np.einsum('abc,w->abcw',num2,brdf(wrange, -evals[j], eta))
                    -np.einsum('acb,w->abcw',num1,brdf(wrange, -evals[j], eta))
                    +np.einsum('acb,w->abcw',num2,brdf(wrange, evals[j], eta)))\
                    + pref2*1j*np.pi*(\
                    +np.einsum('abc,w->abcw',num3,brdf(wrange, evals[j], eta)*(1./(wrange-evals[i]+1j*eta)).real) 
                    -np.einsum('abc,w->abcw',num3,brdf(wrange, evals[i], eta)*(1./(wrange-evals[j]-1j*eta)).real) 
                    -np.einsum('acb,w->abcw',num4,brdf(wrange, -evals[j], eta)*(1./(wrange+evals[i]-1j*eta)).real) 
                    +np.einsum('acb,w->abcw',num4,brdf(wrange, -evals[i], eta)*(1./(wrange+evals[j]+1j*eta)).real))

      sigma0 = comm.allreduce(sigma0)
   
      # write shift current spectrum
      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
          for l,d3 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + d3 + '.txt'
           IO.write_shiftcurrent(tmp, wrange*Ry2eV, sigma0[i,j,l]*au2muAdivV2)
   
      return

   def calc_SHG_with_eh(self, fname='xct-shift_current'):
      """
      Ref. : Pedersen, PRB 92, 235432
             Taghizadeh and Pedersen, PRB 97, 205432   
   
      Shift current with excitonic effects
   
      evals : exciton energy, index evals[n]
      evecs : exciton wf Psi^(n)_cvk, index evecs[kcv,n] 
      xeh  : dipole in exciton picture
      veh  : velocity in exciton picture
             xeh_n = Psi^(n)_cvk * r_vck

      # Implementation 1 
      !! Mix circular and linear polarized components
      sigma0_i = sigma0_i\
                  + np.einsum('abc,w->abcw', num1,\
                                 1./(evals[i]*(evals[j]-(wrange+1j*eta))))\
                  + np.einsum('abc,w->abcw', num2,\
                                 1./(evals[i]*(evals[j]+(wrange+1j*eta))))
   
      Pij = 1j*(evals[i]-evals[j])*Xij
      num1 = np.einsum('a,bc->abc', xeh[i],\
                 np.einsum('b,c->bc', xeh[j].conjugate(), Pij))
   
      sigma0_e = sigma0_e\
                  - np.einsum('abc,w->abcw', num1,\
                  1./((evals[i]+(wrange+1j*eta)*(evals[j]+(wrange+1j*eta)))))

      sigma0_e = comm.allreduce(sigma0_e)
      sigma0_i = comm.allreduce(sigma0_i)
      sigma0 = -sigma0_i + sigma0_e
   
      """
      celvol = self.QP.celvol
      ns = self.sdeg
      nfk = self.QP.kpoint.nrk
      xctdim = self.exciton.xdim
      nevecs = self.exciton.nevecs
      evecs = self.exciton.evecs
      evals = self.exciton.evals
      peh = self.exciton.veh
      xeh = self.exciton.reh

      eta = self.eta / Ry2eV
      wrange = self.w / Ry2eV
      brdf = self.brdfunc

      pref = -eRy**3 / (2*nfk * celvol)
      pref2 = eRy**3 / (2*nfk * celvol)

      # use PRB 97, 205432
      #sigma0_e = np.zeros([3, 3, 3, len(wrange)], dtype=complex)
      #sigma0_i = np.zeros([3, 3, 3, len(wrange)], dtype=complex)
      sigma0 = np.zeros([3, 3, 3, len(wrange)], dtype=complex)

      Dpsi = self.compute_dpsi()

      for j_loc,j in enumerate(self.exciton.my_xcts):

        if rank == 0 and j_loc%int(self.exciton.my_nxct/1) == 0:
          print('  Progress: {0:4.1f}%'.format(j_loc/self.exciton.my_nxct*100))

        for i in range(self.exciton.nevecs):

          Qij = 1j*np.dot(evecs[i].conjugate().flatten(), Dpsi[j_loc])
          #Qij = self.QP.dp.computeQ(evecs[i], evecs[j])
          Yij = self.QP.dp.computeR(evecs[i], evecs[j])
          Xij = Qij + Yij

          # Intraband velocity
          Pij = self.QP.dp.computeP(evecs[i], evecs[j], self.QP.eqpk, eta, dephase=True)
  
          num1 = np.einsum('a,bc->abc', peh[i],\
                     np.einsum('b,c->bc', Xij, xeh[j].conjugate()))
          num2 = num1.conjugate()

          num3 = np.einsum('b,ac->abc', xeh[i].conjugate(),\
                     np.einsum('a,c->ac', Pij, xeh[j]))
          num4 = num3.conjugate()


          # Implementation 2
          sigma0 = sigma0 + pref*1j*np.pi/evals[i]*(\
                    -np.einsum('abc,w->abcw',num1,brdf(wrange, evals[j], eta))
                    +np.einsum('abc,w->abcw',num2,brdf(wrange, -evals[j], eta))
                    -np.einsum('acb,w->abcw',num1,brdf(wrange, -evals[j], eta))
                    +np.einsum('acb,w->abcw',num2,brdf(wrange, evals[j], eta)))\
                    + pref2*1j*np.pi*(\
                    +np.einsum('abc,w->abcw',num3,brdf(wrange, evals[j], eta)*(1./(wrange-evals[i]+1j*eta)).real) 
                    -np.einsum('abc,w->abcw',num3,brdf(wrange, evals[i], eta)*(1./(wrange-evals[j]-1j*eta)).real) 
                    -np.einsum('acb,w->abcw',num4,brdf(wrange, -evals[j], eta)*(1./(wrange+evals[i]-1j*eta)).real) 
                    +np.einsum('acb,w->abcw',num4,brdf(wrange, -evals[i], eta)*(1./(wrange+evals[j]+1j*eta)).real))

      sigma0 = comm.allreduce(sigma0)
   
      # write shift current spectrum
      if rank == 0:
        for i,d1 in enumerate(['x','y','z']): 
         for j,d2 in enumerate(['x','y','z']): 
          for l,d3 in enumerate(['x','y','z']): 
           tmp = fname + '-' + d1 + d2 + d3 + '.txt'
           IO.write_shiftcurrent(tmp, wrange*Ry2eV, sigma0[i,j,l]*au2muAdivV2)
   
      return


   def compute_dpsi(self):
      """
      Prepare covariant derivative of excitons for local exciton states
   
      nonMPI version is equivalent to computeQ in dipole.py

      !!FIX: spin index is 0 

      """
      evecs = self.exciton.evecs
      my_nevec = self.exciton.my_nxct
      nevecs, nfk, ncb, nvb = evecs.shape
      xctdim = self.exciton.xdim

      dpintra = self.QP.dp.dipole - self.QP.dp.int_dipole
      dpiv = dpintra[0,:,:nvb,:nvb,:]
      inds = np.ix_(range(1),range(nfk),range(nvb,nvb+ncb),\
                    range(nvb,nvb+ncb),range(3))
      dpic = dpintra[inds][0]

      Dpsi = np.zeros([my_nevec,nfk,ncb,nvb,3], dtype=complex)

      for i,ii in enumerate(self.exciton.my_xcts):
        for ik in range(nfk):
          Dpsi[i,ik] = self.QP.dp.derivative_Mat(evecs[ii], ik,\
                                                  self.QP.dp.invS[0])

        Dpsi[i] = Dpsi[i] - 1j*(np.einsum('kacn,kcb->kabn', dpic, evecs[ii])\
                   - np.einsum('kab,kbcn->kacn', evecs[ii], dpiv))

      Dpsi = Dpsi.reshape([my_nevec,xctdim,3])

      return Dpsi

