import pickle
import h5py as h5

"""
IO for optical response calculations

"""
def write_Jdos(fname, wrange, dos):
   """
   Joint density of states

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)        Jdos\n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}  {1:16.7f}\n'.
            format(wrange[i], dos[i].real))

   return

def write_absorption(fname, wrange, chi):
   """
   \chi^1 or \epsilon_2 ?

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)        Re[chi^1]     Im[chi^1]\n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}  {1:16.7f}  {2:16.7f}\n'.
            format(wrange[i], chi[i].real, chi[i].imag))

   return

def write_conductivity(fname, wrange, sigma):
   """
   \sigma 

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)        Re[sigma]     Im[sigma]\n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}  {1:16.7f}  {2:16.7f}\n'.
            format(wrange[i], sigma[i].real, sigma[i].imag))

   return

def write_SHG(fname, wrange, chi):
   """
   \chi^2(2w; w, w)

   """
   with open(fname,'w') as f:
     s = '#      omega (eV)       Re[chi^2]       Im[chi^2]\n'
     f.write(s)
     for i in range(len(wrange)):
       f.write('  {0:13.7f}    {1:16.7e}   {2:16.7e}\n'.
               format(wrange[i], chi[i].real, chi[i].imag))

   return

def write_SHG_ie(fname, wrange, chi_inter, chi_intra, chi_real):
   """
   \chi^2(2w; w, w)

   """
   chi_tot = chi_inter + chi_intra
   
   with open(fname,'w') as f:
     s = '#      omega (eV)       Im[chi^2_inter]       Im[chi^2_intra]'\
         '       Re[chi^2]     Im[chi^2]\n'
     f.write(s)
     for i in range(len(wrange)):
       f.write('  {0:13.7f}    {1:16.7e}   {2:16.7e}   {3:16.7e}   {4:16.7e}\n'.
               format(wrange[i], chi_inter[i].real, chi_intra[i].real,\
                      chi_real[i], chi_tot[i].real))

   return

def write_SHG_terms(fname, tmp1, tmp2, tmp3, tmp4, tmp5):
   """
   write each contribution to SHG

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)      term1      term2'\
             '       term3     term4     term5\n')
     for i in range(len(wrange)):
       f.write('  {0:13.7f} {1:16.7e} {2:16.7e} {3:16.7e} {4:16.7e} {5:16.7e}\n'.
               format(wrange[i], tmp1[i].real, tmp2[i].real,\
                      tmp3[i].real, tmp4[i].real, tmp5[i].real))

   return

def write_shiftcurrent(fname, wrange, sigma):
   """
   \sigma^0(0; w, -w)

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)       Re[simga0] (muA/V^2)   Im[sigma0] \n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}    {1:16.7e}   {2:16.7e} \n'.
          format(wrange[i], sigma[i].real, sigma[i].imag))

   return

def write_gyration(fname, wrange, sigma):
   """
   \sigma^0(0; w, -w)

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)       Re[simga0] (muA/V^2)   Im[sigma0] \n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}    {1:16.7e}   {2:16.7e} \n'.
          format(wrange[i], sigma[i].real, sigma[i].imag))

   return

def write_spincurrent(fname, wrange, sigma):
   """
   \sigma^0(0; w, -w)

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)       Re[simga0] (muA/V^2*hbar/(2e))   Im[sigma0] \n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}    {1:16.7e}   {2:16.7e} \n'.
          format(wrange[i], sigma[i].real, sigma[i].imag))

   return

def write_kappa(fname, wrange, kappa):
   """
   \kappa^mt, gyration current 

   """
   with open(fname,'w') as f:
     f.write('#      omega (eV)       Re[kappa] (muA/V^2)   Im[kappa] \n')
     for i in range(len(wrange)):
        f.write('  {0:13.7f}    {1:16.7e}   {2:16.7e} \n'.
          format(wrange[i], kappa[i].real, kappa[i].imag))

   return

def write_matrix_elements(fname, Qij, Yij, Pij, peinfo):
   """
   Inter-exciton matrix elements

   """
   nevecs, my_xcts, nfk, ncb, nvb = peinfo
   with h5.File(fname, 'w') as f: 
      f.create_dataset("nfk", data=nfk)
      f.create_dataset("ncb", data=ncb)
      f.create_dataset("nvb", data=nvb)
      f.create_dataset("nevec", data=nevecs)
      f.create_dataset("my_xcts", data=my_xcts)
      f.create_dataset("Qij", data=Qij)
      f.create_dataset("Yij", data=Yij)
      f.create_dataset("Pij", data=Pij)

   return

