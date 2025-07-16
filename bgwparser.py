import numpy as np

def read_vmtxel(fname):
   """
   Read vmtxel in ascii format

   """

   with open(fname, 'r') as f:
      l1 = f.readline().split()
      content = f.readlines()
   
      # mattype = 0 velocity, = 1 momentum
      nk, ncband, nvband, nspin, mattype = list(map(int, l1))
      ncvb = ncband + nvband
      print( '\n Reading dipole matrix from', name )
      print( '  Header info: nk={0:d} ncband={1:d} nvband={2:d}\
                nspin={3:d} mattype={4:d}\
                '.format(nk, ncband, nvband, nspin, mattype))
     
      if '(' in content[0]: # complex format
         mat = []
         for line in content:
            tmp = list(map(float, line.replace(',',' ').replace('(',' ').replace(')',' ').strip().split()))
            ntmp = int(len(tmp)/2)
            for i in range(ntmp):
               mat.append(tmp[2*i]+1j*tmp[2*i+1])
     
         mat = np.array(mat)
      
      else: # real format
         # Treat last line
         skip = 0
         if nk*ncband*nvband*nspin % 3 > 0:
            skip = 1
        
         mat = np.genfromtxt(fname, skip_header=1, skip_footer=skip)
         if skip == 1:
            mat = mat.reshape([((nk*ncband*nvband*nspin)/3)*3,1])
            mat = np.append(mat, list(map(float, content[-1].strip().split()[:])))
   
   mat = mat.reshape([nk,ncband,nvband,nspin]) * (1j)
   dipole = mat.transpose([3,0,1,2])

   return dipole
