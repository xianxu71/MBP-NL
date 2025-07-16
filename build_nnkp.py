import numpy as np

def parse_nnkp(seed):
  """
  parse nnkp from .mmn file. 
 
  """
  with open(seed+'.nnkp', 'a') as fout:
   fout.write('begin nnkpts\n')
   with open(seed+'.mmn', 'r') as f:
      # skip first line
      dum = f.readline()
      # header
      nband, nk, nz = list(map(int, f.readline().split()))
      print( '\n  nband, nk, nz', nband, nk, nz)
      fout.write('  {0:d}\n'.format(nz))
      knbind = np.zeros([nk,nz], dtype=int)
   
      lines = f.readlines()
      # Read index of neighbor k points
      for i in range(nk):
        ind = i*nz*(nband**2 + 1)
        for j in range(nz):
           ind = (i*nz + j)*(nband**2 + 1)
           fout.write(lines[ind])

   fout.write('end nnkpts')
  
  return

if __name__ == '__main__':

  parse_nnkp('wannier90') 
