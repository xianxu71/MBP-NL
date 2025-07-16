
"""
Utility functions

"""

def print_mat(string, A):

   nbnd = A.shape[0]
   print(string)
   for i in range(nbnd):
     row = ''
     for j in range(nbnd):
       row = row +' {0:6.3f} {1:6.3f}j'.format(A[i,j].real, A[i,j].imag)
     print(row)

   return
