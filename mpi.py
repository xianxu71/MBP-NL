"""MPI distribution of work load"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

i_am_master = bool(rank == 0)


def get_gsize_offset(nrk, dim):
   """
   Prepare for allgatherv

   """
   import numpy as np

   max_nkpt_per_worker = int( np.ceil(nrk *1.0 / size) )

   # Store size of density matrix for every local workers
   # then compute the offsets
   gsizes = np.zeros(size)
   for j in range(size):
     for i in range(max_nkpt_per_worker):
        ikpt = j * max_nkpt_per_worker + i

        if ikpt >= nrk:
            break

        gsizes[j] = gsizes[j] + 1
  
   # FIX: assume always use complex
   gsizes = gsizes* dim * 2
   offsets = np.zeros(size)
   offsets[1:] = np.cumsum(gsizes)[:-1]

   #for i in range(size):
   #   if rank == i:
   #      print( " Rank", i, "gsize", self.gsizes)
   #      print( " offsets", self.offsets)
   #   comm.Barrier()

   return gsizes, offsets
