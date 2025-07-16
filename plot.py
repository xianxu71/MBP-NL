import numpy as np
import sys
import matplotlib.pyplot as plt

fnames = sys.argv[1:]
ind = [1,2,1]

for ii, fname in enumerate(fnames):
  data = np.loadtxt(fname)

  x = data[:,0]
  y = data[:,ind[ii]]

  plt.plot(x,y)

plt.show()
