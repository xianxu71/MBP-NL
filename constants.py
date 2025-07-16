"""Constants"""
# File taken from Electron-phononcoupling

# Tolerance criterions
tol5 = 1E-5
tol6 = 1E-6
tol8 = 1E-8
tol12 = 1E-12

# Conversion factor
Ha2eV = 27.21138386
eV2Ry = 2./Ha2eV
Ry2eV = 1./eV2Ry
a2bohr = 1.88973

# Boltzman constant
kb_HaK = 3.1668154267112283e-06

# Electron mass over atomical mass unit
me_amu = 5.4857990965007152E-4

# charge
eRy = 2**0.5

eps0 = 1./(4*3.141592653589793)

# We use Ry in energy so time unit = hbar/Ry
# convert time unit to nanosec
# hbar/hartree = 2.418884326505*10^{-17} sec
t2nsec = 2.418884326505*2*1E-8

# current density 1 a.u = 8.3627316 x 10^17 A/m^2
# conductivity in 3D, 1 Ry a.u = 2.2999241 x 10^6 A/mV
# Efield is V/m, 1 Ry a.u = 3.6360903 x 10^11 V/m
# We compute sc-sigma = J/E^2
# 8.3627316 x10^17 / (3.6360903 x 10^11)^2 = 0.632526664572245 x 10^-5  A/m^2 * m^2/V^2 
#                                          = 6.32526664572245 mu A / V^2
au2muAdivV2 = 6.32526664572245

# SHG unit 
au2esu = 5.825*1e-8 * 2**0.5
au2pmdivV = 1./0.02893

