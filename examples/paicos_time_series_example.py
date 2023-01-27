import paicos as pa
import numpy as np
import matplotlib.pyplot as plt

snap = pa.Snapshot(pa.root_dir + '/data', 247)

snap.load_data(0, 'Density')

unit = snap['0_MagneticField'].unit

a = np.logspace(-2, 0, 100)
B = pa.units.PaicosTimeSeries(np.ones_like(a), unit, h=snap.h, a=a)

Bc = B.to('uG').no_small_h
Bphys = B.to_physical.to('uG')

plt.figure(1)
plt.clf()
plt.loglog(B.a, Bc, label=r'$B_{\mathrm{comoving}}$')
plt.loglog(B.a, Bphys, label=r'$B_{\mathrm{phys}}$')
plt.xlabel('a')
plt.ylabel(Bc.label() + r'$\;,\;$' + Bphys.label())
plt.legend(frameon=False)
plt.show()

plt.figure(2)
plt.clf()
plt.semilogy(B.age(snap), Bc, label=r'$B_{\mathrm{comoving}}$')
plt.semilogy(B.age(snap), Bphys, label=r'$B_{\mathrm{phys}}$')
plt.xlabel(B.age(snap).unit.to_string(format='latex'))
plt.ylabel(Bc.label() + r'$\;,\;$' + Bphys.label())
plt.legend(frameon=False)
plt.show()
