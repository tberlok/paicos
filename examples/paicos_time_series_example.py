import paicos as pa
import numpy as np
import matplotlib.pyplot as plt

snap = pa.Snapshot(pa.data_dir, 247)

snap.load_data(0, 'Density')

unit = snap['0_MagneticField'].unit

a = np.logspace(-2, 0, 100)
B = pa.units.PaicosTimeSeries(np.ones_like(a), unit, h=snap.h, a=a,
                              comoving_sim=True)

Bc = B.to('uG').no_small_h
Bc2 = B.no_small_h.to('uG')
Bphys = B.to_physical.to('uG')

f = pa.PaicosTimeSeriesWriter(snap, pa.data_dir + 'test_data',
                              basename='time_series')

# Write the time series to file
f.write_data('Bc', Bc)
# f.write_data('Bphys', Bphys)
f.finalize()


plt.figure(1)
plt.clf()
plt.loglog(B.a, Bc, label=r'$B_{\mathrm{comoving}}$')
plt.loglog(B.a, Bphys, label=r'$B_{\mathrm{phys}}$')
plt.xlabel('a')
plt.ylabel(Bc.label() + r'$\;,\;$' + Bphys.label())
plt.legend(frameon=False)


del B, Bc, Bphys
# Read the data and make a new plot

ser = pa.PaicosReader(pa.data_dir + 'test_data', basename="time_series")

plt.figure(2)
plt.clf()

Bc = ser['Bc']
Bphys = Bc.to_physical
age = Bphys.age(ser)
plt.semilogy(age, Bc, label=r'$B_{\mathrm{comoving}}$')
plt.semilogy(age, Bphys, label=r'$B_{\mathrm{phys}}$')
plt.xlabel(age.label(r'\mathrm{age}'))
plt.ylabel(Bc.label() + r'$\;,\;$' + Bphys.label())
plt.legend(frameon=False)
plt.show()
