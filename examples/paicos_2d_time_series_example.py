import paicos as pa
import numpy as np
import matplotlib.pyplot as plt

snap = pa.Snapshot(pa.data_dir, 247)

snap.load_data(0, 'Density')

unit = snap['0_MagneticField'].unit

avec = np.logspace(-2, 0, 100)

# Construct a 2D time series
Bvec = []
for a in avec:
    B = pa.units.PaicosQuantity(np.arange(1, 10), unit, h=snap.h, a=a,
                                comoving_sim=True)
    Bvec.append(B)

B = pa.units.PaicosTimeSeries(Bvec)

Bc = B.to('uG').no_small_h
Bphys = B.to_physical.to('uG')

f = pa.PaicosTimeSeriesWriter(snap, pa.data_dir + 'test_data',
                              basename='time_series2d')

# Write the time series to file
f.write_data('Bc', Bc)
# f.write_data('Bphys', Bphys)
f.finalize()


plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=2, sharey=True)
axes[0].loglog(B.a, Bc)
axes[0].set_title(r'$B_{\mathrm{comoving}}$')
axes[1].loglog(B.a, Bphys)
axes[1].set_title(r'$B_{\mathrm{phys}}$')
for ii in range(2):
    axes[ii].set_xlabel('a')
axes[0].set_ylabel(Bc.label())
axes[1].set_ylabel(Bphys.label())


del B, Bc, Bphys
# Read the data and make a new plot

ser = pa.PaicosReader(pa.data_dir + 'test_data', basename="time_series2d")

plt.figure(2)
plt.clf()

Bc = ser['Bc']
Bphys = Bc.to_physical
age = Bphys.age(ser)
plt.figure(2)
plt.clf()
fig, axes = plt.subplots(num=2, ncols=2, sharey=True)
axes[0].semilogy(age, Bc)
axes[0].set_title(r'$B_{\mathrm{comoving}}$')
axes[1].semilogy(age, Bphys)
axes[1].set_title(r'$B_{\mathrm{phys}}$')
for ii in range(2):
    axes[ii].set_xlabel('a')
axes[0].set_ylabel(Bc.label())
axes[1].set_ylabel(Bphys.label())
plt.show()

Bc * Bc

Bc / Bc

Bc - Bc

Bc**2
