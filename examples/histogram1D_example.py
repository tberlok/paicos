import paicos as pa
import numpy as np

import matplotlib.pyplot as plt

logscale = True

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3, sharex=True)
for ii in range(2):
    snap = pa.Snapshot(pa.data_dir, 247)
    center = snap.Cat.Group["GroupPos"][0]

    r = np.sqrt(np.sum((snap["0_Coordinates"] - center[None, :]) ** 2.0,
                axis=1))

    if pa.settings.use_units:
        r_max = 10000 * r.unit_quantity
    else:
        r_max = 10000

    index = r < r_max * 1.1

    snap = snap.select(index, parttype=0)

    r = np.sqrt(np.sum((snap["0_Coordinates"] - center[None, :]) ** 2.0,
                axis=1))

    bins = [1e-3 * r_max, r_max * 1.1, 50]

    B2 = np.sum((snap["0_MagneticField"]) ** 2, axis=1)
    Volume = snap["0_Volume"]
    Masses = snap["0_Masses"]
    Temperatures = snap["0_Temperatures"]

    if ii == 0:
        h_r = pa.Histogram(r, bins, logscale=logscale, verbose=True)
        B2TimesVolume = h_r.hist((B2 * Volume))
        Volume = h_r.hist(Volume)
        TTimesMasses = h_r.hist((Masses * Temperatures))
        Masses = h_r.hist(Masses)

        axes[0].loglog(h_r.bin_centers, Masses / Volume)
        axes[1].loglog(h_r.bin_centers, B2TimesVolume / Volume)
        axes[2].loglog(h_r.bin_centers, TTimesMasses / Masses)

        if pa.settings.use_units:
            axes[0].set_xlabel(h_r.bin_centers.label(r"\mathrm{radius}\;"))
            axes[0].set_ylabel((Masses / Volume).label("\\rho"))
            axes[1].set_ylabel((B2TimesVolume / Volume).label("B^2"))
            axes[2].set_ylabel((TTimesMasses / Masses).label("T"))
    else:
        bins = np.linspace(0, r_max, 150)
        if pa.settings.use_units:
            r_max = r_max.value
            B2 = B2.value
            Volume = Volume.value
            Temperatures = Temperatures.value
            Masses = Masses.value
            r = r.value

        bins = np.logspace(np.log10(1e-3 * r_max), np.log10(r_max), 50)
        B2TimesVolume, edges = np.histogram(
            r, weights=(B2 * Volume), bins=bins
        )

        Volume, edges = np.histogram(r, weights=Volume, bins=bins)

        TTimesMasses, edges = np.histogram(
            r, weights=(Masses * Temperatures), bins=bins
        )

        Masses, edges = np.histogram(r, weights=Masses, bins=bins)

        bin_centers = 0.5 * (edges[1:] + edges[:-1])

        axes[0].loglog(bin_centers, Masses / Volume, "--")
        axes[1].loglog(bin_centers, B2TimesVolume / Volume, "--")
        axes[2].loglog(bin_centers, TTimesMasses / Masses, "--")

plt.show()
