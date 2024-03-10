# Downloading sample data

The examples require an Arepo snapshot. You can download one [here](https://sid.erda.dk/sharelink/DkVXfdoxIM). Alternatively, open a terminal at
the root directory of the Paicos repo, then run
```
wget -O data/fof_subhalo_tab_247.hdf5 https://sid.erda.dk/share_redirect/BmILDaDPPz
wget -O data/snap_247.hdf5 https://sid.erda.dk/share_redirect/G4pUGFJUpq
```

If you have done a pip-installation, then you will need to tell Paicos where the data is on your system.
This is done by adding something like this
```
# Explicitly set data directory (only needed for pip installations)
data_dir = '/Users/berlok/projects/paicos/data/'
```
to your `paicos_user_settings.py` (see the user configuration tab).

# Downloading high-resolution sample data

A couple of examples require a high-resolution simulation snapshot.
You can download one [here](https://sid.erda.dk/sharelink/aSwKpJ4o7d)
or you can run the following from the terminal (at the root level
of your local version of the Paicos repo)
```
wget -O data/highres/fof_subhalo_tab_247.hdf5 https://sid.erda.dk/share_redirect/aoerIlDtR3
wget -O data/highres/snap_247.hdf5 https://sid.erda.dk/share_redirect/bBs1w2fE0s
```
