# Tests

The tests are located in the `tests` sub-directory and are divided into tests of loading
and analyzing cosmological simluations (in the comoving subfolder) and non-cosmological
simulations (in the non-comoving subfolder).

The data necessary for running the tests is included in the repo. You can run these by moving
into the `tests` directory and running
```
pytest comoving
pytest non-comoving
pytest general
```
but it is also possible to run each test from the command line,
e.g.
```
python3 test_compare_slicer_with_snap_util.py
python3 test_compare_projector_with_snap_util.py
```

will run the test and display some plots where Paicos is compared with the arepo-snap-util Python package.
