# Tests

The tests are locateds in the `tests` and are divided into tests of loading
and analyzing cosmological (in the comoving subfolder) and non-cosmological
simulations (in the non-comoving subfolder). The data necessary for the 
running the tests is including in the repo. You can run these 
using e.g.
```
pytest tests/comoving
pytest tests/non-comoving
```
but it is also possible to run each test from the command line,
e.g.
```
python3 tests/test_non_comoving.py
```
or
```
python3 tests/test_compare_slicer_with_snap_util.py
python3 tests/test_compare_projector_with_snap_util.py
```

will both run the test and present some plots (requires matplotlib).
