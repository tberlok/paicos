# Contributions

Contributions to Paicos are very welcome! 

For new features, we strongly encourage you to create an issue describing the feature that you intend to develop before you start working on it. You are also very welcome to send an email directly to Thomas Berlok at tberlok ad nbi.ku.dk.


## Contribution guidelines

- Pull requests do not necessarily have to be the development of a new feature. Pull requests fixing spelling/grammar or in other ways improving the code base and documentation are just as welcome!

- We recommend that you install the development of Paicos using option 1    from the installation instructions. See [here.](https://paicos.readthedocs.io/en/latest/installation.html#option-1-compile-the-code-and-add-its-path-to-your-pythonpath)


- Pull requests are required to pass all the automated tests. You can check locally that they pass by running the command
```
make dev_checks
```
which runs flake8 and pylint in addition to the automated tests with pytest.
See also [here.](https://paicos.readthedocs.io/en/latest/tests.html)

- New functionality is expected to include a Python script or notebook example illustrating the functionality that has been developed and
we also encourage you to add new tests!
