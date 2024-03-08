# Contributions

Contributions to Paicos are very welcome! 

For new features, we strongly encourage you to create an issue describing the feature that you intend to develop before you start working on it. You are also very welcome to send an email directly to Thomas Berlok at tberlok ad nbi.ku.dk.


## Contribution guidelines

Here are some guidelines. Please don't hesitate to reach out if you have trouble running the commands below - it could be that they are outdated.

- Pull requests do not necessarily have to be the development of a new feature. Pull requests fixing spelling/grammar or in other ways improving the code base and/or documentation are just as welcome!

- To get started, we recommend that you install the development version of Paicos using option 1 from the installation instructions. See [here.](https://paicos.readthedocs.io/en/latest/installation.html#option-1-compile-the-code-and-add-its-path-to-your-pythonpath)

- Pull requests are required to pass all the automated tests. You can check locally that they pass by running the command
```
make dev_checks
```
which runs flake8 and pylint in addition to the automated tests with pytest.
You might need to install these if you don't have them installed already:
```
pip install flake8
pip install pylint
```
or simply 
```
pip install -r dev_requirements.txt
```
Description of current tests can be found [here.](https://paicos.readthedocs.io/en/latest/tests.html)

- New functionality is expected to include a Python script or notebook example illustrating the functionality that has been developed and
we also encourage you to add new tests!


- Functions and classes should have documentation strings, in the style of numpy or Google, so that the automated documentation on [readthedocs](https://paicos.readthedocs.io/en/latest/?badge=latest) works and is kept up to date. You can build the documentation locally as follows from the root directory:
```
pip install -r docs/requirements.txt
make -B docs
```
