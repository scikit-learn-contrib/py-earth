Py-earth documentation
----------------------

This folder contains all the necessary files for building py-earth
documentation (based on sphinx <http://www.sphinx-doc.org>).
Building the documentation requires the following packages : matploblib, numpydoc, sphinx-gallery and sphinxcontrib-bibtex.You can install them with pip :

```
pip install matplotlib
pip install numpydoc
pip install sphinx-gallery
pip install sphinxcontrib-bibtex
```

You can then generate html documentation by running :

```
make html
```

Other formats are supported by Sphinx, you can check the supported formats using :


```
make help
```
