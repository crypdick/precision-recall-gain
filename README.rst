========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |coveralls| |codecov|
        | |scrutinizer|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/precision-recall-gain/badge/?style=flat
    :target: https://precision-recall-gain.readthedocs.io/
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/crypdick/precision-recall-gain.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/crypdick/precision-recall-gain

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/crypdick/precision-recall-gain?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/crypdick/precision-recall-gain

.. |requires| image:: https://requires.io/github/crypdick/precision-recall-gain/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/crypdick/precision-recall-gain/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/crypdick/precision-recall-gain/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/crypdick/precision-recall-gain

.. |codecov| image:: https://codecov.io/gh/crypdick/precision-recall-gain/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/crypdick/precision-recall-gain

.. |version| image:: https://img.shields.io/pypi/v/precision-recall-gain.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/precision-recall-gain

.. |wheel| image:: https://img.shields.io/pypi/wheel/precision-recall-gain.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/precision-recall-gain

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/precision-recall-gain.svg
    :alt: Supported versions
    :target: https://pypi.org/project/precision-recall-gain

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/precision-recall-gain.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/precision-recall-gain

.. |commits-since| image:: https://img.shields.io/github/commits-since/crypdick/precision-recall-gain/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/crypdick/precision-recall-gain/compare/v0.0.0...master


.. |scrutinizer| image:: https://img.shields.io/scrutinizer/quality/g/crypdick/precision-recall-gain/master.svg
    :alt: Scrutinizer Status
    :target: https://scrutinizer-ci.com/g/crypdick/precision-recall-gain/


.. end-badges

Precision-recall-gain curves for Python

* Free software: MIT license

Installation
============

::

    pip install precision-recall-gain

You can also install the in-development version with::

    pip install https://github.com/crypdick/precision-recall-gain/archive/master.zip


Documentation
=============


https://precision-recall-gain.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

References
===========
* [Precision-Recall-Gain Curves: PR Analysis Done Right (2015) by Peter A. Flach and Meelis Kull](https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf)
* [sklearn-compatible implementation](https://github.com/scikit-learn/scikit-learn/pull/24121) by [Bradley Fowler](https://github.com/bradleyfowler123)
* [PRG curves](https://www.biostat.wisc.edu/~page/rocprg.pdf) by [David Page](https://www.biostat.wisc.edu/~page/)
* [Blog post by Bradley Fowler](https://snorkel.ai/improving-upon-precision-recall-and-f1-with-gain-metrics/)
* [Original implementation](https://github.com/meeliskull/prg) by [Meelis Kull](https://github.com/meeliskull)
