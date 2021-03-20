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
        | |scrutinizer| |codacy| |codeclimate|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-precision-recall-gain/badge/?style=flat
    :target: https://python-precision-recall-gain.readthedocs.io/
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/crypdick/python-precision-recall-gain.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/crypdick/python-precision-recall-gain

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/crypdick/python-precision-recall-gain?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/crypdick/python-precision-recall-gain

.. |requires| image:: https://requires.io/github/crypdick/python-precision-recall-gain/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/crypdick/python-precision-recall-gain/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/crypdick/python-precision-recall-gain/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/crypdick/python-precision-recall-gain

.. |codecov| image:: https://codecov.io/gh/crypdick/python-precision-recall-gain/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/crypdick/python-precision-recall-gain

.. |codacy| image:: https://img.shields.io/codacy/grade/[Get ID from https://app.codacy.com/app/crypdick/python-precision-recall-gain/settings].svg
    :target: https://www.codacy.com/app/crypdick/python-precision-recall-gain
    :alt: Codacy Code Quality Status

.. |codeclimate| image:: https://codeclimate.com/github/crypdick/python-precision-recall-gain/badges/gpa.svg
   :target: https://codeclimate.com/github/crypdick/python-precision-recall-gain
   :alt: CodeClimate Quality Status

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

.. |commits-since| image:: https://img.shields.io/github/commits-since/crypdick/python-precision-recall-gain/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/crypdick/python-precision-recall-gain/compare/v0.0.0...master


.. |scrutinizer| image:: https://img.shields.io/scrutinizer/quality/g/crypdick/python-precision-recall-gain/master.svg
    :alt: Scrutinizer Status
    :target: https://scrutinizer-ci.com/g/crypdick/python-precision-recall-gain/


.. end-badges

Precision-recall-gain curves for Python

* Free software: MIT license

Installation
============

::

    pip install precision-recall-gain

You can also install the in-development version with::

    pip install https://github.com/crypdick/python-precision-recall-gain/archive/master.zip


Documentation
=============


https://python-precision-recall-gain.readthedocs.io/


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
