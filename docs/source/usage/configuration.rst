Configuration File Reference
============================

The mejiro pipeline is driven entirely by a YAML configuration file. Example configurations live in ``mejiro/data/mejiro_config/`` (``simple.yaml``, ``test.yaml``, ``roman_test.yaml``, ``lsst.yaml``); the recommended starting point is to copy one and edit it.

This page documents every top-level section and attribute. Units are given only where they are documented in the YAML comments or in the code. Pipeline scripts referenced below all live in ``mejiro/pipeline/``.

Each pipeline step checks the config when it starts, and ``Pipeline.run()`` checks it for every step it will run before the first one starts. A missing required key, a value of the wrong type or an unrecognized key stops the run with a ``ConfigError`` that lists every problem. A config only needs the keys that the steps you run require, which each entry below lists.

The entries below are generated from the schema in :mod:`mejiro.pipeline.config_schema`, which also supplies the defaults.

.. include:: ../_generated/config_reference.inc
