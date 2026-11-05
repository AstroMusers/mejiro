Running the Pipeline
====================

Simplest Example
----------------

The following script runs a simple test version of the end-to-end pipeline, using ``simple.yaml``. It needs a clone of the repository, installed with the ``roman`` optional dependency (see :doc:`installation`):

.. code-block:: bash

    pip install -e ".[roman]"

Save this as a script and run it, changing ``path/to/output`` to wherever the output should go:

.. code-block:: python

    from mejiro.pipeline.mejiro_pipeline import Pipeline

    if __name__ == '__main__':
        pipeline = Pipeline(data_dir='path/to/output')
        pipeline.run()

Keep the ``if __name__ == '__main__':`` line. Some pipeline steps start worker processes that import your script again, and without this line they would start the pipeline again too, which stops the run with an error.

``simple.yaml`` leaves ``data_dir`` unset, so ``Pipeline()`` without ``data_dir`` raises a ``ValueError``. Output goes to ``<data_dir>/simple_dev/``, because ``simple.yaml`` sets ``pipeline_label: simple`` and ``dev: True``. See :doc:`outputs` for what each step writes.

``STPSF``'s data files are only needed to compute PSFs that aren't cached yet. The PSFs ``simple.yaml`` uses are cached in ``tests/test_data/``, so this example doesn't need the data files. That directory only exists in a clone of the repository, which is why this example needs ``pip install -e`` from a clone.

Tweaking the Configuration
--------------------------

To run the pipeline with customized settings:

1. Create a configuration file by copying one of the ``mejiro`` configuration files in ``mejiro/data/mejiro_config/`` to your working directory and modifying it as needed. See :doc:`configuration` for a reference of every attribute.
2. Set ``data_dir`` in your configuration file, or pass it to ``Pipeline`` as in the example above.
3. Create a ``Pipeline`` object with the path to your configuration file as an argument, then run the pipeline.

.. code-block:: python

    from mejiro.pipeline.mejiro_pipeline import Pipeline

    if __name__ == '__main__':
        config_file = 'path/to/your_config_file.yml'

        pipeline = Pipeline(config_file=config_file)
        pipeline.run()
