Running the Pipeline
====================

Simplest Example
----------------

The following lines will run a simple test version of the end-to-end pipeline.

.. code-block:: python

    from mejiro.pipeline.mejiro_pipeline import Pipeline
    
    pipeline = Pipeline()
    pipeline.run()

Tweaking the Configuration
--------------------------

The test version of the pipeline is not scientifically useful. To run the pipeline with customized settings:

1. Create a configuration file by copying one of the ``mejiro`` configuration files in ``mejiro/data/mejiro_config/`` to your working directory and modifying it as needed. See the comments in the file for explanations of the various options.
2. Create a ``Pipeline`` object with the path to your configuration file as an argument, then run the pipeline.

.. code-block:: python

    from mejiro.pipeline.mejiro_pipeline import Pipeline
    
    config_file = 'path/to/your_config_file.yml'

    pipeline = Pipeline(config_file=config_file)
    pipeline.run()
