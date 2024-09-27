Running ``mejiro``
##################

Update pipeline parameters in ``config/config.yaml`` and ensure ``defaults.machine`` points to the name (without extension) of the yaml file created during the first-time setup. The bash script ``execute_pipeline.sh`` will execute the pipeline end-to-end:

.. code-block:: bash    

    time bash execute_pipeline.sh