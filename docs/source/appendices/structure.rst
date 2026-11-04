Structure
=========

``mejiro.utils`` are foundational modules that shouldn't call anything else in ``mejiro``. ``mejiro.engines`` wrap a single package. Similarly, ``instruments`` wrap a single package or reference file.

The diagram below shows which ``mejiro`` modules import which. It's generated with ``pydeps`` every time the docs are built, so it always matches the code. Pipeline scripts are left out to keep it readable. Modules drawn as blue boxes are part of an import cycle.

.. image:: /_generated/mejiro.svg
    :alt: mejiro module dependency diagram
    :align: center