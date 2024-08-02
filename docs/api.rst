.. _api:

API Reference
==============

Model
-----

.. autoclass:: epik.src.model.EpiK
    :members: set_data, fit, predict, simulate, save, load

Kernels
-------

.. autoclass:: epik.src.kernel.AdditiveKernel
    :members: forward

.. autoclass:: epik.src.kernel.PairwiseKernel
    :members: forward

.. autoclass:: epik.src.kernel.ExponentialKernel
    :members: forward

.. autoclass:: epik.src.kernel.VarianceComponentKernel
    :members: forward

.. autoclass:: epik.src.kernel.ConnectednessKernel
    :members: forward

.. autoclass:: epik.src.kernel.JengaKernel
    :members: forward

.. autoclass:: epik.src.kernel.GeneralProductKernel
    :members: forward

.. autoclass:: epik.src.kernel.AdditiveHeteroskedasticKernel
    :members: forward

Utilities
----------

.. autofunction:: epik.src.utils.split_training_test
.. autofunction:: epik.src.utils.encode_seqs