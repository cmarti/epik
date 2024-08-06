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
.. autoclass:: epik.src.kernel.PairwiseKernel
.. autoclass:: epik.src.kernel.ExponentialKernel
.. autoclass:: epik.src.kernel.VarianceComponentKernel
.. autoclass:: epik.src.kernel.ConnectednessKernel
.. autoclass:: epik.src.kernel.JengaKernel
.. autoclass:: epik.src.kernel.GeneralProductKernel
.. autoclass:: epik.src.kernel.AdditiveHeteroskedasticKernel

Utilities
----------

.. autofunction:: epik.src.utils.split_training_test
.. autofunction:: epik.src.utils.encode_seqs