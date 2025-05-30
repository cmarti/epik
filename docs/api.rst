.. _api:

API Reference
==============

Model
-----

.. autoclass:: epik.model.EpiK
    :members: set_data, simulate, fit, predict, make_contrasts, 
        predict_mut_effects, predict_epistatic_coeffs, save, load


Kernels
-------

.. autoclass:: epik.kernel.AdditiveKernel
    :members: forward
.. autoclass:: epik.kernel.PairwiseKernel
    :members: forward
.. autoclass:: epik.kernel.VarianceComponentKernel
    :members: forward
.. autoclass:: epik.kernel.ExponentialKernel
    :members: forward
.. autoclass:: epik.kernel.ConnectednessKernel
    :members: forward, get_delta
.. autoclass:: epik.kernel.JengaKernel
    :members: forward, get_delta, get_mutation_delta
.. autoclass:: epik.kernel.GeneralProductKernel
    :members: forward, get_delta

Utilities
----------

.. autofunction:: epik.utils.split_training_test
.. autofunction:: epik.utils.encode_seqs