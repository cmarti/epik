=======================================================================
EpiK: Epistatic Kernels for GPU-accelerated Gaussian process regression
=======================================================================

EpiK is a python library to infer sequence-function relationships using Gaussian 
Process models. EpiK builts on top of `GPyTorch <https://docs.gpytorch.ai>`_ and `KeOps <https://www.kernel-operations.io>`_
to enable fitting these models on large datasets comprising hundreds of thousands to millions
of sequence measurements. We provide a series of kernel functions whose parameters have
clear biological interpretations and provide better predictive performance
over typical kernel functions on continuous spaces. 


EpiK is written for Python 3 and is provided under an MIT open source license.
The documentation provided here is meant guide users through the basic application of the 
implemented models and the interpretation of the inferred hyperparameters.

Please do not hesitate to contact us with any questions or suggestions for improvements.

* For technical assistance or to report bugs, please contact Carlos Marti (`Email: martigo@cshl.edu <martigo@cshl.edu>`_, `Twitter: @cmarti_ga <https://twitter.com/cmarti_ga>`_).

* For more general correspondence, please contact David McCandlish (`Email: mccandlish@cshl.edu <mccandlish@cshl.edu>`_, `Twitter: @TheDMMcC <https://twitter.com/TheDMMcC>`_).


.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    installation
    tutorial
    api 

References
==========

.. [#mem] `Zhou J and McCandlsih DM.
    Minimum epistasis interpolation for sequence-function relationships (2020)
    <https://www.nature.com/articles/s41467-020-15512-5>`_

.. [#vc] `Zhou J, Wong MS, Chen WC, Krainer AR, Kinney JB, McCandlsih DM.
    Higher order epistasis and phenotypic prediction (2022) 
    <https://www.pnas.org/doi/full/10.1073/pnas.2204233119>`_


Links
=====

- `McCandlish Lab <https://www.cshl.edu/research/faculty-staff/david-mccandlish/#research>`_
