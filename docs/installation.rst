.. _installation:

Installation Instructions
=========================

We recommend using an new independent environment with python3.8, as used during 
development and testing of gpmap-tools to minimize problems with dependencies. For instance,
one can create and activate a new conda environment as follows: ::

    $ conda create -n gpmap python=3.8
    $ conda activate epik

Now you can clone EpiK from `GitHub <https://github.com/cmarti/epik>`_ as follows: ::

    $ git clone https://github.com/cmarti/epik.git

and install it in the current python environment: ::
    
    $ cd epik
    $ pip install .

To test installation is working properly you can run all tests or a
subset of them. Running all of them may take some time. ::

    $ python -m unittest epik/test/*py

