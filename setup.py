#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.1.0'


def main():
    description = 'GPU-accelerated Gaussian Process regression on sequence space'
    setup(
        name='epik',
        version=VERSION,
        description=description,
        license='MIT',
        author='Carlos Martí-Gómez',
        author_email='martigo@cshl.edu',
        url='https://bitbucket.org/cmartiga/epik',
        packages=find_packages(),
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'EpiK = bin.EpiK:main',
            ]},
        install_requires=['numpy', 'pandas', 'sklearn',
                          'torch', 'gpytorch==1.11', 'tqdm'],
        python_requires='>=3',
        platforms='ALL',
        keywords=['genotype-phenotyp maps', 'fitness landscape', 
                  'gaussian process'],
        classifiers=[
            "Programming Language :: Python :: 3",
            'Intended Audience :: Science/Research',
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
    return


if __name__ == '__main__':
    main()
