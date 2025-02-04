from setuptools import setup, find_packages

setup(
    name='dim',
    version='0.1.0',
    author='Namindu De Silva',
    author_email='nami.rangana@gmail.com',
    description='Predict pucker conformations in a DNA with arbitrary length',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/PDNALab/DIM.git',
    packages=find_packages(),
    classifiers=[               # Optional classifiers for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[  
        'pickel',
        'ittertools',
        'numpy'
    ]
)