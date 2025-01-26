from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'scipy',
        'pandas'
    ],
    package_data={
        'src': [
            'Training/*.py',
            'Models/*.py'
        ]
    },
    entry_points={
        'console_scripts': [
            'train=src.Training.trainer:main',
            'solve=src.Training.solver:main',
            'pino=src.Training.NeuralPINN:main'
        ]
    }
)