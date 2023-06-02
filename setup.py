from setuptools import setup, find_packages

setup(
    name='deep_forest',
    version='0.0.1.dev0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'joblib',
        'scikit-learn',
        'imbalanced-learn',
    ],
)
