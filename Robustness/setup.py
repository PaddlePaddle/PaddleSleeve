from setuptools import setup
from setuptools import find_packages

requirements = [
    'numpy',
    'matplotlib',
    'urllib3',
    'tqdm',
    'pillow',
    'scipy',
    'pandas'
]

setup(
    description='Robustness benchmark for deep learning models',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)
