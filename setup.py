from setuptools import setup, find_packages

setup(
    name='inscd-tookit',
    version='1.3.0.post1',
    description='InsCD: A Modularized, Comprehensive and User-Friendly Toolkit for Machine Learning Empowered Cognitive Diagnosis',
    author='Zihan Zhou',
    author_email='',
    url='https://github.com/ECNU-ILOG/InsCD',
    packages=find_packages(),
    py_modules=['inscd_run'],
    python_requires='>=3.7',
    install_requires=[
    'gdown>=5.2.0',
    'pandas>=1.0.0',
    'numpy>=1.20',
    'torch>=2.0',
    'scikit-learn>=1.0',
    'scipy>=1.5',
    'dgl>=1.0',
    'joblib>=1.2',
    'tqdm>=4.60',
    'accelerate>=0.20',
    'pyyaml>=6.0',
    'wandb>=0.15',
    'deap>=1.4.0',
    'networkx>=2.6',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'inscd_run = inscd_run:cli', 
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
)
