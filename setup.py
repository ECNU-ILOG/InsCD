from setuptools import setup, find_packages

setup(
    name='inscd-tookit',
    version='1.3.2',
    description='InsCD: A Modularized, Comprehensive and User-Friendly Toolkit for Machine Learning Empowered Cognitive Diagnosis',
    author='Zihan Zhou',
    author_email='',
    url='https://github.com/ECNU-ILOG/InsCD',
    packages=find_packages(),
    py_modules=['inscd_run'],
    python_requires='>=3.7',
    install_requires=[
    'gdown==5.2.0',
    'pandas==2.2.3',
    'numpy==1.26.4',
    'torch==2.4.0',
    'scikit-learn==1.5.2',
    'scipy==1.13.1',
    'joblib==1.4.2',
    'tqdm==4.66.5',
    'accelerate==1.1.1',
    'pyyaml==6.0.2',
    'wandb==0.18.5',
    'deap==1.4.1',
    'networkx==3.2.1',
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
