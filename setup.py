from setuptools import setup, find_packages

setup(
    name='extendtr',
    version='0.1.0',
    description='Extended TR: An extended Python package for topological regression for quantitative structure-activity relationship modeling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yixiang Mao',
    author_email='yixmao@ttu.edu',
    url='https://github.com/yourusername/your_package_name',  # Project URL
    license='MIT',            
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas', 
        'scikit-learn',
        'fastparquet',
        'rdkit',
        'tensorflow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Specify Python version compatibility
)