from setuptools import setup, find_packages

setup(
    name=‘FHAI’,
    version='0.0.1',
    packages=find_packages(),
    url=‘https://github.com/CalebFikes/FHAI’,
    license='MIT',
    author=‘CalebFikes’,
    author_email=‘cjf6@rice.edu’,
    description='',
    install_requires=['torch', 'torchvision', 'matplotlib', 'numpy', 'scikit-learn']
)
