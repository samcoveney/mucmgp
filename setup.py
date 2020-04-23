from setuptools import setup

setup(name = 'mucmgp',
      version = '0.1',
      description = 'Gaussian Process implementation from Managing Uncertainty for Complex Models',
      url = 'http://github.com/samcoveney/mucmgp',
      author = 'Sam Coveney',
      author_email = 'coveney.sam@gmail.com',
      license = 'GPL-3.0+',
      packages = ['mucmgp'],
      install_requires = [
          'numpy',
          'scipy',
          'future',
      ],
      zip_safe = False)
