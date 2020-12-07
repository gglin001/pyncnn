from setuptools import setup, find_packages

import sys
if sys.version_info < (3,0):
  sys.exit('Sorry, Python < 3.0 is not supported')

requirements = [
    'numpy',
    #'tqdm',
    #'requests',
    #'portalocker',
    #'opencv-python'
]

setup(
  name          = 'ncnn',
  version       = '${PACKAGE_VERSION}',
  url           = 'https://github.com/gglin001/pyncnn',
  packages      = find_packages(),
  package_dir   = {'': '.'},
  package_data  = {'ncnn': ['ncnn.*.so', 'ncnn.*.pyd', 'ncnn.*.dll']},
  install_requires = requirements
)
