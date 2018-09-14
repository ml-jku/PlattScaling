#Copyright (C) 2018 Andreas Mayr, Guenter Klambauer
#Licensed under GNU General Public License v3.0 (see https://github.com/ml-jku/PlattScaling/blob/master/LICENSE)

import subprocess
#import distutils
#import distutils.core
import setuptools
import setuptools.command.build_py

class MyBuildCommand(setuptools.command.build_py.build_py):
  def run(self):
    setuptools.command.build_py.build_py.run(self)
    if not self.dry_run:
      subprocess.call(['make', '-C', 'src', 'target=../'+self.build_lib])

setuptools.setup(
  cmdclass={'build_py': MyBuildCommand},
  name='platt',
  version='1.0',
  description='implements Platt Scaling',
  author='Andreas Mayr, Guenter Klambauer',
  author_email='mayr@bioinf.jku.at',
  packages=['platt'],
  package_data={'platt': ['libPlatt.so']},
  install_requires=['numpy', 'scipy']
)





