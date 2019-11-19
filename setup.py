from setuptools import setup

setup(name='tensorcox',
      version='0.1',
      description='Coxs partial likelihood in Tensorflow',
      url='http://github.com/alexwjung/TensorCox',
      author='awj',
      author_email='alexwjung@googlemail.com',
      license='MIT',
      packages=['tensorcox', 'tensorcox.test'],
      install_requires=['Tensorflow'],
      long_description=open('README.txt').read(),
      zip_safe=False)
