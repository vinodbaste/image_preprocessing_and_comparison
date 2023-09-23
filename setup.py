from setuptools import find_packages, setup

classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Data Science',
  'Operating System :: Microsoft :: Windows',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

setup(
    name='image_preprocessing_comparison',
    packages=find_packages(),
    version='1.0.0',
    classifiers=classifiers,
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    description='This Python library provides a comprehensive set of functions for image preprocessing and '
                'comparison. It includes various filters for image enhancement, as well as metrics for comparing '
                'images.',
    author='Vinod Baste',
    url='https://github.com/vinodbaste/image_preprocessing_and_comparison/blob/main/Readme.md',
    author_email='bastevinod@gmail.com',
    keywords=['image preprocessing', 'image comparison'],
    license='MIT'
)