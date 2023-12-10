import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='RVENet',
   version='1.0beta',
   description='Source code related to the research paper entitled RVENet: A Large Echocardiographic Dataset for the Deep Learning-Based Assessment of Right Ventricular Function',
   author='Bálint Magyar',
   author_email='magyar.balint@itk.ppke.hu',
   install_requires=required
)