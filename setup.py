from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)-> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]


setup(
name='DynamicGridManagement',
version='0.0.1',
author='Sanjay',
author_email='sanjay.devarajan02@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)