# encoding: utf-8
from setuptools import setup, find_packages


setup(
    name='ElevatorSimulator',
    version='1.0.0',
    author='Ge Hangli',
    author_email='hangli.ge@koshizula-lab.org',
    description='Proactive Traffic Manageable Elevator Simulation for Python.',
    long_description='\n\n'.join(
        open(f, 'rb').read().decode('utf-8')
        for f in ['README.txt', 'CHANGES.txt', 'AUTHORS.txt']),
    #url='https://simpy.readthedocs.io',
    license='MIT License',
    install_requires=[],
)
