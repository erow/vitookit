from setuptools import setup, find_packages

setup(
    name='vitookit',
    author='Gent',
    author_email='clouderow@gmail.com',
    decscription='A toolkit for evaluating and analyzing vision models',
    # keywords='evaluation',    
    version='0.1',
    packages=find_packages(),
    # package_dir={"": "vitookit"},
    package_data={
        'evaluation': ['*.py'],  
        'config': ['*.gin'],
    },
    scripts=[
        'bin/vitrun',
        'bin/submitit',
    ]
)