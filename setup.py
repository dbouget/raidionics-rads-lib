from setuptools import find_packages, setup

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', errors='ignore') as ff:
    required = ff.read().splitlines()

setup(
    name='raidionicsrads',
    packages=find_packages(
        include=[
            'raidionicsrads',
            'raidionicsrads.Utils',
            'raidionicsrads.Processing',
            'raidionicsrads.NeuroDiagnosis',
            'raidionicsrads.MediastinumDiagnosis',
            'Atlases',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsrads = raidionicsrads.__main__:main'
        ]
    },
    install_requires=required,
    include_package_data=True,
    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     "raidionicsrads": ["Atlases/Schaefer400/*.csv", "Atlases/Schaefer400/*.nii.gz"],
    # },
    python_requires=">=3.6",
    version='0.1.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='MIT',
    description='Raidionics reporting and data system (RADS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
)