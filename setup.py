from distutils.core import setup

setup(
    name="personal_sg",
    version="0.1",
    author="Benshi_Mei",
    packages=["personal_sg"],
    install_requires=[
        "gym==0.9.4",
        "gymnasium~=0.29.1",
        "pathos>=0.3.1",
        "scipy>=1.11.0",
        "matplotlib>=3.7.2",
        "numpy>=1.25.0",
        "pandas>=2.0.3",
    ],
    include_package_data=True,
    zip_safe=False,
)
