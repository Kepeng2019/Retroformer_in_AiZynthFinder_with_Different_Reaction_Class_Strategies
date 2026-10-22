from setuptools import setup, find_packages

setup(
    name="main_project",
    version="0.1",
    packages=find_packages(where="external"),
    package_dir={"": "external"},  
 
    # extras_require={
    #     "dev": ["pytest>=6.0", "black>=21.0"]
    # },
    # python_requires=">=3.8",
)