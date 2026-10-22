from setuptools import setup, find_packages

setup(
    name='Retroformer_as_one_step',  # 包名
    version='0.1',  # 包版本
    packages=find_packages(where='.', exclude=['tests*']), 
    install_requires=[  
        'numpy',  
        'pandas',
        'torch' 
    ],
    
    author='kepeng',
    author_email='chenkepengjy@163.com',
    description='One step model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.6',  
)

