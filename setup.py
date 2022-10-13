
import setuptools

with open("README.md", "r") as fh:
   long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

   
setuptools.setup(
    name='utils',
    version='0.1',
    author="Sajjad Ayoubi",
    author_email="sadeveloper360@gmail.com",
    description="utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sajjjadayobi/CLIPfa",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
