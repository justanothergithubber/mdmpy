import setuptools

with open(README.md, r) as fh
    long_description = fh.read()

setuptools.setup(
    name=mdmpy,
    version=0.0.1,
    author=Example Author,
    author_email=author@example.com,
    description=A small example package,
    long_description=long_description,
    long_description_content_type=textmarkdown,
    url=httpsgithub.compypasampleproject,
    packages=setuptools.find_packages(),
    classifiers=[
        Programming Language  Python  3,
        License  OSI Approved  MIT License,
        Operating System  OS Independent,
    ],
)