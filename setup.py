import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cascading cache",
    version="0.0.1",
    author="Jeffrey Willette",
    author_email="jwillette@kaist.ac.kr",
    description="Cascading Cache",
    long_description=long_description,
    long_description_content_type="jeffwillette/cascading_cache",
    url="https://github.com/jeffwillette",
    project_urls={
        "Bug Tracker": "https://github.com/jeffwillette/cascading_cache",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"cascading_cache": "cascading_cache"},
    packages=["cascading_cache"],
    python_requires=">=3.6",
)
