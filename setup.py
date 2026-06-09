from setuptools import setup, find_packages

setup(
    name="beniget-analyzer",
    version="0.1.0",
    description="Static dead-code and unused-variable analyzer using beniget",
    python_requires=">=3.6",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "gast>=0.5.0",
        "beniget>=0.4.1",
    ],
    extras_require={
        "dev": ["pytest>=4.6.0"],
    },
    entry_points={
        "console_scripts": [
            "beniget-analyzer=analyzer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
