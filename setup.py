from setuptools import setup, find_packages

setup(
    name="irbns",
    version="0.0.1",
    description="Inhomogeneous radius-bounded noise systems (dynamic set-valued map on GPU)",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "scipy",
    ],
    python_requires=">=3.8",
)
