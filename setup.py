from setuptools import setup, find_packages

setup(
    name="bio-nn",
    version="0.1.0",
    description="Biologically Inspired Neural Network Research Framework",
    author="Bio-NN Research Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "snnTorch>=0.7.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dashboard": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "jinja2>=3.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "brian": [
            "brian2>=2.5.0",
        ],
    },
)
