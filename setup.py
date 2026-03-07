import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wimusim",
    version="0.2.0",
    description="PhyNeSim — Physics-Neural IMU Simulator (WIMUSim + SMPL + neural residual corrector)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/si-runnan/PhyNeSim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=1.5.0",
        "requests",
        "pybullet>=3.2.5",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "smplx>=0.1.28",
        "matplotlib>=3.7.0",
        # torch and pytorch3d must be installed manually before pip install -e .
        # See requirements.txt for instructions (CUDA 12.8 required for RTX 50-series)
        "torch>=2.10.0",
        "pytorch3d>=0.7.9",
    ],
    extras_require={
        "dev": [
            "jupyterlab>=4.0.0",
            "pytest>=7.0.0",
        ]
    },
    python_requires='>=3.10',
)
