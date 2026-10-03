from setuptools import setup, find_packages

setup(
    name="metalliger",
    version="0.3.0",
    description="Fused Metal kernels for PyTorch MPS training & inference on Apple Silicon",
    author="MetalLiger Contributors",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.14.0",
        "torchvision>=0.29.0,
        "transformers>=5.17.0",
    ],
)
