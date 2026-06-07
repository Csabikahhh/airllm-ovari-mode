import setuptools

# Windows uses a different default encoding (use a consistent encoding)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="airllm",
    version="2.11.0",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="AirLLM allows single 4GB GPU card to run 70B large language models without quantization, distillation or pruning. 8GB vmem to run 405B Llama3.1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyogavin/airllm",
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm>=4.66',
        'torch>=2.5',
        'transformers>=4.45,<6',
        'accelerate>=1.0,<2',
        'safetensors>=0.4',
        'optimum>=1.23',
        'huggingface-hub>=0.26',
        'scipy',
        #'bitsandbytes' set it to optional to support fallback when not installable
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
