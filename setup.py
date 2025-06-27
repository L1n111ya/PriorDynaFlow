from setuptools import setup, find_packages

# 读取 README.md 作为 long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mainagent",
    version="0.1.0",
    author="yilin",
    author_email="",
    description="",
    packages=find_packages(),  # 自动发现所有包
    python_requires='>=3.8',
)