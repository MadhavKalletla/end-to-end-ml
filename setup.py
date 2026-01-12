from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path) as file:
        requirements = []
        for line in file:
            line = line.strip()
            if line != "-e .":     
                requirements.append(line)
        return requirements

setup(
    name="endtoend ml project",
    version="0.0.1",
    author="Madhav",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
