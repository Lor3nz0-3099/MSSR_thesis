from glob import glob
from setuptools import find_packages, setup


package_name = "mssr_expert"


setup(
    name=package_name,
    version="0.7.33",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config",
            glob("config/*.yaml") + glob("config/*.json"),
        ),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lorenzo",
    maintainer_email="lorenzo@todo.todo",
    description=(
        "Robot-family-agnostic graph, curriculum, and deterministic expert "
        "infrastructure for modular self-reconfigurable robots."
    ),
    license="TODO",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "mssr_expert_node = mssr_expert.nodes.expert_node:main",
            "mssr_curriculum_node = mssr_expert.nodes.curriculum_node:main",
            "mssr_smores_self_assembly_node = "
            "mssr_expert.nodes.smores_parallel_self_assembly_node:main",
            "mssr_smores_morphology_behavior_node = "
            "mssr_expert.nodes.smores_morphology_behavior_node:main",
            "mssr_smores_morphology_command_client = "
            "mssr_expert.nodes.smores_morphology_command_client:main",
            "mssr_smores_self_reconfiguration_node = "
            "mssr_expert.nodes.smores_self_reconfiguration_node:main",
            "mssr_smores_obstacle_course_node = "
            "mssr_expert.nodes.smores_obstacle_course_node:main",
        ],
    },
)
