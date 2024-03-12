#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()

setup(
    name="llm_coordination",
    version="0.0.1",
    description="Framework for facilating multi-agent coordination using Large Langugage Models",
    author="Saaket Agashe",
    author_email="saagashe@ucsc.edu",
    packages=find_packages("src"),
    keywords=["LLM", "Agents", "Overcooked", "AI"],
    package_dir={"": "src"},
    package_data={
        "overcooked_ai_py": [
            "data/layouts/*.layout",
            "data/planners/*.py",
            "data/human_data/*.pickle",
            "data/graphics/*.png",
            "data/graphics/*.json",
            "data/fonts/*.ttf",
        ],
        "reasong_evals": [
            "data/*.csv"
        ]
    },

)
