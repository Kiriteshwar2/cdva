from __future__ import annotations

from importlib import import_module

__all__ = [
    "DatasetArtifacts",
    "MPDownloadConfig",
    "MaterialsProjectDatasetBuilder",
    "build_dataset",
    "CrystalGraphBuilder",
    "CrystalGraphDataset",
    "GraphBuildConfig",
]


def __getattr__(name: str):
    if name in {"DatasetArtifacts", "MPDownloadConfig", "MaterialsProjectDatasetBuilder", "build_dataset"}:
        module = import_module("data.mp_dataset")
        return getattr(module, name)
    if name in {"CrystalGraphBuilder", "CrystalGraphDataset", "GraphBuildConfig"}:
        module = import_module("data.graph_builder")
        return getattr(module, name)
    raise AttributeError(f"module 'data' has no attribute {name}")
