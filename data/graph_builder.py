from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.preprocessing import (
    canonicalize_structure,
    extract_lattice_parameters,
    read_jsonl_record_at_offset,
    record_to_structure,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []


@dataclass(frozen=True)
class GraphBuildConfig:
    cutoff_angstrom: float = 5.0
    max_neighbors: int = 32
    graph_cache_dir: Path | None = None
    force_rebuild: bool = False


class CrystalGraphBuilder:
    """Convert processed Materials Project records into PyG graphs."""

    def __init__(self, cutoff_angstrom: float = 5.0, max_neighbors: int = 32):
        self.cutoff_angstrom = cutoff_angstrom
        self.max_neighbors = max_neighbors

    def build_graph(self, record: dict[str, Any], species_vocab: dict[str, int]) -> Data:
        structure = record_to_structure(record)
        if structure is None:
            return None  # skip invalid structure
        structure = canonicalize_structure(structure)
        atom_symbols = [site.specie.symbol for site in structure.sites]
        atom_types = torch.tensor([species_vocab[symbol] for symbol in atom_symbols], dtype=torch.long)
        atomic_numbers = torch.tensor([site.specie.Z for site in structure.sites], dtype=torch.long)
        frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
        cart_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
        lattice_params = torch.tensor(extract_lattice_parameters(structure), dtype=torch.float32).unsqueeze(0)
        lattice_matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        edge_index, edge_attr, edge_vec = self._build_edges(structure)

        return Data(
            x=atom_types.unsqueeze(-1),
            atom_types=atom_types,
            atomic_numbers=atomic_numbers,
            frac_coords=frac_coords,
            cart_coords=cart_coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_vec=edge_vec,
            lattice_params=lattice_params,
            lattice_matrix=lattice_matrix,
            num_atoms=torch.tensor([len(structure)], dtype=torch.long),
            num_elements=torch.tensor([len(set(atom_symbols))], dtype=torch.long),
            material_id=record["material_id"],
            num_nodes=len(structure),
        )

    def _build_edges(self, structure) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_indices, dst_indices, images, distances = structure.get_neighbor_list(self.cutoff_angstrom)
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords
        per_source: dict[int, list[tuple[float, int, list[float]]]] = {idx: [] for idx in range(len(structure))}

        for src, dst, image, distance in zip(src_indices, dst_indices, images, distances):
            src = int(src)
            dst = int(dst)
            distance = float(distance)
            if src == dst and distance < 1e-8:
                continue
            neighbor_cart = cart_coords[dst] + image @ lattice_matrix
            rel_vec = (neighbor_cart - cart_coords[src]).tolist()
            per_source[src].append((distance, dst, rel_vec))

        rows: list[int] = []
        cols: list[int] = []
        edge_distances: list[float] = []
        edge_vectors: list[list[float]] = []
        for src, neighbors in per_source.items():
            neighbors.sort(key=lambda item: item[0])
            for distance, dst, rel_vec in neighbors[: self.max_neighbors]:
                rows.append(src)
                cols.append(dst)
                edge_distances.append(distance)
                edge_vectors.append(rel_vec)

            rows.append(src)
            cols.append(src)
            edge_distances.append(0.0)
            edge_vectors.append([0.0, 0.0, 0.0])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(-1)
        edge_vec = torch.tensor(edge_vectors, dtype=torch.float32)
        return edge_index, edge_attr, edge_vec


class CrystalGraphDataset(Dataset):
    """
    Lazily load processed structure records and cache graph tensors to disk.

    The dataset keeps a compact byte-offset index, which scales well to 10k-100k
    cached structures without loading the entire JSONL file into memory.
    """

    def __init__(
        self,
        processed_jsonl_path: str | Path,
        species_vocab_path: str | Path,
        graph_config: GraphBuildConfig,
    ) -> None:
        self.processed_jsonl_path = Path(processed_jsonl_path)
        self.species_vocab_path = Path(species_vocab_path)
        self.graph_config = graph_config
        self.species_vocab = json.loads(self.species_vocab_path.read_text(encoding="utf-8"))
        cache_dir = graph_config.graph_cache_dir
        if cache_dir is None:
            cutoff_token = str(graph_config.cutoff_angstrom).replace(".", "p")
            cache_dir = self.processed_jsonl_path.parent / f"graphs_cutoff_{cutoff_token}"
        self.graph_cache_dir = Path(cache_dir)
        self.graph_cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.processed_jsonl_path.with_suffix(".index.json")
        self.builder = CrystalGraphBuilder(
            cutoff_angstrom=graph_config.cutoff_angstrom,
            max_neighbors=graph_config.max_neighbors,
        )
        self._index = self._load_or_create_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Data:
        entry = self._index[index]
        graph_path = self.graph_cache_dir / f"{entry['material_id']}.pt"

        # ✅ Load cached graph if exists
        if graph_path.exists() and not self.graph_config.force_rebuild:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            if hasattr(graph, "edge_vec") and hasattr(graph, "cart_coords"):
                return graph

        # ✅ Load raw record
        record = read_jsonl_record_at_offset(self.processed_jsonl_path, entry["offset"])

        # ✅ Build graph safely
        graph = self.builder.build_graph(record, self.species_vocab)

        # 🚨 KEY FIX: skip invalid graphs
        if graph is None:
            # Option 1 (safe fallback): try next sample
            return self.__getitem__((index + 1) % len(self))

        # ✅ Save + return
        torch.save(graph, graph_path)
        return graph

    @property
    def num_species(self) -> int:
        return len(self.species_vocab)

    @property
    def max_atoms(self) -> int:
        return max(int(entry["num_atoms"]) for entry in self._index)

    def precompute_graphs(self, indices: Iterable[int] | None = None, show_progress: bool = True) -> None:
        selected = list(indices) if indices is not None else list(range(len(self)))
        iterator = tqdm(selected, desc="Caching crystal graphs", unit="graph") if show_progress else selected
        for index in iterator:
            _ = self[index]

    def compute_lattice_bounds(
        self,
        indices: Iterable[int] | None = None,
        margin_fraction: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = list(indices) if indices is not None else list(range(len(self)))
        params: list[list[float]] = []
        iterator = tqdm(selected, desc="Computing lattice bounds", unit="graph")
        for index in iterator:
            record = read_jsonl_record_at_offset(self.processed_jsonl_path, self._index[index]["offset"])
            lattice = record["lattice"]
            params.append(
                [
                    float(lattice["a"]),
                    float(lattice["b"]),
                    float(lattice["c"]),
                    float(lattice["alpha"]),
                    float(lattice["beta"]),
                    float(lattice["gamma"]),
                ]
            )

        values = torch.tensor(params, dtype=torch.float32)
        minima = values.min(dim=0).values
        maxima = values.max(dim=0).values
        span = (maxima - minima).clamp_min(1e-3)
        minima = minima - margin_fraction * span
        maxima = maxima + margin_fraction * span
        minima[:3] = minima[:3].clamp_min(0.1)
        minima[3:] = minima[3:].clamp(10.0, 170.0)
        maxima[3:] = maxima[3:].clamp(20.0, 179.5)
        return minima, maxima

    def get_material_ids(self) -> list[str]:
        return [entry["material_id"] for entry in self._index]

    def _load_or_create_index(self) -> list[dict[str, int | str]]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))

        index: list[dict[str, int | str]] = []
        with self.processed_jsonl_path.open("r", encoding="utf-8") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                index.append(
                    {
                        "offset": offset,
                        "material_id": record["material_id"],
                        "num_atoms": int(record["num_atoms"]),
                    }
                )

        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index
