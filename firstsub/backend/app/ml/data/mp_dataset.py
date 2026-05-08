from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pymatgen.core import Element, Structure


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPDownloadConfig:
    """
    Offline dataset config.

    This keeps the old class name for compatibility while explicitly disabling any
    network/API usage and sourcing data only from local mp_20 CSV files.
    """

    cache_root: Path = Path("mp_20")
    dataset_name: str = "mp_20"
    max_materials: int | None = None
    max_elements: int = 20
    max_atoms: int = 20
    source_csvs: tuple[str, ...] = ("train.csv", "val.csv", "test.csv")


@dataclass(frozen=True)
class DatasetArtifacts:
    cache_dir: Path
    metadata_path: Path
    raw_jsonl_path: Path
    processed_jsonl_path: Path
    species_vocab_path: Path


class MaterialsProjectDatasetBuilder:
    """Build processed artifacts from local mp_20 CSV files only."""

    def __init__(self, config: MPDownloadConfig):
        self.config = config
        self.cache_dir = config.cache_root
        self.metadata_path = self.cache_dir / "metadata.json"
        self.raw_jsonl_path = self.cache_dir / "raw_structures.jsonl"
        self.processed_jsonl_path = self.cache_dir / "processed_structures.jsonl"
        self.species_vocab_path = self.cache_dir / "species_vocab.json"

    @property
    def artifacts(self) -> DatasetArtifacts:
        return DatasetArtifacts(
            cache_dir=self.cache_dir,
            metadata_path=self.metadata_path,
            raw_jsonl_path=self.raw_jsonl_path,
            processed_jsonl_path=self.processed_jsonl_path,
            species_vocab_path=self.species_vocab_path,
        )

    def build(self, force_refresh: bool = False) -> DatasetArtifacts:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not force_refresh and self.processed_jsonl_path.exists() and self.species_vocab_path.exists():
            LOGGER.info("Using existing offline mp_20 artifacts in %s", self.cache_dir)
            return self.artifacts

        for path in [self.raw_jsonl_path, self.processed_jsonl_path, self.species_vocab_path, self.metadata_path]:
            if path.exists():
                path.unlink()

        stats = self._build_raw_records()
        species_vocab = self._build_species_vocab()
        processed_count = self._build_processed_records(species_vocab)

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": self.config.dataset_name,
            "source": "offline_local_mp_20",
            "stats": stats,
            "processed_count": processed_count,
            "num_species": len(species_vocab),
            "paths": {
                "raw_jsonl": str(self.raw_jsonl_path),
                "processed_jsonl": str(self.processed_jsonl_path),
                "species_vocab": str(self.species_vocab_path),
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return self.artifacts

    def _build_raw_records(self) -> dict[str, int]:
        converted = 0
        failed_conversions = 0
        skipped_validation = 0
        seen_ids: set[str] = set()
        duplicates = 0

        with self.raw_jsonl_path.open("w", encoding="utf-8") as raw_fh:
            for csv_name in self.config.source_csvs:
                csv_path = self.cache_dir / csv_name
                if not csv_path.exists():
                    continue
                frame = pd.read_csv(csv_path)
                cif_col = self._find_cif_column(frame)
                id_col = "material_id" if "material_id" in frame.columns else None
                for row_idx, row in frame.iterrows():
                    material_id = str(row[id_col]) if id_col else f"{csv_name}:{row_idx}"
                    if material_id in seen_ids:
                        duplicates += 1
                        continue
                    try:
                        structure = Structure.from_str(str(row[cif_col]), fmt="cif")
                    except Exception as exc:
                        failed_conversions += 1
                        LOGGER.warning("Failed CIF conversion for %s: %s", material_id, exc)
                        continue
                    if not self._is_valid_structure(structure):
                        skipped_validation += 1
                        continue
                    raw_record = {
                        "material_id": material_id,
                        "structure": structure.as_dict(),
                        "nsites": int(len(structure)),
                        "nelements": int(len(structure.composition.elements)),
                    }
                    raw_fh.write(json.dumps(raw_record) + "\n")
                    seen_ids.add(material_id)
                    converted += 1
                    if self.config.max_materials is not None and converted >= self.config.max_materials:
                        break

        return {
            "downloaded": converted,
            "failed_conversions": failed_conversions,
            "skipped_validation": skipped_validation,
            "duplicates": duplicates,
        }

    def _build_species_vocab(self) -> dict[str, int]:
        species_set: set[str] = set()
        for raw in self._iter_jsonl(self.raw_jsonl_path):
            structure = Structure.from_dict(raw["structure"])
            for site in structure.sites:
                species_set.add(site.specie.symbol)
        sorted_species = sorted(species_set, key=lambda symbol: Element(symbol).Z)
        vocab = {symbol: idx for idx, symbol in enumerate(sorted_species)}
        self.species_vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        return vocab

    def _build_processed_records(self, species_vocab: dict[str, int]) -> int:
        count = 0
        with self.processed_jsonl_path.open("w", encoding="utf-8") as out_fh:
            for raw in self._iter_jsonl(self.raw_jsonl_path):
                try:
                    structure = Structure.from_dict(raw["structure"])
                except Exception as exc:
                    LOGGER.warning("Failed Structure.from_dict for %s: %s", raw.get("material_id"), exc)
                    continue
                if not self._is_valid_structure(structure):
                    continue
                lattice = structure.lattice
                species_symbols = [site.specie.symbol for site in structure.sites]
                if not species_symbols:
                    continue
                species_ids = [species_vocab[symbol] for symbol in species_symbols]
                atomic_numbers = [int(Element(symbol).Z) for symbol in species_symbols]
                processed = {
                    "material_id": raw["material_id"],
                    "num_atoms": int(len(structure)),
                    "num_elements": int(len(set(species_symbols))),
                    "lattice": {
                        "a": float(lattice.a),
                        "b": float(lattice.b),
                        "c": float(lattice.c),
                        "alpha": float(lattice.alpha),
                        "beta": float(lattice.beta),
                        "gamma": float(lattice.gamma),
                    },
                    "species": species_symbols,
                    "species_ids": species_ids,
                    "atomic_numbers": atomic_numbers,
                    "frac_coords": structure.frac_coords.tolist(),
                }
                out_fh.write(json.dumps(processed) + "\n")
                count += 1
        return count

    def iter_processed_records(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_jsonl(self.processed_jsonl_path)

    def load_species_vocab(self) -> dict[str, int]:
        return json.loads(self.species_vocab_path.read_text(encoding="utf-8"))

    @staticmethod
    def _find_cif_column(frame: pd.DataFrame) -> str:
        for col in frame.columns:
            if "cif" in col.lower():
                return col
        raise ValueError("No CIF column found in mp_20 CSV.")

    def _is_valid_structure(self, structure: Structure) -> bool:
        if len(structure) == 0 or len(structure) > self.config.max_atoms:
            return False
        if not structure.species:
            return False
        lattice = structure.lattice
        lattice_values = [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma]
        if any(not float(v) == float(v) for v in lattice_values):
            return False
        if lattice.a <= 0 or lattice.b <= 0 or lattice.c <= 0:
            return False
        if lattice.alpha <= 0 or lattice.beta <= 0 or lattice.gamma <= 0:
            return False
        if lattice.alpha >= 180 or lattice.beta >= 180 or lattice.gamma >= 180:
            return False
        frac = structure.frac_coords
        if frac.size == 0:
            return False
        if not ((frac == frac).all()):
            return False
        return True

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def build_dataset(config: MPDownloadConfig, force_refresh: bool = False) -> DatasetArtifacts:
    return MaterialsProjectDatasetBuilder(config).build(force_refresh=force_refresh)


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline mp_20 artifacts for CDVAE.")
    parser.add_argument("--cache-root", type=Path, default=Path("mp_20"), help="Local mp_20 directory.")
    parser.add_argument("--dataset-name", type=str, default="mp_20", help="Dataset identifier.")
    parser.add_argument("--max-materials", type=int, default=None, help="Optional cap while testing.")
    parser.add_argument("--max-elements", type=int, default=20, help="Max unique elements.")
    parser.add_argument("--max-atoms", type=int, default=20, help="Max atoms per structure.")
    parser.add_argument("--force-refresh", action="store_true", help="Rebuild artifacts.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _make_arg_parser().parse_args()
    config = MPDownloadConfig(
        cache_root=args.cache_root,
        dataset_name=args.dataset_name,
        max_materials=args.max_materials,
        max_elements=args.max_elements,
        max_atoms=args.max_atoms,
    )
    artifacts = MaterialsProjectDatasetBuilder(config).build(force_refresh=args.force_refresh)
    LOGGER.info("Processed records: %s", artifacts.processed_jsonl_path)
    LOGGER.info("Species vocabulary: %s", artifacts.species_vocab_path)


if __name__ == "__main__":
    main()
