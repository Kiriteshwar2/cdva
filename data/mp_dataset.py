from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from mp_api.client import MPRester
from requests import RequestException
from pymatgen.core import Element, Structure

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPDownloadConfig:
    """Configuration for downloading and caching Materials Project structures."""

    api_key: str | None = None
    cache_root: Path = Path("cache/mp")
    dataset_name: str = "mp_structures"
    max_materials: int | None = 10_000
    max_elements: int = 3
    max_atoms: int = 50
    chunk_size: int = 500
    include_gnome: bool = False
    include_deprecated: bool = False
    allow_disordered: bool = False
    timeout_seconds: int = 60
    request_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @property
    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("MP_API_KEY")
        if not key:
            raise ValueError(
                "Materials Project API key not found. Pass `api_key` or set the `MP_API_KEY` environment variable."
            )
        return key

    @property
    def query_signature_payload(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "max_materials": self.max_materials,
            "max_elements": self.max_elements,
            "max_atoms": self.max_atoms,
            "chunk_size": self.chunk_size,
            "include_gnome": self.include_gnome,
            "include_deprecated": self.include_deprecated,
            "allow_disordered": self.allow_disordered,
            "request_retries": self.request_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "fields": ["material_id", "structure"],
        }


@dataclass(frozen=True)
class DatasetArtifacts:
    """Filesystem artifacts produced by the downloader."""

    cache_dir: Path
    metadata_path: Path
    raw_jsonl_path: Path
    processed_jsonl_path: Path
    species_vocab_path: Path


class MaterialsProjectDatasetBuilder:
    """
    Download crystal structures from Materials Project, cache raw payloads, and
    build processed JSONL records for downstream graph construction.
    """

    def __init__(self, config: MPDownloadConfig):
        self.config = config
        query_hash = self._compute_query_hash(config.query_signature_payload)
        self.cache_dir = config.cache_root / f"{config.dataset_name}_{query_hash}"
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
        """
        Build or load a cached dataset.

        If cache files exist and match the current query signature, they are reused.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not force_refresh and self._is_cache_valid():
            LOGGER.info("Using cached dataset at %s", self.cache_dir)
            return self.artifacts

        self._remove_existing_cache_files()

        download_stats = self._download_raw_structures()
        species_vocab = self._build_species_vocab()
        processed_count = self._build_processed_records(species_vocab)

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_signature": self.config.query_signature_payload,
            "download_stats": download_stats,
            "processed_count": processed_count,
            "num_species": len(species_vocab),
            "paths": {
                "raw_jsonl": str(self.raw_jsonl_path),
                "processed_jsonl": str(self.processed_jsonl_path),
                "species_vocab": str(self.species_vocab_path),
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info(
            "Dataset ready. raw=%d processed=%d unique_species=%d",
            download_stats["downloaded"],
            processed_count,
            len(species_vocab),
        )
        return self.artifacts

    def iter_processed_records(self) -> Iterator[dict[str, Any]]:
        """Stream processed records from disk."""
        if not self.processed_jsonl_path.exists():
            raise FileNotFoundError(
                f"Processed cache does not exist at {self.processed_jsonl_path}. Call `build()` first."
            )
        yield from self._iter_jsonl(self.processed_jsonl_path)

    def load_species_vocab(self) -> dict[str, int]:
        """Load element symbol to integer-id mapping."""
        if not self.species_vocab_path.exists():
            raise FileNotFoundError(
                f"Species vocab does not exist at {self.species_vocab_path}. Call `build()` first."
            )
        return json.loads(self.species_vocab_path.read_text(encoding="utf-8"))

    def _download_raw_structures(self) -> dict[str, int]:
        downloaded = 0
        duplicate_ids = 0
        skipped_disordered = 0
        skipped_filter = 0
        seen_ids: set[str] = set()

        # query_kwargs = {
        #     "num_elements": (1, self.config.max_elements),
        #     "num_sites": (1, self.config.max_atoms),
        #     "deprecated": self.config.include_deprecated,
        #     "include_gnome": self.config.include_gnome,
        #     "fields": ["material_id", "structure"],
        #     "all_fields": False,
        #     "chunk_size": self.config.chunk_size,
        # }
        query_kwargs = {
            "fields": ["material_id", "structure", "nsites", "nelements"],
            "chunk_size": self.config.chunk_size,
        }
        if self.config.max_materials is not None:
            query_kwargs["num_chunks"] = max(1, (self.config.max_materials + self.config.chunk_size - 1) // self.config.chunk_size)

        with self.raw_jsonl_path.open("w", encoding="utf-8") as raw_fh:
            docs = self._query_summary_docs(query_kwargs)
            progress = tqdm(
                total=self.config.max_materials,
                desc="Downloading MP structures",
                unit="structure",
            )

            for doc in docs:
                material_id = self._extract_material_id(doc)
                if material_id in seen_ids:
                    duplicate_ids += 1
                    continue

                structure = self._extract_structure(doc)
                if structure is None:
                    skipped_filter += 1
                    continue

                if (not self.config.allow_disordered) and (not structure.is_ordered):
                    skipped_disordered += 1
                    continue

                if len(structure) > self.config.max_atoms or len(structure.composition.elements) > self.config.max_elements:
                    skipped_filter += 1
                    continue

                raw_record = {
                    "material_id": material_id,
                    "nelements": int(len(structure.composition.elements)),
                    "nsites": int(len(structure)),
                    "structure": structure.as_dict(),
                }
                raw_fh.write(json.dumps(raw_record) + "\n")
                seen_ids.add(material_id)
                downloaded += 1
                progress.update(1)

                if self.config.max_materials is not None and downloaded >= self.config.max_materials:
                    break

            progress.close()

        return {
            "downloaded": downloaded,
            "duplicates_skipped": duplicate_ids,
            "disordered_skipped": skipped_disordered,
            "filtered_skipped": skipped_filter,
        }

    def _query_summary_docs(self, query_kwargs: dict[str, Any]) -> list[Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.request_retries + 1):
            try:
                # with MPRester(
                #     self.config.resolved_api_key,
                #     use_document_model=False,
                #     timeout=self.config.timeout_seconds,
                #     mute_progress_bars=True,
                # ) as mpr:
                with MPRester(self.config.resolved_api_key) as mpr:
                    return mpr.materials.summary.search(**query_kwargs)
            except (RequestException, OSError, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.request_retries:
                    break
                sleep_seconds = self.config.retry_backoff_seconds * attempt
                LOGGER.warning(
                    "Materials Project query failed on attempt %d/%d: %s. Retrying in %.1fs.",
                    attempt,
                    self.config.request_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Unexpected Materials Project API failure: {exc}") from exc

        raise RuntimeError(
            "Failed to download Materials Project structures after "
            f"{self.config.request_retries} attempts. Last error: {last_error}"
        ) from last_error

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
            for raw in tqdm(
                self._iter_jsonl(self.raw_jsonl_path),
                desc="Processing structures",
                unit="structure",
            ):
                structure = Structure.from_dict(raw["structure"])
                lattice = structure.lattice

                species_symbols = [site.specie.symbol for site in structure.sites]
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

    def _is_cache_valid(self) -> bool:
        required_files = [
            self.metadata_path,
            self.raw_jsonl_path,
            self.processed_jsonl_path,
            self.species_vocab_path,
        ]
        if not all(path.exists() for path in required_files):
            return False

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        expected_signature = self.config.query_signature_payload
        return metadata.get("query_signature") == expected_signature

    def _remove_existing_cache_files(self) -> None:
        for path in [
            self.metadata_path,
            self.raw_jsonl_path,
            self.processed_jsonl_path,
            self.species_vocab_path,
        ]:
            if path.exists():
                path.unlink()

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def _compute_query_hash(payload: dict[str, Any]) -> str:
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _extract_material_id(doc: Any) -> str:
        if isinstance(doc, dict):
            return str(doc["material_id"])
        return str(getattr(doc, "material_id"))

    @staticmethod
    def _extract_structure(doc: Any) -> Structure | None:
        structure_obj = doc.get("structure") if isinstance(doc, dict) else getattr(doc, "structure", None)
        if structure_obj is None:
            return None
        if isinstance(structure_obj, Structure):
            return structure_obj
        if isinstance(structure_obj, dict):
            return Structure.from_dict(structure_obj)
        return None


def build_dataset(config: MPDownloadConfig, force_refresh: bool = False) -> DatasetArtifacts:
    """
    Convenience function for one-shot dataset creation.
    """
    return MaterialsProjectDatasetBuilder(config).build(force_refresh=force_refresh)


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and cache Materials Project crystal structures for CDVAE training."
    )
    parser.add_argument("--api-key", type=str, default=None, help="Materials Project API key (falls back to MP_API_KEY).")
    parser.add_argument("--cache-root", type=Path, default=Path("cache/mp"), help="Directory where cache artifacts are stored.")
    parser.add_argument("--dataset-name", type=str, default="mp_structures", help="Name prefix for this dataset cache.")
    parser.add_argument("--max-materials", type=int, default=10_000, help="Maximum number of structures to download.")
    parser.add_argument("--max-elements", type=int, default=3, help="Maximum number of unique elements in a crystal.")
    parser.add_argument("--max-atoms", type=int, default=50, help="Maximum number of atoms/sites in a crystal.")
    parser.add_argument("--chunk-size", type=int, default=500, help="API query chunk size.")
    parser.add_argument("--include-gnome", action="store_true", help="Include GNoMe entries in results.")
    parser.add_argument("--include-deprecated", action="store_true", help="Include deprecated entries.")
    parser.add_argument("--allow-disordered", action="store_true", help="Keep disordered structures (default: skip).")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="HTTP timeout for mp-api requests.")
    parser.add_argument("--request-retries", type=int, default=3, help="Number of retry attempts for mp-api calls.")
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds between failed mp-api attempts.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and redownload.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _make_arg_parser().parse_args()

    config = MPDownloadConfig(
        api_key=args.api_key,
        cache_root=args.cache_root,
        dataset_name=args.dataset_name,
        max_materials=args.max_materials,
        max_elements=args.max_elements,
        max_atoms=args.max_atoms,
        chunk_size=args.chunk_size,
        include_gnome=args.include_gnome,
        include_deprecated=args.include_deprecated,
        allow_disordered=args.allow_disordered,
        timeout_seconds=args.timeout_seconds,
        request_retries=args.request_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )

    builder = MaterialsProjectDatasetBuilder(config)
    artifacts = builder.build(force_refresh=args.force_refresh)
    LOGGER.info("Cache directory: %s", artifacts.cache_dir)
    LOGGER.info("Raw records: %s", artifacts.raw_jsonl_path)
    LOGGER.info("Processed records: %s", artifacts.processed_jsonl_path)
    LOGGER.info("Species vocabulary: %s", artifacts.species_vocab_path)


if __name__ == "__main__":
    main()
