"""Replay cache for remote LLM API decisions.

The cache stores every remote decision as JSONL plus a manifest. A later run
can load the manifest and reuse stored phases without creating an API client.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.training.multi_backend_api_client import APIBackend, TokenUsageLog


SCHEMA_VERSION = "api-replay-cache-v1"


class APIReplayCacheError(Exception):
    """Base error for API replay cache failures."""


class CacheMiss(APIReplayCacheError):
    """Raised when a decision is not present in the cache."""


class CacheMismatch(APIReplayCacheError):
    """Raised when a cached decision exists but hashes do not match."""


class ManifestChecksumError(APIReplayCacheError):
    """Raised when a manifest checksum does not match the JSONL cache."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_text(text: str) -> str:
    """Return a stable SHA256 hash for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash_json(value: Any) -> str:
    """Return a stable SHA256 hash for JSON-compatible data."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_hash_text(encoded)


def _backend_to_str(backend: APIBackend | str) -> str:
    if isinstance(backend, APIBackend):
        return backend.value
    return str(backend)


def _backend_from_str(value: APIBackend | str) -> APIBackend | str:
    if isinstance(value, APIBackend):
        return value
    try:
        return APIBackend(str(value))
    except ValueError:
        return str(value)


@dataclass(frozen=True)
class APIDecisionRecord:
    """One cached phase decision for a timestep/intersection pair."""

    run_fingerprint: str
    dataset: str
    phase: int
    mode: str
    method: str
    backend: APIBackend | str
    model: str
    run_id: int
    seed: int
    timestep: int
    intersection_id: str
    state_hash: str
    prompt_hash: str
    prompt: str
    raw_response: str
    parsed_phase: str
    fallback_used: bool
    error_type: str | None
    input_tokens: int
    output_tokens: int
    request_started_at: str
    response_received_at: str
    latency_ms: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "APIDecisionRecord.schema_version must be "
                f"{SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        for field_name in (
            "run_fingerprint",
            "dataset",
            "mode",
            "method",
            "model",
            "intersection_id",
            "state_hash",
            "prompt_hash",
            "prompt",
            "parsed_phase",
            "request_started_at",
            "response_received_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"APIDecisionRecord.{field_name} must be a non-empty str"
                )
        for field_name in (
            "phase",
            "run_id",
            "seed",
            "timestep",
            "input_tokens",
            "output_tokens",
            "latency_ms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"APIDecisionRecord.{field_name} must be int")
            if field_name != "seed" and value < 0:
                raise ValueError(
                    f"APIDecisionRecord.{field_name} must be >= 0"
                )
        if not isinstance(self.fallback_used, bool):
            raise ValueError("APIDecisionRecord.fallback_used must be bool")
        if self.error_type is not None and not isinstance(self.error_type, str):
            raise ValueError("APIDecisionRecord.error_type must be str or None")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend"] = _backend_to_str(self.backend)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "APIDecisionRecord":
        data = dict(payload)
        data["backend"] = _backend_from_str(data["backend"])
        return cls(**data)


@dataclass(frozen=True)
class APIRunManifest:
    """Metadata needed to replay a cached API run."""

    run_fingerprint: str
    jsonl_path: str
    jsonl_sha256: str
    dataset: str
    phase: int
    mode: str
    method: str
    backend: APIBackend | str
    model: str
    run_id: int
    seed: int
    roadnet_path: str
    roadnet_sha256: str
    flow_path: str
    flow_sha256: str
    simulation_config_sha256: str
    prompt_template_version: str
    code_commit: str | None
    replay_file: str | None
    metrics_file: str
    llm_log_dir: str
    token_usage: TokenUsageLog
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "APIRunManifest.schema_version must be "
                f"{SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        for field_name in (
            "run_fingerprint",
            "jsonl_path",
            "jsonl_sha256",
            "dataset",
            "mode",
            "method",
            "model",
            "roadnet_path",
            "roadnet_sha256",
            "flow_path",
            "flow_sha256",
            "simulation_config_sha256",
            "prompt_template_version",
            "metrics_file",
            "llm_log_dir",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"APIRunManifest.{field_name} must be a non-empty str"
                )
        for field_name in ("phase", "run_id", "seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"APIRunManifest.{field_name} must be int")
            if field_name != "seed" and value < 0:
                raise ValueError(f"APIRunManifest.{field_name} must be >= 0")
        if self.code_commit is not None and not isinstance(
            self.code_commit, str
        ):
            raise ValueError("APIRunManifest.code_commit must be str or None")
        if self.replay_file is not None and not isinstance(
            self.replay_file, str
        ):
            raise ValueError("APIRunManifest.replay_file must be str or None")
        if not isinstance(self.token_usage, TokenUsageLog):
            raise ValueError("APIRunManifest.token_usage must be TokenUsageLog")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend"] = _backend_to_str(self.backend)
        payload["token_usage"]["backend"] = _backend_to_str(
            self.token_usage.backend
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "APIRunManifest":
        data = dict(payload)
        data["backend"] = _backend_from_str(data["backend"])
        usage = dict(data["token_usage"])
        usage["backend"] = _backend_from_str(usage["backend"])
        data["token_usage"] = TokenUsageLog(**usage)
        return cls(**data)


@dataclass
class APIReplayCache:
    """JSONL-backed cache for API decisions."""

    root_dir: Path | str = Path("results/api_cache")
    _records: dict[tuple[int, str], APIDecisionRecord] = field(
        default_factory=dict, init=False, repr=False
    )
    _manifest: APIRunManifest | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)

    def jsonl_path_for(
        self, dataset: str, method: str, phase: int, run_id: int
    ) -> Path:
        return (
            self.root_dir
            / dataset
            / method
            / f"phase{phase}"
            / f"run_{run_id}.jsonl"
        )

    def manifest_path_for(
        self, dataset: str, method: str, phase: int, run_id: int
    ) -> Path:
        return self.jsonl_path_for(dataset, method, phase, run_id).with_suffix(
            ".manifest.json"
        )

    def load_existing(
        self, dataset: str, method: str, phase: int, run_id: int
    ) -> int:
        """Load any existing JSONL cache records into memory.

        Used to resume an interrupted run: previously-cached decisions become
        replayable via ``get_decision`` while new decisions still append to
        the same file. Returns the number of records loaded (0 if the file
        does not exist).
        """
        path = self.jsonl_path_for(dataset, method, phase, run_id)
        if not path.exists():
            return 0
        self._records = self._load_records(path, allow_duplicates=True)
        return len(self._records)

    def aggregate_token_usage(self) -> tuple[int, int, int]:
        """Sum ``input_tokens``, ``output_tokens`` and request count across
        all currently-loaded records. Used by the runner to build an accurate
        manifest when resuming a partially-completed run.
        """
        total_in = sum(r.input_tokens for r in self._records.values())
        total_out = sum(r.output_tokens for r in self._records.values())
        return total_in, total_out, len(self._records)

    def append(self, record: APIDecisionRecord) -> Path:
        path = self.jsonl_path_for(
            record.dataset, record.method, record.phase, record.run_id
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
        self._records[(record.timestep, record.intersection_id)] = record
        return path

    def write_manifest(self, manifest: APIRunManifest) -> Path:
        jsonl_path = Path(manifest.jsonl_path)
        if not jsonl_path.is_absolute() and not jsonl_path.exists():
            candidate = self.root_dir / jsonl_path
            if candidate.exists():
                jsonl_path = candidate
        actual_sha = _sha256_file(jsonl_path)
        if actual_sha != manifest.jsonl_sha256:
            raise ManifestChecksumError(
                "Manifest jsonl_sha256 does not match cache file: "
                f"expected {manifest.jsonl_sha256}, got {actual_sha}"
            )
        path = self.manifest_path_for(
            manifest.dataset, manifest.method, manifest.phase, manifest.run_id
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._manifest = manifest
        return path

    def build_manifest(
        self,
        *,
        dataset: str,
        method: str,
        phase: int,
        mode: str,
        backend: APIBackend | str,
        model: str,
        run_id: int,
        seed: int,
        run_fingerprint: str,
        roadnet_path: str,
        roadnet_sha256: str,
        flow_path: str,
        flow_sha256: str,
        simulation_config_sha256: str,
        prompt_template_version: str,
        code_commit: str | None,
        replay_file: str | None,
        metrics_file: str,
        llm_log_dir: str,
        token_usage: TokenUsageLog,
    ) -> APIRunManifest:
        jsonl_path = self.jsonl_path_for(dataset, method, phase, run_id)
        return APIRunManifest(
            run_fingerprint=run_fingerprint,
            jsonl_path=str(jsonl_path),
            jsonl_sha256=_sha256_file(jsonl_path),
            dataset=dataset,
            phase=phase,
            mode=mode,
            method=method,
            backend=backend,
            model=model,
            run_id=run_id,
            seed=seed,
            roadnet_path=roadnet_path,
            roadnet_sha256=roadnet_sha256,
            flow_path=flow_path,
            flow_sha256=flow_sha256,
            simulation_config_sha256=simulation_config_sha256,
            prompt_template_version=prompt_template_version,
            code_commit=code_commit,
            replay_file=replay_file,
            metrics_file=metrics_file,
            llm_log_dir=llm_log_dir,
            token_usage=token_usage,
        )

    def load_manifest(self, manifest_path: str | Path) -> APIRunManifest:
        path = Path(manifest_path)
        manifest = APIRunManifest.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        jsonl_path = Path(manifest.jsonl_path)
        if not jsonl_path.is_absolute() and not jsonl_path.exists():
            jsonl_path = path.parent / jsonl_path.name.replace(
                ".manifest.json", ".jsonl"
            )
        actual_sha = _sha256_file(jsonl_path)
        if actual_sha != manifest.jsonl_sha256:
            raise ManifestChecksumError(
                "Manifest checksum mismatch for API cache JSONL: "
                f"expected {manifest.jsonl_sha256}, got {actual_sha}"
            )
        self._manifest = manifest
        # allow_duplicates=True: parallel writes from earlier interrupted runs
        # may have written multiple records for the same (timestep, intersection)
        # — keep last-wins so resume + replay are robust.
        self._records = self._load_records(jsonl_path, allow_duplicates=True)
        return manifest

    def get_decision(
        self,
        timestep: int,
        intersection_id: str,
        state_hash: str,
        prompt_hash: str,
    ) -> APIDecisionRecord:
        key = (timestep, intersection_id)
        record = self._records.get(key)
        if record is None:
            raise CacheMiss(
                "No cached API decision for "
                f"timestep={timestep}, intersection_id={intersection_id!r}"
            )
        if record.state_hash != state_hash or record.prompt_hash != prompt_hash:
            raise CacheMismatch(
                "Cached API decision hash mismatch for "
                f"timestep={timestep}, intersection_id={intersection_id!r}: "
                f"state expected {record.state_hash}, got {state_hash}; "
                f"prompt expected {record.prompt_hash}, got {prompt_hash}"
            )
        return record

    def verify_decisions(
        self,
        observed: Iterable[APIDecisionRecord],
        mismatch_log_path: str | Path,
    ) -> bool:
        mismatches: list[str] = []
        for item in observed:
            try:
                cached = self.get_decision(
                    item.timestep,
                    item.intersection_id,
                    item.state_hash,
                    item.prompt_hash,
                )
            except APIReplayCacheError as exc:
                mismatches.append(str(exc))
                continue
            if cached.parsed_phase != item.parsed_phase:
                mismatches.append(
                    "Phase mismatch for "
                    f"t={item.timestep} intersection={item.intersection_id}: "
                    f"expected {cached.parsed_phase}, got {item.parsed_phase}"
                )
        if not mismatches:
            return True
        path = Path(mismatch_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(mismatches) + "\n", encoding="utf-8")
        return False

    @staticmethod
    def _load_records(
        path: Path, *, allow_duplicates: bool = False
    ) -> dict[tuple[int, str], APIDecisionRecord]:
        """Load all records from a JSONL cache file.

        When ``allow_duplicates`` is True, later records overwrite earlier
        ones with the same (timestep, intersection_id) key — needed when
        resuming interrupted runs that may have re-attempted some steps.
        """
        records: dict[tuple[int, str], APIDecisionRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = APIDecisionRecord.from_dict(json.loads(stripped))
                key = (record.timestep, record.intersection_id)
                if not allow_duplicates and key in records:
                    raise ValueError(
                        "Duplicate API cache record at "
                        f"line {line_number}: timestep={record.timestep}, "
                        f"intersection_id={record.intersection_id!r}"
                    )
                records[key] = record
        return records

    def compact(
        self, dataset: str, method: str, phase: int, run_id: int
    ) -> int:
        """Rewrite the JSONL cache file with duplicate keys removed (last wins).

        Returns the number of unique records kept. Used after an interrupted
        run wrote multiple records for the same (timestep, intersection_id).
        """
        path = self.jsonl_path_for(dataset, method, phase, run_id)
        if not path.exists():
            return 0
        records = self._load_records(path, allow_duplicates=True)
        # Sort by (timestep, intersection_id) for stable, readable output.
        ordered = sorted(records.values(), key=lambda r: (r.timestep, r.intersection_id))
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in ordered:
                handle.write(
                    json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        tmp.replace(path)
        self._records = records
        return len(records)


__all__ = [
    "APIDecisionRecord",
    "APIReplayCache",
    "APIReplayCacheError",
    "APIRunManifest",
    "CacheMiss",
    "CacheMismatch",
    "ManifestChecksumError",
    "SCHEMA_VERSION",
    "stable_hash_json",
    "stable_hash_text",
]
