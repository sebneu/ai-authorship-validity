"""Detector interface and registry.

Adapters register themselves here so the driver can enumerate what is available on the
current machine without importing GPU dependencies it cannot satisfy. Importing an
adapter that needs torch on a machine without torch raises at registration time and is
reported as unavailable rather than crashing the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    """Text in, score out. Higher means more likely machine-generated."""

    name: str
    version: str

    def score(self, texts: list[str]) -> list[float]:
        ...


@dataclass
class Registration:
    name: str
    kind: str  # "zero-shot", "heuristic", "supervised", "llm"
    needs_gpu: bool
    factory: Callable[[], Detector]
    note: str = ""
    unavailable: str | None = field(default=None)


_REGISTRY: dict[str, Registration] = {}


def register(
    name: str, kind: str, needs_gpu: bool = False, note: str = ""
) -> Callable[[Callable[[], Detector]], Callable[[], Detector]]:
    def wrap(factory: Callable[[], Detector]) -> Callable[[], Detector]:
        _REGISTRY[name] = Registration(name, kind, needs_gpu, factory, note)
        return factory

    return wrap


def available() -> dict[str, Registration]:
    """Registrations whose dependencies import cleanly on this machine."""
    out = {}
    for name, reg in _REGISTRY.items():
        try:
            reg.factory  # noqa: B018 - presence check only; construction is deferred
            out[name] = reg
        except Exception as exc:  # noqa: BLE001
            reg.unavailable = str(exc)[:100]
    return out


def get(name: str) -> Registration:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def load_adapters() -> None:
    """Import adapter modules, tolerating ones whose dependencies are missing.

    The CPU machine has no torch and the GPU host has everything; both should be able
    to run the driver and see a sensible list rather than a traceback.
    """
    import importlib

    for module in ("heuristics", "selfadmission", "fingerprint", "fast_detect_gpt",
                   "binoculars", "detect_code_gpt", "llm_judge"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            _REGISTRY.setdefault(
                module,
                Registration(
                    module,
                    kind="unknown",
                    needs_gpu=True,
                    factory=lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
                    note="not importable here",
                    unavailable=str(exc)[:100],
                ),
            )
