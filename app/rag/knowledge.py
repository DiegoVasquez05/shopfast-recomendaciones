"""Base de conocimiento (FuenteConocimiento del modelo de dominio EP2 §8.4)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KnowledgeDoc:
    id: str
    titulo: str
    intents: list[str]
    contenido: str
    fuente: str
    actualizado: str

    @property
    def text_for_embedding(self) -> str:
        return f"{self.titulo}. {self.contenido}"


def load_kb(path: str | Path) -> list[KnowledgeDoc]:
    return [KnowledgeDoc(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
