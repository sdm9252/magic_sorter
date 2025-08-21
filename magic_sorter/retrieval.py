"""Image retrieval utilities combining pHash and CLIP+FAISS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import faiss  # type: ignore
    import imagehash
    import open_clip
except Exception as exc:  # pragma: no cover - used for environments without deps
    faiss = None  # type: ignore[assignment]
    imagehash = None  # type: ignore[assignment]
    open_clip = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None


@dataclass
class GalleryItem:
    path: Path
    phash: int
    embedding: np.ndarray


class CardGallery:
    """In-memory gallery of known cards."""

    def __init__(self, items: List[GalleryItem], index) -> None:
        self.items = items
        self.index = index

    @classmethod
    def from_directory(cls, directory: str, model_name: str = "ViT-B-16") -> "CardGallery":
        if faiss is None or imagehash is None or open_clip is None:
            raise RuntimeError(
                "faiss, imagehash and open_clip are required"  # pragma: no cover
            ) from _import_error
        model, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        model.eval()
        items: List[GalleryItem] = []
        for path in sorted(Path(directory).glob("*.jpg")):
            img = Image.open(path).convert("RGB")
            phash = int(str(imagehash.phash(img)), 16)
            tensor = preprocess(img).unsqueeze(0)
            with open_clip.torch.no_grad():
                emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            items.append(GalleryItem(path, phash, emb.squeeze(0).cpu().numpy()))
        if not items:
            raise ValueError(f"no images found in gallery: {directory}")
        dim = items[0].embedding.shape[0]
        index = faiss.IndexFlatIP(dim)
        vecs = np.stack([it.embedding for it in items]).astype("float32")
        index.add(vecs)
        return cls(items, index)

    def phash_shortlist(self, image: np.ndarray, max_hamming: int = 16) -> List[int]:
        if imagehash is None:
            raise RuntimeError("imagehash required") from _import_error
        img = Image.fromarray(image).convert("RGB")
        qhash = int(str(imagehash.phash(img)), 16)
        candidate_idxs: List[int] = []
        for idx, item in enumerate(self.items):
            if bin(item.phash ^ qhash).count("1") <= max_hamming:
                candidate_idxs.append(idx)
        return candidate_idxs

    def search(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[Path, float]]:
        if open_clip is None:
            raise RuntimeError("open_clip required") from _import_error
        img = Image.fromarray(image).convert("RGB")
        model, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        model.eval()
        tensor = preprocess(img).unsqueeze(0)
        with open_clip.torch.no_grad():
            emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.cpu().numpy().astype("float32")
        _, idxs = self.index.search(vec, top_k)
        results = []
        for idx in idxs[0]:
            item = self.items[idx]
            score = float(np.dot(vec, item.embedding))
            results.append((item.path, score))
        return results
