from __future__ import annotations

import hashlib
import io

from cdml import model_store


def test_download_model_uses_cache_and_verifies_checksum(tmp_path, monkeypatch) -> None:
    payload = b"test checkpoint"
    target = tmp_path / "cache" / "fade_detector.pt"
    monkeypatch.setattr(model_store, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(model_store, "urlopen", lambda _url: io.BytesIO(payload))
    assert model_store.download_model(target) == target
    assert target.read_bytes() == payload
    monkeypatch.setattr(model_store, "urlopen", lambda _url: (_ for _ in ()).throw(AssertionError()))
    assert model_store.download_model(target) == target
