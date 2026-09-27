"""Escuta dos niveis: uma pagina local para ouvir e comparar os arms nivel a nivel.

`gefx disent tune --recording minha.wav` sobe um servidor em localhost que
renderiza a gravacao em cada arm, no knob que `calibrate` gravou para cada nivel,
na cadeia do render (loudness -> arm -> loudness). A pagina alterna A/B
entre a referencia e o arm no mesmo ponto da gravacao.

So para ver e ouvir: os niveis vem sempre do pareamento pelo Rnonlin, e a pagina
nao grava nada.

O servidor atende uma requisicao por vez porque os plugins nao sao thread-safe.
"""
from __future__ import annotations

import io
import json
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from gefx.audio import load_audio_file, normalize_loudness
from gefx.disent.arms import LoadedArm, load_levels, load_roster

PAGE = Path(__file__).with_name("tune.html")


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(np.asarray(audio).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Tuner:
    def __init__(self, recording: Path, roster_path: Path, levels_path: Path) -> None:
        self.levels_path = levels_path
        self.roster = load_roster(roster_path, levels_path)
        audio, self.sr = load_audio_file(recording)
        self.dry = normalize_loudness(audio.mean(axis=0, keepdims=True), self.sr)
        self.loaded = {arm.key: LoadedArm(arm, self.sr) for arm in self.roster.arms}
        self.cache: Dict[Tuple[str, float], bytes] = {}

    def state(self) -> Dict[str, object]:
        data = load_levels(self.levels_path)
        arms = [{"key": arm.key, "stratum": arm.stratum, "levels": list(arm.levels),
                 "unmatched": data["arms"][arm.key].get("unmatched") or []}
                for arm in self.roster.arms]
        return {"reference": self.roster.reference, "arms": arms,
                "targets": data.get("targets") or [],
                "seconds": self.dry.shape[1] / self.sr}

    def render(self, key: str, level: int) -> bytes:
        """O arm no knob gravado para o nivel (base 0); so niveis do arquivo."""
        knob = self.loaded[key].arm.levels[level]
        cache_key = (key, round(knob, 4))
        if cache_key not in self.cache:
            wet = normalize_loudness(self.loaded[key].render(self.dry, self.sr, knob), self.sr)
            self.cache[cache_key] = wav_bytes(wet, self.sr)
        return self.cache[cache_key]


def serve(tuner: Tuner, port: int = 8765, open_browser: bool = True) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/state":
                self._send(json.dumps(tuner.state()).encode(), "application/json")
            elif url.path == "/render":
                query = parse_qs(url.query)
                try:
                    body = tuner.render(query["arm"][0], int(query["level"][0]))
                except (KeyError, IndexError, ValueError) as exc:
                    self._send(str(exc).encode(), "text/plain; charset=utf-8", 400)
                    return
                self._send(body, "audio/wav")
            else:
                self._send(b"", "text/plain", 404)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"escuta dos niveis em {url} (Ctrl+C para sair)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
