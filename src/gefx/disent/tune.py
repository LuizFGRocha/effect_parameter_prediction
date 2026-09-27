"""Pareamento de ouvido: uma pagina local com um slider por arm e nivel.

`gefx disent tune --recording minha.wav` sobe um servidor em localhost que
renderiza a gravacao no arm com o knob do slider, na cadeia do render sem o tone
(loudness -> arm -> loudness). A pagina alterna A/B entre a referencia e o arm no
mesmo ponto da gravacao, e `Salvar` grava `levels` no arquivo de niveis; `auto`
fica como estava. A referencia nao se ajusta: ela e a ancora da unidade.

O servidor atende uma requisicao por vez porque os plugins nao sao thread-safe.
"""
from __future__ import annotations

import io
import json
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from gefx.audio import load_audio_file, normalize_loudness
from gefx.disent.arms import LoadedArm, arm_levels, load_levels, load_roster, write_levels
from gefx.disent.calibrate import sweep_knobs

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


def slider_bounds(values: List[float], sweep: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Cada slider vai do nivel vizinho de baixo ao de cima: resolucao fina e ordem."""
    padded = [sweep[0]] + list(values) + [sweep[1]]
    return [(padded[i], padded[i + 2]) for i in range(len(values))]


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
        arms = []
        for arm in self.roster.arms:
            spec = data["arms"][arm.key]
            knobs = sweep_knobs(self.loaded[arm.key], 2)
            anchor = spec.get("auto") or spec["levels"]
            arms.append({"key": arm.key, "levels": spec["levels"], "auto": spec.get("auto"),
                         "unmatched": spec.get("unmatched") or [],
                         "bounds": slider_bounds(anchor, (float(knobs[0]), float(knobs[-1])))})
        return {"reference": self.roster.reference, "arms": arms,
                "seconds": self.dry.shape[1] / self.sr}

    def render(self, key: str, knob: float) -> bytes:
        cache_key = (key, round(knob, 4))
        if cache_key not in self.cache:
            wet = normalize_loudness(self.loaded[key].render(self.dry, self.sr, knob), self.sr)
            self.cache[cache_key] = wav_bytes(wet, self.sr)
        return self.cache[cache_key]

    def save(self, levels: Dict[str, List[float]]) -> None:
        data = load_levels(self.levels_path)
        for key, values in levels.items():
            if key == self.roster.reference:
                raise ValueError("a referencia nao se ajusta de ouvido")
            arm_levels(key, {key: values})  # crescente, ao menos 2
            if len(values) != len(data["arms"][key]["levels"]):
                raise ValueError(f"{key}: numero de niveis mudou")
            data["arms"][key]["levels"] = [round(float(v), 4) for v in values]
        write_levels(self.levels_path, data)


def serve(tuner: Tuner, port: int = 8765, open_browser: bool = True) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, status: int = 200, extra: Dict[str, str] = {}):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            for name, value in extra.items():
                self.send_header(name, value)
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
                self._send(tuner.render(query["arm"][0], float(query["knob"][0])), "audio/wav")
            else:
                self._send(b"", "text/plain", 404)

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                tuner.save(payload)
                self._send(b"ok", "text/plain")
            except (ValueError, KeyError) as exc:
                self._send(str(exc).encode(), "text/plain; charset=utf-8", 400)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"pareamento em {url} (Ctrl+C para sair)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
