"""Inspecao de um plugin VST3/AU: quais parametros expoe e como sao mapeados.

Serve a dois propositos na validacao cross-implementation:

1. Descobrir os parametros e seus nomes (`--list`).
2. Recuperar o mapeamento knob normalizado -> unidade fisica, varrendo
   `raw_value` de 0 a 1 e lendo o valor exibido (`--sweep`). E isso que permite
   renderizar um plugin de terceiro em 2.0 Hz de verdade, em vez de assumir que o
   knob 0.5 dele equivale ao 0.5 do Pedalboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _describe(param) -> Dict[str, Any]:
    def safe(getter, default=None):
        try:
            return getter()
        except Exception:
            return default

    valid = safe(lambda: list(param.valid_values) if param.valid_values else None)
    return {
        "min_value": safe(lambda: param.min_value),
        "max_value": safe(lambda: param.max_value),
        "step_size": safe(lambda: param.step_size),
        "approximate_step_size": safe(lambda: param.approximate_step_size),
        "units": safe(lambda: param.units),
        "label": safe(lambda: param.label),
        "type": safe(lambda: getattr(param.type, "__name__", str(param.type))),
        "raw_value": safe(lambda: param.raw_value),
        "n_valid_values": len(valid) if valid else None,
        # Poucos valores distintos => parametro discreto (modo, on/off, tipo de fx).
        "valid_values_head": valid[:8] if valid else None,
        "discrete": bool(valid) and len(valid) <= 32,
    }


def _sweep(param, steps: int) -> List[Dict[str, Any]]:
    original = param.raw_value
    points = []
    try:
        for index in range(steps):
            raw = index / (steps - 1) if steps > 1 else 0.0
            param.raw_value = raw
            points.append({"raw": round(raw, 6), "display": str(param.string_value)})
    finally:
        try:
            param.raw_value = original
        except Exception:
            pass
    return points


def _roundtrip_check(param) -> Optional[Dict[str, Any]]:
    """Testa se `get_raw_value_for()` inverte o mapeamento de forma confiavel.

    Se inverter, da para pedir "2.0 Hz" diretamente em vez de calibrar na mao.
    """
    try:
        lo, hi = param.min_value, param.max_value
        if lo is None or hi is None or not isinstance(lo, (int, float)):
            return None
        target = lo + 0.37 * (hi - lo)
        raw = param.get_raw_value_for(target)
        original = param.raw_value
        param.raw_value = raw
        display = str(param.string_value)
        param.raw_value = original
        return {"asked": target, "raw": raw, "display_at_raw": display}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def inspect_plugin(
    plugin_path: str,
    sweep: Sequence[str] = (),
    sweep_all: bool = False,
    steps: int = 11,
    json_out: Optional[Path] = None,
    plugin_name: Optional[str] = None,
) -> Dict[str, Any]:
    from pedalboard import load_plugin

    kwargs = {"plugin_name": plugin_name} if plugin_name else {}
    try:
        plugin = load_plugin(plugin_path, **kwargs)
    except Exception as exc:
        # Um .vst3 com varios sub-plugins falha e lista os nomes disponiveis.
        raise SystemExit(f"Falha ao carregar {plugin_path}: {type(exc).__name__}: {exc}")

    params = plugin.parameters
    report: Dict[str, Any] = {
        "plugin_path": str(plugin_path),
        "name": getattr(plugin, "name", None),
        "is_effect": getattr(plugin, "is_effect", None),
        "n_parameters": len(params),
        "parameters": {},
    }

    to_sweep = set(sweep)
    print(f"### {report['name']}  ({plugin_path})")
    print(f"{report['n_parameters']} parametros\n")

    header = f"{'parametro':<34} {'faixa':<26} {'un.':<6} {'tipo':<8} n_val"
    print(header)
    print("-" * len(header))

    for name, param in params.items():
        info = _describe(param)
        report["parameters"][name] = info

        lo, hi = info["min_value"], info["max_value"]
        faixa = f"[{lo}, {hi}]" if lo is not None else "-"
        kind = "discreto" if info["discrete"] else "continuo"
        print(
            f"{name:<34} {faixa[:26]:<26} {str(info['units'] or '')[:6]:<6} "
            f"{kind:<8} {info['n_valid_values'] or '-'}"
        )
        if info["discrete"] and info["valid_values_head"]:
            print(f"{'':<34}   valores: {info['valid_values_head']}")

        if sweep_all and not info["discrete"]:
            to_sweep.add(name)

    for name in sorted(to_sweep):
        if name not in params:
            print(f"\n[aviso] parametro '{name}' nao existe neste plugin")
            continue
        points = _sweep(params[name], steps)
        report["parameters"][name]["sweep"] = points
        roundtrip = _roundtrip_check(params[name])
        report["parameters"][name]["roundtrip"] = roundtrip
        print(f"\n--- varredura: {name} (units={params[name].units!r}) ---")
        print("  " + "  ".join(f"{p['raw']:.2f}={p['display']}" for p in points))
        if roundtrip and "error" not in roundtrip:
            print(
                f"  get_raw_value_for({roundtrip['asked']:.4g}) -> raw={roundtrip['raw']:.4g}"
                f" -> exibe {roundtrip['display_at_raw']!r}"
            )
        elif roundtrip:
            print(f"  get_raw_value_for indisponivel: {roundtrip['error']}")

    if json_out:
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nRelatorio salvo em {json_out}")

    return report
