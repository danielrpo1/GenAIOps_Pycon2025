# app/metricas_recuperacion.py
"""
Métrica de recuperación sin LLM: ¿el buscador trajo la página correcta?

Cada pregunta del dataset (tests/eval_dataset.json) anota en `source` el documento y la página
donde está la respuesta. Para cada ejecución de evaluación, este script lee las trazas, mira qué
fragmentos devolvió el retriever y compara con esas páginas. Es gratis (no llama a ningún modelo)
y es determinista, así que sirve para separar "falló la búsqueda" de "falló la redacción".

Registra en cada ejecución de `rag_evaluation`:
  - retrieval_hit_any       fracción de preguntas con al menos una página esperada recuperada
  - retrieval_hit_all_docs  fracción de preguntas donde se recuperó algo de CADA documento esperado
y una tabla `recuperacion_por_pregunta.json` con el detalle.

Uso:
    python -m app.metricas_recuperacion            # procesa las ejecuciones que aún no la tienen
    python -m app.metricas_recuperacion --todas    # recalcula todas
"""

import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import SpanType

EVAL_EXPERIMENT = "rag_evaluation"
DATASET_PATH = "tests/eval_dataset.json"


def paginas_esperadas(source: str) -> dict[str, set[int]]:
    """'Reglamento_Pregrado.pdf, p. 21-22 (Art. 46); Reglamento_Posgrado.pdf, p. 18 y p. 29' → {doc: {páginas}}"""
    esperadas: dict[str, set[int]] = {}
    for segmento in source.split(";"):
        doc = re.search(r"([\w]+\.pdf)", segmento)
        if not doc:
            continue
        paginas: set[int] = set()
        for a, b in re.findall(r"p\.\s*(\d+)(?:-(\d+))?", segmento):
            paginas.update(range(int(a), int(b or a) + 1))
        if paginas:
            esperadas[doc.group(1)] = paginas
    return esperadas


def recuperadas(trace) -> set[tuple[str, int]]:
    """Pares (documento, página 1-based) devueltos por el retriever en una traza."""
    pares = set()
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        for doc in span.outputs or []:
            meta = doc.get("metadata") or {}
            if "source" in meta and isinstance(meta.get("page"), int):
                pares.add((os.path.basename(meta["source"]), meta["page"] + 1))
    return pares


def pregunta_de(trace) -> str | None:
    try:
        return json.loads(trace.data.request)["question"]
    except Exception:
        return None


def evaluar_ejecucion(run_id: str, experiment_id: str, dataset: dict[str, dict]) -> pd.DataFrame:
    filas = []
    for trace in mlflow.search_traces(run_id=run_id, locations=[experiment_id], return_type="list"):
        pregunta = pregunta_de(trace)
        item = dataset.get(pregunta)
        if item is None or not trace.search_spans(span_type=SpanType.RETRIEVER):
            continue  # trazas de jueces u otras
        if item["type"] in ("sin respuesta", "adversaria"):
            continue  # no hay una página que buscar: el comportamiento correcto es abstenerse o rechazar
        esperadas = paginas_esperadas(item["source"])
        if not esperadas:
            continue
        traidas = recuperadas(trace)
        hit_any = any((d, p) in traidas for d, ps in esperadas.items() for p in ps)
        hit_docs = all(any(d == td for td, _ in traidas) for d in esperadas)
        filas.append({
            "tipo": item["type"],
            "pregunta": pregunta,
            "esperado": item["source"],
            "recuperado": "; ".join(f"{d} p.{p}" for d, p in sorted(traidas)),
            "hit_any": hit_any,
            "hit_all_docs": hit_docs,
        })
    return pd.DataFrame(filas)


def main(todas: bool = False):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    experimento = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experimento is None:
        print("No hay evaluaciones todavía.")
        return
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = {x["question"]: x for x in json.load(f)}

    cliente = MlflowClient()
    runs = cliente.search_runs([experimento.experiment_id], filter_string="attributes.status = 'FINISHED'")
    for run in runs:
        if not todas and "retrieval_hit_any" in run.data.metrics:
            continue
        df = evaluar_ejecucion(run.info.run_id, experimento.experiment_id, dataset)
        if df.empty:
            print(f"- {run.info.run_name}: sin trazas con retriever")
            continue
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_metrics({
                "retrieval_hit_any": float(df["hit_any"].mean()),
                "retrieval_hit_all_docs": float(df["hit_all_docs"].mean()),
            })
            mlflow.log_table(df, "recuperacion_por_pregunta.json")
        print(f"- {run.info.run_name}: hit_any={df['hit_any'].mean():.2f} "
              f"hit_all_docs={df['hit_all_docs'].mean():.2f} ({len(df)} preguntas)")


if __name__ == "__main__":
    main(todas="--todas" in sys.argv)
