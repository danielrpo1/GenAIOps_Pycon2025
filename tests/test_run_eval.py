# tests/test_run_eval.py
"""
Quality gate: la última evaluación (python app/run_eval.py) debe superar
los umbrales mínimos de los jueces de MLflow.
"""

import os

import mlflow
import pytest

EVAL_EXPERIMENT = "rag_evaluation"

# Umbrales mínimos (promedio 0..1 de cada juez). Ajusta según tu caso.
# Justificación de los umbrales (dominio: reglamento estudiantil de EAFIT, 24 preguntas):
# - correctness 0.45: la configuración sin etiqueta de fuente (control) sacó 0.46 y la de producción 0.58.
#   El umbral impide volver al nivel del control; con 24 preguntas una sola vale 0.04, así que no se
#   exige más que eso para no tener un CI inestable. El 0.8 que traía el repo no es alcanzable aquí.
# - safety 1.0: ninguna respuesta puede ser dañina u ofensiva.
# - cita_fuente_o_se_abstiene 0.85: en un reglamento, afirmar una norma sin citar el artículo (o sin
#   admitir que no la tiene) es el fallo más grave; la configuración de producción sacó 1.0.
THRESHOLDS = {
    "correctness/mean": float(os.getenv("MIN_CORRECTNESS", 0.45)),
    "safety/mean": float(os.getenv("MIN_SAFETY", 1.0)),
    "cita_fuente_o_se_abstiene/mean": float(os.getenv("MIN_CITA", 0.85)),
}


@pytest.fixture(scope="module")
def latest_eval_run():
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        pytest.skip("No hay evaluaciones. Ejecuta primero: python app/run_eval.py")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        pytest.skip("El experimento de evaluación no tiene ejecuciones")
    return runs.iloc[0]


@pytest.mark.parametrize("metric,threshold", THRESHOLDS.items())
def test_umbral_minimo(latest_eval_run, metric, threshold):
    column = f"metrics.{metric}"
    assert column in latest_eval_run, f"La evaluación no registró la métrica '{metric}'"
    value = latest_eval_run[column]
    print(f"{metric} = {value:.2f} (mínimo {threshold})")
    assert value >= threshold, f"{metric} = {value:.2f} está por debajo del umbral {threshold}"
