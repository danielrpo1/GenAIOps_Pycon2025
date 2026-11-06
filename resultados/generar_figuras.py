"""Genera las figuras del informe a partir de las ejecuciones guardadas en mlflow.db."""
import warnings; warnings.filterwarnings("ignore")
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("sqlite:///mlflow.db")
ex = mlflow.get_experiment_by_name("rag_evaluation")
runs = {r.info.run_name: r.data.metrics for r in mlflow.search_runs([ex.experiment_id], filter_string="attributes.status = 'FINISHED'", output_format="list")}

Q = "juez-qwen3.8-27b"
def c(nombre): return runs[f"groq|asistente_eafit/{nombre}"]["correctness/mean"] * 100
etapas = [
    ("Base: examen original\n(respuestas largas)", c(f"1|k4|c512|{Q}|ds1"), "#9e9e9e"),
    ("Prompt corto (v2)", c(f"2|k4|c512|{Q}|ds1"), "#9e9e9e"),
    ("2 fragmentos", c(f"1|k2|c512|{Q}|ds1"), "#9e9e9e"),
    ("6 fragmentos", c(f"1|k6|c512|{Q}|ds1"), "#9e9e9e"),
    ("Fragmentos de 256", c(f"1|k4|c256|{Q}|ds1"), "#9e9e9e"),
    ("Fragmentos de 1024", c(f"1|k4|c1024|{Q}|ds1"), "#9e9e9e"),
    ("Examen corregido\n(respuestas cortas)", c(f"1|k4|c512|{Q}|ds2"), "#4f83cc"),
    ("+ etiqueta de fuente\n(PRODUCCIÓN)", c(f"1|k4|c512|{Q}|fuente|ds2"), "#2e7d32"),
    ("+ etiqueta, 6 fragmentos", c(f"1|k6|c512|{Q}|fuente|ds2"), "#4f83cc"),
    ("+ etiqueta, prompt corto", c(f"2|k4|c512|{Q}|fuente|ds2"), "#4f83cc"),
]
fig, ax = plt.subplots(figsize=(9, 5.2))
y = range(len(etapas))[::-1]
ax.barh(list(y), [e[1] for e in etapas], color=[e[2] for e in etapas])
ax.set_yticks(list(y)); ax.set_yticklabels([e[0] for e in etapas], fontsize=8)
for yi, e in zip(y, etapas): ax.text(e[1] + 0.8, yi, f"{e[1]:.0f}%", va="center", fontsize=9)
ax.set_xlim(0, 80); ax.set_xlabel("Respuestas correctas según el juez (Qwen3.8-27B)")
ax.set_title("Comparaciones de configuración (24 preguntas; una pregunta = 4 puntos)", fontsize=10)
ax.axvline(45, color="#c62828", ls="--", lw=1); ax.text(45.5, len(etapas) - 0.6, "umbral del CI: 45%", color="#c62828", fontsize=8)
plt.tight_layout(); plt.savefig("resultados/fig_experimentos.png", dpi=150); plt.close()

# Figura 2: por tipo de pregunta (configuración de producción, juez Qwen)
tipos = ["directas\n(9)", "varios documentos\n(6)", "sin respuesta\n(4)", "trampas\n(5)"]
vals = [33, 33, 100, 100]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(tipos, vals, color=["#c62828", "#c62828", "#2e7d32", "#2e7d32"])
for i, v in enumerate(vals): ax.text(i, v + 2, f"{v}%", ha="center", fontsize=9)
ax.set_ylim(0, 115); ax.set_ylabel("Respuestas correctas"); ax.set_title("Configuración de producción, por tipo de pregunta", fontsize=10)
plt.tight_layout(); plt.savefig("resultados/fig_tipos.png", dpi=150); plt.close()
print("figuras generadas")
