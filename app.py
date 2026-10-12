import math
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

# ----------------------------
# Núcleo estadístico
# ----------------------------
def poisson_pmf(k, lam):
    if lam < 0:
        return 0.0
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_distribution(lam, max_k=12):
    probs = [poisson_pmf(k, lam) for k in range(max_k + 1)]
    total = sum(probs)
    # Normalizar el pequeño resto que queda fuera del rango mostrado.
    return [p / total for p in probs]

def over_probability(lam, line, max_k=30):
    # P(X > line), válida también para líneas como 2.5 o 9.5
    return sum(poisson_pmf(k, lam) for k in range(max_k + 1) if k > line)

def under_probability(lam, line, max_k=30):
    return sum(poisson_pmf(k, lam) for k in range(max_k + 1) if k < line)

def match_matrix(home_lambda, away_lambda, max_goals=10):
    hp = poisson_distribution(home_lambda, max_goals)
    ap = poisson_distribution(away_lambda, max_goals)
    matrix = [[hp[i] * ap[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    total = sum(sum(row) for row in matrix)
    return [[v / total for v in row] for row in matrix]

def predict_result(home_lambda, away_lambda):
    m = match_matrix(home_lambda, away_lambda)
    home = sum(m[i][j] for i in range(len(m)) for j in range(len(m)) if i > j)
    draw = sum(m[i][j] for i in range(len(m)) for j in range(len(m)) if i == j)
    away = sum(m[i][j] for i in range(len(m)) for j in range(len(m)) if i < j)
    return home, draw, away, m

def pct(x):
    return f"{100*x:.1f}%"

st.title("⚽ Football Predictor")
st.caption("Prototipo gratuito de pronósticos estadísticos. Los resultados dependen de los parámetros introducidos y NO son garantías.")

with st.sidebar:
    st.header("1. Partido")
    home_team = st.text_input("Equipo local", "Equipo Local")
    away_team = st.text_input("Equipo visitante", "Equipo Visitante")
    competition = st.text_input("Competición", "Liga de ejemplo")
    match_date = st.date_input("Fecha del partido", value=datetime.today().date())
    st.divider()
    st.header("2. Parámetros de goles")
    st.caption("Para la primera prueba, deja los valores de ejemplo. Después se sustituyen por estimaciones calculadas con datos históricos.")
    home_goals = st.number_input("Goles esperados local (λ)", min_value=0.05, max_value=6.0, value=1.55, step=0.05)
    away_goals = st.number_input("Goles esperados visitante (λ)", min_value=0.05, max_value=6.0, value=1.10, step=0.05)
    st.subheader("Goles del primer tiempo")
    home_ht = st.number_input("Local HT", min_value=0.01, max_value=4.0, value=0.65, step=0.05)
    away_ht = st.number_input("Visitante HT", min_value=0.01, max_value=4.0, value=0.40, step=0.05)
    st.divider()
    st.header("3. Córners esperados")
    corners_total = st.number_input("Córners totales", min_value=0.1, max_value=25.0, value=9.2, step=0.1)
    corners_ht = st.number_input("Córners primer tiempo", min_value=0.1, max_value=15.0, value=4.1, step=0.1)
    corners_2h = st.number_input("Córners segundo tiempo", min_value=0.1, max_value=15.0, value=5.1, step=0.1)
    st.divider()
    st.header("4. Tarjetas esperadas")
    yellows = st.number_input("Amarillas totales", min_value=0.0, max_value=15.0, value=4.2, step=0.1)
    red_prob = st.slider("Probabilidad de al menos una roja (%)", min_value=0, max_value=80, value=18)
    calculate = st.button("🔍 Calcular pronósticos", type="primary", use_container_width=True)

# Calculate live on every rerun; button is included as a clear user action.
p_home, p_draw, p_away, matrix = predict_result(home_goals, away_goals)
p_1x = p_home + p_draw
p_x2 = p_draw + p_away
p_12 = p_home + p_away

st.subheader(f"{home_team} vs. {away_team}")
st.write(f"**Competición:** {competition} · **Fecha:** {match_date.strftime('%d/%m/%Y')}")
st.info("Demo con parámetros manuales. No está conectada a una API ni ha sido entrenada con estadísticas reales. Los porcentajes ilustran cómo funciona el cálculo.")

c1, c2, c3 = st.columns(3)
c1.metric("Victoria local (1)", pct(p_home))
c2.metric("Empate (X)", pct(p_draw))
c3.metric("Victoria visitante (2)", pct(p_away))

st.markdown("### Los 12 mercados")
result_df = pd.DataFrame([
    ["1. Ganador - local", pct(p_home)],
    ["2. Doble oportunidad 1X", pct(p_1x)],
    ["3. Empate X", pct(p_draw)],
    ["4. Doble oportunidad X2", pct(p_x2)],
    ["5. Goles totales esperados", f"{home_goals + away_goals:.2f}"],
    ["6. Goles esperados HT", f"{home_ht + away_ht:.2f}"],
    ["7. Marcador exacto más probable", ""],
    ["8. Córners totales esperados", f"{corners_total:.1f}"],
    ["9. Córners HT esperados", f"{corners_ht:.1f}"],
    ["10. Córners segundo tiempo", f"{corners_2h:.1f}"],
    ["11. Amarillas esperadas", f"{yellows:.1f}"],
    ["12. Probabilidad de al menos una roja", f"{red_prob}%"],
], columns=["Mercado", "Estimación"])
scores = []
for i in range(len(matrix)):
    for j in range(len(matrix)):
        scores.append((matrix[i][j], f"{i}-{j}"))
scores.sort(reverse=True)
result_df.loc[result_df["Mercado"] == "7. Marcador exacto más probable", "Estimación"] = (
    f"{scores[0][1]} ({pct(scores[0][0])})"
)
st.dataframe(result_df, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("### Marcadores exactos más probables")
    score_rows = [{"Marcador": score, "Probabilidad": pct(prob)} for prob, score in scores[:10]]
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
with right:
    st.markdown("### Líneas de goles")
    goal_lines = []
    total_goals = home_goals + away_goals
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        goal_lines.append({
            "Línea": f"{line:.1f}",
            "Más de": pct(over_probability(total_goals, line)),
            "Menos de": pct(under_probability(total_goals, line))
        })
    st.dataframe(pd.DataFrame(goal_lines), use_container_width=True, hide_index=True)

st.markdown("### Córners: líneas de ejemplo")
corner_rows = []
for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
    corner_rows.append({
        "Línea": f"{line:.1f}",
        "Más de": pct(over_probability(corners_total, line)),
        "Menos de": pct(under_probability(corners_total, line))
    })
st.dataframe(pd.DataFrame(corner_rows), use_container_width=True, hide_index=True)

st.markdown("### Tarjetas amarillas: líneas de ejemplo")
card_rows = []
for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
    card_rows.append({
        "Línea": f"{line:.1f}",
        "Más de": pct(over_probability(yellows, line)),
        "Menos de": pct(under_probability(yellows, line))
    })
st.dataframe(pd.DataFrame(card_rows), use_container_width=True, hide_index=True)

st.markdown("### Primer tiempo: goles esperados")
ht_total = home_ht + away_ht
st.write(f"Total HT estimado: **{ht_total:.2f}** goles.")
ht_rows = []
for line in [0.5, 1.5, 2.5]:
    ht_rows.append({"Línea HT": f"{line:.1f}", "Más de": pct(over_probability(ht_total, line)), "Menos de": pct(under_probability(ht_total, line))})
st.dataframe(pd.DataFrame(ht_rows), use_container_width=True, hide_index=True)

st.markdown("### Descargar los pronósticos")
csv = result_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Descargar resultados en CSV", data=csv,
                   file_name=f"pronostico_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}.csv",
                   mime="text/csv")

st.caption("Importante: este prototipo usa distribuciones de Poisson y parámetros introducidos manualmente. Las estimaciones de córners/tarjetas y rojas son aproximaciones simplificadas. Para uso real se debe calibrar y validar cada mercado con datos históricos, sin fuga de información.")
