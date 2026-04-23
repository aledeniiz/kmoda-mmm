# K-Moda · Marketing Mix Modeling Dashboard

Dashboard interactivo del modelo MMM de K-Moda — Universidad Alfonso X el Sabio, Caso Práctico de Marketing Analytics con IA.

## Páginas

1. **🏠 Resumen Ejecutivo** — KPIs clave, mix recomendado 12 M€, mROI por canal, tabla histórica 2020-2025.
2. **📈 Resultados del Modelo** — Coeficientes Hill, ecuación maestra, verificación matemática (sec 2.3 y 2.4).
3. **💡 Simulador de Presupuesto** — Ajuste interactivo del presupuesto con proyección de ventas.
4. **📊 Forecast DN vs DS** — Comparativa *Do Nothing* vs *Do Something* con descomposición del uplift.
5. **📅 Comparativa Anual 2020-2025** — Evolución histórica + proyección 2025 validada.

## Stack

- Streamlit, pandas, plotly, scikit-learn, scipy, numpy
- Modelo MMM v4.1 con Ridge + features Hill + caps de industria (Analytic Partners 2024 TQ)

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

Configurada para desplegarse en Streamlit Community Cloud directamente desde este repositorio.
