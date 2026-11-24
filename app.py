import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="SVM - Airbnb Valencia", layout="wide")

COLORES = {1: "#d7191c", 0: "#2c7bb6"}      # rojo = Caro, azul = No caro


@st.cache_data
def cargar():
    return pd.read_csv("resultados_svm.csv"), pd.read_csv("metricas_svm.csv")


datos, metricas = cargar()
corridas = [c.removeprefix("pred_") for c in datos.columns if c.startswith("pred_")]


def a_geojson(df_, col_pred, col_conj):
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [x, y]},
              "properties": {"pred": int(p), "real": int(r), "grupo": g}}
             for x, y, p, r, g in zip(df_["longitud"], df_["latitud"],
                                      df_[col_pred], df_["caro"], df_[col_conj])]
    return {"type": "FeatureCollection", "features": feats}


def construir_mapa(seleccion, solo_prueba, paso):
    lat, lon = datos["latitud"], datos["longitud"]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=11,
                   tiles=None, control_scale=True)

    # Mapas base (sin llave de API)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors", name="OpenStreetMap",
        max_zoom=19, control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO", name="CartoDB claro",
        subdomains="abcd", max_zoom=19, show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Satélite (Esri)", max_zoom=19, show=False,
    ).add_to(m)
    folium.TileLayer(tiles=None, name="Sin mapa base", attr=" ", show=False).add_to(m)

    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    
    # Calor por frecuencia
    capa = folium.FeatureGroup(name="Calor: frecuencia", show=True)
    HeatMap(list(zip(lat, lon)), radius=12, blur=15, min_opacity=0.3).add_to(capa)
    capa.add_to(m)

    # Calor por precio
    capa = folium.FeatureGroup(name="Calor: precio", show=False)
    p = datos["precio"]
    peso = (p - p.min()) / (p.max() - p.min())
    HeatMap(list(zip(lat, lon, peso)), radius=12, blur=15, min_opacity=0.3).add_to(capa)
    capa.add_to(m)

    # Grilla
    capa = folium.FeatureGroup(name="Grilla", show=False)
    for y in np.arange(np.floor(lat.min() / paso) * paso, lat.max() + paso, paso):
        folium.PolyLine([[y, lon.min() - paso], [y, lon.max() + paso]],
                        color="gray", weight=0.7, opacity=0.6).add_to(capa)
    for x in np.arange(np.floor(lon.min() / paso) * paso, lon.max() + paso, paso):
        folium.PolyLine([[lat.min() - paso, x], [lat.max() + paso, x]],
                        color="gray", weight=0.7, opacity=0.6).add_to(capa)
    capa.add_to(m)

    # Una capa por corrida seleccionada
    for corrida in seleccion:
        col_pred, col_conj = f"pred_{corrida}", f"conjunto_{corrida}"
        d = datos[datos[col_conj] == "prueba"] if solo_prueba else datos
        capa = folium.FeatureGroup(name=f"SVM: {corrida}", show=False)
        folium.GeoJson(
            a_geojson(d, col_pred, col_conj),
            marker=folium.CircleMarker(radius=3, weight=0, fill=True, fill_opacity=0.8),
            style_function=lambda f: {"fillColor": COLORES[f["properties"]["pred"]]},
            tooltip=folium.GeoJsonTooltip(fields=["pred", "real", "grupo"],
                                          aliases=["Predicho (1=caro)", "Real (1=caro)", "Grupo"]),
        ).add_to(capa)
        capa.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


st.title("Clasificación de anuncios caros con SVM · Airbnb Valencia")
st.caption("Caro = precio sobre la mediana (anuncios hasta 1 000, solo Entire home/apt y Private room).")

tab_mapa, tab_metricas = st.tabs(["🗺️ Mapa", "📊 Métricas"])

# ------------------------- Pestaña 1: mapa -------------------------
with tab_mapa:
    c1, c2, c3 = st.columns([3, 1, 1])
    seleccion = c1.multiselect("Corridas SVM a incluir como capas", corridas, default=corridas[-2:])
    solo_prueba = c2.checkbox("Solo puntos de prueba", value=True)
    paso = c3.select_slider("Paso de la grilla (°)", options=[0.01, 0.02, 0.05], value=0.02)
    st.caption("Rojo = caro, azul = no caro. Usa el control de capas (arriba a la derecha) para prender y apagar.")
    mapa = construir_mapa(seleccion, solo_prueba, paso)
    st_folium(mapa, height=650, use_container_width=True, returned_objects=[])

# ------------------------- Pestaña 2: métricas -------------------------
with tab_metricas:
    cols_num = ["Exactitud", "Precisión", "Recall", "F1", "ROC-AUC"]
    tabla = metricas.set_index("Corrida")
    st.subheader("Métricas en el conjunto de prueba")
    st.dataframe(tabla.style.format({c: "{:.4f}" for c in cols_num})
                      .highlight_max(subset=cols_num, color="#c8e6c9"),
                 use_container_width=True)

    metrica = st.selectbox("Métrica a comparar", cols_num, index=4)
    st.bar_chart(tabla[metrica])

    st.subheader("Matriz de confusión (conjunto de prueba)")
    corrida_cm = st.selectbox("Corrida", corridas, key="cm")
    prueba = datos[datos[f"conjunto_{corrida_cm}"] == "prueba"]
    cm = (pd.crosstab(prueba["caro"], prueba[f"pred_{corrida_cm}"])
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0))
    cm.index = ["Real: No caro", "Real: Caro"]
    cm.columns = ["Pred: No caro", "Pred: Caro"]
    st.dataframe(cm, use_container_width=True)
