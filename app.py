import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Sistema Inteligente Completo", layout="wide")
st.title("Proyecto Integrador: Ciclo de Vida de IA (Etapas 1-4)")
st.write("Dibuja un número. Un modelo generativo limpiará tu trazo y 4 arquitecturas distintas intentarán predecirlo.")

# 1. Cargar los 5 modelos
@st.cache_resource
def cargar_modelos():
    modelo_lr = joblib.load('modelo_baseline_lr.pkl')
    modelo_cnn = load_model('modelo_cnn.h5')
    modelo_rnn = load_model('modelo_rnn.h5')
    modelo_tl = load_model('modelo_tl.h5')
    autoenc = load_model('autoencoder.h5')
    return modelo_lr, modelo_cnn, modelo_rnn, modelo_tl, autoenc

modelo_lr, modelo_cnn, modelo_rnn, modelo_tl, autoencoder = cargar_modelos()

# 2. Lienzo de dibujo
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True
)

if st.button("Procesar y Predecir"):
    if canvas_result.image_data is not None:
        # Preprocesamiento inicial
        imagen = canvas_result.image_data
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGBA2GRAY)
        img_28 = cv2.resize(imagen_gris, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalizar para el Autoencoder
        entrada_ae = img_28.reshape(1, 28, 28, 1) / 255.0
        
        # Etapa 4: Reconstrucción Generativa
        img_limpia_array = autoencoder.predict(entrada_ae, verbose=0)
        img_limpia_28 = img_limpia_array.reshape(28, 28)
        
        # Mostrar comparativa de limpieza
        st.write("### Reconstrucción del Trazo (Autoencoder)")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img_28, caption="Trazo Original (28x28)", width=150)
        with col_img2:
            st.image(img_limpia_28, caption="Trazo Limpio/Reconstruido", width=150)
        
        # Inferencia con la imagen LIMPIA
        entrada_lr = img_limpia_28.reshape(1, -1)
        pred_lr = modelo_lr.predict(entrada_lr)[0]
        
        entrada_cnn = img_limpia_28.reshape(1, 28, 28, 1)
        pred_cnn = np.argmax(modelo_cnn.predict(entrada_cnn, verbose=0))
        
        entrada_rnn = img_limpia_28.reshape(1, 28, 28)
        pred_rnn = np.argmax(modelo_rnn.predict(entrada_rnn, verbose=0))
        
        # Preparar la imagen limpia para Transfer Learning (32x32x3)
        img_limpia_32 = cv2.resize(img_limpia_28, (32, 32), interpolation=cv2.INTER_AREA)
        img_limpia_32_rgb = cv2.cvtColor((img_limpia_32 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        entrada_tl = (img_limpia_32_rgb.reshape(1, 32, 32, 3) / 127.5) - 1.0
        pred_tl = np.argmax(modelo_tl.predict(entrada_tl, verbose=0))
        
        # Mostrar predicciones
        st.write("### Resultados de Clasificación")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Clásico (LR):** {pred_lr}")
        with col2:
            st.success(f"**Red CNN:** {pred_cnn}")
        with col3:
            st.warning(f"**Red RNN:** {pred_rnn}")
        with col4:
            st.error(f"**MobileNetV2:** {pred_tl}")
            
    else:
        st.warning("Por favor, dibuja algo antes de predecir.")

# Etapa 4: Panel de Responsabilidad e Impacto
st.markdown("---")
st.header("Panel de Responsabilidad e Impacto")
st.markdown("""
El desarrollo de arquitecturas generativas, como el **Autoencoder** implementado en este proyecto para la reconstrucción y limpieza de trazos, demuestra el poderoso alcance de la Inteligencia Artificial para alterar y sintetizar información visual. Sin embargo, esta capacidad de "reconstrucción" a partir de ruido es la misma base tecnológica utilizada para la creación de **Deepfakes** y la generación de contenido hiperrealista falso.

**Riesgos Éticos y de Seguridad:**
* **Suplantación de Identidad:** Algoritmos entrenados para reconocer y replicar patrones biométricos (como firmas, rostros o huellas) pueden ser manipulados para evadir sistemas de seguridad, generando falsos positivos que vulneran la privacidad de los usuarios.
* **Desinformación:** La facilidad con la que arquitecturas de Transfer Learning pueden acoplarse a modelos generativos permite la creación masiva de contenido sintético, dificultando la distinción entre trazos/imágenes reales y generadas por computadora.
* **Sesgo del Dataset:** Nuestro modelo fuerza cualquier garabato a convertirse en un dígito estilizado similar a los del dataset MNIST. En aplicaciones críticas (médicas, penales), obligar a un modelo a "reconstruir" información faltante basándose en sus sesgos de entrenamiento puede llevar a invenciones peligrosas (alucinaciones) que no corresponden a la realidad.

**Conclusión:**
El despliegue de sistemas inteligentes debe ir acompañado de mecanismos de trazabilidad y validación. Como ingenieros, es nuestra responsabilidad garantizar que los modelos generativos se utilicen para asistir y corregir errores, manteniendo siempre la transparencia sobre cuándo y cómo los datos originales han sido alterados.
""")