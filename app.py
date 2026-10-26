import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# 1. Configuración de página
st.set_page_config(page_title="Proyecto Integrador IA", layout="wide")

# 2. Inyección de CSS para la estética Sage Green y tipografías
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@300;400;700&family=Marcellus&display=swap');
    
    .stApp {
        background-color: #cbd3c5;
    }
    
    .custom-title {
        font-family: 'Marcellus', serif;
        font-size: 3rem;
        color: #4a443b;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 20px;
    }
    
    .custom-subtitle {
        font-family: 'Josefin Sans', sans-serif;
        font-size: 1rem;
        color: #4a443b;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 40px;
    }
    
    .polaroid-box {
        background-color: #f4efea;
        padding: 15px;
        border: 1px solid #4a443b;
        text-align: center;
        font-family: 'Josefin Sans', sans-serif;
        color: #4a443b;
        margin-bottom: 15px;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.05);
    }
    
    /* Estilo del botón Procesar centrado y en negritas */
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 15px;
    }
    div.stButton > button:first-child {
        background-color: transparent !important;
        color: #4a443b !important;
        border: 1px solid #4a443b !important;
        border-radius: 0px !important;
        font-family: 'Josefin Sans', sans-serif !important;
        font-weight: bold !important;
        letter-spacing: 2px !important;
        padding: 10px 30px !important;
        width: 80% !important;
        transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #4a443b !important;
        color: #f4efea !important;
    }
    
    /* Estilos para el panel de impacto */
    .impact-panel {
        background-color: #f4efea;
        padding: 35px;
        border: 1px solid #4a443b;
        font-family: 'Josefin Sans', sans-serif;
        color: #4a443b;
        margin-top: 50px;
        margin-bottom: 30px;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.05);
        line-height: 1.6;
    }
    .impact-panel h2 {
        font-family: 'Marcellus', serif;
        color: #4a443b;
        margin-top: 0;
        text-align: center;
        margin-bottom: 25px;
    }
    .impact-panel strong {
        color: #2c2923;
    }
    .impact-panel ul {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Carga de Modelos
@st.cache_resource
def cargar_modelos():
    modelo_lr = joblib.load('modelo_baseline_lr.pkl')
    modelo_cnn = load_model('modelo_cnn.h5')
    modelo_rnn = load_model('modelo_rnn.h5')
    modelo_tl = load_model('modelo_tl.h5')
    autoenc = load_model('autoencoder.h5')
    return modelo_lr, modelo_cnn, modelo_rnn, modelo_tl, autoenc

modelo_lr, modelo_cnn, modelo_rnn, modelo_tl, autoencoder = cargar_modelos()

# 4. Estructura de Columnas (Izquierda 55%, Derecha 45%)
col_izq, col_espacio, col_der = st.columns([5.5, 0.5, 4.0])

# Variables para guardar resultados
pred_lr, pred_cnn, pred_rnn, pred_tl = "-", "-", "-", "-"
img_28 = None
img_limpia_28 = None
procesado = False

with col_der:
    # Lienzo de dibujo
    canvas_result = st_canvas(
        fill_color="#f4efea",
        stroke_width=25,
        stroke_color="#4a443b",
        background_color="#f4efea",
        width=350,
        height=350,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True
    )
    
    if st.button("PROCESAR Y PREDECIR"):
        if canvas_result.image_data is not None:
            imagen = canvas_result.image_data
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGBA2GRAY)
            imagen_invertida = cv2.bitwise_not(imagen_gris)
            img_28 = cv2.resize(imagen_invertida, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Reconstrucción (Autoencoder)
            entrada_ae = img_28.reshape(1, 28, 28, 1) / 255.0
            img_limpia_array = autoencoder.predict(entrada_ae, verbose=0)
            img_limpia_28 = img_limpia_array.reshape(28, 28)
            
            # Predicciones
            entrada_lr = img_limpia_28.reshape(1, -1)
            pred_lr = modelo_lr.predict(entrada_lr)[0]
            
            entrada_cnn = img_limpia_28.reshape(1, 28, 28, 1)
            pred_cnn = np.argmax(modelo_cnn.predict(entrada_cnn, verbose=0))
            
            entrada_rnn = img_limpia_28.reshape(1, 28, 28)
            pred_rnn = np.argmax(modelo_rnn.predict(entrada_rnn, verbose=0))
            
            img_limpia_32 = cv2.resize(img_limpia_28, (32, 32), interpolation=cv2.INTER_AREA)
            img_limpia_32_rgb = cv2.cvtColor((img_limpia_32 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            entrada_tl = (img_limpia_32_rgb.reshape(1, 32, 32, 3) / 127.5) - 1.0
            pred_tl = np.argmax(modelo_tl.predict(entrada_tl, verbose=0))
            
            procesado = True
        else:
            st.warning("Dibuja un número primero.")

with col_izq:
    st.markdown('<div class="custom-title">PROYECTO INTEGRADOR:<br>CICLO DE VIDA DE IA<br>(ETAPAS 1-4)</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-subtitle">DIBUJA UN NÚMERO. UN MODELO GENERATIVO LIMPIARÁ TU TRAZO Y 4 ARQUITECTURAS DISTINTAS INTENTARÁN PREDECIRLO.</div>', unsafe_allow_html=True)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        if procesado and img_28 is not None:
            st.image(cv2.bitwise_not(img_28), width=150)
        else:
            st.image(np.ones((150, 150, 3)), width=150)
        st.markdown('<div style="width: 150px; text-align: center; font-family: \'Josefin Sans\', sans-serif; color: #4a443b; margin-top: 10px; font-weight: bold; letter-spacing: 1px;">TRAZO ORIGINAL</div>', unsafe_allow_html=True)
        
    with col_img2:
        if procesado and img_limpia_28 is not None:
            st.image(cv2.bitwise_not((img_limpia_28 * 255).astype(np.uint8)), width=150)
        else:
            st.image(np.ones((150, 150, 3)), width=150)
        st.markdown('<div style="width: 150px; text-align: center; font-family: \'Josefin Sans\', sans-serif; color: #4a443b; margin-top: 10px; font-weight: bold; letter-spacing: 1px;">TRAZO LIMPIO</div>', unsafe_allow_html=True)
        
    st.write("")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.markdown(f'<div class="polaroid-box">CLÁSICO (LR):<br><b>{pred_lr}</b></div>', unsafe_allow_html=True)
    with col_res2:
        st.markdown(f'<div class="polaroid-box">RED CNN:<br><b>{pred_cnn}</b></div>', unsafe_allow_html=True)
    with col_res3:
        st.markdown(f'<div class="polaroid-box">RED RNN:<br><b>{pred_rnn}</b></div>', unsafe_allow_html=True)
        
    col_res_empty, col_res4, col_res_empty2 = st.columns([1, 1, 1])
    with col_res4:
        st.markdown(f'<div class="polaroid-box">MOBILENETV2:<br><b>{pred_tl}</b></div>', unsafe_allow_html=True)

    # 5. Panel de Impacto
    st.markdown("""
    <div class="impact-panel">
        <h2>Panel de Responsabilidad e Impacto</h2>
        <p>El desarrollo de arquitecturas generativas, como el <strong>Autoencoder</strong> implementado en este proyecto para la reconstrucción y limpieza de trazos, demuestra el poderoso alcance de la Inteligencia Artificial para alterar y sintetizar información visual. Sin embargo, esta capacidad de "reconstrucción" a partir de ruido es la misma base tecnológica utilizada para la creación de <strong>Deepfakes</strong> y la generación de contenido hiperrealista falso.</p>
        <p><strong>Riesgos Éticos y de Seguridad:</strong></p>
        <ul>
         <li><strong>Suplantación de Identidad:</strong> Algoritmos entrenados para reconocer y replicar patrones biométricos (como firmas, rostros o huellas) pueden ser manipulados para evadir sistemas de seguridad, generando falsos positivos que vulneran la privacidad de los usuarios.</li>
         <li><strong>Desinformación:</strong> La facilidad con la que arquitecturas de Transfer Learning pueden acoplarse a modelos generativos permite la creación masiva de contenido sintético, dificultando la distinción entre trazos/imágenes reales y generadas por computadora.</li>
         <li><strong>Sesgo del Dataset:</strong> Nuestro modelo fuerza cualquier garabato a convertirse en un dígito estilizado similar a los del dataset MNIST. En aplicaciones críticas (médicas, penales), obligar a un modelo a "reconstruir" información faltante basándose en sus sesgos de entrenamiento puede llevar a invenciones peligrosas (alucinaciones) que no corresponden a la realidad.</li>
     </ul>
     <p><strong>Conclusión:</strong></p>
        <p>El despliegue de sistemas inteligentes debe ir acompañado de mecanismos de trazabilidad y validación. Como ingenieros, es nuestra responsabilidad garantizar que los modelos generativos se utilicen para asistir y corregir errores, manteniendo siempre la transparencia sobre cuándo y cómo los datos originales han sido alterados.</p>
        </div>
            """, unsafe_allow_html=True)