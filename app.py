import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from skimage.morphology import skeletonize
import base64
from io import BytesIO
from ultralytics import YOLO
import gc

import tensorflow.keras.backend as K
import matplotlib.cm as cm

import hashlib
import psutil, os

# ======== Funciones personalizadas ========
def mostrar_memoria():
    proceso = psutil.Process(os.getpid())
    ram_mb = proceso.memory_info().rss / 1024**2

    st.sidebar.write(f"🧠 RAM usada: {ram_mb:.0f} MB")

    if ram_mb > 800:
        st.sidebar.error("⚠️ Muy cerca del límite")
    elif ram_mb > 600:
        st.sidebar.warning("Cuidado con la memoria")

    return ram_mb
    
# ======== Funciones personalizadas ========
def Weighted_Cross_Entropy(beta):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)
        return tf.reduce_mean(loss)

    return loss

# ======== Cargar modelos ========
@st.cache_resource
def cargar_segmentador():
    return load_model('model_SG.h5', custom_objects={'loss': Weighted_Cross_Entropy(10.0)}, safe_mode=False)

@st.cache_resource
def cargar_clasificador_grieta():
    return load_model('model_CG.h5', safe_mode=False)

@st.cache_resource
def cargar_detector_murosc():
    return YOLO("best.pt")

@st.cache_resource
def cargar_clasificador_ladrillo():
    return load_model('model_CL.h5', safe_mode=False)


# ==================================================
# FUNCION GRAD - CAM
# ==================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

# ==================================================
# FUNCIONES PARA FOTOS TOMADAS A MAYOR DISTANCIA
# ==================================================

def es_formato_valido(w, h, tolerancia=0.01):
    ratio = w / h
    if abs(ratio - 1.0) < tolerancia:
        return True, "1:1"
    if abs(ratio - (3/4)) < tolerancia:
        return True, "3:4"
    if abs(ratio - (4/3)) < tolerancia:
        return True, "4:3"
    return False, None


def redimensionar_imagen_grande(img, lado_objetivo):
    h, w = img.shape[:2]

    valido, formato = es_formato_valido(w, h)

    if not valido:
        return None, None, None

    # Cuadrada
    if formato == "1:1":

        img_resize = cv2.resize(
            img,
            (lado_objetivo, lado_objetivo),
            interpolation=cv2.INTER_AREA
        )

        return img_resize, lado_objetivo, formato

    # 3:4
    elif formato == "3:4":

        nuevo_w = lado_objetivo
        nuevo_h = int(lado_objetivo * 4 / 3)

        img_resize = cv2.resize(
            img,
            (nuevo_w, nuevo_h),
            interpolation=cv2.INTER_AREA
        )

        return img_resize, lado_objetivo, formato

    # 4:3
    elif formato == "4:3":

        nuevo_h = lado_objetivo
        nuevo_w = int(lado_objetivo * 4 / 3)

        img_resize = cv2.resize(
            img,
            (nuevo_w, nuevo_h),
            interpolation=cv2.INTER_AREA
        )

        return img_resize, lado_objetivo, formato

    
def existe_grieta_en_parche(parche_rgb, umbral_clasificador):
    patch = np.expand_dims(parche_rgb, axis=0)
    pred = model_clasificador_grieta.predict(patch, verbose=0)[0][0]
    return pred >= umbral_clasificador

def segmentar_parche(parche_rgb, umbral):
    patch = np.expand_dims(parche_rgb,axis=0)
    pred = model_segmentador.predict(patch,verbose=0)[0]
    if pred.shape[-1] == 1:
        pred = pred[:, :, 0]
    return (pred > umbral).astype(np.uint8)


# ==================================================
# MENÚ PRINCIPAL
# ==================================================

tarea = st.sidebar.selectbox(
    "Seleccione una tarea",
    [
        "Segmentación de Grietas",
        "Detección de Muros Confinados"
    ]
)


if tarea == "Segmentación de Grietas":

    subcampo = st.sidebar.radio(
        "Tipo de fotografía",
        [
            "Fotos tomadas desde cerca",
            "Fotos tomadas a mayor distancia"
        ]
    )
    

elif tarea == "Detección de Muros Confinados":

    subcampo = None

mostrar_memoria()

# ======== Interfaz ========
if tarea == "Segmentación de Grietas":

    # Liberar modelos que NO se usarán
    cargar_detector_murosc.clear()
    cargar_clasificador_ladrillo.clear()
    model_segmentador = cargar_segmentador()
    model_clasificador_grieta = cargar_clasificador_grieta()

    if subcampo == "Fotos tomadas desde cerca":

        st.title("Detección y Segmentación automática de Grietas - Fotos tomadas desde cerca ( d ≤ 0.5m )")

        st.markdown("""
        ### Instrucciones para subir imágenes:
        1. Suba **imágenes cuadradas** (idealmente 1:1).
        2. **No garantiza buenos resultados** en todas las superficies de albañilería confinada.
        3. También se aceptan distancias mayores a 0.5m si la imagen proviene del recorte de una imagen más grande
        4. Para el método 2 (Disparo láser o distancia (m) al muro) usar lente principal 1x (Sin zoom).
        5. Se automatiza la detección, extracción geométrica y predicción de anchos de grieta a nivel milimétrico.
        """)


        # Parámetros
        st.markdown("### Parámetros")
        umbral = st.slider("Umbral de segmentación", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        ancho_mm = st.number_input("Ancho real de la escala cuadrada (mm) - Método 1", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
        usar_escala_verde = st.checkbox("Activar detección automática de escala verde",value=True)
        distancia_muro = st.number_input("Disparo Láser o distancia de captura al muro (m) - Método 2", min_value=0.0, value=0.0, step=0.001, format="%.3f")

        with st.expander("⚙️ Configuración de ecuación mm/pixel "):
            R = st.number_input("R : Resolución (pixeles) de las imágenes ensayadas",value=2048,step=1)
            K1 = st.number_input("K1 : Coeficiente 1",value=0.47102,format="%.6f")
            K2 = st.number_input("K2 : Coeficiente 2",value=-0.013403,format="%.6f")
            st.markdown("**Función obtenida producto de la regresión lineal ( d(m) vs mm/pixel ) :**")
            st.latex(r"\text{Ecuación Base} = \frac{R}{\text{Redimensión}}\,(K_1 d + K_2)")

        R_original = st.number_input("Si la imagen proviene del recorte de una imagen más grande, indicar la resolución de la imagen original (pixeles):",min_value=0,value=0,step=1)

        # Subida de imagen
        uploaded_file = st.file_uploader("Sube una imagen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None: 
            
            # detectar si es nueva imagen
            new_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            if "last_image_hash" not in st.session_state:
                st.session_state.last_image_hash = None
            if st.session_state.last_image_hash != new_hash:
                st.session_state.last_image_hash = new_hash
                # BORRAR TODO LO ANTERIOR
                gc.collect()
                plt.close("all")
                # borrar variables de ejecución anterior
                for k in list(st.session_state.keys()):
                    if k != "last_image_hash":
                        del st.session_state[k]
                # 3. limpiar TensorFlow (CLAVE)
                tf.keras.backend.clear_session()
                # 4. forzar garbage collector
                gc.collect()
        
            # recién aquí procesas la imagen
            image = Image.open(uploaded_file).convert("RGB")
            w_original, h_original = image.size
            resized_image = image.resize((512, 512))
            img_input = np.expand_dims(resized_image, axis=0)

            # Segmentación
            prediction = model_segmentador.predict(img_input)[0]
            if prediction.shape[-1] == 1:
                prediction = prediction[:, :, 0]
            mask = (prediction > umbral).astype(np.uint8)

            # Grad CAM
            img_array = img_input.copy()
            heatmap = make_gradcam_heatmap(img_array, model_clasificador_grieta,last_conv_layer_name="conv_pw_13_relu")
            gradcam_img = overlay_gradcam(np.array(resized_image), heatmap)

            
            # Esqueletización
            skeleton = skeletonize(mask).astype(np.uint8)
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            crack_width_map = dist_transform * skeleton * 2
            mean_width = crack_width_map[crack_width_map > 0].mean() if np.any(crack_width_map > 0) else 0
            max_idx = np.unravel_index(np.argmax(crack_width_map), crack_width_map.shape)
            max_width = crack_width_map[max_idx]

            # Detección de escala

            escala_detectada = False
            mm_per_pixel = None
            img_escala_rgb = None

            if usar_escala_verde:

                cv_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
                img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).copy()
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35,40,40])
                upper_green = np.array([90,255,255])
                mask1 = cv2.inRange(hsv, lower_green, upper_green)
                kernel = np.ones((5,5), np.uint8)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

                contornos, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contornos:
                    contorno_mayor = max(contornos, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contorno_mayor)
                    img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(img_escala_rgb, (x,y), (x+w,y+h), (0,255,0), 5)
                    cv2.putText(img_escala_rgb, "Detected", (x,y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 2)
                    escala_detectada = True
                    mm_per_pixel = ancho_mm / np.mean([w,h])

                if contornos:
                    contorno_mayor = max(contornos, key=cv2.contourArea)
                    epsilon = 0.02 * cv2.arcLength(contorno_mayor, True)
                    approx = cv2.approxPolyDP(contorno_mayor, epsilon, True)

                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                        cv2.rectangle(img_escala_rgb, (x,y), (x+w,y+h), (0,255,0), 5)
                        cv2.putText(img_escala_rgb, "Detected", (x,y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 2)
                        escala_detectada = True
                        mm_per_pixel = ancho_mm / np.mean([w,h])

            # ==================================================
            # SECCIÓN 1
            # ==================================================

            # Mostrar resultados: Imagen Original  - Gradcam
            col1, col2 = st.columns(2)

            with col1:
                st.image(resized_image, caption="Imagen original", use_container_width=True)
            with col2:
                st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)

            # Mostrar Mascara
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.image(mask * 255, caption=f"Máscara predicha (umbral: {umbral})", use_container_width=True)

            # Mapa de ancho de grietas
            fig_width, ax_width = plt.subplots(figsize=(5, 4))
            im = ax_width.imshow(crack_width_map, cmap='jet')
            ax_width.scatter(max_idx[1], max_idx[0], color='white', s=80, edgecolors='black', label='Ancho máximo')
            ax_width.set_title("Mapa de ancho de grietas")
            ax_width.axis('off')
            plt.colorbar(im, ax=ax_width, fraction=0.046, pad=0.04, label='Ancho (píxeles)')
            ax_width.legend()

            buf_width = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf_width, format="png")
            plt.close(fig_width)

            # Preparar imagen de escala
            caption_escala = "Escala detectada" if escala_detectada else "Escala no detectada"

            # ==================================================
            # SECCIÓN 2
            # ==================================================
            st.markdown("---")
            st.markdown("### Visualización de resultados")

            col_mapa, col_escala = st.columns(2)

            with col_mapa:
                st.image(buf_width.getvalue(),caption="Mapa de ancho de grietas",use_container_width=True)

            if usar_escala_verde:
                with col_escala:
                    st.image(img_escala_rgb, caption=caption_escala, use_container_width=True)

            st.markdown(f"**Formato:** 1 : 1")
            st.markdown(f"**Resolución procesada:** 512 x 512 pixeles")
            st.markdown(f"**Ancho promedio en pixeles:** {mean_width:.2f}")
            st.markdown(f"**Ancho máximo en pixeles:** {max_width:.2f}")

            # ==================================================
            # SECCIÓN 3
            # ==================================================
            st.markdown("---")
            st.markdown("### Estimación automática de ancho de grieta")

            metodo1, metodo2 = st.columns(2)

            # Método plantilla verde

            with metodo1:
                st.markdown("#### Método 1: Mediante plantilla verde en el muro")

                if not usar_escala_verde:
                    st.info("Detección de escala verde desactivada")

                elif escala_detectada and mm_per_pixel:
                    mean_mm = mean_width * mm_per_pixel
                    max_mm = max_width * mm_per_pixel
                    st.markdown(f"**Escala:** {mm_per_pixel:.2f} mm/pixel")
                    st.markdown(f"**Ancho promedio:** {mean_mm:.2f} mm")
                    st.markdown(f"**Ancho máximo:** {max_mm:.2f} mm")

                else:

                    st.markdown("*No se pudo calcular porque no se detectó la plantilla verde*")

            # Método distancia al muro

            with metodo2:
                st.markdown("#### Método 2: Mediante disparo láser o distancia(m) al muro")

                if distancia_muro == 0:
                    st.markdown("*No se cuenta con distancia de captura al muro*")

                elif distancia_muro < 0.10 or distancia_muro > 1.40 :
                    st.error("Distancias fuera de rangos (0.10 m - 1.40 m)")

                else:

                    escala_mm_pixel = (1 / 512) * ( R * (K1 * distancia_muro + K2) )

                    if R_original == 0:
                        escala_mm_pixel = (1 / 512) * ( R * (K1 * distancia_muro + K2) )

                    else:

                        lado_recorte = max(w_original, h_original)

                        escala_mm_pixel =  (lado_recorte / R_original) * (1 / 512) * ( R * (K1 * distancia_muro + K2) )

                    mean_mm_dist = mean_width * escala_mm_pixel
                    max_mm_dist = max_width * escala_mm_pixel

                    st.markdown(f"**Escala:** {escala_mm_pixel:.2f} mm/pixel")
                    st.markdown(f"**Ancho promedio:** {mean_mm_dist:.2f} mm")
                    st.markdown(f"**Ancho máximo:** {max_mm_dist:.2f} mm")

            # ==================================================
            # SECCION 4: TABLAS DE REFERENCIA
            # ==================================================

            st.markdown("---")

            with st.expander("📚 Clasificación de daños según grietas por movimientos sísmicos y movimiento progresivo de cimientos"):

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        "tabla1.png",
                        caption="Escala de daño sísmico en albañilería Confinada.",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        "tabla2.png",
                        caption="BRE Digest 251 - Evaluación de daños en edificios de poca altura debido al movimiento progresivo de los cimientos.",
                        use_container_width=True
                    )

            st.markdown("---")
            st.markdown("#### Referencias usadas para las tablas de clasificación de daños :")
            st.markdown("""
            1. BRE Digest 251. Driscoll, R. (1995). *Assessment of Damage in Low-Rise Buildings, with Particular Reference to Progressive Foundation Movement*, United Kingdom.
            2. Astroza, M. y Figueroa, S. (2000). *Escalas para calificar los daños sísmicos en los muros de edificios de albañilería*. XXIX Jornadas Sudamericanas de Ingeniería Estructural, Montevideo, Uruguay.
            """)
            
            image.close()
            del image
            del resized_image
            del img_input
            del prediction
            del heatmap
            del gradcam_img
            del skeleton
            del dist_transform
            del crack_width_map
            del mask
            if img_escala_rgb is not None:
                del img_escala_rgb
            gc.collect()

    elif subcampo == "Fotos tomadas a mayor distancia":

        st.title("Detección y Segmentación Automática de Grietas - Fotos tomadas a mayores distancias del muro ( d > 0.5 m )")

        st.markdown("""
        ### Instrucciones
        1. **No garantiza buenos resultados** en todas las superficies de albañilería confinada.
        2. Para el método 2 (Disparo láser o distancia (m) al muro) usar lente principal 1x (Sin zoom).
        3. Se automatiza la detección, extracción geométrica y predicción de anchos de grieta a nivel milimétrico.

        - Formato permitido:
            - Cuadrado (1:1)
            - Horizontal (4:3)
            - Vertical (3:4)
        - La imagen será dividida en parches de 512x512.
        - Solo los parches con grietas serán segmentados.

        """)

        st.markdown("""
        ### Rango de captura (Cámara <-> Muro):

        - **0.5 m < Distancia de Captura ≤ 1.4 m (Opcion A)**
          - Requiere imágenes con una resolución mínima de **1536 px** en el lado menor.
          - También requiere que la imagen sea de **ALTA CALIDAD**.

        - **1.4 m < Distancia de Captura ≤ 4.0 m (Opcion B)**
          - Requiere imágenes con una resolución mínima de **4608 px** en el lado menor.
          - También requiere que la imagen sea de **MUY ALTA CALIDAD**.
        """)

        col1, col2 = st.columns(2)

        with col1:
            resolucion_objetivo = st.radio(
                "Rango de distancia : Cámara <-> Muro",
                [
                    "OpcionA",
                    "OpcionB"
                ],
                index=0,
                horizontal=True
            )

        st.markdown("### Parámetros")
        umbral_clasificador = st.slider("Umbral de clasificación",min_value=0.0,max_value=1.0,value=0.50, step=0.01)
        umbral = st.slider("Umbral de segmentación", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="distancia_umbral")

        ancho_mm = st.number_input("Ancho real de la escala cuadrada (mm) - Método 1", min_value=1.0, max_value=1000.0, value=20.0, step=1.0, key="distancia_plantilla")
        usar_escala_verde = st.checkbox("Activar detección automática de escala verde",value=True)
        distancia_muro = st.number_input("Disparo Láser o distancia de captura al muro (m) - Método 2", min_value=0.0, value=0.0, step=0.001, format="%.3f", key="distancia_muro_lejos")

        with st.expander("⚙️ Configuración de ecuación mm/pixel "):

            R = st.number_input("R : Resolución (pixeles) de las imágenes ensayadas",value=2048,step=1)
            K1 = st.number_input("K1 : Coeficiente 1",value=0.47102,format="%.6f")
            K2 = st.number_input("K2 : Coeficiente 2",value=-0.013403,format="%.6f")
            st.markdown("**Función obtenida producto de la regresión lineal ( d(m) vs mm/pixel ) :**")
            st.latex(r"\text{Ecuación Base} = \frac{R}{\text{Redimensión}}\,(K_1 d + K_2)")

        uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="upload_distancia")

        if uploaded_file is not None:
            
            # detectar si es nueva imagen
            new_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            if "last_image_hash" not in st.session_state:
                st.session_state.last_image_hash = None
            if st.session_state.last_image_hash != new_hash:
                st.session_state.last_image_hash = new_hash
                # BORRAR TODO LO ANTERIOR
                gc.collect()
                plt.close("all")
                # borrar variables de ejecución anterior
                for k in list(st.session_state.keys()):
                    if k != "last_image_hash":
                        del st.session_state[k]
                # 3. limpiar TensorFlow (CLAVE)
                tf.keras.backend.clear_session()
                # 4. forzar garbage collector
                gc.collect()

            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            img_original = np.array(image)
            h0, w0 = img_original.shape[:2]
            valido, formato = es_formato_valido(w0, h0)

            if not valido:
                st.error("Dimensiones de foto no aceptadas")
                st.stop()

            if resolucion_objetivo == "OpcionA":
                lado_objetivo = 1536
            else:
                lado_objetivo = 4608

            img_resize, lado_menor, formato = redimensionar_imagen_grande(img_original,lado_objetivo)

            H, W = img_resize.shape[:2]
            mascara_global = np.zeros((H, W), dtype=np.uint8)
            parches_con_grieta = 0
            parches_sin_grieta = 0

            # Imagen para visualización superpuesta
            img_overlay = img_resize.copy()

            for y in range(0, H, 512):
                for x in range(0, W, 512):
                    parche = img_resize[y:y+512, x:x+512]
                    hay_grieta = existe_grieta_en_parche(parche, umbral_clasificador)
                    if hay_grieta:
                        parches_con_grieta += 1
                        mask_patch = segmentar_parche(parche, umbral)
                        mascara_global[y:y+512, x:x+512] = mask_patch
                        
                        # ==============================
                        # Visualización del parche segmentado
                        # ==============================
                        # Crear parche negro
                        parche_visual = np.zeros((512, 512, 3), dtype=np.uint8)
                        # Dibujar la grieta en blanco
                        parche_visual[mask_patch == 1] = (255, 255, 255)
                        # Colocar el parche sobre la imagen reconstruida
                        img_overlay[y:y+512, x:x+512] = parche_visual

                    else:
                        parches_sin_grieta += 1

            mask = mascara_global
            total_parches = parches_con_grieta + parches_sin_grieta

            # Esqueletización
            skeleton = skeletonize(mask).astype(np.uint8)
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            crack_width_map = dist_transform * skeleton * 2
            mean_width = crack_width_map[crack_width_map > 0].mean() if np.any(crack_width_map > 0) else 0
            max_idx = np.unravel_index(np.argmax(crack_width_map), crack_width_map.shape)
            max_width = crack_width_map[max_idx]

            # ==================================================
            # DETECCIÓN DE ESCALA VERDE
            # ==================================================

            escala_detectada = False
            mm_per_pixel = None
            img_escala_rgb = None

            if usar_escala_verde:

                cv_image = cv2.cvtColor(np.array(img_resize), cv2.COLOR_RGB2BGR)
                img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).copy()
                hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35,40,40])
                upper_green = np.array([90,255,255])
                mask1 = cv2.inRange(hsv, lower_green, upper_green)
                kernel = np.ones((5,5), np.uint8)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
                mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

                contornos, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contornos:
                    contorno_mayor = max(contornos, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contorno_mayor)
                    img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(img_escala_rgb, (x,y), (x+w,y+h), (0,255,0), 5)
                    cv2.putText(img_escala_rgb, "Detected", (x,y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 2)
                    escala_detectada = True
                    mm_per_pixel = ancho_mm / np.mean([w,h])

                if contornos:
                    contorno_mayor = max(contornos, key=cv2.contourArea)
                    epsilon = 0.02 * cv2.arcLength(contorno_mayor, True)
                    approx = cv2.approxPolyDP(contorno_mayor, epsilon, True)

                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        img_escala_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                        cv2.rectangle(img_escala_rgb, (x,y), (x+w,y+h), (0,255,0), 5)
                        cv2.putText(img_escala_rgb, "Detected", (x,y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 2)
                        escala_detectada = True
                        mm_per_pixel = ancho_mm / np.mean([w,h])

            # ==================================================
            # SECCION 1 : IMAGEN + SEGMENTACIÓN
            # ==================================================

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.image(img_resize,
                        caption="Imagen original",
                        use_container_width=True)

            with col2:
                st.image(img_overlay,
                        caption="Segmentación superpuesta",
                        use_container_width=True)

            # Máscara binaria debajo
            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                st.image(mask*255,
                        caption="Máscara reconstruida",
                        use_container_width=True)


            # ==================================================
            # SECCION 2 : MAPA DE ANCHOS Y PLANTILLA DETECTADA
            # ==================================================

            # Mapa de ancho de grietas
            fig_width, ax_width = plt.subplots(figsize=(5, 4))
            im = ax_width.imshow(crack_width_map, cmap='jet')
            ax_width.scatter(max_idx[1], max_idx[0], color='white', s=80, edgecolors='black', label='Ancho máximo')
            ax_width.set_title("Mapa de ancho de grietas")
            ax_width.axis('off')
            plt.colorbar(im, ax=ax_width, fraction=0.046, pad=0.04, label='Ancho (píxeles)')
            ax_width.legend()

            buf_width = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf_width, format="png")
            plt.close(fig_width)

            # Escala verde
            caption_escala = "Escala detectada" if escala_detectada else "Escala no detectada"

            # Mostrar en seccion
            st.markdown("---")
            st.markdown("### Visualización de resultados")

            col_mapa, col_escala = st.columns(2)

            with col_mapa:
                st.image(buf_width.getvalue(), caption="Mapa de ancho de grietas", use_container_width=True)

            if usar_escala_verde:
                with col_escala:
                    st.image(img_escala_rgb, caption=caption_escala, use_container_width=True)

            st.markdown(f"**Formato detectado:** {formato}")
            st.markdown(f"**Resolución procesada:** {W} x {H}")
            st.markdown(f"**Lado menor utilizado:** {lado_menor}")

            st.markdown(f"**Ancho promedio en pixeles:** {mean_width:.2f}")
            st.markdown(f"**Ancho máximo en pixeles:** {max_width:.2f}")

            st.markdown("---")
            st.markdown("#### Clasificación de parches")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total de parches", total_parches)
            with col_b:
                st.metric("Parches con grieta", parches_con_grieta)
            with col_c:
                st.metric("Parches sin grieta", parches_sin_grieta)

            # ==================================================
            # SECCION 3 : RESULTADOS
            # ==================================================

            st.markdown("---")
            st.markdown("### Estimación automática de ancho de grieta")
            metodo1, metodo2 = st.columns(2)

            with metodo1:
                st.markdown("#### Método 1: Mediante plantilla verde en el muro")
                if not usar_escala_verde:
                    st.info("Detección de escala verde desactivada")
                elif escala_detectada and mm_per_pixel:
                    mean_mm = mean_width * mm_per_pixel
                    max_mm = max_width * mm_per_pixel
                    st.markdown(f"**Escala:** {mm_per_pixel:.2f} mm/pixel")
                    st.markdown(f"**Ancho promedio:** {mean_mm:.2f} mm")
                    st.markdown(f"**Ancho máximo:** {max_mm:.2f} mm")
                else:
                    st.markdown("*No se pudo calcular porque no se detectó la plantilla verde*")


            with metodo2:
                st.markdown("##### Método 2: Mediante disparo láser o distancia (m) al muro ")
                if distancia_muro == 0:
                    st.markdown("*No se cuenta con distancia de captura al muro*")
                elif distancia_muro < 0.4 or distancia_muro > 4:
                    st.error("Distancias fuera de rangos (0.4 m - 4 m)")
                else:
                    escala_mm_pixel = (1 / lado_menor) * ( R * (K1 * distancia_muro + K2) )
                    mean_mm_dist = mean_width * escala_mm_pixel
                    max_mm_dist = max_width * escala_mm_pixel
                    st.markdown(f"**Escala:** {escala_mm_pixel:.2f} mm/pixel")
                    st.markdown(f"**Ancho promedio:** {mean_mm_dist:.2f} mm")
                    st.markdown(f"**Ancho máximo:** {max_mm_dist:.2f} mm")

            # ==================================================
            # TABLAS DE REFERENCIA
            # ==================================================

            st.markdown("---")

            with st.expander("📚 Clasificación de daños según grietas por movimientos sísmicos y movimiento progresivo de cimientos"):

                col1, col2 = st.columns(2)
                with col1:
                    st.image("tabla1.png", caption="Escala de daño sísmico en albañilería Confinada.",use_container_width=True)
                with col2:
                    st.image("tabla2.png", caption="BRE Digest 251 - Evaluación de daños en edificios de poca altura debido al movimiento progresivo de los cimientos.",use_container_width=True)

            st.markdown("---")
            st.markdown("#### Referencias usadas para las tablas de diagnóstico")
            st.markdown("""
            1. BRE Digest 251. Driscoll, R. (1995). *Assessment of Damage in Low-Rise Buildings, with Particular Reference to Progressive Foundation Movement*, United Kingdom.
            2. Astroza, M. y Figueroa, S. (2000). *Escalas para calificar los daños sísmicos en los muros de edificios de albañilería*. XXIX Jornadas Sudamericanas de Ingeniería Estructural, Montevideo, Uruguay.
            """)
            
            image.close()
            del image
            del img_original
            del img_resize
            del img_overlay
            del mascara_global
            del mask
            del skeleton
            del dist_transform
            del crack_width_map
            if img_escala_rgb is not None:
                del img_escala_rgb
            gc.collect()

elif tarea == "Detección de Muros Confinados":

    # Liberar modelos que NO se usarán
    cargar_segmentador.clear()
    cargar_clasificador_grieta.clear()
    
    model_detector_murosc = cargar_detector_murosc()
    model_clasificador_ladrillo = cargar_clasificador_ladrillo()

    st.title("Detección, Conteo y Clasificación Automática de Muros Confinados con Unidades Tubulares")

    st.markdown("""
    ### Instrucciones
    1. En edificaciones grandes se recomienda el empleo de vehículos aéreos no tripulados (UAVs), para una captura  centrada y de buena resolución.
    2. Se recomienda capturar únicamente la edificación de interés, minimizando la presencia de fondo de otras edificaciones en la imagen.
        - Se rellena con píxeles negros hacia ARRIBA o DERECHA hasta formar imagen cuadrada LxL.
        - Se aceptan imágenes en cualquier formato W:H.
        - Se redimensiona (1024x1024) y se procesa con YOLO 11 para detección de muros confinados en sus cuatro lados.
        - Las coordenadas de las cajas delimitadoras se reescalan a la imagen LxL para la extraccion de muros.
        - Se extraen los muros manteniendo la resolucion, se vuelven cuadradas y se redimensionan a 512x512 para su clasificación mediante un modelo CNN.
    - Se identifican y se hace un conteo automático de los muros confinados con unidades tubulares (Norma E070).
    """)

    st.markdown("### Parámetros")
    conf_yolo = st.slider("Umbral de confianza del detector YOLO11l",min_value=0.0,max_value=1.0, value=0.80,step=0.01)
    umbral_clasificador = st.slider("Umbral del clasificador",min_value=0.0,max_value=1.0,value=0.50,step=0.01)
    st.markdown("### Filtro de Muros Redundantes % (Opcional)")
    porcentaje_minimo = st.slider("Eliminar detecciones con área menor al (%) del muro más grande",min_value=0,max_value=100,value=25,step=1)

    
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","jpeg","png","JPG"])

    if uploaded_file is not None:
        
        # detectar si es nueva imagen
        new_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if "last_image_hash" not in st.session_state:
            st.session_state.last_image_hash = None
        if st.session_state.last_image_hash != new_hash:
            st.session_state.last_image_hash = new_hash
            # BORRAR TODO LO ANTERIOR
            gc.collect()
            plt.close("all")
            # borrar variables de ejecución anterior
            for k in list(st.session_state.keys()):
                if k != "last_image_hash":
                    del st.session_state[k]
            # 3. limpiar TensorFlow (CLAVE)
            tf.keras.backend.clear_session()
            # 4. forzar garbage collector
            gc.collect()
            
        # =========================
        # 1. CARGA IMAGEN
        # =========================
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        img = np.array(image)
        h, w = img.shape[:2]

        # =========================
        # 2. PAD A CUADRADO LxL (ARRIBA o DERECHA)
        # =========================
        L = max(h, w)

        padded = np.zeros((L, L, 3), dtype=np.uint8)

        padded[L-h : L, 0 : w] = img

        st.image(padded, caption="Imagen cuadrada LxL", use_container_width=True)

        # =========================
        # 3. YOLO 1024x1024
        # =========================
        img_1024 = cv2.resize(padded, (1024, 1024), interpolation=cv2.INTER_AREA)

        # seguridad anti artefactos
        img_1024 = np.ascontiguousarray(img_1024)

        # =========================
        # 4. DETECCIÓN YOLO
        # =========================

        results = model_detector_murosc(img_1024, conf=conf_yolo)[0]

        if len(results.boxes) == 0:
            st.warning("No se detectaron muros confinados en la imagen.")
            st.image(img_1024, caption="Imagen procesada (sin detecciones)", use_container_width=True)
            st.stop()


        boxes_1024 = []
        conf_yolo_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes_1024.append([x1, y1, x2, y2])
            conf_yolo_boxes.append(float(box.conf[0]))
        
        # =========================
        # 5. REESCALAR A LxL
        # =========================

        scale = L / 1024
        boxes_L = []
        for (x1, y1, x2, y2) in boxes_1024:
            boxes_L.append([
                x1 * scale,
                y1 * scale,
                x2 * scale,
                y2 * scale
            ])

        # =========================
        # FILTRAR POR ÁREA
        # =========================

        if len(boxes_L) > 0:

            areas = []

            for x1, y1, x2, y2 in boxes_L:
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)

            area_max = max(areas)
            umbral_area = area_max * (porcentaje_minimo / 100)

            boxes_L_filtradas = []
            boxes_1024_filtradas = []
            conf_yolo_filtradas = []

            for box_L, box_1024, conf, area in zip(
                    boxes_L, boxes_1024, conf_yolo_boxes, areas):

                if area >= umbral_area:

                    boxes_L_filtradas.append(box_L)
                    boxes_1024_filtradas.append(box_1024)
                    conf_yolo_filtradas.append(conf)

            boxes_L = boxes_L_filtradas
            boxes_1024 = boxes_1024_filtradas
            conf_yolo_boxes = conf_yolo_filtradas

        st.info(
            f"Muros detectados : {len(boxes_L)}"
        )

        # =========================
        # 6. EXTRAER PARCHES + CLASIFICAR
        # =========================
        labels = []

        for (x1, y1, x2, y2) in boxes_L:

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            crop = padded[y1:y2, x1:x2]

            if crop.size == 0:
                labels.append(0)
                continue

            # hacer cuadrado rellenando hacia ARRIBA o DERECHA
            h2, w2 = crop.shape[:2]
            Lc = max(h2, w2)

            square = np.zeros((Lc, Lc, 3), dtype=np.uint8)

            # ✔ anclado abajo-izquierda dentro del parche
            square[Lc-h2:Lc, 0:w2] = crop

            patch_512 = cv2.resize(square, (512, 512))

            pred = model_clasificador_ladrillo.predict(
                np.expand_dims(patch_512.astype(np.float32), axis=0),
                verbose=0
            )[0][0]

            labels.append(pred)

        # =========================
        # 7. DIBUJAR RESULTADO
        # =========================

        output = img_1024.copy()

        for i, (box, score, conf_det) in enumerate(zip(boxes_1024, labels, conf_yolo_boxes), start=1):
            
            x1, y1, x2, y2 = map(int, box)

            is_pandereta = score >= umbral_clasificador
            color = (150, 0, 0) if is_pandereta else (0, 150, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            # TEXTO MULTILÍNEA
            clase = "TubularBrick" if is_pandereta else "NoTubularBrick"

            lines = [
                f"Muro {i}",
                "Muro Confinado",
                f"Conf: {conf_det:.2f}",
                clase,
                f"Pred: {score:.2f}"
            ]

            cx = x1 + 10
            cy = y1 + 20

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            for j, line in enumerate(lines):

                (text_w, text_h), baseline = cv2.getTextSize(
                    line, font, font_scale, thickness
                )

                y_text = cy + j * (text_h + 6)

                # color del fondo = mismo color de la caja
                box_color = color

                # coordenadas del rectángulo de fondo
                cv2.rectangle(
                    output,
                    (cx - 2, y_text - text_h - 2),
                    (cx + text_w + 2, y_text + baseline + 2),
                    box_color,
                    -1
                )

                # texto  encima
                cv2.putText(
                    output,
                    line,
                    (cx, y_text),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
                
        st.image(output, caption="Resultado YOLO + Clasificación", use_container_width=True)

        st.markdown(f"**Resolución recibida:** {w} x {h}")
        st.markdown(f"**Resolución procesada:** {L} x {L}")

        # =========================
        # 8. LEYENDA
        # =========================

        total = len(labels)
        pandereta = sum([1 for s in labels if s >= umbral_clasificador])
        no_pandereta = total - pandereta

        relaciones_LH = []

        for (x1, y1, x2, y2) in boxes_L:

            ancho = x2 - x1
            alto = y2 - y1

            if alto > 0:
                relaciones_LH.append(ancho / alto)
            else:
                relaciones_LH.append(0)

        st.markdown("---")
        st.markdown("#### Resumen de Muros Confinados Detectados")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total de muros confinados", total)
        with col_b:
            st.metric("Muros con unidades Tubulares", pandereta)
        with col_c:
            st.metric("Muros sin unidades Tubulares", no_pandereta)

        st.markdown("---")

        # =========================
        # GRÁFICO CIRCULAR (CENTRADO)
        # =========================
        st.markdown("#### Diagrama Circular : Muros Detectados (%)")
        
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            fig_pie, ax_pie = plt.subplots(figsize=(7,5))

            ax_pie.pie(
                [pandereta, no_pandereta],
                labels=["Con U. Tubular", "Sin U. Tubular"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#FFA500", "#1f77b4"]
            )

            ax_pie.set_title("Diagrama Circular - Muros Detectados (%)")
            ax_pie.legend(labels=["Tubular", "Sin Tubular"], loc="upper right", bbox_to_anchor=(1.25, 1))
            st.pyplot(fig_pie)

        st.markdown("---")

        # =========================
        # GRÁFICO BARRAS (CENTRADO)
        # =========================
        st.markdown("#### Gráfico de barras : Relación Longitud - Altura (L/A)")
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            fig_bar, ax_bar = plt.subplots(figsize=(8,5))

            numeros_muro = np.arange(1, len(relaciones_LH)+1)

            ax_bar.bar(
                numeros_muro,
                relaciones_LH,
                color="#2ecc71"
            )

            ax_bar.set_xlabel("Número de Muro")
            ax_bar.set_ylabel("L/A")
            ax_bar.set_title("Relación Longitud/Altura por paño de muro")

            ax_bar.set_xticks(numeros_muro)

            for x, y in zip(numeros_muro, relaciones_LH):
                ax_bar.text(
                    x,
                    y,
                    f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            
            st.pyplot(fig_bar)

        # =========================
        # 9. GUIAS
        # =========================

        st.markdown("---")

        with st.expander("📚 Restricciones del uso de Unidades de Albañileria en Muros Confinados según Zona Sísmica."):
          st.image("tabla3.png", caption="Restricciones del uso de Unidades de Albañileria en Muros Confinados según Zona Sísmica",use_container_width=True)

        # =========================
        # 9. VISUALIZACIÓN DESLIZABLE DE MUROS
        # =========================
        st.markdown("## Detalle de Muros Confinados Detectados")

        with st.container():
            for i, (box, score, conf) in enumerate(zip(boxes_L, labels, conf_yolo_boxes), start=1):
            
                x1, y1, x2, y2 = map(int, box)
                crop = padded[y1:y2, x1:x2]
            
                if crop.size == 0:
                    continue
            
                # Relación L/A
                ancho = x2 - x1
                alto = y2 - y1
                relacion = (ancho / alto) if alto > 0 else 0
            
                # Clasificación
                es_tubular = score >= umbral_clasificador
            
                clase_texto = (
                    "Muro confinado con unidades tubulares (Pandereta)"
                    if es_tubular
                    else "Muro confinado sin unidades tubulares"
                )
            
                st.markdown(
                    f"""
                    ### Muro {i}
            
                    **Tipo:** {clase_texto}  
                    **Score clasificación:** `{score:.3f}`  
                    **Confianza YOLO:** `{conf:.3f}`  
                    **Relación L/A:** `{relacion:.2f}`
                    """
                )
            
                st.image(crop, width=800)


        st.markdown("---")
        st.markdown("#### Referencias usadas para la tabla y YOLO11l :")
        st.markdown("""
        1. C. y S. Ministerio de Vivienda, “Norma Técnica E.070 Albañilería,” Lima, Perú, 2020. 
        2. Ultralytics YOLO11 | Ultralytics Docs,” Ultralytics Docs. 
        """)

        image.close()   
        del image
        del img
        del padded
        del img_1024
        del output    
        del boxes_1024
        del boxes_L
        del conf_yolo_boxes 
        del labels
        del relaciones_LH
        gc.collect()
        
        

