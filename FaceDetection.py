import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Titre de l'application
st.title("🔍 Face Detection with Viola-Jones Algorithm")

# ✅ Instructions
st.markdown("""
### ℹ️ Instructions:
1. Téléchargez une image avec des visages.
2. Choisissez la couleur des rectangles de détection.
3. Ajustez les paramètres `scaleFactor` et `minNeighbors`.
4. Cliquez sur "Détecter les visages".
5. Cliquez sur "📥 Enregistrer l'image" pour sauvegarder le résultat sur votre appareil.
""")

# ✅ Chargement du modèle pré-entraîné de détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Chargement d'une image
uploaded_file = st.file_uploader("📤 Téléchargez une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="🖼️ Image originale", use_column_width=True)

    # ✅ Choix de la couleur du rectangle
    rectangle_color = st.color_picker("🎨 Choisissez la couleur du rectangle", "#00FF00")
    bgr_color = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # convert to BGR

    # ✅ Ajustement des paramètres
    scaleFactor = st.slider("📏 scaleFactor (échelle)", 1.1, 2.0, 1.1, 0.1)
    minNeighbors = st.slider("👥 minNeighbors (voisins minimum)", 1, 10, 5)

    if st.button("▶️ Détecter les visages"):
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(image_cv2, (x, y), (x+w, y+h), bgr_color, 2)

        st.image(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB), caption="🟩 Visages détectés", use_column_width=True)

        # ✅ Sauvegarde de l'image
        save_button = st.button("📥 Enregistrer l'image")
        if save_button:
            result_filename = "detected_faces.png"
            cv2.imwrite(result_filename, image_cv2)
            with open(result_filename, "rb") as file:
                st.download_button(label="📥 Cliquez ici pour télécharger", data=file, file_name=result_filename, mime="image/png")
