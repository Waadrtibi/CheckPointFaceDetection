import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🔍 Détection de Visages avec Viola-Jones (Amélioré)")

# ✅ Instructions
st.markdown("""
### ℹ️ Instructions :
1. Téléchargez une image contenant des visages.
2. Choisissez la couleur du rectangle de détection.
3. Ajustez les paramètres `scaleFactor` et `minNeighbors`.
4. Cliquez sur "Détecter les visages".
5. Cliquez sur "📥 Enregistrer l'image" pour sauvegarder le résultat.
""")

# ✅ Classificateur Haar Cascade amélioré
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# ✅ Chargement de l'image
uploaded_file = st.file_uploader("📤 Téléchargez une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="🖼️ Image originale", use_column_width=True)

    # ✅ Paramètres utilisateur
    rectangle_color = st.color_picker("🎨 Choisissez la couleur du rectangle", "#00FF00")
    bgr_color = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # RGB → BGR
    scaleFactor = st.slider("📏 scaleFactor (échelle)", 1.1, 2.0, 1.1, 0.1)
    minNeighbors = st.slider("👥 minNeighbors (voisins minimum)", 1, 10, 5)

    if st.button("▶️ Détecter les visages"):
        # ✅ Prétraitement de l'image
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # amélioration du contraste
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # réduction du bruit

        # ✅ Détection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        if len(faces) == 0:
            st.warning("😕 Aucun visage détecté. Essayez d'ajuster les paramètres.")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(image_cv2, (x, y), (x+w, y+h), bgr_color, 2)

            st.success(f"✅ {len(faces)} visage(s) détecté(s)")
            st.image(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB), caption="🟩 Visages détectés", use_column_width=True)

            if st.button("📥 Enregistrer l'image"):
                result_filename = "detected_faces.png"
                cv2.imwrite(result_filename, image_cv2)
                with open(result_filename, "rb") as file:
                    st.download_button(
                        label="📥 Cliquez ici pour télécharger",
                        data=file,
                        file_name=result_filename,
                        mime="image/png"
                    )
