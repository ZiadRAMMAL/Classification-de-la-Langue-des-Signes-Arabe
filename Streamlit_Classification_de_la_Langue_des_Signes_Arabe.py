# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:58:21 2026

@author: Ziad RAMMAL et Melissa YESGUER
"""

from __future__ import annotations
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import model_from_json

# --- 1. INITIALISATION ET MÉMOIRE ---

# On utilise st.session_state pour que les lettres ne s'effacent pas quand la page se rafraîchit
if 'phrase' not in st.session_state:
    st.session_state.phrase = ""

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

nom_classe = [
    "Ain", "Al", "Alef", "Beh", "Dad", "Dal", "Feh", "Ghain", "Hah", "Heh",
    "Jeem", "Kaf", "Khah", "Laa", "Lam", "Meem", "Noon", "Qaf", "Reh", "Sad",
    "Seen", "Sheen", "Tah", "Teh", "Teh_Marbuta", "Thal", "Theh", "Waw", "Yeh", 
    "Zah", "Zain"
]
# Mapping entre le nom de la classe et le caractère arabe
mapping_arabe = {
    "Ain": "ع", "Al": "ال", "Alef": "أ", "Beh": "ب", "Dad": "ض",
    "Dal": "د", "Feh": "ف", "Ghain": "غ", "Hah": "ح", "Heh": "ه",
    "Jeem": "ج", "Kaf": "ك", "Khah": "خ", "Laa": "لا", "Lam": "ل",
    "Meem": "م", "Noon": "ن", "Qaf": "ق", "Reh": "ر", "Sad": "ص",
    "Seen": "س", "Sheen": "ش", "Tah": "ط", "Teh": "ت", "Teh_Marbuta": "ة",
    "Thal": "ذ", "Theh": "ث", "Waw": "و", "Yeh": "ي", "Zah": "ظ",
    "Zain": "ز"
}

@st.cache_data
def load_trained_model():
    try:
        model = model_from_json(open('model.json').read())
        model.load_weights('model.h5')  
        return model
    except Exception as e:
        st.error(f"Erreur modèle : {e}")
        return None

model = load_trained_model()

# --- 2. TRAITEMENT ---

def get_hand_crop(image_pil):
    img = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    results = hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            center_x, center_y = (min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2
            box_size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)) * 1.4
            
            x1, y1 = int(max(0, center_x - box_size/2)), int(max(0, center_y - box_size/2))
            x2, y2 = int(min(w, center_x + box_size/2)), int(min(h, center_y + box_size/2))
            
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0: return None, None, False
            
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            img_tensor = roi_rgb.astype('float32') / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            return img_tensor, roi_rgb, True
    return None, None, False

# --- 3. INTERFACE ---

st.title("Classification des lettres de signe (vesion arabe) : Mode Traduction")

# Zone de texte pour afficher la phrase construite
st.markdown(
    f"""
    <div style="text-align: right; direction: rtl; font-size: 40px; background-color: #000000; padding: 20px; border-radius: 10px;">
        {st.session_state.phrase}
    </div>
    """, 
    unsafe_allow_html=True
)

source = st.radio("Source :", ("Webcam", "Upload"))
img_file = st.camera_input("Faites votre signe") if source == "Webcam" else st.file_uploader("Image", type=["jpg", "png", "jpeg"])

if img_file:
    image_pil = Image.open(img_file)
    input_ia, img_display, detected = get_hand_crop(image_pil)
    
    if detected:
        col_img, col_act = st.columns([1, 1])
        
        with col_img:
            st.image(img_display, caption="Signe détecté", use_column_width=True)
            
        with col_act:
            # Prédiction
            preds = model.predict(input_ia)[0]
            top_idx = np.argmax(preds)
            lettre_detectee = nom_classe[top_idx]
            confiance = preds[top_idx] * 100

            st.subheader(f"Lettre : :blue[{lettre_detectee}]")
            st.write(f"Confiance : {confiance:.1f}%")

            # --- BOUTONS DE TRADUCTION MIS À JOUR ---
            st.write("---")
            # On récupère le vrai caractère arabe
            caractere_arabe = mapping_arabe.get(lettre_detectee, lettre_detectee)

            if st.button(f"✅ Ajouter '{caractere_arabe}' ({lettre_detectee})"):
                # On ajoute le caractère arabe à la phrase
                st.session_state.phrase += caractere_arabe
                st.experimental_rerun()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⌨️ Espace"):
                    st.session_state.phrase += " "
                    st.experimental_rerun()
            with col_btn2:
                if st.button("🗑️ Effacer"):
                    st.session_state.phrase = ""
                    st.experimental_rerun()
    else:
        st.warning("Main non détectée.")