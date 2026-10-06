# 🤲 Arabic Sign Language (ArSL) Recognition App

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

Application web interactive de reconnaissance de lettres en Langue des Signes Arabe (ArSL) basée sur le Deep Learning, conçue pour faciliter la communication et l'accessibilité.

---

## 🚀 Aperçu du Projet
Ce projet implémente un pipeline complet de vision par ordinateur pour classifier les signes de l'alphabet arabe en temps réel ou à partir d'images statiques. Il combine l'extraction de caractéristiques et des architectures de réseaux de neurones profonds, le tout intégré dans une interface utilisateur fluide développée avec Streamlit.

## 🛠️ Stack Technique
* **Deep Learning :** PyTorch, TensorFlow / Keras
* **Traitement d'Image :** OpenCV, NumPy, Scikit-image
* **Interface Utilisateur :** Streamlit
* **Gestion de code :** Git / GitHub

## 📂 Structure du Dépôt
```text
├── data/               # Scripts de prétraitement et d'augmentation de données
├── models/             # Définition des architectures (CNN, Transfer Learning) et poids entraînés
├── notebooks/          # Jupyter Notebooks pour l'exploration et l'entraînement
├── app.py              # Application principale Streamlit
├── requirements.txt    # Dépendances du projet
└── README.md
