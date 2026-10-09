# 🤟 Arabic Sign Language Recognition (ARSL) using Deep Learning

**Auteurs :** Ziad RAMMAL & Melissa YESGUER  
**Cadre :** Master 2 SIA2 / EEA - Université de Toulouse  
**Application en ligne :** [Tester l'application Streamlit](https://classification-de-la-langue-des-signes-arabe-h2stpn8kfz2aaksxd.streamlit.app/)

---

## 📌 Présentation du Projet
Ce projet vise à traduire en temps réel les signes de l'alphabet de la langue des signes arabe (31 classes) en caractères arabes. L'application permet à l'utilisateur d'utiliser sa webcam ou de charger une image pour prédire la lettre correspondante et composer une phrase complète.

---

## 📊 Résultats & Performances
- **Modèle CNN Baseline (from scratch) :** ~51% d'accuracy (Sujet à un surapprentissage important).
- **Modèle Transfer Learning (VGG16) + Data Augmentation :** **~96.1% d'accuracy** sur l'ensemble de test.
- **Détection des mains :** Intégration de **MediaPipe Hands** pour le découpage automatique de la zone d'intérêt (*ROI*) avant la prédiction.

---

## 📁 Structure du Dépôt
```text
.
├── app/
│   └── Streamlit_Classification_de_la_langue_des_signes.py  # Application Streamlit
├── docs/
│   └── Rapport_Classification_de_la_langue_des_signes.pdf   # Rapport détaillé
├── models/
│   ├── model.json                                           # Architecture du réseau
│   └── model.h5                                             # Poids (téléchargés via Drive)
├── notebooks/                                               # Scripts d'entraînement
├── .gitignore
├── packages.txt                                             # Dépendances système
├── README.md
└── requirements.txt                                         # Dépendances Python
