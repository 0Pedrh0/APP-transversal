import cv2
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
SCENE_VIDEO_PATH = 'scene.mp4'           # Ta vidéo
GAZE_CSV_PATH = 'gaze.csv'               # Ton fichier de regard
TIMESTAMPS_CSV_PATH = 'world_timestamps.csv' # Les timestamps de la vidéo (souvent fourni par Pupil)
OUTPUT_VIDEO_PATH = 'output_gaze_overlay.mp4'

# --- 1. CHARGEMENT DES DONNÉES ---
print("Chargement des données en cours... ⏳")

# Chargement du CSV de regard
# On suppose que les colonnes sont 'timestamp [ns]', 'gaze x [px]', 'gaze y [px]'
# Si tes colonnes sont différentes (ex: normalisées 0-1), le code s'adapte plus bas.
gaze_df = pd.read_csv(GAZE_CSV_PATH)

# Chargement des timestamps de la vidéo
# Ce fichier lie chaque frame de la vidéo à un temps précis
video_timestamps = pd.read_csv(TIMESTAMPS_CSV_PATH)

# Renommer les colonnes pour être sûr (adapte si tes CSV ont des noms différents)
# Vérifie si la colonne s'appelle 'timestamp [ns]' ou 'timestamp'
time_col_gaze = [c for c in gaze_df.columns if 'timestamp' in c.lower()][0]
time_col_vid = [c for c in video_timestamps.columns if 'timestamp' in c.lower()][0]

gaze_df = gaze_df.rename(columns={time_col_gaze: 'timestamp'})
video_timestamps = video_timestamps.rename(columns={time_col_vid: 'timestamp'})

# Trier par temps (crucial pour la synchronisation)
gaze_df = gaze_df.sort_values('timestamp')
video_timestamps = video_timestamps.sort_values('timestamp')

# --- 2. SYNCHRONISATION (La partie magique ✨) ---
# On cherche pour chaque frame vidéo, le point de regard le plus proche dans le temps
print("Synchronisation des données... 🔗")

# merge_asof est super puissant : il trouve la valeur la plus proche (nearest)
merged_data = pd.merge_asof(
    video_timestamps, 
    gaze_df, 
    on='timestamp', 
    direction='nearest'
)

# --- 3. TRAITEMENT VIDÉO ---
print("Génération de la vidéo... 🎥")

cap = cv2.VideoCapture(SCENE_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Création du writer pour sauvegarder la vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Identification des colonnes de coordonnées
# Pupil Labs Neon exporte souvent en pixels ('gaze x [px]'), mais parfois en normalisé (0-1)
col_x = [c for c in gaze_df.columns if 'x' in c.lower()][0]
col_y = [c for c in gaze_df.columns if 'y' in c.lower()][0]

# Vérification simple : si max < 2, c'est probablement normalisé
is_normalized = gaze_df[col_x].max() <= 1.5

frame_idx = 0
total_frames = len(video_timestamps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Récupérer les données de regard pour cette frame spécifique
    if frame_idx < len(merged_data):
        row = merged_data.iloc[frame_idx]
        
        gaze_x = row[col_x]
        gaze_y = row[col_y]
        
        # Vérifier si on a des données valides (pas de NaN)
        if not np.isnan(gaze_x) and not np.isnan(gaze_y):
            # Conversion si les données sont normalisées (0 à 1)
            if is_normalized:
                pixel_x = int(gaze_x * width)
                # Attention: l'axe Y peut être inversé selon les versions (essayer height - y si ça semble faux)
                pixel_y = int((1 - gaze_y) * height) # Souvent inversé dans Pupil
            else:
                pixel_x = int(gaze_x)
                pixel_y = int(gaze_y)
            
            # DESSINER LE CERCLE (Le "Gaze Overlay")
            # Couleur (B, G, R) -> (0, 0, 255) est Rouge
            cv2.circle(frame, (pixel_x, pixel_y), 20, (0, 0, 255), 3) # Cercle
            cv2.circle(frame, (pixel_x, pixel_y), 4, (0, 255, 255), -1) # Point central
            
    out.write(frame)
    
    if frame_idx % 100 == 0:
        print(f"Traitement : {frame_idx}/{total_frames} frames")
        
    frame_idx += 1

cap.release()
out.release()
print(f"Terminé ! Vidéo sauvegardée sous : {OUTPUT_VIDEO_PATH} 🚀")