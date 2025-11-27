import cv2
import pandas as pd
import numpy as np
import os

def add_gaze_to_video(
    video_path, 
    gaze_csv_path, 
    timestamps_csv_path, 
    output_path, 
    show_progress=True
):
    """
    Incruste la position du regard sur une vidéo à partir des données Pupil Labs.
    
    Args:
        video_path (str): Chemin vers la vidéo source (.mp4)
        gaze_csv_path (str): Chemin vers gaze.csv
        timestamps_csv_path (str): Chemin vers world_timestamps.csv
        output_path (str): Chemin où sauvegarder la vidéo finale
        show_progress (bool): Afficher l'avancement dans la console
        
    Returns:
        bool: True si succès, False sinon.
    """
    
    # 1. VÉRIFICATION DES FICHIERS
    if not os.path.exists(video_path):
        print(f"Erreur : Vidéo introuvable -> {video_path}")
        return False
    if not os.path.exists(gaze_csv_path):
        print(f"Erreur : CSV Gaze introuvable -> {gaze_csv_path}")
        return False

    try:
        # 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
        if show_progress: print(f"Chargement des données pour {os.path.basename(video_path)}...")
        
        gaze_df = pd.read_csv(gaze_csv_path)
        video_timestamps = pd.read_csv(timestamps_csv_path)

        # Harmonisation des noms de colonnes (timestamp)
        time_col_gaze = [c for c in gaze_df.columns if 'timestamp' in c.lower()][0]
        time_col_vid = [c for c in video_timestamps.columns if 'timestamp' in c.lower()][0]

        gaze_df = gaze_df.rename(columns={time_col_gaze: 'timestamp'})
        video_timestamps = video_timestamps.rename(columns={time_col_vid: 'timestamp'})

        # Tri nécessaire pour merge_asof
        gaze_df = gaze_df.sort_values('timestamp')
        video_timestamps = video_timestamps.sort_values('timestamp')

        # Synchronisation
        merged_data = pd.merge_asof(
            video_timestamps, 
            gaze_df, 
            on='timestamp', 
            direction='nearest'
        )

        # 3. CONFIGURATION VIDÉO
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Détection des colonnes X et Y
        col_x = [c for c in gaze_df.columns if 'x' in c.lower()][0]
        col_y = [c for c in gaze_df.columns if 'y' in c.lower()][0]

        # Détection Normalisé vs Pixels 
        is_normalized = gaze_df[col_x].max() <= 1.5

        # 4. TRAITEMENT FRAME PAR FRAME
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(merged_data):
                row = merged_data.iloc[frame_idx]
                gaze_x = row[col_x]
                gaze_y = row[col_y]
                
                if not np.isnan(gaze_x) and not np.isnan(gaze_y):
                    if is_normalized:
                        pixel_x = int(gaze_x * width)
                        # Inversion Y pour coordonnées normalisées (standard Pupil Labs)
                        pixel_y = int((1 - gaze_y) * height) 
                    else:
                        pixel_x = int(gaze_x)
                        pixel_y = int(gaze_y)
                    
                    # Design du pointeur
                    cv2.circle(frame, (pixel_x, pixel_y), 20, (0, 0, 255), 3) # Cercle rouge
                    cv2.circle(frame, (pixel_x, pixel_y), 4, (0, 255, 255), -1) # Point jaune
                    
            out.write(frame)
            
            if show_progress and frame_idx % 100 == 0:
                print(f"Traitement : {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.0f}%)", end='\r')
                
            frame_idx += 1

        cap.release()
        out.release()
        if show_progress: print(f"\nVidéo sauvegardée : {output_path}")
        return True

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return False



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paramètres pour le SUJET 1
    sujet_folder = os.path.join(script_dir, '..', 'AcquisitionsEyeTracker', 'sujet1_f-42e0d11a')
    
    video_in = os.path.join(sujet_folder, 'e0b2c246_0.0-138.011.mp4')
    csv_gaze = os.path.join(sujet_folder, 'gaze.csv')
    csv_time = os.path.join(sujet_folder, 'world_timestamps.csv')
    video_out = os.path.join(script_dir, 'gaze_and_video_sujet1.mp4')

    # Appel de la fonction
    add_gaze_to_video(video_in, csv_gaze, csv_time, video_out)