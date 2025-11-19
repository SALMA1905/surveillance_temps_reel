import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
from datetime import datetime
import shutil

# Chemin vers les fichiers nécessaires
yolov3_weights_path = r'C:\Users\Salma\Desktop\yolo\yolov3\yolov3-wider_16000.weights'
yolov3_config_path = r'C:\Users\Salma\Desktop\yolo\yolov3\yolov3-face.cfg'
names_path = r'C:\Users\Salma\Desktop\yolo\yolov3\face.names'
model_path = r'C:\Users\Salma\model_pfe.h5'
output_folder = r'C:\Users\Salma\Desktop\yolo\yolov3\detected_faces'  # Dossier pour enregistrer les visages détectés
face_output_folder = os.path.join(output_folder, 'face')  # Dossier pour enregistrer les visages avec leur nom

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)
os.makedirs(face_output_folder, exist_ok=True)

# Charger les noms des classes (généralement juste "face")
with open(names_path) as f:
    classes = [line.strip() for line in f.readlines()]

# Charger le réseau YOLO pour la détection de visage
net = cv2.dnn.readNet(yolov3_weights_path, yolov3_config_path)

# Charger le modèle de classification
model = tf.keras.models.load_model(model_path)
classification_classes = ['bill_gates', 'elon_musk', 'Inconnu', 'jeff_bezos', 'mark_zuckerberg', 'steve_jobs']
# Fonction de prédiction pour la classification
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    return classification_classes[np.argmax(pred)]

# Fonction de prétraitement des visages
def preprocess_face(face):
    img = cv2.resize(face, (224, 224))  # Redimensionner l'image à 224x224
    img = img.astype('float32') / 255.0  # Normaliser les pixels à la plage [0, 1]
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour le batch
    return img

# Fonction pour vérifier la proximité des visages déjà détectés
def is_nearby(new_face_center, detected_faces_centers, threshold=50):
    for center in detected_faces_centers:
        distance = np.linalg.norm(np.array(new_face_center) - np.array(center))
        if distance < threshold:
            return True
    return False

# Détection continue des visages depuis la caméra
def detect_and_save_faces_continuous():
    # Liste pour stocker les noms des visages détectés
    detected_faces = []

    # Ouvrir la capture vidéo avec DirectShow
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Liste pour stocker les centres des visages déjà détectés
    detected_faces_centers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de l'image depuis la caméra.")
            break

        height, width = frame.shape[:2]

        # Préparation de l'image pour YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Récupération des noms des couches de sortie
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Détection des objets
        outputs = net.forward(output_layers)

        # Analyse des détections
        conf_threshold = 0.5
        nms_threshold = 0.4
        detected_faces = []  # Liste pour stocker les étiquettes des visages détectés

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                confidence = max(scores)
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Vérifier les dimensions valides du visage
                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        face = frame[y:y+h, x:x+w]
                        face_center = (center_x, center_y)

                        # Vérifier que l'image de visage n'est pas vide et non déjà détectée
                        if face is not None and face.size != 0 and not is_nearby(face_center, detected_faces_centers):
                            # Nommer le fichier d'image avec le timestamp actuel
                            face_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            face_path = os.path.join(output_folder, face_filename)

                            # Enregistrer l'image du visage avec son nom
                            cv2.imwrite(face_path, face)

                            # Prédire la classe du visage
                            class_label = predict_image(face_path)

                            # Ajouter le label détecté à la liste des visages détectés
                            detected_faces.append((face_filename, class_label))

                            # Afficher et annoter l'image avec la classe prédite
                            print(f"Detected: {class_label} ")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            # Ajouter le centre du visage à la liste des centres détectés
                            detected_faces_centers.append(face_center)

        # Afficher le résultat avec les visages détectés
        cv2.imshow('Face Detection and Classification', frame)

        # Attendre une petite durée entre chaque frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presser 'q' pour quitter la boucle
            break

    # Libérer la capture et détruire la fenêtre OpenCV
    cap.release()
    cv2.destroyAllWindows()

    # Enregistrer les visages détectés avec leur nom dans le dossier `face`
    for face_filename, class_label in detected_faces:
        source_path = os.path.join(output_folder, face_filename)
        dest_path = os.path.join(face_output_folder, f"{class_label}_{face_filename}")
        shutil.copy(source_path, dest_path)
        print(f"Saved: {dest_path}")

    # Nettoyer les fichiers temporaires du dossier de sortie
    for face_filename, _ in detected_faces:
        face_path = os.path.join(output_folder, face_filename)
        if os.path.exists(face_path):
            os.remove(face_path)
            print(f"Removed: {face_path}")

    # Imprimer les noms des visages détectés
    for _, class_label in detected_faces:
        print(f"Detected: {class_label}")

# Appeler la fonction pour détecter et enregistrer les visages depuis la caméra en continu
detect_and_save_faces_continuous()
