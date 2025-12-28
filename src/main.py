import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from googletrans import Translator
from pathlib import Path
import logging
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ManhwaTranslatorLongImage:
    def __init__(self, input_folder, output_folder, source_lang='ko', target_lang='fr', 
                 chunk_height='auto', overlap=200, confidence_threshold=0.3, debug_mode=False):
        """
        Initialise le traducteur de manhwa pour images longues
        
        Args:
            input_folder: Dossier contenant les images à traduire
            output_folder: Dossier de sortie pour les images traduites
            source_lang: Langue source (par défaut: 'ko' pour coréen)
            target_lang: Langue cible (par défaut: 'fr' pour français)
            chunk_height: Hauteur de chaque section ('auto' pour détection automatique, ou pixels)
            overlap: Chevauchement entre sections pour ne rien manquer
            confidence_threshold: Seuil de confiance OCR (0.0 à 1.0, plus bas = plus de détections)
            debug_mode: Si True, sauvegarde les images avec détections visualisées
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.chunk_height = chunk_height
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        
        # Créer le dossier de sortie s'il n'existe pas
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Créer dossier debug si nécessaire
        if self.debug_mode:
            self.debug_folder = self.output_folder / "debug"
            self.debug_folder.mkdir(exist_ok=True)
            logger.info(f"🔍 Mode debug activé: {self.debug_folder}")
        
        # Initialiser EasyOCR et le traducteur
        logger.info(f"Initialisation d'EasyOCR pour la langue: {source_lang}")
        self.reader = easyocr.Reader([source_lang, 'en'], gpu=True)
        self.translator = Translator()
        
        # Extensions d'images supportées
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # MODIFIÉ: Ajout de la fonction de sauvegarde debug (déplacée depuis v2.py)
    def save_debug_image(self, image_cv, ocr_results, filename_prefix):
        """
        Sauvegarde une image avec les détections visualisées (mode debug)
        
        Args:
            image_cv: Image OpenCV
            ocr_results: Résultats OCR
            filename_prefix: Préfixe du nom de fichier
        """
        if not self.debug_mode:
            return
        
        debug_img = image_cv.copy()
        
        for bbox, text, confidence in ocr_results:
            if confidence > self.confidence_threshold:
                # Dessiner la bbox en rouge
                points = np.array(bbox, dtype=np.int32)
                cv2.polylines(debug_img, [points], True, (0, 0, 255), 2)
                
                # Ajouter le texte et la confiance
                x, y = int(points[0][0]), int(points[0][1]) - 10
                label = f"{text[:20]} ({confidence:.2f})"
                cv2.putText(debug_img, label, (x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Sauvegarder
        debug_path = self.debug_folder / f"{filename_prefix}_detections.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"🔍 Debug sauvegardé: {debug_path.name}")

    def get_image_files(self):
        """Récupère tous les fichiers images du dossier d'entrée"""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.input_folder.glob(f'*{ext}'))
            image_files.extend(self.input_folder.glob(f'*{ext.upper()}'))
        return sorted(image_files)
    
    def calculate_optimal_chunk_height(self, image_height):
        """
        Calcule automatiquement la hauteur optimale de découpage
        selon la hauteur de l'image
        
        Args:
            image_height: Hauteur de l'image en pixels
            
        Returns:
            Hauteur de chunk optimale
        """
        if image_height <= 3000:
            # Image courte/moyenne : pas de découpage nécessaire
            return image_height
        elif image_height <= 6000:
            # Image moyenne-longue : 2 sections
            return 3000
        elif image_height <= 10000:
            # Image longue : découpage standard
            return 2500
        elif image_height <= 20000:
            # Image très longue (webtoon standard) : découpage fin
            return 2000
        else:
            # Image extrêmement longue : découpage très fin
            return 1800
    
    def split_long_image(self, image_cv, auto_adjust=True):
        """
        Découpe une longue image en sections avec chevauchement
        Détecte automatiquement la meilleure stratégie de découpage
        
        Args:
            image_cv: Image OpenCV
            auto_adjust: Si True, ajuste automatiquement chunk_height
            
        Returns:
            Liste de tuples (chunk_image, y_offset)
        """
        height, width = image_cv.shape[:2]
        
        # Afficher les dimensions
        logger.info(f"📏 Dimensions détectées: {width}x{height} pixels")
        
        # Calculer la hauteur optimale de chunk si mode auto
        if self.chunk_height == 'auto' or auto_adjust:
            calculated_chunk = self.calculate_optimal_chunk_height(height)
            logger.info(f"🎯 Hauteur de section optimale calculée: {calculated_chunk}px")
            chunk_height_to_use = calculated_chunk
        else:
            chunk_height_to_use = self.chunk_height
        
        chunks = []
        
        if height <= chunk_height_to_use:
            # Image courte, pas besoin de découper
            logger.info(f"✅ Image assez courte, pas de découpage nécessaire")
            return [(image_cv, 0)]
        
        # Calculer le nombre de sections
        num_sections = int(np.ceil((height - self.overlap) / (chunk_height_to_use - self.overlap)))
        logger.info(f"✂️  Image longue détectée!")
        logger.info(f"   Hauteur totale: {height}px")
        logger.info(f"   Découpage en {num_sections} section(s) de ~{chunk_height_to_use}px")
        logger.info(f"   Chevauchement: {self.overlap}px")
        
        y_start = 0
        chunk_index = 0
        
        while y_start < height:
            y_end = min(y_start + chunk_height_to_use, height)
            
            # Extraire la section
            chunk = image_cv[y_start:y_end, :]
            chunks.append((chunk, y_start))
            
            chunk_index += 1
            chunk_actual_height = y_end - y_start
            logger.info(f"   📄 Section {chunk_index}/{num_sections}: y={y_start}-{y_end} ({chunk_actual_height}px)")
            
            # Calculer le prochain départ avec chevauchement
            if y_end >= height:
                break
            
            y_start = y_end - self.overlap
        
        logger.info(f"✅ Découpage terminé: {len(chunks)} sections créées")
        return chunks
    
    # MODIFIÉ: Ajout de prétraitements (inversés) pour mieux détecter
    def preprocess_image_for_ocr(self, chunk_image):
        """
        Prétraite l'image pour améliorer la détection OCR
        Ajoute des versions inversées pour le texte clair sur fond sombre
        
        Args:
            chunk_image: Section d'image (numpy array)
            
        Returns:
            Liste d'images prétraitées à tester
        """
        images_to_process = [chunk_image]
        
        gray = cv2.cvtColor(chunk_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Version en niveaux de gris avec contraste amélioré (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        images_to_process.append(enhanced_bgr)
        
        # 2. Version avec seuillage adaptatif
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        images_to_process.append(thresh_bgr)
        
        # 3. Version inversée + CLAHE (pour texte blanc sur fond noir)
        inverted = cv2.bitwise_not(gray)
        enhanced_inverted = clahe.apply(inverted)
        enhanced_inverted_bgr = cv2.cvtColor(enhanced_inverted, cv2.COLOR_GRAY2BGR)
        images_to_process.append(enhanced_inverted_bgr)
        
        # 4. Version inversée + Seuillage adaptatif
        thresh_inverted = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        thresh_inverted_bgr = cv2.cvtColor(thresh_inverted, cv2.COLOR_GRAY2BGR)
        images_to_process.append(thresh_inverted_bgr)

        return images_to_process
    
    # MODIFIÉ: Optimisation majeure - passe le numpy array directement à EasyOCR
    def extract_text_from_chunk(self, chunk_image):
        """
        Extrait le texte d'une section d'image avec prétraitement multiple
        
        Args:
            chunk_image: Section d'image (numpy array)
            
        Returns:
            Liste de tuples (bbox, texte, confidence)
        """
        try:
            all_detections = []
            
            # Obtenir plusieurs versions de l'image
            processed_images = self.preprocess_image_for_ocr(chunk_image)
            
            for idx, processed_img in enumerate(processed_images):
                
                # OCR avec paramètres plus agressifs
                # Passe directement le numpy array (BGR) à readtext
                result = self.reader.readtext(
                    processed_img,
                    detail=1,
                    paragraph=False,
                    min_size=5,
                    text_threshold=0.5,
                    low_text=0.3,
                    link_threshold=0.3,
                    canvas_size=2800,
                    mag_ratio=1.5
                )
                
                all_detections.extend(result)
            
            # Fusionner et dédupliquer les détections
            unique_detections = self.deduplicate_detections(all_detections)
            
            return unique_detections
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte: {e}")
            return []
    
    def deduplicate_detections(self, detections):
        """
        Supprime les détections en double (même position, texte similaire)
        AMÉLIORÉ : Utilise IoU et similarité de texte
        
        Args:
            detections: Liste de détections
            
        Returns:
            Liste de détections uniques
        """
        if not detections:
            return []
        
        unique = []
        skip_indices = set()
        
        for i, (bbox, text, conf) in enumerate(detections):
            if i in skip_indices:
                continue
            
            # Calculer le centre de cette bbox
            points = np.array(bbox, dtype=np.float32)
            center_x = points[:, 0].mean()
            center_y = points[:, 1].mean()
            
            # Vérifier les doublons potentiels
            is_duplicate = False
            for j in range(i + 1, len(detections)):
                if j in skip_indices:
                    continue
                
                other_bbox, other_text, other_conf = detections[j]
                other_points = np.array(other_bbox, dtype=np.float32)
                other_center_x = other_points[:, 0].mean()
                other_center_y = other_points[:, 1].mean()
                
                # Distance entre centres
                distance = np.sqrt((center_x - other_center_x)**2 + 
                                 (center_y - other_center_y)**2)
                
                # IoU
                iou = self.calculate_iou(bbox, other_bbox)
                
                # Textes normalisés
                text_norm = text.strip().lower().replace(" ", "")
                other_text_norm = other_text.strip().lower().replace(" ", "")
                
                # Doublon si proche ET texte similaire
                if (distance < 30 or iou > 0.6) and text_norm == other_text_norm:
                    # Garder le meilleur score
                    if other_conf > conf:
                        is_duplicate = True
                        break
                    else:
                        skip_indices.add(j)
            
            if not is_duplicate:
                unique.append((bbox, text, conf))
        
        if len(detections) > len(unique):
            logger.info(f"   Déduplication: {len(detections)} → {len(unique)} détections")
        
        return unique
    
    def merge_overlapping_detections(self, all_detections, image_height):
        """
        Fusionne les détections qui se chevauchent entre sections
        AMÉLIORÉ : Détecte aussi les variations de casse (He Is / HE Is)
        
        Args:
            all_detections: Liste de toutes les détections avec offsets
            image_height: Hauteur totale de l'image
            
        Returns:
            Liste de détections uniques
        """
        if not all_detections:
            return []
        
        # Trier par position Y
        sorted_detections = sorted(all_detections, key=lambda x: x[0][0][1])
        
        merged = []
        skip_indices = set()
        
        for i, (bbox, text, conf) in enumerate(sorted_detections):
            if i in skip_indices:
                continue
            
            # Calculer le centre de cette détection
            current_points = np.array(bbox, dtype=np.float32)
            current_center_x = current_points[:, 0].mean()
            current_center_y = current_points[:, 1].mean()
            current_area = self.calculate_bbox_area(bbox)
            
            # Chercher les doublons potentiels
            is_duplicate = False
            for j in range(i + 1, len(sorted_detections)):
                if j in skip_indices:
                    continue
                
                other_bbox, other_text, other_conf = sorted_detections[j]
                other_points = np.array(other_bbox, dtype=np.float32)
                other_center_x = other_points[:, 0].mean()
                other_center_y = other_points[:, 1].mean()
                
                # Calculer la distance entre les centres
                distance = np.sqrt((current_center_x - other_center_x)**2 + 
                                 (current_center_y - other_center_y)**2)
                
                # Calculer l'IoU (Intersection over Union) des bboxes
                iou = self.calculate_iou(bbox, other_bbox)
                
                # Normaliser les textes pour comparaison (ignorer la casse et espaces)
                text_normalized = text.strip().lower().replace(" ", "")
                other_text_normalized = other_text.strip().lower().replace(" ", "")
                
                # C'est un doublon si :
                # 1. Les centres sont très proches (< 50 pixels)
                # 2. OU IoU élevé (> 0.5)
                # 3. ET texte identique ou très similaire
                if ((distance < 50 or iou > 0.5) and 
                    (text_normalized == other_text_normalized or 
                     self.similar_text(text_normalized, other_text_normalized))):
                    
                    # Garder celui avec meilleure confiance
                    if other_conf > conf:
                        is_duplicate = True
                        break
                    else:
                        skip_indices.add(j)
            
            if not is_duplicate:
                merged.append((bbox, text, conf))
        
        logger.info(f"Détections fusionnées: {len(all_detections)} → {len(merged)} (éliminé {len(all_detections) - len(merged)} doublons)")
        return merged
    
    def calculate_bbox_area(self, bbox):
        """Calcule l'aire d'une bbox"""
        points = np.array(bbox, dtype=np.float32)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        width = x_coords.max() - x_coords.min()
        height = y_coords.max() - y_coords.min()
        return width * height
    
    def calculate_iou(self, bbox1, bbox2):
        """
        Calcule l'IoU (Intersection over Union) entre deux bboxes
        
        Args:
            bbox1, bbox2: Boîtes englobantes
            
        Returns:
            IoU score (0.0 à 1.0)
        """
        points1 = np.array(bbox1, dtype=np.float32)
        points2 = np.array(bbox2, dtype=np.float32)
        
        x1_min, y1_min = points1[:, 0].min(), points1[:, 1].min()
        x1_max, y1_max = points1[:, 0].max(), points1[:, 1].max()
        
        x2_min, y2_min = points2[:, 0].min(), points2[:, 1].min()
        x2_max, y2_max = points2[:, 0].max(), points2[:, 1].max()
        
        # Calculer l'intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y_inter_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculer l'union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def similar_text(self, text1, text2):
        """
        Vérifie si deux textes sont similaires (distance de Levenshtein simplifiée)
        
        Args:
            text1, text2: Textes normalisés à comparer
            
        Returns:
            True si similaires
        """
        if not text1 or not text2:
            return False
        
        # Si l'un contient l'autre
        if text1 in text2 or text2 in text1:
            return True
        
        # Si longueurs très différentes, pas similaires
        if abs(len(text1) - len(text2)) > 3:
            return False
        
        # Compter les caractères différents
        max_len = max(len(text1), len(text2))
        min_len = min(len(text1), len(text2))
        
        differences = abs(len(text1) - len(text2))
        for i in range(min_len):
            if text1[i] != text2[i]:
                differences += 1
        
        # Similaires si moins de 20% de différences
        similarity_ratio = 1 - (differences / max_len)
        return similarity_ratio > 0.8
    
    def deduplicate_text_regions(self, text_regions):
        """
        Déduplique les régions de texte traduit avant de les dessiner
        Évite d'écrire plusieurs fois au même endroit
        
        Args:
            text_regions: Liste de régions de texte
            
        Returns:
            Liste dédupliquée
        """
        if not text_regions:
            return []
        
        unique_regions = []
        skip_indices = set()
        
        for i, region in enumerate(text_regions):
            if i in skip_indices:
                continue
            
            bbox = region['bbox']
            text = region['text']
            confidence = region.get('confidence', 1.0)
            
            # Calculer le centre
            points = np.array(bbox, dtype=np.float32)
            center_x = points[:, 0].mean()
            center_y = points[:, 1].mean()
            
            # Chercher les doublons
            is_duplicate = False
            for j in range(i + 1, len(text_regions)):
                if j in skip_indices:
                    continue
                
                other_region = text_regions[j]
                other_bbox = other_region['bbox']
                other_text = other_region['text']
                other_confidence = other_region.get('confidence', 1.0)
                
                other_points = np.array(other_bbox, dtype=np.float32)
                other_center_x = other_points[:, 0].mean()
                other_center_y = other_points[:, 1].mean()
                
                # Distance
                distance = np.sqrt((center_x - other_center_x)**2 + 
                                 (center_y - other_center_y)**2)
                
                # IoU
                iou = self.calculate_iou(bbox, other_bbox)
                
                # Textes normalisés
                text_norm = text.strip().lower().replace(" ", "")
                other_text_norm = other_text.strip().lower().replace(" ", "")
                
                # Doublon si très proche ET texte identique/similaire
                if (distance < 40 or iou > 0.5) and (text_norm == other_text_norm):
                    logger.info(f"   🔄 Doublon détecté: '{text}' et '{other_text}' (distance: {distance:.1f}px, IoU: {iou:.2f})")
                    
                    # Garder celui avec meilleure confiance
                    if other_confidence > confidence:
                        is_duplicate = True
                        logger.info(f"      → Gardé: '{other_text}' (conf: {other_confidence:.2f})")
                        break
                    else:
                        skip_indices.add(j)
                        logger.info(f"      → Gardé: '{text}' (conf: {confidence:.2f})")
            
            if not is_duplicate:
                unique_regions.append(region)
        
        if len(text_regions) > len(unique_regions):
            logger.info(f"🎯 Déduplication finale: {len(text_regions)} → {len(unique_regions)} régions uniques")
        
        return unique_regions
    
    def extract_text_from_long_image(self, image_path):
        """
        Extrait le texte d'une image longue en la découpant
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Liste de tuples (bbox, texte, confidence) avec coordonnées absolues
        """
        try:
            logger.info(f"Extraction du texte de: {image_path.name}")
            
            # Charger l'image
            image_cv = cv2.imread(str(image_path))
            height, width = image_cv.shape[:2]
            
            # Découper en sections
            chunks = self.split_long_image(image_cv)
            
            # Extraire le texte de chaque section
            all_detections = []
            
            for chunk_idx, (chunk, y_offset) in enumerate(chunks):
                logger.info(f"Traitement section {chunk_idx + 1}/{len(chunks)}...")
                
                chunk_detections = self.extract_text_from_chunk(chunk)
                logger.info(f"  Trouvé {len(chunk_detections)} zones de texte")
                
                # Ajuster les coordonnées bbox avec l'offset
                for bbox, text, confidence in chunk_detections:
                    adjusted_bbox = []
                    for point in bbox:
                        adjusted_point = [point[0], point[1] + y_offset]
                        adjusted_bbox.append(adjusted_point)
                    
                    all_detections.append((adjusted_bbox, text, confidence))
            
            # Fusionner les détections qui se chevauchent
            merged_detections = self.merge_overlapping_detections(all_detections, height)
            
            logger.info(f"Total: {len(merged_detections)} zones de texte détectées")
            return merged_detections
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_text_mask(self, image_shape, bbox, expansion=5):
        """
        Crée un masque pour la zone de texte SIMPLE
        
        Args:
            image_shape: Forme de l'image (height, width)
            bbox: Boîte englobante du texte
            expansion: Pixels à ajouter autour du texte
            
        Returns:
            Masque binaire
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Extraire et convertir les coordonnées
        points = np.array(bbox, dtype=np.int32)
        
        # Calculer le centre et agrandir légèrement la région
        center = points.mean(axis=0)
        expanded_points = []
        for point in points:
            direction = point - center
            dist = np.linalg.norm(direction)
            if dist > 0:
                # Agrandir de 'expansion' pixels
                expanded_point = point + direction * (expansion / dist)
            else:
                expanded_point = point
            expanded_points.append(expanded_point)
        
        expanded_points = np.array(expanded_points, dtype=np.int32)
        
        # Remplir le polygone
        cv2.fillPoly(mask, [expanded_points], 255)
        
        return mask
    
    # SUPPRIMÉ: Remplacé par detect_text_and_bg_color
    # def detect_bubble_color(self, image_cv, bbox): ...

    # SUPPRIMÉ: Remplacé par cv2.inpaint dans process_image
    # def simple_erase_text(self, image_cv, mask, bbox): ...

    # NOUVEAU: Utilise K-Means pour trouver la couleur du texte ET du fond
    def detect_text_and_bg_color(self, image_cv, bbox):
        """
        Détecte la couleur dominante du texte et du fond via k-means
        
        Args:
            image_cv: Image OpenCV ORIGINALE
            bbox: Boîte englobante du texte
            
        Returns:
            Tuple (bg_color_bgr, text_color_rgb)
        """
        try:
            # Extraire la région
            points = np.array(bbox, dtype=np.int32)
            x_min = max(0, int(points[:, 0].min()))
            x_max = min(image_cv.shape[1], int(points[:, 0].max()))
            y_min = max(0, int(points[:, 1].min()))
            y_max = min(image_cv.shape[0], int(points[:, 1].max()))
            
            region = image_cv[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                return (255, 255, 255), (0, 0, 0) # Défaut: Fond Blanc, Texte Noir

            # Reshape en liste de pixels
            pixels = region.reshape(-1, 3).astype(np.float32)
            
            # Utiliser k-means pour trouver les 2 couleurs dominantes
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # `centers` contient les 2 couleurs (BGR)
            color1_bgr = tuple(map(int, centers[0]))
            color2_bgr = tuple(map(int, centers[1]))
            
            # Compter quelle couleur est la plus fréquente (fond)
            count1 = np.count_nonzero(labels == 0)
            count2 = np.count_nonzero(labels == 1)
            
            if count1 > count2:
                bg_color_bgr = color1_bgr
                text_color_bgr = color2_bgr
            else:
                bg_color_bgr = color2_bgr
                text_color_bgr = color1_bgr
            
            # Vérifier que les couleurs sont assez différentes
            color_dist = np.linalg.norm(np.array(bg_color_bgr) - np.array(text_color_bgr))
            
            if color_dist < 40:  # Couleurs trop similaires
                # Utiliser la méthode simple basée sur la luminosité
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                if gray_region.mean() > 127:
                    bg_color_bgr = (255, 255, 255)
                    text_color_bgr = (0, 0, 0)
                else:
                    bg_color_bgr = (0, 0, 0)
                    text_color_bgr = (255, 255, 255)
            
            # Convertir la couleur du texte en RGB pour PIL
            text_color_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])
            
            return bg_color_bgr, text_color_rgb

        except Exception as e:
            logger.error(f"Erreur détection couleur k-means: {e}. Défaut noir/blanc.")
            return (255, 255, 255), (0, 0, 0)

    
    # SUPPRIMÉ: Remplacé par detect_text_and_bg_color
    # def detect_text_color(self, image_cv, bbox): ...
    
    def translate_text(self, text):
        """
        Traduit le texte (utilisé en fallback)
        
        Args:
            text: Texte à traduire
            
        Returns:
            Texte traduit
        """
        try:
            if not text or text.strip() == '':
                return text
            
            translation = self.translator.translate(text, src=self.source_lang, dest=self.target_lang)
            return translation.text
        except Exception as e:
            logger.error(f"Erreur lors de la traduction: {e}")
            return text
    
    # NOUVEAU: Traduction par lots pour plus de rapidité
    def batch_translate_text(self, texts):
        """
        Traduit une liste de textes en une seule requête
        
        Args:
            texts: Liste de textes à traduire
            
        Returns:
            Liste de textes traduits
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Traduction de {len(texts)} textes en lot...")
            # Googletrans accepte une liste
            translations = self.translator.translate(texts, src=self.source_lang, dest=self.target_lang)
            translated_texts = [t.text for t in translations]
            logger.info("Traduction par lot terminée.")
            return translated_texts
        except Exception as e:
            logger.error(f"Erreur de traduction batch: {e}. Passage en mode individuel.")
            # Fallback vers la traduction individuelle
            return [self.translate_text(t) for t in texts]
    
    def get_text_style_settings(self, bbox_width, bbox_height):
        """
        Détermine les paramètres de style pour un texte plus naturel
        
        Args:
            bbox_width: Largeur de la zone
            bbox_height: Hauteur de la zone
            
        Returns:
            Dict avec les paramètres de style
        """
        base_size = min(bbox_height * 0.6, bbox_width * 0.2)
        
        return {
            'font_size': int(base_size),
            'stroke_width': max(1, int(base_size * 0.05)),
            'line_spacing': int(base_size * 0.2)
        }
    
    # MODIFIÉ: S'assure que text_color est un tuple (R, G, B)
    def draw_text_with_style(self, image_pil, bbox, text, text_color, font_path=None):
        """
        Dessine le texte avec un style plus naturel
        
        Args:
            image_pil: Image PIL
            bbox: Coordonnées de la boîte englobante
            text: Texte à dessiner
            text_color: Couleur du texte (tuple R, G, B)
            font_path: Chemin vers la police personnalisée
        """
        draw = ImageDraw.Draw(image_pil, 'RGBA')
        
        # S'assurer que la couleur est RGB
        if not isinstance(text_color, tuple) or len(text_color) != 3:
            logger.warning(f"Couleur de texte invalide: {text_color}. Utilisation du noir.")
            text_color = (0, 0, 0) # Noir par défaut
        
        # Calculer les dimensions
        points = np.array(bbox, dtype=np.int32)
        x_min = int(points[:, 0].min())
        x_max = int(points[:, 0].max())
        y_min = int(points[:, 1].min())
        y_max = int(points[:, 1].max())
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Obtenir les paramètres de style
        style = self.get_text_style_settings(bbox_width, bbox_height)
        
        # Charger la police
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, style['font_size'])
            else:
                font_options = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", 
                               "NotoSans-Regular.ttf", "seguiui.ttf"]
                font = None
                for font_name in font_options:
                    try:
                        font = ImageFont.truetype(font_name, style['font_size'])
                        break
                    except:
                        continue
                if font is None:
                    font = ImageFont.load_default()
        except Exception as e:
            font = ImageFont.load_default()
        
        # Découper le texte en lignes
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox_test = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox_test[2] - bbox_test[0]
            
            if text_width <= bbox_width * 0.85:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Ajuster la taille si nécessaire
        total_height = len(lines) * (style['font_size'] + style['line_spacing'])
        if total_height > bbox_height * 0.9:
            reduction_factor = (bbox_height * 0.9) / total_height
            style['font_size'] = int(style['font_size'] * reduction_factor)
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, style['font_size'])
                else:
                    for font_name in font_options:
                        try:
                            font = ImageFont.truetype(font_name, style['font_size'])
                            break
                        except:
                            continue
            except:
                font = ImageFont.load_default()
        
        # Position verticale
        y_offset = y_min + (bbox_height - total_height) // 2
        
        # Couleur du contour (basé sur la luminosité de la couleur du texte)
        luminance = 0.299*text_color[0] + 0.587*text_color[1] + 0.114*text_color[2]
        stroke_color = (255, 255, 255) if luminance < 127 else (0, 0, 0)
        
        # Dessiner chaque ligne
        for line in lines:
            bbox_line = draw.textbbox((0, 0), line, font=font)
            text_width = bbox_line[2] - bbox_line[0]
            x_offset = x_min + (bbox_width - text_width) // 2
            
            # Contour
            for offset_x in range(-style['stroke_width'], style['stroke_width'] + 1):
                for offset_y in range(-style['stroke_width'], style['stroke_width'] + 1):
                    if offset_x != 0 or offset_y != 0:
                        draw.text((x_offset + offset_x, y_offset + offset_y),
                                line, fill=stroke_color, font=font)
            
            # Texte principal
            draw.text((x_offset, y_offset), line, fill=text_color, font=font)
            y_offset += style['font_size'] + style['line_spacing']
    
    # MODIFIÉ: Flux de traitement entièrement révisé
    def process_image(self, image_path, font_path=None):
        """
        Traite une image complète (gère les images longues)
        Nouveau flux: 
        1. Détecter tout
        2. Détecter couleurs sur l'original
        3. Effacer tout (inpainting) sur une copie
        4. Traduire tout (batch)
        5. Dessiner tout
        
        Args:
            image_path: Chemin vers l'image
            font_path: Chemin vers la police personnalisée
            
        Returns:
            Image traduite
        """
        try:
            # Charger l'image
            original_image_cv = cv2.imread(str(image_path))
            if original_image_cv is None:
                logger.error(f"Impossible de charger l'image: {image_path}")
                return None
            
            # Copie pour l'effacement
            erased_image_cv = original_image_cv.copy()
            
            height, width = original_image_cv.shape[:2]
            
            # Afficher les dimensions avec des catégories
            if height <= 3000:
                img_type = "🖼️  Image courte/normale"
            elif height <= 6000:
                img_type = "📄 Image moyenne"
            elif height <= 10000:
                img_type = "📜 Image longue"
            elif height <= 20000:
                img_type = "📏 Webtoon standard"
            else:
                img_type = "🎢 Webtoon très long"
            
            logger.info(f"\n{img_type}")
            logger.info(f"📐 Dimensions: {width}x{height} pixels")
            
            # 1. Extraire le texte (avec gestion des longues images)
            ocr_results = self.extract_text_from_long_image(image_path)
            
            if not ocr_results:
                logger.warning(f"Aucun texte trouvé dans {image_path.name}")
                return Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))
            
            # Sauvegarder image debug avec détections
            if self.debug_mode:
                self.save_debug_image(original_image_cv, ocr_results, image_path.stem)
            
            
            text_regions_to_process = []
            
            # 2. & 3. Détecter couleurs (sur l'original) et Effacer (sur la copie)
            logger.info("Effacement du texte original (Inpainting)...")
            for bbox, text, confidence in ocr_results:
                if confidence > self.confidence_threshold:
                    logger.info(f"✓ Texte: '{text}' (confiance: {confidence:.2f})")
                    
                    # 2. Détecter la couleur sur l'image ORIGINALE
                    _bg_color, text_color_rgb = self.detect_text_and_bg_color(original_image_cv, bbox)
                    
                    # 3. EFFACER avec Inpainting sur l'image COPIÉE
                    # Expansion de 3-5px pour que l'inpaint "mange" un peu du texte
                    mask = self.create_text_mask(erased_image_cv.shape, bbox, expansion=5)
                    erased_image_cv = cv2.inpaint(erased_image_cv, mask, 
                                                 inpaintRadius=3, 
                                                 flags=cv2.INPAINT_TELEA)
                    
                    text_regions_to_process.append({
                        'bbox': bbox,
                        'text': text,
                        'color': text_color_rgb,
                        'confidence': confidence
                    })
                else:
                    logger.info(f"✗ Ignoré (confiance trop basse): '{text}' ({confidence:.2f})")
            
            if not text_regions_to_process:
                logger.warning("Aucun texte au-dessus du seuil de confiance.")
                return Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))

            logger.info("✅ Texte original effacé proprement")

            # 4. Traduire tous les textes en un seul lot
            original_texts = [r['text'] for r in text_regions_to_process]
            translated_texts = self.batch_translate_text(original_texts)
            
            # Combiner les traductions
            text_regions_final = []
            for region, translated_text in zip(text_regions_to_process, translated_texts):
                region['text'] = translated_text
                text_regions_final.append(region)
                logger.info(f"  '{region['text']}' → '{translated_text}'")

            # DÉDUPLICATION FINALE des régions (au cas où)
            text_regions_final = self.deduplicate_text_regions(text_regions_final)
            logger.info(f"📝 {len(text_regions_final)} textes uniques à dessiner")
            
            # Convertir l'image effacée pour PIL
            image_pil = Image.fromarray(cv2.cvtColor(erased_image_cv, cv2.COLOR_BGR2RGB))
            
            # 5. Deuxième passe : dessiner le texte traduit
            logger.info("Ajout du texte traduit...")
            for region in text_regions_final:
                self.draw_text_with_style(image_pil, region['bbox'], 
                                         region['text'], region['color'], font_path)
            
            return image_pil
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def translate_all(self, font_path=None):
        """
        Traduit toutes les images
        
        Args:
            font_path: Chemin vers la police personnalisée
        """
        image_files = self.get_image_files()
        
        if not image_files:
            logger.warning(f"Aucune image trouvée dans {self.input_folder}")
            return
        
        logger.info(f"Traitement de {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Traduction manhwa"):
            try:
                translated_image = self.process_image(image_path, font_path)
                
                if translated_image:
                    output_path = self.output_folder / image_path.name
                    # S'assurer que le format est géré (ex: convertir en PNG)
                    output_path = output_path.with_suffix('.png')
                    translated_image.save(output_path, quality=95)
                    logger.info(f"✅ Sauvegardé: {output_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Erreur: {image_path.name}: {e}")
                continue
        
        logger.info(f"Terminé! Images dans: {self.output_folder}")


def main():
    """Fonction principale"""
    print("=" * 70)
    print("  TRADUCTEUR MANHWA - Version Améliorée 🚀 (v3)")
    print("=" * 70)
    print("\n✨ Améliorations de cette version:")
    print("   • 🎨 Effacement par Inpainting (plus de 'taches')")
    print("   • 🔍 Détection OCR améliorée (normal + inversé)")
    print("   • ⚡ Traduction par lots (plus rapide)")
    print("   • 🐞 Correction de bugs critiques (détection couleur)")
    print()
    
    # Configuration
    input_folder = input("📁 Dossier contenant les images: ").strip()
    output_folder = input("💾 Dossier de sortie (défaut: 'output_traduit'): ").strip() or "output_traduit"
    
    print("\n🌍 Langues: ko (coréen), ja (japonais), zh (chinois), en (anglais)")
    source_lang = input("🔤 Langue source (défaut: ko): ").strip() or "ko"
    
    # Mode automatique par défaut
    print("\n⚙️  Mode de découpage:")
    print("   1. AUTO (recommandé) - Détection intelligente")
    print("   2. Manuel - Spécifier la hauteur")
    
    mode_choice = input("Choix (défaut: 1): ").strip() or "1"
    
    if mode_choice == "1":
        chunk_height = 'auto'
        overlap = 200
        print("✅ Mode automatique activé!")
    else:
        chunk_input = input("Hauteur de section en pixels (défaut: 2000): ").strip()
        chunk_height = int(chunk_input) if chunk_input else 2000
        
        overlap_input = input("Chevauchement entre sections (défaut: 200): ").strip()
        overlap = int(overlap_input) if overlap_input else 200
    
    # Seuil de confiance
    print("\n🎯 Seuil de confiance OCR:")
    print("   Plus bas = plus de détections (mais aussi plus de faux positifs)")
    print("   Recommandé: 0.3 à 0.4")
    confidence_input = input("Seuil (défaut: 0.3): ").strip()
    confidence_threshold = float(confidence_input) if confidence_input else 0.3
    
    # Mode debug
    print("\n🔍 Mode debug:")
    print("   Sauvegarde les images avec détections visualisées")
    debug_input = input("Activer? (o/n, défaut: n): ").strip().lower()
    debug_mode = debug_input == 'o' or debug_input == 'oui' or debug_input == 'y' or debug_input == 'yes'
    
    # Police
    print("\n🎨 Police personnalisée (optionnel)")
    font_path = input("Chemin vers police TTF (Enter pour défaut): ").strip()
    font_path = font_path if font_path and os.path.exists(font_path) else None
    
    # Créer le traducteur
    translator = ManhwaTranslatorLongImage(
        input_folder=input_folder,
        output_folder=output_folder,
        source_lang=source_lang,
        target_lang='fr',
        chunk_height=chunk_height,
        overlap=overlap,
        confidence_threshold=confidence_threshold,
        debug_mode=debug_mode
    )
    
    print("\n" + "="*70)
    print("🚀 DÉMARRAGE DE LA TRADUCTION")
    print("="*70)
    
    if chunk_height == 'auto':
        print("🤖 Mode: Détection automatique intelligente")
    else:
        print(f"⚙️  Mode: Manuel ({chunk_height}px, chevauchement {overlap}px)")
    
    print(f"🎯 Seuil de confiance: {confidence_threshold}")
    
    if debug_mode:
        print(f"🔍 Mode debug: OUI (images dans {output_folder}/debug/)")
    else:
        print("🔍 Mode debug: NON")
    
    print()
    
    translator.translate_all(font_path=font_path)
    
    print("\n" + "=" * 70)
    print("  ✅ TRADUCTION TERMINÉE!")
    print("=" * 70)
    
    if debug_mode:
        print(f"\n💡 Consultez le dossier 'debug' pour voir ce qui a été détecté")


if __name__ == "__main__":
    main()