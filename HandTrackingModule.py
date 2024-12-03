import mediapipe as mp
import cv2
import time
import numpy as np
from typing import List, Tuple, Optional, Dict


class HandDetector:
    """
    Une classe pour détecter et suivre les mains dans des images ou flux vidéo en utilisant MediaPipe.

    Cette classe fournit des fonctionnalités pour :
    - Détecter les mains dans une image
    - Suivre les points de repère des mains
    - Calculer les distances entre les points de repère
    - Mesurer la distance entre deux mains

    Attributes:
        static_image_mode (bool): Si True, traite chaque image indépendamment
        max_num_hands (int): Nombre maximum de mains à détecter
        min_detection_confidence (float): Seuil minimal de confiance pour la détection
        min_tracking_confidence (float): Seuil minimal de confiance pour le suivi
        detection_color (Tuple[int, int, int]): Couleur BGR pour le dessin des connexions
        landmark_color (Tuple[int, int, int]): Couleur BGR pour le dessin des points de repère
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        detection_color: Tuple[int, int, int] = (0, 255, 0),
        landmark_color: Tuple[int, int, int] = (255, 0, 0),
    ):
        """
        Initialise le détecteur de mains avec les paramètres spécifiés.

        Args:
            static_image_mode: Si True, traite chaque image indépendamment
            max_num_hands: Nombre maximum de mains à détecter
            min_detection_confidence: Seuil minimal de confiance pour la détection
            min_tracking_confidence: Seuil minimal de confiance pour le suivi
            detection_color: Couleur BGR pour le dessin des connexions
            landmark_color: Couleur BGR pour le dessin des points de repère
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detection_color = detection_color
        self.landmark_color = landmark_color

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.fps = 0
        self.prev_time = 0

    def findHands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Détecte les mains dans une image et dessine optionnellement les repères.

        Args:
            img: Image d'entrée au format BGR (OpenCV)
            draw: Si True, dessine les points de repère et les connexions sur l'image

        Returns:
            Image avec les annotations si draw=True, sinon image originale
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=self.detection_color),
                        self.mpDraw.DrawingSpec(color=self.detection_color),
                    )
        return img

    def findPosition(
        self, img: np.ndarray, handNo: int = 0, draw: bool = True
    ) -> List[List[int]]:
        """
        Trouve les positions des points de repère pour une main spécifique.

        Args:
            img: Image d'entrée
            handNo: Index de la main à analyser (0 pour la première main détectée)
            draw: Si True, dessine les points sur l'image

        Returns:
            Liste des positions [id, x, y] pour chaque point de repère de la main
        """
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, self.landmark_color, cv2.FILLED)
        return lmList

    def getAllHandsPositions(
        self, img: np.ndarray, draw: bool = True
    ) -> Dict[str, List[List[int]]]:
        """
        Récupère les positions pour toutes les mains détectées et les classe par type (gauche/droite).

        Args:
            img: Image d'entrée
            draw: Si True, dessine les points sur l'image

        Returns:
            Dictionnaire avec les positions pour chaque main {'Left': [...], 'Right': [...]}
            Chaque position est une liste [id, x, y]
        """
        hands_positions = {"Left": [], "Right": []}

        if self.results and self.results.multi_hand_landmarks:
            for idx, (hand_landmarks, hand_info) in enumerate(
                zip(self.results.multi_hand_landmarks, self.results.multi_handedness)
            ):
                hand_type = hand_info.classification[0].label
                positions = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    positions.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, self.landmark_color, cv2.FILLED)
                hands_positions[hand_type] = positions

        return hands_positions

    def calculateDistance(self, p1: List[int], p2: List[int]) -> float:
        """
        Calcule la distance euclidienne entre deux points.

        Args:
            p1: Premier point [id, x, y]
            p2: Second point [id, x, y]

        Returns:
            Distance en pixels entre les deux points
        """
        return np.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def measureHandsDistance(
        self, img: np.ndarray, landmark_id: int = 4, draw: bool = True
    ) -> Optional[float]:
        """
        Mesure la distance entre un point spécifique des deux mains.

        Args:
            img: Image d'entrée
            landmark_id: ID du point à mesurer (4 pour le pouce par défaut)
            draw: Si True, dessine une ligne entre les points et affiche la distance

        Returns:
            Distance en pixels entre les points spécifiés des deux mains,
            ou None si moins de 2 mains sont détectées
        """
        hands_positions = self.getAllHandsPositions(img, draw=False)

        if len(hands_positions["Left"]) > 0 and len(hands_positions["Right"]) > 0:
            left_point = hands_positions["Left"][landmark_id]
            right_point = hands_positions["Right"][landmark_id]

            if draw:
                cv2.line(
                    img,
                    (left_point[1], left_point[2]),
                    (right_point[1], right_point[2]),
                    (0, 255, 255),
                    2,
                )

                distance = self.calculateDistance(left_point, right_point)
                center_point = (
                    (left_point[1] + right_point[1]) // 2,
                    (left_point[2] + right_point[2]) // 2,
                )
                cv2.putText(
                    img,
                    f"{int(distance)}px",
                    center_point,
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 255),
                    2,
                )

            return distance
        return None

    def updateFPS(self) -> float:
        """
        Met à jour et retourne le taux de rafraîchissement actuel (FPS).

        Returns:
            FPS actuel
        """
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps


def main():
    """
    Fonction principale de démonstration du détecteur de mains.

    Initialise une capture vidéo et affiche en temps réel :
    - Les mains détectées avec leurs points de repère
    - La distance entre les pouces des deux mains
    - La distance entre les centres des paumes
    - Le taux de rafraîchissement (FPS)

    Pour quitter, appuyez sur 'q'.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = HandDetector(detection_color=(0, 255, 0), landmark_color=(255, 0, 0))

    while True:
        success, img = cap.read()
        if not success:
            print("Échec de la capture vidéo")
            break

        img = detector.findHands(img)

        thumb_distance = detector.measureHandsDistance(img, landmark_id=4, draw=True)
        palm_distance = detector.measureHandsDistance(img, landmark_id=0, draw=True)

        fps = detector.updateFPS()
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 255),
            2,
        )

        cv2.imshow("Hand Tracking", img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
