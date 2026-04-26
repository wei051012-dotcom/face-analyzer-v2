import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from pathlib import Path


class FaceAnalyzer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).with_name("face_landmarker.task")
        model_path = str(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def analyze_image(self, image_path):
        image = mp.Image.create_from_file(image_path)
        result = self.detector.detect(image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        points = []
        for lm in landmarks:
            points.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })

        return {
            "landmarks": points,
            "blendshapes": result.face_blendshapes,
            "matrixes": result.facial_transformation_matrixes
        }
    def calculate_distance(self, p1, p2, width, height):
        x1, y1 = p1.x * width, p1.y * height
        x2, y2 = p2.x * width, p2.y * height
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def get_point(self, landmarks, idx, width, height):
        return np.array([landmarks[idx].x * width, landmarks[idx].y * height])

    def analyze(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None, image

        landmarks = results.face_landmarks[0]
        h, w, _ = image.shape
        
        # Landmarks Selection
        # Face
        top = self.get_point(landmarks, 10, w, h)
        bottom = self.get_point(landmarks, 152, w, h)
        left = self.get_point(landmarks, 234, w, h)
        right = self.get_point(landmarks, 454, w, h)
        jaw_left = self.get_point(landmarks, 132, w, h)
        jaw_right = self.get_point(landmarks, 361, w, h)
        forehead_left = self.get_point(landmarks, 71, w, h)
        forehead_right = self.get_point(landmarks, 301, w, h)
        
        # Eyes
        # Left Eye (from camera perspective)
        left_eye_inner = self.get_point(landmarks, 133, w, h)
        left_eye_outer = self.get_point(landmarks, 33, w, h)
        left_eye_top = self.get_point(landmarks, 159, w, h)
        left_eye_bottom = self.get_point(landmarks, 145, w, h)
        
        # Right Eye
        right_eye_inner = self.get_point(landmarks, 362, w, h)
        right_eye_outer = self.get_point(landmarks, 263, w, h)
        right_eye_top = self.get_point(landmarks, 386, w, h)
        right_eye_bottom = self.get_point(landmarks, 374, w, h)
        
        # Eyebrows
        # Left Eyebrow
        left_brow_inner = self.get_point(landmarks, 46, w, h)
        left_brow_peak = self.get_point(landmarks, 105, w, h)
        left_brow_outer = self.get_point(landmarks, 107, w, h)
        
        # Right Eyebrow
        right_brow_inner = self.get_point(landmarks, 276, w, h)
        right_brow_peak = self.get_point(landmarks, 334, w, h)
        right_brow_outer = self.get_point(landmarks, 336, w, h)
        
        # Nose
        nose_bridge = self.get_point(landmarks, 168, w, h)
        nose_tip = self.get_point(landmarks, 1, w, h)
        nose_left = self.get_point(landmarks, 129, w, h)
        nose_right = self.get_point(landmarks, 358, w, h)
        
        # Lips
        mouth_left = self.get_point(landmarks, 61, w, h)
        mouth_right = self.get_point(landmarks, 291, w, h)
        upper_lip_top = self.get_point(landmarks, 0, w, h)
        upper_lip_bottom = self.get_point(landmarks, 13, w, h)
        lower_lip_top = self.get_point(landmarks, 14, w, h)
        lower_lip_bottom = self.get_point(landmarks, 17, w, h)

        # --- Rule-based Calculations ---
        analysis = {}

        # 1. Face Shape
        face_height = np.linalg.norm(bottom - top)
        face_width = np.linalg.norm(right - left)
        jaw_width = np.linalg.norm(jaw_right - jaw_left)
        forehead_width = np.linalg.norm(forehead_right - forehead_left)
        
        ratio_hw = face_height / face_width
        ratio_jf = jaw_width / forehead_width
        
        if ratio_hw > 1.5:
            analysis["face_shape"] = "長臉 (Long)"
        elif forehead_width > jaw_width * 1.3:
            analysis["face_shape"] = "心形臉 (Heart)"
        elif ratio_hw < 1.25 and ratio_jf > 0.9:
            analysis["face_shape"] = "方臉 (Square)"
        elif ratio_hw < 1.25:
            analysis["face_shape"] = "圓臉 (Round)"
        else:
            analysis["face_shape"] = "橢圓臉 (Oval)"

        # 2. Eye Shape (Average of both eyes)
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_bottom - left_eye_top)
        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_bottom - right_eye_top)
        
        eye_ratio = ((left_eye_width/left_eye_height) + (right_eye_width/right_eye_height)) / 2
        
        # Tilt (y decreases upwards in images)
        left_tilt = left_eye_inner[1] - left_eye_outer[1]
        right_tilt = right_eye_inner[1] - right_eye_outer[1]
        avg_tilt = (left_tilt + right_tilt) / 2
        
        if avg_tilt > 5:  # Outer corner higher (y is smaller)
            analysis["eye_shape"] = "上揚眼 (Upturned)"
        elif avg_tilt < -5: # Outer corner lower
            analysis["eye_shape"] = "下垂眼 (Downturned)"
        elif eye_ratio < 2.0:
            analysis["eye_shape"] = "圓眼 (Round)"
        elif eye_ratio > 3.0:
            analysis["eye_shape"] = "細長眼 (Slender)"
        else:
            analysis["eye_shape"] = "杏仁眼 (Almond)"

        # 3. Eyebrow Shape
        left_brow_tilt = left_brow_inner[1] - left_brow_outer[1]
        right_brow_tilt = right_brow_inner[1] - right_brow_outer[1]
        avg_brow_tilt = (left_brow_tilt + right_brow_tilt) / 2
        
        left_arch = left_brow_inner[1] - left_brow_peak[1]
        right_arch = right_brow_inner[1] - right_brow_peak[1]
        avg_arch = (left_arch + right_arch) / 2
        
        if avg_arch > 10:
            analysis["eyebrow_shape"] = "拱眉 (Arched)"
        elif avg_brow_tilt > 10:
            analysis["eyebrow_shape"] = "上揚眉 (Upturned)"
        elif avg_brow_tilt < -10:
            analysis["eyebrow_shape"] = "下垂眉 (Downturned)"
        else:
            analysis["eyebrow_shape"] = "平眉 (Straight)"

        # 4. Nose Shape
        nose_width = np.linalg.norm(nose_right - nose_left)
        nose_length = np.linalg.norm(nose_tip - nose_bridge)
        nose_ratio = nose_width / face_width
        
        # Simplified bridge estimate (distance between inner eyes vs bridge width)
        inter_eye_dist = np.linalg.norm(right_eye_inner - left_eye_inner)
        bridge_ratio = nose_length / inter_eye_dist
        
        if nose_ratio > 0.3:
            analysis["nose_shape"] = "寬鼻 (Wide)"
        elif nose_ratio < 0.2:
            analysis["nose_shape"] = "窄鼻 (Narrow)"
        elif bridge_ratio > 1.2:
            analysis["nose_shape"] = "高鼻樑 (High bridge)"
        elif bridge_ratio < 0.8:
            analysis["nose_shape"] = "低鼻樑 (Low bridge)"
        else:
            analysis["nose_shape"] = "中等鼻 (Medium)"

        # 5. Lips
        upper_lip = np.linalg.norm(upper_lip_bottom - upper_lip_top)
        lower_lip = np.linalg.norm(lower_lip_bottom - lower_lip_top)
        total_lip = upper_lip + lower_lip
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        lip_face_ratio = total_lip / face_height
        
        if upper_lip > lower_lip * 1.3:
            analysis["lips"] = "上唇較厚 (Thicker upper lip)"
        elif lower_lip > upper_lip * 1.5:
            analysis["lips"] = "下唇較厚 (Thicker lower lip)"
        elif lip_face_ratio > 0.12:
            analysis["lips"] = "厚唇 (Thick)"
        elif lip_face_ratio < 0.08:
            analysis["lips"] = "薄唇 (Thin)"
        else:
            analysis["lips"] = "中等唇 (Medium)"

        # 6. Eye-distance ratios
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        analysis["eye_width_eye_dist_ratio"] = round(avg_eye_width / inter_eye_dist, 3)
        analysis["nose_eye_ratio"] = round(nose_width / inter_eye_dist, 3)
        analysis["mouth_eye_ratio"] = round(mouth_width / inter_eye_dist, 3)

        # --- Draw Annotations ---
        annotated_image = image.copy()
        
        # Helper to draw lines
        def draw_pts(pts, color, closed=False):
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_image, [pts], closed, color, 2)
            
        # Draw Face
        draw_pts([top, left, bottom, right], (0, 255, 0), True)
        
        # Draw Eyes
        draw_pts([left_eye_inner, left_eye_top, left_eye_outer, left_eye_bottom], (255, 0, 0), True)
        draw_pts([right_eye_inner, right_eye_top, right_eye_outer, right_eye_bottom], (255, 0, 0), True)
        
        # Draw Eyebrows
        draw_pts([left_brow_inner, left_brow_peak, left_brow_outer], (0, 0, 255))
        draw_pts([right_brow_inner, right_brow_peak, right_brow_outer], (0, 0, 255))
        
        # Draw Nose
        draw_pts([nose_bridge, nose_tip, nose_left, nose_right, nose_tip], (0, 255, 255))
        
        # Draw Lips
        draw_pts([mouth_left, upper_lip_top, mouth_right, lower_lip_bottom], (255, 0, 255), True)

        return analysis, annotated_image
