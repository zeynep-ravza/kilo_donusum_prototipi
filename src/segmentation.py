import cv2
import mediapipe as mp
import numpy as np

def segment_body(image_rgb, threshold=0.3, draw_pose=False):
    mp_seg = mp.solutions.selfie_segmentation
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 🔧 Ön işleme: kontrast ve parlaklık artırma
    image_rgb = enhance_image(image_rgb)

    # Segmentasyon işlemi
    with mp_seg.SelfieSegmentation(model_selection=1) as segmenter:
        seg_result = segmenter.process(image_rgb)
        raw_mask = seg_result.segmentation_mask
        mask = (raw_mask > threshold).astype(np.uint8) * 255

    # ⚠️ Maske boşsa uyar
    if mask.sum() == 0:
        print("\u26a0\ufe0f Maske boş: Vücut segmentasyonu başarısız oldu!")

    # Poz gösterimi isteniyorsa
    if draw_pose:
        with mp_pose.Pose(static_image_mode=True) as pose:
            pose_result = pose.process(image_rgb)
            annotated_image = image_rgb.copy()
            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
            return mask, annotated_image

    return mask, image_rgb

def enhance_image(image_rgb):
    """
    Kontrast ve parlaklığı iyileştirerek segmentasyon için daha belirgin hale getirir.
    """
    # YUV renk uzayına çevir ve histogram eşitleme uygula
    img_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Y kanalı (parlaklık)
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Hafif kontrast artırma (gamma correction)
    gamma = 1.1
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_img = cv2.LUT(enhanced_img, look_up_table)

    return enhanced_img
