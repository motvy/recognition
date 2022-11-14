models = {
    'face_detection': r"C:\All\Python\projects\recognition\models\face_detection_yunet_2022mar.onnx",
    'face_recognition': r"C:\All\Python\projects\recognition\models\face_recognition_sface_2021dec.onnx",
}

detector_params = {
    'score_threshold': 0.9,
    'nms_threshold': 0.3,
    'top_k': 5000,
}

recognition_params = {
    'cosine_similarity_threshold': 0.40,
    'l2_similarity_threshold': 1.09,
}

log_path = r"C:\All\Python\projects\recognition\logs"

visualize_flag = False