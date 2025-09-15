# # #!/usr/bin/env python3
# # """
# # Test script for the trained face detection model
# # """

# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # import os
# # import logging
# # from tqdm import tqdm

# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)

# # def preprocess_image(image_path, target_size=(224, 224)):
# #     """
# #     Preprocess an image for model inference
# #     """
# #     # Read and resize image
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         raise ValueError(f"Could not load image: {image_path}")
    
# #     img = cv2.resize(img, target_size)
    
# #     # Convert to float32 and normalize
# #     img = img.astype(np.float32) / 255.0
    
# #     # Add batch dimension
# #     img = np.expand_dims(img, axis=0)
    
# #     return img

# # def test_single_image(model, image_path, threshold=0.5):
# #     """
# #     Test the model on a single image
# #     """
# #     try:
# #         # Preprocess image
# #         img = preprocess_image(image_path)
        
# #         # Make prediction
# #         prediction = model.predict(img, verbose=0)[0][0]
        
# #         # Get result
# #         is_target = prediction >= threshold
# #         confidence = prediction if is_target else 1 - prediction
        
# #         return is_target, confidence
        
# #     except Exception as e:
# #         logger.error(f"Error processing {image_path}: {str(e)}")
# #         return None, None

# # def visualize_result(image_path, is_target, confidence):
# #     """
# #     Visualize the detection result on the image
# #     """
# #     # Read image
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         return None
    
# #     # Add text with result
# #     result_text = f"Target: {'Yes' if is_target else 'No'} ({confidence:.2%})"
# #     color = (0, 255, 0) if is_target else (0, 0, 255)
    
# #     # Get text size and position
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     font_scale = 1
# #     thickness = 2
# #     text_size = cv2.getTextSize(result_text, font, font_scale, thickness)[0]
    
# #     # Add background rectangle for text
# #     cv2.rectangle(img, (10, 30), (10 + text_size[0], 30 + text_size[1] + 10), (255, 255, 255), -1)
    
# #     # Add text
# #     cv2.putText(img, result_text, (10, 30 + text_size[1]), font, font_scale, color, thickness)
    
# #     return img

# # def main():
# #     try:
# #         # Load the model
# #         logger.info("Loading model...")
# #         model = load_model('best_model.h5')
        
# #         # Create output directory for visualizations
# #         output_dir = "test_results"
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         # Test directory paths
# #         test_dirs = {
# #             "Positive": "test_faces",  # Directory with target person faces
# #             "Negative": "test_negative"  # Directory with other people's faces
# #         }
        
# #         # Results dictionary
# #         results = {
# #             "Positive": {"correct": 0, "total": 0},
# #             "Negative": {"correct": 0, "total": 0}
# #         }
        
# #         # Process each test directory
# #         for test_type, test_dir in test_dirs.items():
# #             if not os.path.exists(test_dir):
# #                 logger.warning(f"{test_type} test directory '{test_dir}' not found, skipping...")
# #                 continue
                
# #             logger.info(f"Processing {test_type} samples from {test_dir}...")
            
# #             # Get all jpg files in the directory
# #             image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
            
# #             for image_file in tqdm(image_files, desc=f"Testing {test_type} samples"):
# #                 image_path = os.path.join(test_dir, image_file)
                
# #                 # Test image
# #                 is_target, confidence = test_single_image(model, image_path)
                
# #                 if is_target is not None:
# #                     # Update results
# #                     results[test_type]["total"] += 1
# #                     if (test_type == "Positive" and is_target) or (test_type == "Negative" and not is_target):
# #                         results[test_type]["correct"] += 1
                    
# #                     # Visualize result
# #                     result_img = visualize_result(image_path, is_target, confidence)
# #                     if result_img is not None:
# #                         output_path = os.path.join(output_dir, f"{test_type.lower()}_{image_file}")
# #                         cv2.imwrite(output_path, result_img)
        
# #         # Print results
# #         logger.info("\nTest Results:")
# #         for test_type, counts in results.items():
# #             if counts["total"] > 0:
# #                 accuracy = counts["correct"] / counts["total"] * 100
# #                 logger.info(f"{test_type} Accuracy: {accuracy:.2f}% ({counts['correct']}/{counts['total']})")
        
# #         logger.info(f"\nVisualization results saved in '{output_dir}' directory")
        
# #     except Exception as e:
# #         logger.error(f"An error occurred: {str(e)}")
# #         raise

# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# Test script for the trained face detection model on live webcam feed
# """
# #!/usr/bin/env python3
# """
# Face detection on uploaded video using trained model
# """

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import os
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def preprocess_image(frame, target_size=(224, 224)):
#     img = cv2.resize(frame, target_size)
#     img = img.astype(np.float32) / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# def test_frame(model, frame, threshold=0.5):
#     try:
#         img = preprocess_image(frame)
#         prediction = model.predict(img, verbose=0)[0][0]
#         is_target = prediction >= threshold
#         confidence = prediction if is_target else 1 - prediction
#         return is_target, confidence
#     except Exception as e:
#         logger.error(f"Error processing frame: {str(e)}")
#         return None, None

# def visualize_result(frame, is_target, confidence):
#     result_text = f"Target: {'Yes' if is_target else 'No'} ({confidence:.2%})"
#     color = (0, 255, 0) if is_target else (0, 0, 255)
    
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     thickness = 2
#     text_size = cv2.getTextSize(result_text, font, font_scale, thickness)[0]
    
#     cv2.rectangle(frame, (10, 30), (10 + text_size[0], 30 + text_size[1] + 10), (255, 255, 255), -1)
#     cv2.putText(frame, result_text, (10, 30 + text_size[1]), font, font_scale, color, thickness)
    
#     return frame

# def process_video(model, video_path, output_path="test_results/output_video.mp4"):
#     if not os.path.exists(video_path):
#         logger.error(f"Video file '{video_path}' not found!")
#         return
    
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logger.error(f"Failed to open video: {video_path}")
#         return
    
#     # Video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 20
    
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     logger.info("Processing video...")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         is_target, confidence = test_frame(model, frame)
#         if is_target is not None:
#             frame = visualize_result(frame, is_target, confidence)
        
#         cv2.imshow("Face Detection", frame)
#         out.write(frame)
        
#         # Press q to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     logger.info(f"Processed video saved at '{output_path}'")

# def main():
#     logger.info("Loading model...")
#     model = load_model('best_model.h5')
    
#     video_path = input("Enter path to your video file (e.g., E:\\HOD_detection\\video.m3): ").strip()
#     output_path = "test_results/output_video.mp4"
    
#     process_video(model, video_path, output_path)

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(face_img, target_size=(224, 224)):
    img = cv2.resize(face_img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def test_face(model, face_img, threshold=0.5):
    try:
        img = preprocess_image(face_img)
        prediction = model.predict(img, verbose=0)[0][0]
        is_target = prediction >= threshold
        confidence = prediction if is_target else 1 - prediction
        return is_target, confidence
    except Exception as e:
        logger.error(f"Error processing face: {str(e)}")
        return None, None

def process_video(model, video_path, output_path="test_results/output_video.mp4"):
    if not os.path.exists(video_path):
        logger.error(f"Video file '{video_path}' not found!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logger.info("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            is_target, confidence = test_face(model, face_img)

            if is_target is not None:
                color = (0, 255, 0) if is_target else (0, 0, 255)
                label = f"{'Target' if is_target else 'Not Target'} ({confidence:.2%})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"Processed video saved at '{output_path}'")

def main():
    logger.info("Loading model...")
    model = load_model('best_model.h5')
    video_path = input("Enter path to your video file: ").strip()
    process_video(model, video_path)

if __name__ == "__main__":
    main()
