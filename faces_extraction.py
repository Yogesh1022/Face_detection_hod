import cv2
import os
import glob
import mediapipe as mp

# Folders
frames_folder = "frames"
output_folder = "train_data"
os.makedirs(output_folder, exist_ok=True)

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

face_count = 0

# Loop through all images in frames folder
for img_path in glob.glob(os.path.join(frames_folder, "*.jpg")):
    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            # Get bounding box relative coordinates
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)

            # Clip values to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_min + box_w)
            y_max = min(h, y_min + box_h)

            # Crop and save face
            face_img = image[y_min:y_max, x_min:x_max]
            if face_img.size == 0:
                continue
            face_filename = os.path.join(output_folder, f"face_{face_count:05d}.jpg")
            cv2.imwrite(face_filename, face_img)
            face_count += 1

    if face_count % 10 == 0 and face_count > 0:
        print(f"Saved {face_count} faces so far...")

print(f"\n✅ Done! Extracted {face_count} faces into '{output_folder}'")

face_detection.close()
