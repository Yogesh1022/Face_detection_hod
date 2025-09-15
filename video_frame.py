import cv2
import os
import time
import glob

def extract_frames(video_path, output_dir="frames"):
    """
    Extract all frames from a video
    """
    os.makedirs(output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n🎬 Processing: {os.path.basename(video_path)}")
    print(f"  FPS: {fps:.2f} | Duration: {duration:.2f}s | Total Frames: {total_frames}")
    print(f"  Extracting all frames")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        saved_count += 1
        frame_count += 1

    video.release()
    print(f"✅ Saved {saved_count} frames from {os.path.basename(video_path)}")

def main():
    video_folder = "Frame_data"
    output_folder = "frames"
    os.makedirs(output_folder, exist_ok=True)

    # Get all video files
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))

    if not video_files:
        print("No video files found!")
        return

    start = time.time()

    # Process all videos
    for video_path in video_files:
        extract_frames(video_path, output_dir=output_folder)  # Extract all frames

    elapsed = time.time() - start
    print(f"\n⏱ Done! Processed {len(video_files)} videos in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
