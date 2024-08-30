import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Parameters for the license plate
plate_text = "ABC123"
plate_size = (250, 100)
frame_size = (640, 480)
n_frames = 10
movement = (frame_size[0] - plate_size[0], 0)  # Move the plate from left to right, staying within the frame

# Create the license plate image (sharp)
def create_license_plate(text, size):
    plate = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255  # white plate
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    cv2.putText(plate, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return plate

# Generate a set of frames with a blurry moving license plate
def generate_frames_with_moving_plate(plate, frame_size, movement, n_frames):
    frames = []
    for i in range(n_frames):
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # white background
        x_offset = int(i * movement[0] / n_frames)
        y_offset = int(i * movement[1] / n_frames)
        frame[y_offset:y_offset+plate.shape[0], x_offset:x_offset+plate.shape[1]] = plate
        frame = cv2.GaussianBlur(frame, (21, 21), 0)  # blur the frame to simulate motion blur
        frames.append(frame)
    return frames

# Create the sharp license plate
sharp_plate = create_license_plate(plate_text, plate_size)

# Generate the frames
frames = generate_frames_with_moving_plate(sharp_plate, frame_size, movement, n_frames)

# Save the generated frames and the sharp plate for inspection
output_dir = "frames"
# Delete the directory if it is there to clear old frames
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Ensure the aligned_frames directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Save the frames
for idx, frame in enumerate(frames):
    cv2.imwrite(f"{output_dir}/frame_{idx + 1}.png", frame)

# Save the sharp plate as ground truth
cv2.imwrite(f"ground_truth_plate.png", sharp_plate)
