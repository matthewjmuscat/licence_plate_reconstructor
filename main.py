import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import shutil
import re


# Define a simple VSR model (This is a basic example)
class BasicVSRNet(nn.Module):
    def __init__(self):
        super(BasicVSRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)
        return x

class RotatableRectangleSelector:
    def __init__(self, ax, img):
        self.ax = ax
        self.img = img
        self.rect = None
        self.angle = 0
        self.start_point = None
        self.end_point = None
        self.dragging = False
        self.rectangle_patch = None

        # Connect to the matplotlib events
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Add a button to finalize the selection
        self.finalize_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.finalize_button = Button(self.finalize_button_ax, 'Finalize')
        self.finalize_button.on_clicked(self.finalize)

        self.finalized = False

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (event.xdata, event.ydata)
        self.end_point = self.start_point
        self.rect = None
        self.dragging = True
        self.draw_rectangle()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.end_point = (event.xdata, event.ydata)
        self.dragging = False
        self.draw_rectangle()

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        self.end_point = (event.xdata, event.ydata)
        self.draw_rectangle()

    def draw_rectangle(self):
        if self.rectangle_patch:
            self.rectangle_patch.remove()

        x0, y0 = self.start_point
        x1, y1 = self.end_point
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        self.rect = (min(x0, x1), min(y0, y1), width, height)

        self.rectangle_patch = Rectangle((self.rect[0], self.rect[1]), width, height,
                                         linewidth=1, edgecolor='r', facecolor='none', angle=self.angle)
        self.ax.add_patch(self.rectangle_patch)
        self.ax.figure.canvas.draw()

    def finalize(self, event):
        self.finalized = True
        plt.close()

    def get_final_rect(self):
        if not self.finalized:
            return None
        return self.rect

def manual_rotatable_roi_selection(image):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    selector = RotatableRectangleSelector(ax, image)
    plt.show()

    return selector.get_final_rect()

def load_frames(frame_paths):
    frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(frame))
    return torch.stack(frames)

def visualize_optical_flow(flow, reference_roi):
    # Convert flow to an RGB image for visualization
    hsv = np.zeros_like(cv2.cvtColor(reference_roi, cv2.COLOR_RGB2BGR))
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical Flow', flow_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_remapped_roi(aligned_roi_np):
    plt.imshow(cv2.cvtColor(aligned_roi_np, cv2.COLOR_RGB2BGR))
    plt.title("Remapped ROI")
    plt.axis("off")
    plt.show()


import matplotlib.pyplot as plt

def visualize_image(image, title="Image", cmap=None):
    plt.imshow(image if cmap is None else image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_optical_flow_vectors(flow, reference_frame, current_frame):
    # Draw the flow vectors on the image that blends reference and current frames
    h, w = flow.shape[:2]
    
    # Blend reference and current frames for better visualization
    blended_frame = cv2.addWeighted(reference_frame, 0.5, current_frame, 0.5, 0)

    # Overlay flow vectors on the blended frame
    flow_image = blended_frame.copy()
    step = 4  # Step size for sampling vectors
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(flow_image, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.3)
    
    # Visualize the result
    visualize_image(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB), title="Optical Flow Vectors")


def align_frames_with_optical_flow(frames, roi, vis_opt_flow=True):
    x, y, w, h = map(int, roi)
    aligned_frames = []
    reference_frame = frames[0].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to NumPy
    reference_roi = reference_frame[y:y+h, x:x+w]  # Original ROI

    # Visualize the original reference frame
    visualize_image(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB), title="Reference Frame")

    for i, frame in enumerate(frames):
        frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # Convert tensor to NumPy

        # Visualize the current frame
        visualize_image(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB), title=f"Current Frame {i + 1}")

        if i == 0:
            aligned_frames.append(to_tensor(reference_roi))  # Append the original ROI for the first frame
            visualize_image(cv2.cvtColor(reference_roi, cv2.COLOR_BGR2RGB), title="Original ROI")
            continue

        # Calculate optical flow from reference full frame to current full frame
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(reference_frame, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY),
            None,
            pyr_scale=0.1,  # Pyramid scale: Image scale (<1) to build pyramids for each image
            levels=5,       # Number of pyramid layers
            winsize=100,     # Averaging window size
            iterations=10,   # Number of iterations the algorithm does at each pyramid level
            poly_n=100,       # Size of the pixel neighborhood
            poly_sigma=1.1, # Standard deviation of the Gaussian used to smooth derivatives
            flags=0
        )

        if vis_opt_flow:
            visualize_optical_flow_vectors(flow, reference_frame, frame_np)  # Visualize flow vectors on blended image

        # Create the remapping coordinates for the full frame
        h_full, w_full = reference_frame.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w_full), np.arange(h_full))

        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        # Remap the full current frame to align with the reference frame
        remapped_frame_np = cv2.remap(frame_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Visualize the remapped full frame
        visualize_image(cv2.cvtColor(remapped_frame_np, cv2.COLOR_BGR2RGB), title=f"Remapped Frame {i + 1}")

        # Extract the ROI from the remapped full frame
        aligned_roi_np = remapped_frame_np[y:y+h, x:x+w]

        # Visualize the extracted ROI
        visualize_image(cv2.cvtColor(aligned_roi_np, cv2.COLOR_BGR2RGB), title=f"Aligned ROI {i + 1}")

        if np.all(aligned_roi_np == 0):
            print(f"Frame {i + 1}: All pixels are black after remapping.")
        elif np.any(np.isnan(aligned_roi_np)):
            print(f"Frame {i + 1}: Found NaN values after remapping.")
        else:
            aligned_roi_tensor = to_tensor(aligned_roi_np)
            aligned_frames.append(aligned_roi_tensor)

    return torch.stack(aligned_frames)





def save_aligned_frames(aligned_frames, frame_paths, aligned_dir):
    os.makedirs(aligned_dir, exist_ok=True)
    for i, aligned_frame in enumerate(aligned_frames):
        aligned_frame_np = aligned_frame.cpu().numpy().transpose(1, 2, 0)
        aligned_frame_bgr = cv2.cvtColor(aligned_frame_np, cv2.COLOR_RGB2BGR)
        aligned_frame_path = os.path.join(aligned_dir, os.path.basename(frame_paths[i]))
        cv2.imwrite(aligned_frame_path, aligned_frame_bgr)

def multi_frame_super_resolve(frame_paths, roi, model, aligned_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load and stack frames
    frames = load_frames(frame_paths).to(device)

    # Align frames using optical flow and the selected ROI
    aligned_frames = align_frames_with_optical_flow(frames, roi)

    # Save the aligned frames for further processing or inspection
    save_aligned_frames(aligned_frames, frame_paths, aligned_dir)

    # Perform multi-frame super-resolution
    with torch.no_grad():
        sr_image = model(aligned_frames.mean(dim=0, keepdim=True).to(device))  # Ensure tensor is on the correct device

    # Convert back to PIL image and save
    sr_image = to_pil_image(sr_image.squeeze().cpu())
    sr_image.save('multi_frame_super_resolved_image.png')


# Load and sort the frames numerically
def numeric_sort_key(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

# Paths
frames_dir = 'frames'
aligned_dir = 'aligned_frames'

# Delete the directory if it is there to clear old frames
if os.path.exists(aligned_dir):
    shutil.rmtree(aligned_dir)

# Ensure the aligned_frames directory exists
if not os.path.exists(aligned_dir):
    os.makedirs(aligned_dir)

# Load and sort the frames
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png') or f.endswith('.jpg')],
                     key=numeric_sort_key)

if not frame_files:
    raise FileNotFoundError(f"No image files found in the directory '{frames_dir}'.")

# Load the first frame after sorting
first_frame_path = os.path.join(frames_dir, frame_files[0])
first_frame = cv2.imread(first_frame_path)

# Check if the image was loaded successfully
if first_frame is None:
    raise FileNotFoundError(f"Image at path '{first_frame_path}' could not be loaded. Check the file path.")

# Use the function to select and rotate ROI
roi = manual_rotatable_roi_selection(first_frame)

if roi is not None:
    # Create a basic VSR model instance
    vsr_model = BasicVSRNet()

    # Apply multi-frame super-resolution to the set of aligned frames
    frame_paths = [os.path.join(frames_dir, frame) for frame in frame_files]  # Corrected: Load from frames_dir
    multi_frame_super_resolve(frame_paths, roi, vsr_model, aligned_dir)

    print("Super-resolved image saved as 'multi_frame_super_resolved_image.png'")
else:
    print("No region was selected.")
