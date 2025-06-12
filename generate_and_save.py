import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from typing import Optional
import gc

from model import Unet
from training_utils import DiffusionHelper


def generate_diffusion_video(checkpoint_path: str, 
                           output_path: str = "./diffusion_video.mp4",
                           num_images: int = 256,  # 16x16 grid
                           batch_size: int = 8,
                           fps: int = 60,
                           image_size: int = 128,
                           num_channels: int = 3,
                           timesteps: int = 1024,
                           final_hold_seconds: float = 3.0,  # Hold final frame for 3 seconds
                           device: Optional[str] = None) -> None:
    """
    Generate a video showing the diffusion denoising process from timestep 1024 to 0.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth file)
        output_path: Path where the output video will be saved
        num_images: Total number of images to generate (default: 256 for 16x16 grid)
        batch_size: Number of images to process at once (default: 8 for memory efficiency)
        fps: Frames per second for the output video
        image_size: Size of each generated image
        num_channels: Number of channels in the images
        timesteps: Number of diffusion timesteps
        final_hold_seconds: How long to hold the final frame (timestep 0)
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load model and diffusion helper
    model = Unet(num_channels=num_channels)
    diffusion_helper = DiffusionHelper(timesteps=timesteps, device=device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    # Calculate grid dimensions
    grid_size = int(np.sqrt(num_images))  # 16 for 256 images
    if grid_size * grid_size != num_images:
        raise ValueError(f"num_images ({num_images}) must be a perfect square")
    
    # Initialize all images with random noise
    all_images = []
    num_batches = (num_images + batch_size - 1) // batch_size
    
    print(f"Generating {num_images} images in {num_batches} batches of {batch_size}")
    
    # Generate initial noise for all images
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_images)
        current_batch_size = batch_end - batch_start
        
        noise = torch.randn((current_batch_size, num_channels, image_size, image_size), device=device)
        all_images.append(noise)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Prepare video writer
    frame_height = grid_size * image_size + 100  # Extra space for timestep text
    frame_width = grid_size * image_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Starting denoising process for {timesteps} timesteps...")
    
    # Variable to store the final frame
    final_frame = None
    
    # Denoising loop
    with torch.no_grad():
        for t_val in tqdm(reversed(range(timesteps)), desc="Denoising", total=timesteps):
            # Process each batch
            denoised_batches = []
            
            for batch_idx, batch_images in enumerate(all_images):
                current_batch_size = batch_images.shape[0]
                t = torch.full((current_batch_size,), t_val, device=device, dtype=torch.long)
                
                # Predict noise
                noise_pred = model(batch_images, t)
                
                # Get previous timestep
                denoised = diffusion_helper.get_prev_timestep(
                    xt=batch_images, 
                    noise_pred=noise_pred, 
                    t=t
                )
                
                denoised_batches.append(denoised)
            
            # Update all images
            all_images = denoised_batches
            
            # Create frame for current timestep
            frame = create_frame(all_images, grid_size, image_size, t_val, diffusion_helper)
            
            # Store the final frame (timestep 0) for later use
            if t_val == 0:
                final_frame = frame.copy()
            
            # Write frame to video
            video_writer.write(frame)
            
            # Clear cache periodically
            if t_val % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    # Add additional frames for the final timestep (timestep 0)
    if final_frame is not None:
        additional_frames = int(final_hold_seconds * fps)
        print(f"Adding {additional_frames} additional frames ({final_hold_seconds} seconds) of timestep 0...")
        
        for _ in tqdm(range(additional_frames), desc="Adding final frames"):
            video_writer.write(final_frame)
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to {output_path}")


def create_frame(all_images: list, grid_size: int, image_size: int, timestep: int, diffusion_helper: DiffusionHelper) -> np.ndarray:
    """
    Create a single frame showing the current state of all images in a grid layout.
    
    Args:
        all_images: List of image batches
        grid_size: Size of the grid (e.g., 16 for 16x16)
        image_size: Size of each individual image
        timestep: Current timestep value
        diffusion_helper: DiffusionHelper instance for denormalization
    
    Returns:
        Frame as numpy array in BGR format for OpenCV
    """
    
    # Concatenate all batches
    all_imgs_tensor = torch.cat(all_images, dim=0)
    
    # Denormalize images
    denormalized = diffusion_helper.denormalize(all_imgs_tensor.cpu())
    
    # Convert to numpy and ensure proper range [0, 1]
    images_np = denormalized.clamp(0, 1).numpy()
    
    # Create grid
    grid = np.zeros((grid_size * image_size, grid_size * image_size, 3))
    
    for i in range(grid_size):
        for j in range(grid_size):
            img_idx = i * grid_size + j
            if img_idx < len(images_np):
                # Convert from CHW to HWC format
                img = np.transpose(images_np[img_idx], (1, 2, 0))
                
                # Place image in grid
                start_h = i * image_size
                end_h = start_h + image_size
                start_w = j * image_size
                end_w = start_w + image_size
                
                grid[start_h:end_h, start_w:end_w] = img
    
    # Convert to 0-255 range
    grid = (grid * 255).astype(np.uint8)
    
    # Create frame with timestep text
    frame_height = grid_size * image_size + 100
    frame_width = grid_size * image_size
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Add grid to frame
    frame[100:, :] = grid
    
    # Add timestep text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # White text
    thickness = 3
    
    text = f"Timestep: {timestep}"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = 60
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame


def main():

    checkpoint = 'Train Data/Checkpoints/train 1 celeb/model_epoch_46.pth'
    output = 'Results/video_2.mp4'
    num_images = 64
    batch_size = 8
    fps = 60
    image_size = 64
    timesteps = 1024
    final_hold_seconds = 3.0  # Hold final frame for 3 seconds
    
    try:
        generate_diffusion_video(
            checkpoint_path=checkpoint,
            output_path=output,
            num_images=num_images,
            batch_size=batch_size,
            fps=fps,
            image_size=image_size,
            timesteps=timesteps,
            final_hold_seconds=final_hold_seconds,
        )
    except Exception as e:
        print(f"Error generating video: {e}")
        raise


if __name__ == "__main__":
    main()