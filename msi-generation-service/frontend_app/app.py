import base64
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image

# --- Configuration ---
st.set_page_config(layout="wide", page_title="RGB to Multispectral Image Generator")

# URL of your FastAPI model server.
# If running locally in Docker, this will be http://localhost:8000
# If deployed, this will be your service's URL.
API_URL = "http://localhost:8000/predict"
MODEL_INPUT_SIZE = 256 # Should match the model's expected input size

# --- Helper Functions ---
def split_image_into_blocks(image: Image, block_size: int):
    """Yields blocks of the image of a specified size."""
    width, height = image.size
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            box = (j, i, j + block_size, i + block_size)
            yield image.crop(box), box

def stitch_blocks_together(blocks_with_coords: list, image_size: tuple, num_channels: int):
    """Stitches processed blocks back into full-sized channel images."""
    width, height = image_size
    
    # Create empty canvases for each of the 6 output channels
    stitched_channels = [Image.new('L', (width, height)) for _ in range(num_channels)]
    
    for block_channels, coords in blocks_with_coords:
        for i in range(num_channels):
            stitched_channels[i].paste(block_channels[i], coords)
            
    return stitched_channels

# --- Streamlit UI ---
st.title("🎨 RGB to Multispectral Image Generator")
st.write("Upload a high-resolution RGB image to generate its corresponding 6-channel multispectral version.")
st.info(f"The model processes the image in **{MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}** blocks.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original RGB Image")
    st.image(input_image, caption="Your uploaded image.", use_column_width=True)

    if st.button("Generate Multispectral Image"):
        # 2. Split image into blocks and prepare for processing
        blocks_generator = split_image_into_blocks(input_image, MODEL_INPUT_SIZE)
        list_of_blocks = list(blocks_generator)
        total_blocks = len(list_of_blocks)
        
        processed_blocks_with_coords = []
        
        # 3. Process each block
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (block, coords) in enumerate(list_of_blocks):
            status_text.text(f"Processing block {i+1}/{total_blocks}...")
            
            # Convert block to base64
            buffered = io.BytesIO()
            block.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Send to API
            try:
                response = requests.post(API_URL, json={"base64_str": img_base64})
                response.raise_for_status() # Raise an exception for bad status codes
                
                # Decode response
                result = response.json()
                output_channels_b64 = result['channels_base64']
                
                decoded_channels = []
                for ch_b64 in output_channels_b64:
                    img_bytes = base64.b64decode(ch_b64)
                    decoded_channels.append(Image.open(io.BytesIO(img_bytes)))
                
                processed_blocks_with_coords.append((decoded_channels, coords))

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the model API. Please ensure the backend is running. Error: {e}")
                break # Exit the loop on failure
                
            progress_bar.progress((i + 1) / total_blocks)
        
        status_text.text("Processing complete! Stitching results...")

        # 4. Stitch the results back together
        num_channels = len(processed_blocks_with_coords[0][0]) if processed_blocks_with_coords else 0
        if num_channels > 0:
            final_channel_images = stitch_blocks_together(
                processed_blocks_with_coords, input_image.size, num_channels
            )

            # 5. Display the final stitched images
            st.subheader("Generated Multispectral Channels")
            
            cols = st.columns(3) # Display in 3 columns
            for i, channel_img in enumerate(final_channel_images):
                with cols[i % 3]:
                    st.image(channel_img, caption=f"Generated Channel {i+1}", use_column_width=True)
            status_text.success("Done!")