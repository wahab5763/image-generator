import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

# Load the Stable Diffusion model
model_id = "dreamlike-art/dreamlike-anime-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

st.title("Anime Image Generator")

# User input for the prompt
prompt = st.text_input("Enter your prompt:", value="anime, masterpiece, high quality")

# Optional: Negative prompt to avoid specific elements
negative_prompt = st.text_area(
    "Negative Prompt (optional):", 
    value="simple background, duplicate, retro style, low quality"
)

# Button to generate the image
if st.button("Generate Image"):
    with st.spinner("Generating..."):
        try:
            # Generate the image using the prompt
            image = pipe(prompt, negative_prompt=negative_prompt).images[0]

            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)

            # Save the image temporarily
            save_path = "./generated_image.jpg"
            image.save(save_path)

            # Provide download option
            with open(save_path, "rb") as file:
                btn = st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="anime_image.jpg",
                    mime="image/jpeg"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
