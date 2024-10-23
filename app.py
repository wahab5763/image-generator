import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Initialize the model pipeline only once to avoid reloading issues
@st.cache_resource
def load_pipeline():
    model_id = "dreamlike-art/dreamlike-anime-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_pipeline()

# Streamlit UI elements
st.title("Anime Image Generator")

# User inputs
prompt = st.text_input("Enter your prompt:", value="anime, masterpiece, high quality")
negative_prompt = st.text_area(
    "Negative Prompt (optional):",
    value="simple background, duplicate, retro style, low quality"
)

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating the image... Please wait."):
        try:
            # Run inference with timeout handling
            image = pipe(prompt, negative_prompt=negative_prompt).images[0]

            # Display and save the generated image
            st.image(image, caption="Generated Image", use_column_width=True)

            # Save and provide download link
            save_path = "generated_image.jpg"
            image.save(save_path)

            with open(save_path, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="anime_image.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
