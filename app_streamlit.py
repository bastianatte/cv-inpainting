import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from inpaint import load_pipeline, inpaint

st.set_page_config(page_title="Inpainting Diffusion", layout="wide")
st.title("🖌️ Inpainting & Object Removal (Diffusion)")

with st.sidebar:
    st.header("Parametri")
    prompt = st.text_input("Prompt (opzionale)", "clean background, realistic")
    negative_prompt = st.text_input("Negative prompt (opz.)", "blurry, low quality, artifacts")
    steps = st.slider("num_inference_steps", 10, 80, 30, 1)
    guidance = st.slider("guidance_scale", 1.0, 12.0, 7.5, 0.5)
    brush = st.slider("Pennello (px)", 5, 80, 30, 1)
    seed = st.number_input("Seed (opz.)", value=0, step=1)
    use_seed = st.checkbox("Usa seed", value=False)

@st.cache_resource
def get_pipe():
    return load_pipeline()

pipe = get_pipe()

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.subheader("1) Carica immagine")
    file = st.file_uploader("Upload", type=["png","jpg","jpeg","webp"])
    if file is None:
        st.stop()
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Originale", use_container_width=True)

with col2:
    st.subheader("2) Disegna maschera (bianco = rimuovi)")
    W, H = image.size
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=brush,
        stroke_color="#FFFFFF",
        background_image=image,
        update_streamlit=True,
        height=H,
        width=W,
        drawing_mode="freedraw",
        key="canvas"
    )
    mask = None
    if canvas_result.image_data is not None:
        # l'immagine del canvas è RGBA; maschera = canale alfa > 0 o pixel bianchi disegnati
        arr = canvas_result.image_data.astype(np.uint8)
        # prendi layer disegnato: differenza dal background (qui semplifichiamo: tutto ciò che non è trasparente è maschera)
        # streamlit-drawable-canvas già scrive il tratto come bianco opaco
        mask = Image.fromarray(arr[...,0])  # usa il canale R come base
        # binarizza
        mask = ImageOps.grayscale(mask).point(lambda p: 255 if p > 10 else 0).convert("L")
        st.image(mask, caption="Maschera", use_container_width=True)

with col3:
    st.subheader("3) Risultato")
    if mask is not None and st.button("Esegui Inpainting", type="primary"):
        out = inpaint(
            pipe, image, mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=int(seed) if use_seed else None
        )
        st.image(out, caption="Output", use_container_width=True)
        # download
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        st.download_button("Scarica PNG", data=buf.getvalue(), file_name="inpainted.png", mime="image/png")
