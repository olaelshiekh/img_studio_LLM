import os
import base64
import io
import time
import requests
import streamlit as st
from typing import Optional
from PIL import Image
from huggingface_hub import InferenceClient

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen Vision Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #0b0c10;
    --surface:  #13151c;
    --card:     #1a1d28;
    --border:   #252836;
    --accent:   #7c6af7;
    --accent2:  #f76ac8;
    --green:    #43e8a4;
    --text:     #e8eaf6;
    --muted:    #6b7280;
    --radius:   14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; height: 0; }
[data-testid="stToolbar"] button { visibility: visible !important; height: auto !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Top title bar */
.title-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.title-bar h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    padding: 0 !important;
}
.badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 50px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Mode pills */
.mode-pills { display: flex; gap: 10px; margin-bottom: 20px; }
.pill {
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    border: 1.5px solid var(--border);
    background: var(--card);
    color: var(--muted);
    transition: all .2s;
}
.pill.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(124,106,247,.12);
}

/* Chat bubbles */
.chat-wrap { display: flex; flex-direction: column; gap: 16px; padding-bottom: 20px; }

.msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #3b2fa0, #5a3580);
    border: 1px solid rgba(124,106,247,.35);
    border-radius: var(--radius) var(--radius) 4px var(--radius);
    padding: 14px 18px;
    max-width: 75%;
    color: var(--text);
    font-size: 0.88rem;
    line-height: 1.6;
}
.msg-bot {
    align-self: flex-start;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) var(--radius) 4px;
    padding: 14px 18px;
    max-width: 82%;
    color: var(--text);
    font-size: 0.88rem;
    line-height: 1.6;
}
.msg-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 6px;
    font-weight: 600;
}
.msg-label.user { color: var(--accent); }
.msg-label.bot  { color: var(--green); }

/* Input area */
.stTextArea textarea {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,.2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 10px 24px !important;
    letter-spacing: 0.04em !important;
    transition: opacity .2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] * { color: var(--text) !important; }

/* Selectbox / radio */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stRadio > div { gap: 12px !important; }
.stRadio label { color: var(--text) !important; font-size: 0.85rem !important; }

/* Slider */
.stSlider > div > div > div { background: var(--accent) !important; }

/* Info / success / error boxes */
.stInfo, .stSuccess, .stError, .stWarning {
    border-radius: var(--radius) !important;
    border: none !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Generated image container */
.gen-img-wrap {
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--card);
    padding: 12px;
    margin-top: 12px;
}

/* Scrollable chat area */
.chat-scroll {
    max-height: 520px;
    overflow-y: auto;
    padding-right: 8px;
}
.chat-scroll::-webkit-scrollbar { width: 5px; }
.chat-scroll::-webkit-scrollbar-track { background: transparent; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

/* Sidebar section headers */
.sidebar-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 20px 0 10px 0;
}

/* Token counter */
.token-info {
    font-size: 0.7rem;
    color: var(--muted);
    text-align: right;
    margin-top: 4px;
}

div[data-testid="stImage"] img {
    border-radius: var(--radius);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client(api_key: str) -> InferenceClient:
    return InferenceClient(api_key=api_key)


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def pil_from_upload(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def img_to_text(client: InferenceClient, image: Image.Image, prompt: str,
                system_prompt: str, model: str, temperature: float,
                max_tokens: int, history: list) -> str:
    """Send image + text to Qwen vision model and stream back the reply."""
    b64 = image_to_base64(image)
    data_url = f"data:image/png;base64,{b64}"

    messages = [{"role": "system", "content": system_prompt}]
    # Add conversation history (text only for past turns)
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    # Current turn with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text",      "text": prompt},
        ],
    })

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    reply = ""
    placeholder = st.empty()
    for chunk in completion:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            placeholder.markdown(f'<div class="msg-bot"><div class="msg-label bot">⚡ Qwen</div>{reply}▌</div>',
                                 unsafe_allow_html=True)
    placeholder.markdown(f'<div class="msg-bot"><div class="msg-label bot">⚡ Qwen</div>{reply}</div>',
                         unsafe_allow_html=True)
    return reply


def text_to_text(client: InferenceClient, prompt: str, system_prompt: str,
                 model: str, temperature: float, max_tokens: int, history: list) -> str:
    """Pure text conversation with streaming."""
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    reply = ""
    placeholder = st.empty()
    for chunk in completion:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            placeholder.markdown(f'<div class="msg-bot"><div class="msg-label bot">⚡ Qwen</div>{reply}▌</div>',
                                 unsafe_allow_html=True)
    placeholder.markdown(f'<div class="msg-bot"><div class="msg-label bot">⚡ Qwen</div>{reply}</div>',
                         unsafe_allow_html=True)
    return reply


def text_to_image(client: InferenceClient, prompt: str, img_model: str,
                  width: int, height: int, num_steps: int) -> Optional[Image.Image]:
    """Generate an image from a text prompt via HF Inference."""
    try:
        result = client.text_to_image(
            prompt=prompt,
            model=img_model,
            width=width,
            height=height,
            num_inference_steps=num_steps,
        )
        return result  # returns PIL Image
    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = "img2txt"
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("HF_TOKEN", "")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="title-bar"><h1>🔮 Studio</h1></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-header">🔑 Authentication</p>', unsafe_allow_html=True)
    api_key_input = st.text_area(
        "Hugging Face Token",
        value=st.session_state.api_key,
        placeholder="hf_...",
        help="Get yours at huggingface.co/settings/tokens",
        label_visibility="collapsed",
        height=60,
    ).strip()
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown('<p class="sidebar-header">🎛 Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        "Select Mode",
        options=["img2txt", "txt2img", "chat"],
        format_func=lambda x: {
            "img2txt": "🖼 Image → Text  (Vision)",
            "txt2img": "✨ Text → Image  (Generative)",
            "chat":    "💬 Text Chat  (Reasoning)",
        }[x],
        index=["img2txt", "txt2img", "chat"].index(st.session_state.mode),
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    st.markdown('<p class="sidebar-header">🤖 Models</p>', unsafe_allow_html=True)

    VISION_MODELS = [
        "Qwen/Qwen3.5-397B-A17B:novita",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
    ]
    IMG_GEN_MODELS = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]

    if mode in ("img2txt", "chat"):
        vision_model = st.selectbox("Vision / Chat Model", VISION_MODELS, label_visibility="collapsed")
    if mode == "txt2img":
        gen_model = st.selectbox("Image Gen Model", IMG_GEN_MODELS, label_visibility="collapsed")

    st.markdown('<p class="sidebar-header">⚙️ Parameters</p>', unsafe_allow_html=True)

    if mode in ("img2txt", "chat"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05,
                                help="Higher = more creative, lower = more focused")
        max_tokens  = st.slider("Max Tokens", 128, 4096, 1024, 128)

    if mode == "txt2img":
        col1, col2 = st.columns(2)
        with col1:
            img_width  = st.selectbox("Width",  [512, 768, 1024], index=0)
        with col2:
            img_height = st.selectbox("Height", [512, 768, 1024], index=0)
        num_steps = st.slider("Inference Steps", 1, 50, 4, 1)

    st.markdown('<p class="sidebar-header">🧠 System Prompt</p>', unsafe_allow_html=True)
    default_sys = {
        "img2txt": (
            "You are an expert multimodal AI assistant powered by Qwen. "
            "When given an image, analyze it thoroughly, describe it in detail, "
            "answer questions about it, extract text, identify objects, describe scenes, "
            "and reason about visual content. Be precise, insightful, and comprehensive."
        ),
        "txt2img": (
            "You are a creative AI assistant specializing in image generation. "
            "Help users craft detailed, evocative prompts for image generation. "
            "If the user gives a simple idea, expand it into a rich visual prompt."
        ),
        "chat": (
            "You are Qwen, a highly capable reasoning AI assistant. "
            "Be helpful, accurate, thoughtful, and concise. "
            "Think step by step when solving complex problems."
        ),
    }
    system_prompt = st.text_area(
        "System Prompt",
        value=default_sys[mode],
        height=140,
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🗑 Clear Conversation"):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        '<p style="font-size:0.65rem;color:#4b5563;margin-top:12px;">'
        'Powered by <a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B" '
        'style="color:#7c6af7;">Qwen3.5-397B-A17B</a> via HF Inference API'
        '</p>',
        unsafe_allow_html=True,
    )


# ── Main panel ────────────────────────────────────────────────────────────────
mode_labels = {
    "img2txt": ("🖼  Image → Text", "Upload an image and ask anything about it"),
    "txt2img": ("✨  Text → Image", "Describe an image and watch it come to life"),
    "chat":    ("💬  Text Chat",    "Have a conversation with Qwen"),
}
title, subtitle = mode_labels[mode]

st.markdown(f"""
<div class="title-bar">
  <div>
    <h1 style="font-family:Syne,sans-serif;font-size:1.75rem;font-weight:800;
               background:linear-gradient(135deg,#7c6af7,#f76ac8);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin:0;padding:0;">{title}</h1>
    <p style="color:#6b7280;font-size:0.8rem;margin:4px 0 0 0;">{subtitle}</p>
  </div>
  <span class="badge">Qwen3.5 · 397B</span>
</div>
""", unsafe_allow_html=True)

# Guard: API key required
if not st.session_state.api_key:
    st.warning("⚠️ Please enter your Hugging Face API token in the sidebar to get started.")
    st.stop()

client = get_client(st.session_state.api_key)


# ────────────────────────────────────────────────────────────────────────────
#  MODE: Image → Text
# ────────────────────────────────────────────────────────────────────────────
if mode == "img2txt":
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown("**Upload Image**")
        uploaded = st.file_uploader(
            "Drop image here",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            label_visibility="collapsed",
        )
        if uploaded:
            img = pil_from_upload(uploaded)
            st.image(img, use_container_width=True)
            st.caption(f"📐 {img.width} × {img.height} px · {uploaded.type}")

        st.markdown('<p style="margin-top:16px;font-weight:600;">— or paste URL —</p>',
                    unsafe_allow_html=True)
        img_url = st.text_input("Image URL", placeholder="https://...", label_visibility="collapsed")

        # Quick-prompt suggestions
        st.markdown("**Quick Prompts**")
        quick_prompts = [
            "Describe this image in detail.",
            "What objects are present?",
            "Extract any visible text.",
            "What is the mood or aesthetic?",
            "Identify the main subject.",
        ]
        for qp in quick_prompts:
            if st.button(qp, key=f"qp_{qp}"):
                st.session_state["prefill"] = qp
                st.rerun()

    with col_right:
        # Chat history
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
        for msg in st.session_state.history:
            role_cls = "msg-user" if msg["role"] == "user" else "msg-bot"
            label_cls = "user" if msg["role"] == "user" else "bot"
            label_txt = "You" if msg["role"] == "user" else "⚡ Qwen"
            # images in history shown as text note
            content = msg["content"] if isinstance(msg["content"], str) else "[Image attached]"
            st.markdown(
                f'<div class="{role_cls}"><div class="msg-label {label_cls}">{label_txt}</div>{content}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        prefill = st.session_state.get("prefill", "")
        user_prompt = st.text_area(
            "Your message",
            value=prefill,
            placeholder="Ask anything about the image…",
            height=90,
            label_visibility="collapsed",
        )

        send = st.button("Send  ➤", use_container_width=True)

        if send and user_prompt.strip():
            # Clear prefill after getting the value
            if "prefill" in st.session_state:
                del st.session_state["prefill"]
            
            # Resolve image
            image = None
            if uploaded:
                image = pil_from_upload(uploaded)
            elif img_url.strip():
                try:
                    resp = requests.get(img_url, timeout=10)
                    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    st.error(f"Could not load image from URL: {e}")

            if image is None:
                st.warning("Please upload an image or provide an image URL.")
            else:
                st.session_state.history.append({"role": "user", "content": user_prompt})
                st.markdown(
                    f'<div class="msg-user"><div class="msg-label user">You</div>{user_prompt}</div>',
                    unsafe_allow_html=True,
                )
                with st.spinner(""):
                    reply = img_to_text(
                        client, image, user_prompt, system_prompt,
                        vision_model, temperature, max_tokens,
                        st.session_state.history[:-1],
                    )
                st.session_state.history.append({"role": "assistant", "content": reply})


# ────────────────────────────────────────────────────────────────────────────
#  MODE: Text → Image
# ────────────────────────────────────────────────────────────────────────────
elif mode == "txt2img":
    col_prompt, col_result = st.columns([1, 1.2], gap="large")

    with col_prompt:
        st.markdown("**Describe your image**")

        enhance_help = st.checkbox("✨ AI-enhance prompt first (uses chat model)", value=False)

        raw_prompt = st.text_area(
            "Prompt",
            placeholder="A neon-lit cyberpunk street market at midnight, rain-soaked, ultra-detailed...",
            height=160,
            label_visibility="collapsed",
        )

        neg_prompt = st.text_area(
            "Negative Prompt (optional)",
            placeholder="blurry, low quality, watermark, text...",
            height=80,
        )

        # Style presets
        st.markdown("**Style Presets**")
        presets = {
            "None":         "",
            "Cinematic":    ", cinematic lighting, anamorphic lens, film grain, award-winning photography",
            "Anime":        ", anime style, Studio Ghibli, vibrant colors, detailed",
            "Photorealism": ", photorealistic, 8K, DSLR, shallow depth of field, sharp focus",
            "Oil Painting": ", oil painting, impressionist, textured canvas, classical art",
            "Pixel Art":    ", pixel art, 16-bit, retro game, vibrant palette",
        }
        preset = st.selectbox("Preset", list(presets.keys()))
        generate = st.button("Generate Image  🎨", use_container_width=True)

    with col_result:
        st.markdown("**Generated Image**")
        result_placeholder = st.empty()

        if generate and raw_prompt.strip():
            final_prompt = raw_prompt + presets[preset]

            # Optionally enhance the prompt via Qwen chat
            if enhance_help:
                with st.spinner("Enhancing prompt with Qwen…"):
                    enhance_sys = (
                        "You are an expert at writing detailed, evocative prompts for AI image generation. "
                        "Take the user's idea and expand it into a rich, detailed image generation prompt "
                        "in 1-3 sentences. Return ONLY the improved prompt, no explanation."
                    )
                    enhance_client = get_client(st.session_state.api_key)
                    msgs = [
                        {"role": "system", "content": enhance_sys},
                        {"role": "user",   "content": final_prompt},
                    ]
                    comp = enhance_client.chat.completions.create(
                        model=VISION_MODELS[0],
                        messages=msgs,
                        temperature=0.8,
                        max_tokens=256,
                    )
                    final_prompt = comp.choices[0].message.content.strip()
                    st.info(f"**Enhanced prompt:** {final_prompt}")

            with st.spinner("Generating… this may take a moment"):
                gen_image = text_to_image(
                    client, final_prompt, gen_model,
                    img_width, img_height, num_steps,
                )

            if gen_image:
                result_placeholder.image(gen_image, use_container_width=True)

                # Download button
                buf = io.BytesIO()
                gen_image.save(buf, format="PNG")
                st.download_button(
                    "⬇ Download PNG",
                    data=buf.getvalue(),
                    file_name=f"qwen_gen_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True,
                )
                st.session_state.history.append({
                    "role": "user",
                    "content": f"[Generated image with prompt]: {final_prompt}",
                })
        else:
            result_placeholder.markdown(
                '<div style="border:1.5px dashed #252836;border-radius:14px;'
                'padding:80px 20px;text-align:center;color:#4b5563;">'
                '🎨 Your generated image will appear here</div>',
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────────
#  MODE: Text Chat
# ────────────────────────────────────────────────────────────────────────────
elif mode == "chat":
    # Initialize chat_input if needed
    if "chat_input_clear" not in st.session_state:
        st.session_state.chat_input_clear = False
    
    if st.session_state.chat_input_clear:
        st.session_state.chat_input = ""
        st.session_state.chat_input_clear = False
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.history:
            role_cls = "msg-user" if msg["role"] == "user" else "msg-bot"
            label_cls = "user" if msg["role"] == "user" else "bot"
            label_txt = "You" if msg["role"] == "user" else "⚡ Qwen"
            content   = msg["content"] if isinstance(msg["content"], str) else ""
            st.markdown(
                f'<div class="{role_cls}"><div class="msg-label {label_cls}">{label_txt}</div>{content}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_area(
            "Message",
            placeholder="Ask Qwen anything…  (Shift+Enter for new line)",
            height=80,
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_send:
        st.markdown("<br>", unsafe_allow_html=True)
        send_chat = st.button("Send ➤", use_container_width=True)

    char_count = len(user_input) if user_input else 0
    st.markdown(f'<div class="token-info">{char_count} chars</div>', unsafe_allow_html=True)

    if send_chat and user_input.strip():
        st.session_state.history.append({"role": "user", "content": user_input})
        st.markdown(
            f'<div class="msg-user"><div class="msg-label user">You</div>{user_input}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner(""):
            reply = text_to_text(
                client, user_input, system_prompt,
                vision_model, temperature, max_tokens,
                st.session_state.history[:-1],
            )
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.session_state.chat_input_clear = True
        st.rerun()