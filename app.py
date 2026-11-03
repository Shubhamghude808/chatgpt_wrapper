import os
import base64
import traceback
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="ChatGPT Wrapper", page_icon="🤖", layout="wide")
st.title("ChatGPT Personal Wrapper")

# Read API key from environment variable (set OPENAI_API_KEY before running)
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("🔑 `OPENAI_API_KEY` environment variable not set. "
             "Please set it in your .env file before running the app.")
    st.stop()

client = OpenAI(api_key=api_key)

# ── Supported file types & Limits ───────────────────────────────────────
IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]
TEXT_TYPES = [
    "txt", "md", "csv", "json", "xml", "html", "css", "js",
    "ts", "py", "java", "c", "cpp", "h", "rb", "go", "rs",
    "sh", "bat", "yaml", "yml", "toml", "ini", "cfg", "log",
    "sql", "r", "kt", "swift", "dart"
]
DOC_TYPES = ["pdf"]
MAX_FILE_CHARS = 25000  # Cap extracted text per file to protect API tokens (~5-6k tokens)


def _encode_image(file_bytes: bytes, mime: str) -> dict:
    """Build an input_image content block for the Responses API with required detail."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{b64}",
        "detail": "auto",
    }


def _extract_text(uploaded_file) -> str | None:
    """Extract readable text from an uploaded file with safety length cap."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    text = ""
    if name.endswith(".pdf"):
        try:
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n\n".join(pages).strip()
            if not text:
                return "(No extractable text found in PDF)"
        except Exception as e:
            return f"(Could not parse PDF: {e})"
    else:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return "(Could not read file as plain text)"

    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n\n[... Truncated to {MAX_FILE_CHARS} characters to conserve tokens ...]"
    return text


def _build_input_messages(messages: list) -> list:
    """Build the clean input list for the Responses API from session history."""
    history = []
    for m in messages:
        role = m["role"]
        # For assistant messages with images, provide text context
        if m.get("_generated_image"):
            content_text = m.get("content") or "[Generated an image]"
            history.append({"role": role, "content": content_text})
            continue

        # Messages with vision/multi-part content
        if "vision_content" in m:
            history.append({"role": role, "content": m["vision_content"]})
        else:
            text_val = m.get("content") or ""
            history.append({"role": role, "content": text_val})
    return history


def _process_response(response):
    """Extract text, refusals, generated images, errors, and token usage."""
    text_parts = []
    images = []  # list of {"bytes": bytes, "revised_prompt": str|None}
    errors = []

    outputs = getattr(response, "output", [])
    if isinstance(response, dict):
        outputs = response.get("output", [])

    for output in outputs:
        out_type = getattr(output, "type", None) or (output.get("type") if isinstance(output, dict) else None)

        if out_type == "message":
            contents = getattr(output, "content", []) or (output.get("content", []) if isinstance(output, dict) else [])
            for c in contents:
                c_type = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
                if c_type == "output_text":
                    txt = getattr(c, "text", "") or (c.get("text", "") if isinstance(c, dict) else "")
                    if txt:
                        text_parts.append(txt)
                elif c_type == "refusal":
                    refusal = getattr(c, "refusal", "") or (c.get("refusal", "") if isinstance(c, dict) else "")
                    if refusal:
                        errors.append(f"Model refusal: {refusal}")

        elif out_type == "image_generation_call":
            status = getattr(output, "status", None) or (output.get("status") if isinstance(output, dict) else None)
            if status == "failed":
                err_obj = getattr(output, "error", None) or (output.get("error") if isinstance(output, dict) else None)
                msg = getattr(err_obj, "message", None) or (err_obj.get("message") if isinstance(err_obj, dict) else None) or "Image generation failed."
                errors.append(f"Image generation failed: {msg}")
            else:
                result = getattr(output, "result", None) or (output.get("result") if isinstance(output, dict) else None)
                if result:
                    img_bytes = base64.b64decode(result)
                    revised = getattr(output, "revised_prompt", None) or (output.get("revised_prompt") if isinstance(output, dict) else None)
                    images.append({"bytes": img_bytes, "revised_prompt": revised})

    # Extract token usage if returned
    usage = getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None)
    usage_info = None
    if usage:
        in_tok = getattr(usage, "input_tokens", None) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
        out_tok = getattr(usage, "output_tokens", None) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
        tot_tok = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
        if in_tok is not None and out_tok is not None:
            usage_info = {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": tot_tok or (in_tok + out_tok)}

    return "\n\n".join(text_parts), images, errors, usage_info


# ── Session state initialization ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑️ Clear Chat & Reset Context", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens_used = 0
        st.session_state.uploader_key += 1
        st.rerun()

    # Token usage monitor
    if st.session_state.total_tokens_used > 0:
        st.info(f"📊 **Session Tokens Used:** {st.session_state.total_tokens_used:,}")

    st.divider()
    st.header("📎 Attach Files & Images")
    all_types = IMAGE_TYPES + TEXT_TYPES + DOC_TYPES
    uploaded_files = st.file_uploader(
        "Attach to next message",
        type=all_types,
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
        help="Attached files are sent with your next prompt and extracted safely to conserve tokens."
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) attached — type your message below.")

    st.divider()
    st.caption("**Supported formats:**")
    st.caption(f"🖼️ Images: {', '.join(IMAGE_TYPES)}")
    st.caption(f"📄 Code/Text: {', '.join(TEXT_TYPES)}")
    st.caption(f"📕 Documents: {', '.join(DOC_TYPES)}")

# ── Render chat history ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show attached images if present
        if msg.get("_attached_images"):
            for img_bytes, fname in msg["_attached_images"]:
                st.image(img_bytes, caption=fname, width=350)

        # Show attached text files in expander
        if msg.get("_attached_texts"):
            for fname, text in msg["_attached_texts"]:
                with st.expander(f"📄 {fname}"):
                    st.code(text[:2000], language=None)

        # Show text content
        if msg.get("content"):
            st.markdown(msg["content"])

        # Show generated images
        if msg.get("_generated_images"):
            for img_item in msg["_generated_images"]:
                if isinstance(img_item, dict):
                    st.image(img_item["bytes"], caption=img_item.get("revised_prompt"))
                else:
                    st.image(img_item)

        # Show token badge
        if msg.get("_tokens"):
            t = msg["_tokens"]
            st.caption(f"⚡ Tokens: {t['input_tokens']:,} in · {t['output_tokens']:,} out")

# ── Chat input ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Type a message or ask to generate an image…"):

    current_files = list(uploaded_files) if uploaded_files else []

    attached_images = []   # (bytes, filename, mime)
    attached_texts = []    # (filename, extracted_text)

    for f in current_files:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        if ext in IMAGE_TYPES:
            mime = f.type or f"image/{ext}"
            f.seek(0)
            attached_images.append((f.read(), f.name, mime))
        else:
            f.seek(0)
            text = _extract_text(f)
            if text:
                attached_texts.append((f.name, text))

    # Prepare user message for session display
    user_msg: dict = {"role": "user", "content": prompt}

    if attached_images:
        user_msg["_attached_images"] = [
            (img_bytes, fname) for img_bytes, fname, _ in attached_images
        ]
    if attached_texts:
        user_msg["_attached_texts"] = attached_texts

    # Build multi-part content if files are attached
    if attached_images or attached_texts:
        parts: list[dict] = []

        if attached_texts:
            file_context = "\n\n".join(
                f"── File: {fname} ──\n{text}" for fname, text in attached_texts
            )
            parts.append({"type": "input_text", "text": f"{file_context}\n\n{prompt}"})
        else:
            parts.append({"type": "input_text", "text": prompt})

        for img_bytes, fname, mime in attached_images:
            parts.append(_encode_image(img_bytes, mime))

        user_msg["vision_content"] = parts

    # Append user message
    st.session_state.messages.append(user_msg)

    # Render user message in UI
    with st.chat_message("user"):
        if attached_images:
            for img_bytes, fname, _ in attached_images:
                st.image(img_bytes, caption=fname, width=350)
        if attached_texts:
            for fname, text in attached_texts:
                with st.expander(f"📄 {fname}"):
                    st.code(text[:2000], language=None)
        st.markdown(prompt)

    # Reset file uploader for subsequent turns so attachments aren't duplicated
    if current_files:
        st.session_state.uploader_key += 1

    # Call OpenAI Responses API
    with st.chat_message("assistant"):
        with st.spinner("Processing…"):
            try:
                input_messages = _build_input_messages(st.session_state.messages)

                response = client.responses.create(
                    model="gpt-4o",
                    input=input_messages,
                    tools=[
                        {
                            "type": "image_generation",
                            "model": "gpt-image-2.5-sunburst",
                        }
                    ],
                )

                text_resp, gen_images, errors, usage = _process_response(response)

                if errors:
                    for err in errors:
                        st.warning(err)

                if text_resp:
                    st.markdown(text_resp)

                for img_item in gen_images:
                    st.image(img_item["bytes"], caption=img_item.get("revised_prompt"))

                if usage:
                    st.session_state.total_tokens_used += usage["total_tokens"]
                    st.caption(f"⚡ Tokens: {usage['input_tokens']:,} in · {usage['output_tokens']:,} out")

                # Store assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": text_resp or "",
                }
                if gen_images:
                    assistant_msg["_generated_images"] = gen_images
                if usage:
                    assistant_msg["_tokens"] = usage

                st.session_state.messages.append(assistant_msg)

            except Exception as e:
                # If API call fails, remove the failed user message so history isn't poisoned
                if st.session_state.messages and st.session_state.messages[-1] == user_msg:
                    st.session_state.messages.pop()
                st.error(f"Error: {e}")
                st.code(traceback.format_exc(), language=None)
