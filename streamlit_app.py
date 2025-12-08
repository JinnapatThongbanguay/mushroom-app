# streamlit_app.py
# 🍄 Mushroom Safety Finder (RAG + CLIP + FAISS) + Gemini (on-demand)
# Safe Mode ON — Minimal UI, User Prompts, Ready for Streamlit Cloud deploy

import os
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
from typing import Tuple, List, Dict, Any

# CONFIG
st.set_page_config(page_title="Mushroom Safety Finder (RAG + Gemini)",
                   page_icon="🍄", layout="centered")

KB_PATH = "mushroom_knowledge_base.pkl"   # created by build_kb.py
TOP_K = 5
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.55
SAFE_CONF_THRESH = 0.60   # threshold for forcing "do not eat" when low confidence

# Load CLIP model & processor
# cached for streamlit
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor, device

clip_model, clip_processor, device = load_clip()

# Load KB and build FAISS index
@st.cache_resource
def load_kb_and_index(kb_path: str = KB_PATH):
    if not os.path.exists(kb_path):
        return None, None, None
    with open(kb_path, "rb") as f:
        kb = pickle.load(f)

    all_feats = []
    metadata = []
    for species, data in kb.items():
        feats = data.get("features")
        if feats is None:
            continue
        # ensure float32
        for i, feat in enumerate(feats):
            all_feats.append(np.asarray(feat, dtype="float32"))
            metadata.append((species, data.get("label", "Unknown"), i, data.get("paths", [])))

    if not all_feats:
        return kb, None, metadata

    all_feats = np.stack(all_feats)
    dim = all_feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    # assume features are normalized in build_kb
    index.add(all_feats)
    return kb, index, metadata

kb, faiss_index, metadata = load_kb_and_index(KB_PATH)

# Setup Gemini (on-demand)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else None)
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # use a supported model
        MODEL_NAME = "models/gemini-2.5-flash"
        gemini_model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        gemini_model = None
        st.warning(f"Gemini init warning: {e}")

# Helpers: feature extraction, retrieval, prediction
def extract_image_features_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]

def retrieve_similar_examples(query_feat: np.ndarray, top_k: int = TOP_K) -> List[Dict[str,Any]]:
    if faiss_index is None:
        return []
    q = query_feat.reshape(1, -1).astype("float32")
    sims, idxs = faiss_index.search(q, top_k)
    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        species, label, img_idx, paths = metadata[idx]
        results.append({"species": species, "label": label, "similarity": float(sim), "image_idx": img_idx, "paths": paths})
    return results

def predict_rag_from_image(img: Image.Image) -> Tuple[str, float, str, List[Dict[str,Any]]]:
    try:
        qfeat = extract_image_features_pil(img)
    except Exception as e:
        return "Unknown", 0.0, "unknown", []
    retrieved = retrieve_similar_examples(qfeat, top_k=TOP_K)
    label_scores = {"Edible": 0.0, "Poisonous": 0.0}
    for r in retrieved:
        lab = r.get("label", "Unknown")
        if lab in label_scores:
            label_scores[lab] += r.get("similarity", 0.0)
    total = sum(label_scores.values())
    if total <= 0:
        probs = {"Edible": 0.5, "Poisonous": 0.5}
    else:
        probs = {k: v/total for k, v in label_scores.items()}
    predicted = max(probs, key=probs.get)
    confidence = float(probs[predicted])
    top_species = retrieved[0]["species"] if retrieved else "unknown"
    return predicted, confidence, top_species, retrieved

# Safe Mode decision
def safe_decision(predicted_label: str, confidence: float, species_key: str) -> Dict[str,Any]:
    has_info = (kb is not None and species_key in kb)
    if predicted_label == "Poisonous":
        return {"risk": "เสี่ยงสูง (เห็ดพิษ)", "advice": "ห้ามรับประทานโดยเด็ดขาด และรีบไปโรงพยาบาลทันที", "safe_to_eat": False}
    if predicted_label == "Edible" and not has_info:
        return {"risk":"ไม่สามารถยืนยันความปลอดภัยได้", "advice":"แม้ระบบคาดว่าเป็นเห็ดกินได้ แต่ไม่มีข้อมูลยืนยัน → ห้ามรับประทานเด็ดขาด", "safe_to_eat": False}
    if predicted_label == "Edible" and has_info and confidence >= CONFIDENCE_HIGH:
        return {"risk":"เสี่ยงต่ำ (มีข้อมูลยืนยัน)", "advice":"สามารถรับประทานได้ เฉพาะเมื่อปรุงสุกแล้วเท่านั้น", "safe_to_eat": True}
    if predicted_label == "Edible" and has_info and confidence >= CONFIDENCE_MEDIUM:
        return {"risk":"ความมั่นใจปานกลาง", "advice":"ยังไม่แนะนำให้รับประทาน ควรให้ผู้เชี่ยวชาญตรวจสอบก่อน", "safe_to_eat": False}
    # fallback
    return {"risk":"ไม่สามารถระบุชนิดได้", "advice":"ห้ามรับประทานโดยเด็ดขาด", "safe_to_eat": False}


# Gemini prompt builder (User / Expert modes)
def build_gemini_prompt(species_key: str, predicted_label: str, confidence: float, expert_mode: bool = False) -> str:
    if expert_mode:
        prompt = f"""
คุณคือผู้เชี่ยวชาญด้านเห็ด (Mycologist)

ระบบคอมพิวเตอร์ให้ข้อมูลเบื้องต้น:
- ชนิดที่ระบบคาด: {species_key}
- การประเมิน: {predicted_label}
- ความมั่นใจ: {confidence:.2%}

ช่วยสรุปข้อมูลเชิงอ้างอิงเชิงวิชาการสำหรับชนิดนี้เป็นภาษาไทย โดยให้มีรายการต่อไปนี้ (หากไม่พบข้อมูลใด ๆ ให้เขียนว่า "ไม่พบข้อมูล"):
- scientific_name
- thai_name
- edibility
- toxicity_level
- physical_characteristics (ละเอียดพอจะใช้อ้างอิง)
- habitat
- symptoms
- first_aid
- warning

ข้อจำกัด:
- หาก confidence < {SAFE_CONF_THRESH*100:.0f}% หรือไม่มีข้อมูลยืนยัน ให้ขึ้นต้นด้วยข้อความชัดเจนว่า "ห้ามรับประทาน" และอย่าแนะนําให้กิน
- ตอบเป็นภาษาไทย แบบเชิงวิชาการ แต่ไม่จำเป็นต้องเป็น JSON
"""
    else:
        # โหมดผู้ใช้ทั่วไป (ตัดคำทักทาย/คำสรุป)
        prompt = f"""
คุณคือผู้เชี่ยวชาญด้านเห็ด (Mycologist)

ระบบคอมพิวเตอร์ให้ข้อมูลเบื้องต้น:
- ชนิดที่ระบบคาด: {species_key}
- การประเมิน: {predicted_label}
- ความมั่นใจ: {confidence:.2%}

ช่วยอธิบายเห็ดชนิดนี้เป็นภาษาคน อ่านเข้าใจง่ายสำหรับผู้ใช้ทั่วไป โดยให้มี:
- ชื่อเห็ด
- ลักษณะเด่น (สั้น ๆ)
- ความเป็นพิษ (ชัดเจน)
- อาการที่อาจเกิดขึ้น (สั้น)
- การปฐมพยาบาล (สั้น)
- คำเตือนด้านความปลอดภัย

ข้อจำกัด:
- ห้ามขึ้นต้นด้วยคำทักทาย เช่น "สวัสดีครับ/ค่ะ" หรือคำอ้างอิงถึงตัวคุณเอง
- ห้ามมีคำสรุปปิดท้าย เช่น "หวังว่าข้อมูลนี้จะเป็นประโยชน์"
- หาก confidence < {SAFE_CONF_THRESH*100:.0f}% หรือไม่มีข้อมูลยืนยัน ให้ชัดเจนว่า 'ห้ามรับประทาน' และอย่าแนะนำให้กิน
- ห้ามตอบเป็น JSON
- ตอบเป็นภาษาไทยแบบเป็นมิตรต่อผู้ใช้
"""
    return prompt.strip()

def ask_gemini_text(species_key: str, predicted_label: str, confidence: float, expert_mode: bool=False) -> str:
    if gemini_model is None:
        return "Gemini API not configured. ตั้งค่า GEMINI_API_KEY ใน environment หรือ Streamlit secrets ก่อน"
    prompt = build_gemini_prompt(species_key, predicted_label, confidence, expert_mode)
    try:
        resp = gemini_model.generate_content(prompt)
        # return textual content only
        return resp.text
    except Exception as e:
        return f"Gemini error: {e}"

# UI
st.title("Mushroom Safety Finder (RAG + Gemini)")
st.write("ระบบประเมินความเสี่ยงเห็ดจากภาพ (CLIP + FAISS + Knowledge Base) — Safe Mode ON")
st.info("เพื่อการศึกษาเท่านั้น ห้ามใช้ตัดสินใจกินจริง")

uploaded = st.file_uploader("อัปโหลดภาพเห็ด (jpg, png, heic)", type=["jpg","jpeg","png", "heic"])
# expert_mode ถูกตั้งค่าตายตัวเป็น False หรือย้ายไป sidebar แทน
expert_mode = False # ใช้ User Mode เสมอ

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
    except Exception:
        st.error("ไม่สามารถเปิดไฟล์รูปภาพได้ กรุณาอัปโหลดไฟล์รูปที่ถูกต้อง")
        st.stop()

    if kb is None or faiss_index is None:
        st.warning("ฐานความรู้ (mushroom_knowledge_base.pkl) หรือ FAISS index ยังไม่พร้อม — ให้รัน build_kb.py เพื่อสร้างฐานก่อนใช้งานเต็มรูปแบบ")

    with st.spinner("กำลังวิเคราะห์ (RAG)..."):
        predicted_label, confidence, top_species, retrieved = predict_rag_from_image(image)

    decision = safe_decision(predicted_label, confidence, top_species)

    st.markdown("### ผลการประเมินความปลอดภัย")
    st.write(f"- **ความมั่นใจ (model):** {confidence*100:.2f}%")
    st.write(f"- **ระดับความเสี่ยง:** {decision['risk']}")
    st.write(f"- **คำแนะนำ:** {decision['advice']}")

    st.markdown("#### ข้อมูลเชิงลึก (On-demand)")
    st.write("หากต้องการข้อมูลเชิงอ้างอิง / คำอธิบายจากผู้เชี่ยวชาญ ให้กดปุ่มด้านล่าง")

    if st.button("ขอคำอธิบายเพิ่มเติมจาก AI (Gemini)"):
        with st.spinner("Gemini กำลังสร้างคำอธิบาย..."):
            # ใช้ expert_mode ที่ถูกกำหนดไว้ (ในที่นี้คือ False)
            gemini_text = ask_gemini_text(top_species, predicted_label, confidence, expert_mode=expert_mode)
        st.markdown("#### คำอธิบายจาก AI ผู้ช่วย")
        st.write(gemini_text)
        

else:
    st.caption("อัปโหลดรูปเพื่อประเมิน — ถ่ายมุมบน, ใต้ดอก, และโคนก้านช่วยให้จำแนกได้แม่นยำขึ้น")

# footer note about gemini
if gemini_model is None:
    st.caption("Gemini: Not configured. เพื่อเปิดใช้งาน ให้ตั้งค่า GEMINI_API_KEY ใน environment หรือ Streamlit secrets.")
else:
    st.caption("Gemini: Enabled (on-demand). API calls made only when user requests details.")
