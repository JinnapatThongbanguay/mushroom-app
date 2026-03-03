# streamlit_app.py
"""
🍄 Mushroom Safety Classifier
ใช้ RAG + CLIP (ตรงกับ evaluation code 100%)
Accuracy: 86.98% | Safe Mode ON
"""

import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pickle
import faiss
import os
import google.generativeai as genai
# นำเข้า exceptions เพื่อจัดการ Error 429
from google.api_core import exceptions as gcp_exceptions 
from typing import Tuple, List, Dict

# CONFIG 
st.set_page_config(
    page_title="ระบบวิเคราะห์เห็ด",
    page_icon="🍄",
    layout="centered"
)

# Paths and settings (SAME AS EVALUATION)
KB_PATH = "mushroom_knowledge_base.pkl"
TOP_K = 3  
SAFETY_THRESHOLD = 0.55 
MIN_CONFIDENCE = 0.55

# LOAD MODELS (CACHED) 
@st.cache_resource
def load_clip_model():
    """Load CLIP model (cached)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor, device

@st.cache_resource
def load_kb_and_faiss(kb_path: str = KB_PATH):
    """Load KB and build FAISS index (cached)"""
    
    # ตรวจสอบว่าไฟล์มีอยู่จริง
    if not os.path.exists(kb_path):
        st.error(f"ไม่พบไฟล์ Knowledge Base")
        st.info(f"กำลังค้นหา: `{kb_path}`")
        st.info(f"ไฟล์ต้องอยู่ที่: root directory ของโปรเจค")
        st.info("""
        **วิธีแก้:**
        1. ตรวจสอบว่าไฟล์ `mushroom_knowledge_base.pkl` อยู่ที่ root
        2. ถ้าใช้ GitHub: อย่าลืม push ไฟล์นี้ด้วย
        3. ถ้าไฟล์ใหญ่เกินไป: ใช้ Git LFS หรือ upload แยก
        """)
        return None, None, None
    
    # โหลด KB
    try:
        with open(kb_path, "rb") as f:
            kb = pickle.load(f)
        
        # Build FAISS index (SAME AS EVALUATION)
        all_features = []
        metadata = []
        
        for species, data in kb.items():
            features = data.get("features")
            if features is None:
                continue
            for i, feat in enumerate(features):
                all_features.append(np.asarray(feat, dtype="float32"))
                metadata.append((species, data.get("label", "Unknown"), i))
        
        if not all_features:
            st.error("KB ไม่มีข้อมูล features")
            return kb, None, metadata
        
        all_features = np.stack(all_features)
        dimension = all_features.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(all_features)
        
        return kb, index, metadata
        
    except Exception as e:
        st.error(f"Error loading KB: {e}")
        return None, None, None

# Load models
clip_model, clip_processor, device = load_clip_model()
kb, faiss_index, metadata = load_kb_and_faiss()

#CLIP PROMPTS (FOR HYBRID)
EDIBLE_PROMPTS = [
    "a safe edible mushroom suitable for cooking",
    "a choice edible mushroom prized by foragers",
    "an edible Boletus with thick stem and pores",
    "an edible oyster mushroom or similar safe species"
]

POISONOUS_PROMPTS = [
    "a deadly poisonous mushroom that is toxic",
    "a dangerous Amanita mushroom with white gills",
    "a toxic Galerina or Cortinarius species",
    "a poisonous mushroom with warning colors"
]

# PREDICTION FUNCTIONS (EXACT COPY FROM EVALUATION)
def extract_image_features(image: Image.Image) -> np.ndarray:
    """Extract CLIP features (normalized)"""
    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        
        # แก้ตรงนี้: บาง version return BaseModelOutputWithPooling
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        elif hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state[:, 0, :]
        # else: features เป็น tensor ปกติอยู่แล้ว
        
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy()[0]

def retrieve_similar_examples(query_features: np.ndarray, top_k: int = TOP_K) -> List[Dict]:
    """Retrieve top-k similar examples from KB"""
    if faiss_index is None:
        return []
    
    query_features = query_features.reshape(1, -1).astype("float32")
    similarities, indices = faiss_index.search(query_features, top_k)
    
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx >= len(metadata):
            continue
        species, label, img_idx = metadata[idx]
        results.append({
            'species': species,
            'label': label,
            'similarity': float(sim),
            'image_idx': img_idx
        })
    
    return results

def predict_with_prompts(image: Image.Image) -> float:
    """Get CLIP prompt-based prediction"""
    try:
        image = image.convert("RGB")
        all_prompts = EDIBLE_PROMPTS + POISONOUS_PROMPTS
        
        inputs = clip_processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        edible_prob = probs[:len(EDIBLE_PROMPTS)].mean()
        poison_prob = probs[len(EDIBLE_PROMPTS):].mean()
        
        total = edible_prob + poison_prob
        edible_ratio = edible_prob / total if total > 0 else 0.5
        
        return edible_ratio
    except:
        return 0.5

def predict_with_enhanced_rag(image: Image.Image) -> Tuple[str, float, str, List[Dict]]:
    """
    Enhanced RAG prediction (EXACT SAME AS EVALUATION CODE)
    Returns: (classification, confidence, reasoning, retrieved_examples)
    """
    try:
        # Step 1: Extract features
        query_features = extract_image_features(image)
        
        # Step 2: Retrieve similar examples
        similar_examples = retrieve_similar_examples(query_features, top_k=TOP_K)
        
        if not similar_examples:
            return "Unknown", 0.0, "ไม่พบข้อมูลในฐานความรู้", []
        
        # Step 3: Calculate RAG scores with distance weighting (IMPORTANT!)
        label_scores = {'Edible': 0.0, 'Poisonous': 0.0}
        total_weight = 0.0
        
        for i, example in enumerate(similar_examples):
            # CRITICAL: Exponential decay weight (ตรงกับ evaluation)
            weight = example['similarity'] * (0.85 ** i)
            label_scores[example['label']] += weight
            total_weight += weight
        
        # Normalize RAG scores
        if total_weight > 0:
            rag_edible_score = label_scores['Edible'] / total_weight
        else:
            rag_edible_score = 0.5
        
        # Step 4: Get CLIP prompt score
        clip_edible_ratio = predict_with_prompts(image)
        
        # Step 5: Hybrid scoring (70% RAG + 30% CLIP)
        hybrid_edible_score = 0.7 * rag_edible_score + 0.3 * clip_edible_ratio
        
        # Step 6: Conservative safety logic (ตรงกับ evaluation)
        base_confidence = hybrid_edible_score
        
        if hybrid_edible_score >= SAFETY_THRESHOLD:
            classification = "Edible"
            confidence = hybrid_edible_score
            
            if base_confidence < MIN_CONFIDENCE:
                classification = "Poisonous"
                confidence = 1.0 - hybrid_edible_score
                reasoning = f"Confidence ต่ำ ({base_confidence:.1%}) → ปลอดภัยกว่าถ้าถือว่ามีพิษ"
            else:
                reasoning = f"RAG: {rag_edible_score:.1%}, CLIP: {clip_edible_ratio:.1%}"
        else:
            classification = "Poisonous"
            confidence = 1.0 - hybrid_edible_score
            reasoning = f"Score: {hybrid_edible_score:.1%} < threshold {SAFETY_THRESHOLD:.0%}"
        
        # Add similar species info
        top_species = [ex['species'] for ex in similar_examples[:3]]
        reasoning += f" | Similar: {', '.join(top_species[:2])}"
        
        return classification, confidence, reasoning, similar_examples
        
    except Exception as e:
        return "Unknown", 0.0, f"Error: {str(e)[:50]}", []

# GEMINI INTEGRATION
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print(f"Gemini configuration error: {e}")
        pass

# นำเข้า similar_species ด้วย
def ask_gemini_for_details(species: str, classification: str, confidence: float, similar_species: list) -> str:
    """Ask Gemini for detailed mushroom information with context-aware prompt"""
    if gemini_model is None:
        return "Gemini API ไม่พร้อมใช้งาน"
    
    # Get top 3 similar species with labels
    similar_info = []
    for ex in similar_species[:3]:
        similar_info.append(f"- {ex['species'].replace('_', ' ').title()} ({ex['label']}) - Similarity: {ex['similarity']:.1%}")
    
    similar_text = "\n".join(similar_info)
    
    # Create context-aware prompt
    if classification == "Poisonous":
        # Focus on poisonous species or safety concerns
        prompt = f"""คุณคือผู้เชี่ยวชาญด้านเห็ด

**สถานการณ์:**
ผู้ใช้ถ่ายภาพเห็ดมาให้วิเคราะห์ ระบบ AI ประเมินว่า:
- การจำแนก: มีพิษ / ไม่แนะนำให้บริโภค
- ความมั่นใจ: {confidence:.1%}

**เห็ดที่คล้ายกันจาก Knowledge Base:**
{similar_text}

**คำแนะนำ:** เนื่องจากระบบประเมินว่ามีความเสี่ยง กรุณาให้ข้อมูลดังนี้:

1. **เห็ดที่คาดว่าเป็น:**
   - ระบุชื่อสายพันธุ์ที่มีแนวโน้มสูงสุด (จากรายการคล้ายกัน)
   - ชื่อวิทยาศาสตร์และชื่อสามัญ

2. **ทำไมระบบจึงประเมินว่ามีความเสี่ยง:**
   - อาจเป็นเพราะคล้ายกับเห็ดพิษบางชนิด
   - หรือความมั่นใจต่ำเกินไป
   - หรือมีการปนเปื้อนในกลุ่ม similar species

3. **เห็ดพิษที่อันตรายที่คล้ายกัน:**
   - ชื่อเห็ดพิษที่อาจสับสนได้
   - ลักษณะที่แยกแยะได้

4. **อาการพิษ (กรณีเห็ดพิษ):**
   - อาการที่อาจเกิดขึ้น
   - ระยะเวลาที่อาการปรากฏ

5. **การปฐมพยาบาล:**
   - ขั้นตอนฉุกเฉิน
   - โทร 1669 ทันที

**สำคัญ:**
- ห้ามขึ้นต้นด้วยคำทักทาย
- ห้ามจบด้วยคำปิดท้าย
- **ต้องเน้นย้ำว่า "ห้ามรับประทาน"** เพราะความเสี่ยงสูง
- ตอบแบบตรงประเด็น กระชับ
"""
    else:
        # Focus on edible species but with caution
        prompt = f"""คุณคือผู้เชี่ยวชาญด้านเห็ด

**สถานการณ์:**
ผู้ใช้ถ่ายภาพเห็ดมาให้วิเคราะห์ ระบบ AI ประเมินว่า:
- การจำแนก: **กินได้**
- ความมั่นใจ: {confidence:.1%}

**เห็ดที่คล้ายกันจาก Knowledge Base:**
{similar_text}

**คำแนะนำ:** กรุณาให้ข้อมูลดังนี้:

1. **เห็ดที่คาดว่าเป็น:**
   - ชื่อสายพันธุ์ที่มีแนวโน้มสูงสุด
   - ชื่อวิทยาศาสตร์และชื่อสามัญ

2. **ลักษณะทางกายภาพ:**
   - หมวก (สี, รูปทรง, ขนาด)
   - ครีบ (สี, การเรียงตัว)
   - ก้าน (ลักษณะเด่น)

3. **แหล่งที่พบ:**
   - Habitat ทั่วไป
   - ฤดูกาลที่พบ

4. **เห็ดพิษที่คล้ายกัน (Look-alikes):**
   - ชื่อเห็ดพิษที่อาจสับสน
   - วิธีแยกแยะ

5. **คำเตือนสำคัญ:**
   - แม้ระบบบอกว่ากินได้ แต่ **ควรให้ผู้เชี่ยวชาญตรวจสอบ
   - อย่าใช้ AI เป็นหลักฐานเดียว
   - ปรุงสุกก่อนรับประทาน

**สำคัญ:**
- ห้ามขึ้นต้นด้วยคำทักทาย
- ห้ามจบด้วยคำปิดท้าย
- เน้นความระมัดระวัง แม้ความมั่นใจจะสูง
- ตอบแบบกระชับ เข้าใจง่าย
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    
    except gcp_exceptions.ResourceExhausted:
        # จัดการ Error 429 โดยเฉพาะ
        return "โควต้าการใช้งาน Gemini API เต็มชั่วคราว กรุณารอสักครู่ (1 นาที) แล้วลองกดปุ่มใหม่อีกครั้ง"
        
    except Exception as e:
        return f"Gemini Error: {str(e)[:100]}"

# STREAMLIT UI 
def main():
    # Header
    st.title("🍄 ระบบวิเคราะห์เห็ดด้วย AI")
    st.caption("RAG + CLIP | Accuracy: 86.98% | Safe Mode")
    
    # Check KB
    if kb is None or faiss_index is None:
        st.error("ระบบไม่สามารถโหลดฐานความรู้ได้")
        st.warning("""
        **ไฟล์ที่ต้องการ:** `mushroom_knowledge_base.pkl`
        
        **วิธีแก้ไข:**
        
        1. **Download KB file จาก Google Drive:**
           - เปิด Colab notebook
           - รัน: `from google.colab import files`
           - รัน: `files.download('/content/drive/MyDrive/mushroom_knowledge_base.pkl')`
        
        2. **Upload ไฟล์ไปที่ GitHub repo:**
           - ใส่ไฟล์ `mushroom_knowledge_base.pkl` ที่ root directory
           - Commit และ push
        
        3. **ถ้าไฟล์ใหญ่เกิน 100MB:**
           - ใช้ Git LFS: `git lfs track "*.pkl"`
           - หรือ upload ไปที่ cloud storage แล้วดาวน์โหลดใน app
        
        4. **สำหรับ Local testing:**
           - วางไฟล์ `.pkl` ไว้ใน root folder เดียวกับ `streamlit_app.py`
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.subheader("เกี่ยวกับระบบ")
        
        st.markdown(f"""
        #### ข้อมูลเทคนิค
        - **Model:** CLIP + RAG
        - **Accuracy:** 86.98%
        - **Dataset:** {len(kb)} species
        - **Threshold:** {SAFETY_THRESHOLD:.0%}
        - **Top-K:** {TOP_K}
        
        #### คำเตือน
        - ระบบมี error rate ~13%
        - ห้ามใช้เป็นหลักฐานเดียว
        - ควรปรึกษาผู้เชี่ยวชาญ
        - เห็ดพิษอันตรายมาก
        
        #### วิธีใช้
        1. อัปโหลดภาพเห็ดที่ชัดเจน
        2. ดูผลการวิเคราะห์
        3. กดขอรายละเอียดเพิ่มเติม (ถ้าต้องการ)
        """)
        
        st.divider()
        
        if gemini_model:
            st.success("Gemini: พร้อมใช้งาน")
        else:
            st.warning("Gemini: ไม่พร้อมใช้งาน")
    
    # Main content
    st.subheader("อัปโหลดภาพเห็ด")
    
    uploaded_file = st.file_uploader(
        "เลือกไฟล์รูปภาพ (JPG, PNG, HEIC)",
        type=["jpg", "jpeg", "png", "heic"],
        help="ถ่ายภาพให้ชัด เห็นหมวก ครีบ และก้านเห็ด",
        on_change=lambda: st.session_state.update({
            'analysis_done': False,
            'gemini_details': None
        })
    )
    
    if not uploaded_file:
        st.info("กรุณาอัปโหลดภาพเห็ด")
        
        # Example images (optional)
        st.caption("Tips: ถ่ายภาพหลายมุม (บน, ล่าง, ข้าง) จะช่วยให้วิเคราะห์แม่นยำขึ้น")
        return
    
    # Load and display image
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)
        
    except Exception as e:
        st.error(f"ไม่สามารถเปิดไฟล์ได้: {e}")
        st.stop()
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'classification' not in st.session_state:
        st.session_state.classification = None
    if 'gemini_details' not in st.session_state:
        st.session_state.gemini_details = None
    
    # Analyze button
    st.divider()
    
    analyze_clicked = st.button("🔍 วิเคราะห์เห็ด", type="primary", use_container_width=True)
    
    # Run analysis if button clicked
    if analyze_clicked:
        with st.spinner("กำลังวิเคราะห์ . ."):
            classification, confidence, reasoning, retrieved = predict_with_enhanced_rag(image)
        
        # Store in session state
        st.session_state.analysis_done = True
        st.session_state.classification = classification
        st.session_state.confidence = confidence
        st.session_state.reasoning = reasoning
        st.session_state.retrieved = retrieved
        st.session_state.gemini_details = None  # Reset Gemini details
    
    # Display results if analysis is done
    if st.session_state.analysis_done:
        classification = st.session_state.classification
        confidence = st.session_state.confidence
        reasoning = st.session_state.reasoning
        retrieved = st.session_state.retrieved
        
        # RESULTS 
        st.subheader("ผลการวิเคราะห์")
        
        # Main classification
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if classification == "Edible":
                st.success("### กินได้")
                st.caption("(ตามการประเมินของระบบ)")
            elif classification == "Poisonous":
                st.error("### มีพิษ / ไม่แนะนำ")
                st.caption("(อย่ารับประทาน)")
            else:
                st.warning("### ไม่สามารถระบุได้")
                st.caption("(อย่ารับประทาน)")
        
        with col2:
            st.metric("ความมั่นใจ", f"{confidence:.1%}")
        
        # Warning box - แก้ Logic ให้ถูกต้อง
        if classification == "Poisonous":
            st.error("""
            **คำเตือนสำคัญ:**
            - **ห้ามรับประทานโดยเด็ดขาด**
            - เห็ดพิษอันตรายมาก
            - หากรับประทานไป โทร 1669 ทันที
            """)
        elif classification == "Edible" and confidence < 0.75:
            st.warning("""
            **ข้อควรระวัง:**
            - ระบบมีความมั่นใจปานกลาง (< 75%)
            - **ควรให้ผู้เชี่ยวชาญตรวจสอบก่อน**
            - อย่าใช้ AI เป็นหลักฐานเดียว
            - ปรุงสุกก่อนรับประทาน
            """)
        elif classification == "Edible" and confidence >= 0.75:
            st.info("""
            **ข้อมูล:**
            - ระบบมีความมั่นใจสูง
            - **แต่ยังคงควรให้ผู้เชี่ยวชาญยืนยัน**
            - อย่าใช้ AI เป็นหลักฐานเดียว
            - ปรุงสุกก่อนรับประทาน
            """)
        
        st.divider()
        
        # Similar species
        if retrieved:
            st.subheader("สายพันธุ์ที่คล้ายกัน (จาก Knowledge Base)")
            
            cols = st.columns(min(3, len(retrieved)))
            for i, example in enumerate(retrieved[:3]):
                with cols[i]:
                    species_name = example['species'].replace('_', ' ').title()
                    similarity = example['similarity']
                    label = example['label']
                    
                    st.markdown(f"**{i+1}. {species_name}**")
                    st.caption(f"Similarity: {similarity:.1%}")
                    
                    if label == "Edible":
                        st.success(f"🟢 {label}")
                    else:
                        st.error(f"🔴 {label}")
        
        st.divider()
        

        
        # Gemini detailed info
        st.divider()
        st.subheader("ข้อมูลเพิ่มเติม")
        
        if gemini_model:
            if st.button("ขอข้อมูลละเอียดจาก AI ผู้ช่วย (Gemini)", use_container_width=True):
                primary_species = retrieved[0]['species'] if retrieved else "unknown"
                
                with st.spinner("Gemini กำลังสรุปข้อมูล..."):
                    # แก้ไขการเรียกใช้ฟังก์ชัน: ส่ง 'retrieved' เข้าไปด้วย
                    details = ask_gemini_for_details(
                        primary_species, 
                        classification, 
                        confidence, 
                        retrieved # <--- แก้ไขตรงนี้
                    )
                
                st.markdown("### คำอธิบายจาก AI")
                st.info(details)
                
                st.caption("ข้อมูลจาก AI อาจไม่สมบูรณ์ ควรตรวจสอบกับแหล่งอ้างอิงที่เชื่อถือได้")
        else:
            st.warning("Gemini API ไม่พร้อมใช้งาน - ตั้งค่า GEMINI_API_KEY ใน Streamlit Secrets")
    
    # Footer
    st.divider()
    st.caption("Powered by RAG + CLIP + Gemini | For Educational Purposes Only")
    st.caption("ระบบนี้ไม่ใช่การวินิจฉัยทางการแพทย์ - อย่าใช้ตัดสินใจบริโภคจริง")

if __name__ == "__main__":
    main()
