import streamlit as st
import google.generativeai as genai

# โหลด API key จาก secrets.toml
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ใช้โมเดล Gemini (รุ่นที่เร็วและฟรี)
model = genai.GenerativeModel("gemini-2.0-flash")

# ตั้งค่าหน้า Streamlit
st.set_page_config(page_title="🍄 Mushroom Classifier", page_icon="🍄")
st.title("🍄 Mushroom Classification using Gemini API")
st.write("อัปโหลดภาพเห็ด แล้วให้ Gemini ช่วยวิเคราะห์ว่าเห็ดนี้กินได้หรือไม่")

# ส่วนอัปโหลดภาพ
uploaded_file = st.file_uploader("📸 อัปโหลดภาพเห็ด (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="ภาพที่อัปโหลด", use_column_width=True)
    
    if st.button("🔍 วิเคราะห์ภาพ"):
        with st.spinner("กำลังวิเคราะห์ด้วย Gemini..."):
            try:
                # วิเคราะห์ภาพด้วย Gemini (multimodal)
                response = model.generate_content([
                    "Analyze this image and tell me if the mushroom is edible or poisonous. Give reasoning briefly.",
                    genai.upload_file(uploaded_file)
                ])
                st.success("✅ ผลการวิเคราะห์:")
                st.write(response.text)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    st.info("กรุณาอัปโหลดภาพก่อนครับ")

st.markdown("---")
st.caption("สร้างด้วย ❤️ โดย Streamlit + Gemini API")

try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("ทดสอบการเชื่อมต่อกับ Gemini API")
    st.success("✅ Gemini API พร้อมใช้งานแล้ว!")
    st.write(response.text)
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
