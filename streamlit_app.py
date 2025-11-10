import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

st.title("🍄 Mushroom Classification using Gemini API")

uploaded_file = st.file_uploader("📸 อัปโหลดภาพเห็ด (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # แสดงภาพ
    st.image(uploaded_file, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        with st.spinner("กำลังวิเคราะห์ด้วย Gemini..."):
            try:
                # อ่านข้อมูลไฟล์และระบุ MIME TYPE ให้ชัดเจน
                image_bytes = uploaded_file.read()
                mime_type = uploaded_file.type  # เช่น 'image/jpeg'

                response = model.generate_content(
                    [
                        {
                            "role": "user",
                            "parts": [
                                {"text": "วิเคราะห์ภาพนี้ว่าเป็นเห็ดกินได้หรือมีพิษ พร้อมอธิบายสั้น ๆ เป็นภาษาไทย"},
                                {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
                            ],
                        }
                    ]
                )

                st.success("✅ ผลการวิเคราะห์:")
                st.write(response.text)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    st.info("กรุณาอัปโหลดภาพก่อนครับ")

st.markdown("---")
st.caption("สร้างด้วย ❤️ โดย Streamlit + Gemini API")
