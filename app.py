import streamlit as st
import google.generativeai as genai
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# --- 1. إعدادات Gemini ---
# تأكد إن الـ Key بتاعك صح ومفعل
API_KEY = "AIzaSyC_Bsz_7azkD-T9_309n7UbMmxbW1tfV3I"
genai.configure(api_key=API_KEY)

# جرب نستخدم 1.5 flash لأنه الأحدث ومتاح للجميع مجاناً
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except:
    # لو منفعش جرب الاسم القديم كخطة بديلة
    gemini_model = genai.GenerativeModel('gemini-pro')

# --- 2. تحميل عقل النظام (Model + JSON) ---
@st.cache_resource
def load_medical_assets():
    with open('chest_diseases_db_v2.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
    
    device = torch.device('cpu') # للتشغيل المستقر على اللاب توب
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 14))
    
    try:
        # تحميل الأوزان
        checkpoint = torch.load('best_densenet121.pth', map_location=device, weights_only=False)
        state_dict = checkpoint if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        st.error(f"Error loading pth: {e}")
    return db, model

medical_db, vision_model = load_medical_assets()

# --- 3. الواجهة الرسومية ---
st.set_page_config(page_title="MED:AI Assistant", layout="centered")
st.title("🏥 مساعد MED:AI الطبي")

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# منطقة المدخلات
with st.sidebar:
    st.header("التحكم")
    uploaded_file = st.file_uploader("ارفع صورة الأشعة", type=['png', 'jpg', 'jpeg'])

prompt = st.chat_input("اسألني أي سؤال عن حالتك...")

if prompt or uploaded_file:
    user_input = prompt if prompt else "تحليل صورة الأشعة المرفقة"
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        if uploaded_file: st.image(uploaded_file, width=300)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # أ) تحليل الصورة بموديلك (RAG Step 1)
        vision_info = "لم يتم تزويد أشعة."
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.sigmoid(vision_model(tensor)).numpy()[0]
                disease = list(medical_db.keys())[probs.argmax()]
                info = medical_db.get(disease, {})
                vision_info = f"تشخيص الموديل: {disease}. الأعراض: {info.get('symptoms')}. النصيحة: {info.get('advice')}."

        # ب) توليد الرد بـ Gemini (RAG Step 2)
        full_context = f"""
        أنت مساعد طبي ذكي. السياق: {vision_info}
        سؤال المريض: {prompt}
        المطلوب: اشرح الحالة بالعامية المصرية الودودة، استخدم المعلومات الطبية من السياق فقط، وانصح بزيارة الطبيب.
        """
        
        try:
            response = gemini_model.generate_content(full_context)
            ai_reply = response.text
            placeholder.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        except Exception as e:
            st.error(f"خطأ في Gemini: {e}")