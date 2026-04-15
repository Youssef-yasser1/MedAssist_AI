import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعدادات سريعة للموديل المحلي ---
device = torch.device("cpu")
def load_medical_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, 14))
    if os.path.exists('best_densenet121.pth'):
        try:
            checkpoint = torch.load('best_densenet121.pth', map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            return model
        except: return None
    return None

medical_model = load_medical_model()

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        raw_data = await request.json()
        # أهم سطر: هنطبع اللي Lovable بيبعته عشان نشوفه في الـ Logs
        print(f"DEBUG: Data from Lovable: {raw_data}")

        # محاولة استخراج الرسالة بكل الطرق
        user_message = raw_data.get("message") or raw_data.get("prompt") or ""
        if not user_message and "messages" in raw_data:
            user_message = raw_data["messages"][-1].get("content", "")

        api_key = os.getenv("GEMINI_API_KEY")
        
        # --- الطريقة اليدوية لضمان عدم حدوث 404 ---
        genai.configure(api_key=api_key)
        # جرب نكتب اسم الموديل بالكامل بالمسار بتاعه
        model = genai.GenerativeModel('models/gemini-1.5-flash')

        if user_message:
            # تجربة إرسال النص
            response = model.generate_content(f"أجب باختصار بالعربي: {user_message}")
            return {"response": response.text}
        
        return {"response": f"وصلتني رسالة فاضية! الداتا اللي جاتلي هي: {str(raw_data)}"}

    except Exception as e:
        # لو حصل 404 تاني، اطبعه هنا بالتفصيل
        return {"response": f"حصل إيرور جديد: {str(e)}"}