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

# --- إعداد الموديل المحلي ---
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
        data = await request.json()
        
        # 🎯 الصيد الصحيح بناءً على اللوجز بتاعتك:
        # لوفابل بيبعت داتا شكلها كدة: {'messages': [{'role': 'user', 'content': 'انا يوسف'}]}
        user_message = ""
        if "messages" in data and len(data["messages"]) > 0:
            user_message = data["messages"][-1].get("content", "")
        
        image_data = data.get("image")
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        # 🛑 إجبار Gemini على استخدام النسخة المستقرة v1 للهروب من الـ 404
        genai.configure(api_key=api_key, transport='rest')
        # هنا التكة: بنحدد الموديل بدون أي مقدمات Beta
        model = genai.GenerativeModel('gemini-1.5-flash')

        if user_message:
            # رد Gemini بذكاء
            response = model.generate_content(f"أنت مساعد طبي في نظام Healytics. المريض يقول: {user_message}. رد عليه بالعربي.")
            return {"response": response.text}
        
        return {"response": "أهلاً بك في Healytics، كيف يمكنني مساعدتك؟"}

    except Exception as e:
        # لو حصل إيرور هيطلع لنا هنا بالتفصيل
        return {"response": f"خطأ في السيرفر: {str(e)}"}