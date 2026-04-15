import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import requests

app = FastAPI()

# تفعيل الـ CORS للربط مع Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعداد الموديل المحلي (DenseNet121) ---
device = torch.device("cpu")
def load_medical_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 14)
    )
    
    if os.path.exists('best_densenet121.pth'):
        try:
            checkpoint = torch.load('best_densenet121.pth', map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            print("✅ Medical Model Loaded!")
            return model
        except Exception as e:
            print(f"❌ Error loading .pth: {e}")
            return None
    return None

medical_model = load_medical_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# --- دالة الاتصال المباشر بجوجل (الحل القاطع للـ 404) ---
def ask_gemini_direct(prompt, api_key):
    # استخدام v1beta لأنها النسخة المستقرة حالياً لموديل Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Request Failed: {str(e)}"

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "Healytics Backend is Ready", "model": medical_model is not None}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            return {"response": "❌ خطأ: لم يتم العثور على GEMINI_API_KEY في الـ Secrets."}

        # استقبال الرسالة من Lovable
        user_message = ""
        if "messages" in data and len(data["messages"]) > 0:
            user_message = data["messages"][-1].get("content", "")
        elif "message" in data:
            user_message = data["message"]
            
        image_data = data.get("image")

        # 1. تحليل الأشعة
        if image_data and medical_model:
            header, encoded = image_data.split(",", 1) if "," in image_data else (None, image_data)
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = medical_model(input_tensor)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).nonzero(as_tuple=True)[1]
                detected = [class_names[i] for i in preds]

            res_text = ", ".join(detected) if detected else "سليم"
            prompt = f"المريض رفع أشعة والذكاء الاصطناعي اكتشف: {res_text}. اشرح ده بالعربي بأسلوب طبي مطمئن."
            ai_res = ask_gemini_direct(prompt, api_key)
            return {"response": ai_res, "analysis": detected}

        # 2. دردشة نصية
        if user_message:
            ai_res = ask_gemini_direct(f"أنت مساعد طبي في نظام Healytics. رد بالعربي: {user_message}", api_key)
            return {"response": ai_res}

        return {"response": "أهلاً بك، أنا أسمعك جيداً."}

    except Exception as e:
        return {"response": f"❌ حدث خطأ: {str(e)}"}