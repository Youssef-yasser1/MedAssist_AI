import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import requests  # هنستخدم دي عشان نكلم جوجل مباشرة

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- الموديل المحلي (DenseNet) ---
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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

# دالة ذكية لإرسال النص لجوجل بدون مكتبة (عشان نتفادى الـ 404)
def ask_gemini_direct(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error from Google API: {response.status_code} - {response.text}"

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        api_key = os.getenv("GEMINI_API_KEY")
        
        # التقاط الرسالة من Lovable
        user_message = ""
        if "messages" in data and len(data["messages"]) > 0:
            user_message = data["messages"][-1].get("content", "")
        elif "message" in data:
            user_message = data["message"]
            
        image_data = data.get("image")

        # 1. لو فيه صورة
        if image_data and medical_model:
            encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = medical_model(input_tensor)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).nonzero(as_tuple=True)[1]
                diseases = [class_names[i] for i in preds]
            
            res_text = ", ".join(diseases) if diseases else "نتائج سليمة"
            prompt = f"المريض رفع أشعة والتحليل الأولي: {res_text}. اشرح ده بالعربي باختصار وبأسلوب طبي."
            ai_response = ask_gemini_direct(prompt, api_key)
            return {"response": ai_response, "analysis": diseases}

        # 2. لو نص بس (زي كلمة كحة)
        if user_message:
            ai_response = ask_gemini_direct(f"أنت مساعد طبي في Healytics. رد بالعربي على: {user_message}", api_key)
            return {"response": ai_response}

        return {"response": "أهلاً بك في Healytics! كيف يمكنني مساعدتك؟"}

    except Exception as e:
        return {"response": f"حدث خطأ في النظام: {str(e)}"}