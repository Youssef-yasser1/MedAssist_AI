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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- تحميل موديل الأشعة ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_medical_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, 14))
    if os.path.exists('best_densenet121.pth'):
        checkpoint = torch.load('best_densenet121.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        return model
    return None

medical_model = load_medical_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        
        # 🎯 "الرادار": محاولة استخراج الرسالة مهما كان اسمها في الـ JSON
        user_message = data.get("message") or data.get("prompt") or data.get("text")
        
        # لو مبعوتة كـ Array (نظام OpenAI اللي Lovable بيحبه ساعات)
        if not user_message and "messages" in data:
            user_message = data["messages"][-1].get("content")

        image_data = data.get("image")
        
        # إعداد Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return {"response": "API Key Missing from Secrets!"}
        genai.configure(api_key=api_key)
        model_gemini = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # 1️⃣ حالة الأشعة (السيستم الداخلي + Gemini)
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
            prompt = f"المريض رفع أشعة والتحليل هو: {res_text}. اشرح ده بالعربي بأسلوب طبي مطمئن."
            response = model_gemini.generate_content(prompt)
            return {"response": response.text, "analysis": diseases, "confidence": round(float(probs.max()), 2)}

        # 2️⃣ حالة الدردشة النصية (Gemini API)
        if user_message:
            full_prompt = f"أنت مساعد طبي ذكي في نظام Healytics. أجب بالعربي وبشكل إنساني: {user_message}"
            response = model_gemini.generate_content(full_prompt)
            return {"response": response.text}

        return {"response": "أهلاً بك في Healytics! أنا جاهز لمساعدتك، هل تريد تحليل أشعة أم لديك سؤال طبي؟"}

    except Exception as e:
        return {"response": f"عذراً، حدث خطأ فني: {str(e)}"}