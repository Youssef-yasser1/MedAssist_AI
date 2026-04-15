import os
import json
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

# 1. إعدادات الـ CORS لضمان الربط مع Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. إعداد جهاز العمل (GPU لو متاح)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. تحميل موديل الأشعة (DenseNet121)
def load_medical_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 14)
    )
    
    model_path = 'best_densenet121.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            print("✅ Medical Model Loaded Successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading .pth file: {e}")
            return None
    else:
        print(f"⚠️ {model_path} not found!")
        return None

medical_model = load_medical_model()

# إعداد معالجة الصور
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

# 4. نقطة اختبار السيرفر
@app.get("/")
def home():
    return {"status": "Healytics Backend is LIVE", "model_loaded": medical_model is not None}

# 5. الموزع الرئيسي للطلبات (Text & Images)
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # قراءة الـ API Key من الـ Secrets في كل مرة لضمان التحديث
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"response": "⚠️ خطأ: API Key غير موجود في إعدادات Hugging Face Secrets."}

        # إعداد Gemini بالنسخة المستقرة لتجنب إيرور 404
        genai.configure(api_key=api_key, transport='rest')
        model_gemini = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # استلام البيانات
        data = await request.json()
        
        # محاولة صيد الرسالة النصية بأي مسمى يبعته Lovable
        user_message = data.get("message") or data.get("prompt") or data.get("text")
        if not user_message and "messages" in data:
            user_message = data["messages"][-1].get("content")
            
        image_data = data.get("image")

        # --- الحالة الأولى: تحليل أشعة (صورة + نص أو صورة فقط) ---
        if image_data and medical_model:
            try:
                # تنظيف Base64
                if "," in image_data:
                    encoded = image_data.split(",", 1)[1]
                else:
                    encoded = image_data
                
                image_bytes = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # التحليل بالموديل المحلي
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = medical_model(input_tensor)
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).nonzero(as_tuple=True)[1]
                    detected_diseases = [class_names[i] for i in predictions]

                # صياغة الرد بواسطة Gemini
                findings = ", ".join(detected_diseases) if detected_diseases else "No significant findings"
                prompt = f"المريض رفع صورة أشعة صدر. الموديل اكتشف: {findings}. اشرح هذا بالعربي بأسلوب طبي مطمئن وقدم نصائح عامة."
                
                response = model_gemini.generate_content(prompt)
                
                return {
                    "response": response.text,
                    "analysis": detected_diseases,
                    "confidence": round(float(probabilities.max()), 2) if detected_diseases else 0.0
                }
            except Exception as img_err:
                return {"response": f"❌ مشكلة في معالجة الصورة: {str(img_err)}"}

        # --- الحالة الثانية: دردشة نصية فقط ---
        if user_message:
            full_prompt = f"أنت مساعد طبي ذكي في نظام Healytics. أجب بالعربي بوضوح واختصار: {user_message}"
            response = model_gemini.generate_content(full_prompt)
            return {"response": response.text}

        return {"response": "أهلاً بك في Healytics! أنا جاهز لمساعدتك، هل تود تحليل أشعة أم لديك استفسار طبي؟"}

    except Exception as e:
        # إرجاع الخطأ الحقيقي للمساعدة في الـ Debugging
        return {"response": f"❌ حدث خطأ فني: {str(e)}"}