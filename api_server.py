import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import google.generativeai as genai
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64

app = FastAPI()

# 1. تفعيل الـ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. إعداد Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# 3. تحميل موديل الأشعة (DenseNet121)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تعريف بنية الموديل (تأكد أنها مطابقة لتدريبك)
def load_medical_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    # التعديل هنا: غيرنا الـ classifier عشان يطابق الملف بتاعك
    # إحنا بنخليه Sequential وبنحط Linear في الخانة رقم 1
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), # ده بياخد مكان 0
        nn.Linear(num_ftrs, 14) # وده بياخد مكان 1 اللي الإيرور عايزه
    )
    
    if os.path.exists('best_densenet121.pth'):
        # إضافة ميزة الأمان للأوزان
        checkpoint = torch.load('best_densenet121.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ Medical Model Loaded Successfully!")
        return model
    else:
        print("⚠️ Model file not found!")
        return None

medical_model = load_medical_model()

# تحضير الصورة (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# قائمة الأمراض (ترتيبها لازم يكون نفس ترتيب التدريب)
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

@app.get("/")
def home():
    return {"status": "Healytics Backend is Ready for Text & Images!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_data = data.get("image") # استقبال الصورة كـ Base64 من Lovable

        # الحالة 1: لو فيه صورة (تحليل أشعة)
        if image_data and medical_model:
            # تحويل الـ Base64 لصورة
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # التنبؤ بالموديل
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = medical_model(input_tensor)
                probabilities = torch.sigmoid(outputs)
                # الحصول على الأمراض اللي نسبتها أعلى من 50%
                predictions = (probabilities > 0.5).nonzero(as_tuple=True)[1]
                detected_diseases = [class_names[i] for i in predictions]

            # دمج النتيجة مع Gemini عشان يشرحها
            result_text = ", ".join(detected_diseases) if detected_diseases else "No significant findings"
            prompt = f"المريض رفع صورة أشعة صدر، والذكاء الاصطناعي اكتشف احتمالية وجود: {result_text}. اشرح للمريض ماذا يعني هذا بأسلوب طبي مطمئن بالعربي ووجهه للخطوات القادمة."
            response = model_gemini.generate_content(prompt)
            
            return {
                "response": response.text,
                "analysis": detected_diseases,
                "confidence": probabilities.max().item()
            }

        # الحالة 2: دردشة نصية فقط
        full_prompt = f"أنت مساعد طبي ذكي في نظام Healytics. أجب باختصار: {user_message}"
        response = model_gemini.generate_content(full_prompt)
        return {"response": response.text}

    except Exception as e:
        return {"error": str(e), "response": "عذراً، واجهت مشكلة في تحليل البيانات."}