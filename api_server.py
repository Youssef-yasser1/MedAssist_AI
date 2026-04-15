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

# 1. تفعيل الـ CORS بشكل كامل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. تحميل موديل الأشعة (DenseNet121)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            model.to(device)
            model.eval()
            print("✅ Medical Model Loaded Successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading .pth file: {e}")
            return None
    else:
        print("⚠️ Model file best_densenet121.pth not found!")
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

@app.get("/")
def home():
    return {"status": "Healytics Backend is Online"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # التأكد من وجود الـ API Key في كل طلب
        current_api_key = os.getenv("GEMINI_API_KEY")
        if not current_api_key:
            return {"response": "خطأ: لم يتم العثور على API Key في إعدادات Hugging Face."}
        
        genai.configure(api_key=current_api_key)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')

        data = await request.json()
        user_message = data.get("message", "")
        image_data = data.get("image")

        # الحالة 1: لو فيه صورة
        if image_data and medical_model:
            try:
                # معالجة الـ Base64 مع التأكد من وجود الـ header
                if "," in image_data:
                    header, encoded = image_data.split(",", 1)
                else:
                    encoded = image_data
                
                image_bytes = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = medical_model(input_tensor)
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).nonzero(as_tuple=True)[1]
                    detected_diseases = [class_names[i] for i in predictions]

                result_text = ", ".join(detected_diseases) if detected_diseases else "لا توجد نتائج واضحة"
                prompt = f"المريض رفع صورة أشعة صدر، والتحليل الأولي هو: {result_text}. اشرح للمريض ماذا يعني هذا بالعربي بأسلوب طبي مطمئن ووجهه للخطوات القادمة."
                response = model_gemini.generate_content(prompt)
                
                return {
                    "response": response.text,
                    "analysis": detected_diseases,
                    "confidence": round(float(probabilities.max()), 2)
                }
            except Exception as img_err:
                return {"response": f"حدث خطأ أثناء تحليل الصورة: {str(img_err)}"}

        # الحالة 2: دردشة نصية فقط
        if user_message:
            full_prompt = f"أنت مساعد طبي ذكي في نظام Healytics. أجب باختصار ومودة بالعربي: {user_message}"
            response = model_gemini.generate_content(full_prompt)
            return {"response": response.text}
        
        return {"response": "كيف يمكنني مساعدتك اليوم؟"}

    except Exception as e:
        return {"response": f"عذراً، حدث خطأ فني: {str(e)}"}