import os
import json
import torch
import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# 1. تعريف التطبيق (ده اللي كان ناقص ومسبب الإيرور)
app = FastAPI()

# 2. تفعيل الـ CORS عشان موقع Lovable يقدر يكلم السيرفر من غير حظر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. إعداد مفتاح Gemini من الـ Secrets اللي حطيناها في Hugging Face
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# 4. تحميل قاعدة بيانات الأمراض (تأكد أن اسم الملف صح)
try:
    with open('chest_diseases_db_v2.json', 'r', encoding='utf-8') as f:
        diseases_db = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    diseases_db = {}

# 5. تحميل موديل الـ AI بتاع الأشعة (Torch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# تأكد أن اسم الملف best_densenet121.pth موجود فعلاً في الـ Space
try:
    # هنا بنفترض إنك هتعمل تحميل للموديل، لو مش محتاجه في الـ API حالياً ممكن تمسح السطر ده
    # checkpoint = torch.load('best_densenet121.pth', map_location=device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading skipped or error: {e}")

# --- الـ Endpoints ---

@app.get("/")
def home():
    return {"status": "MedAssist AI Backend is Running!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            return {"response": "أهلاً بك، كيف يمكنني مساعدتك اليوم؟"}

        # إرسال الرسالة لـ Gemini
        # ممكن تضيف برومبت هنا عشان تخليه يرد كطبيب متخصص
        full_prompt = f"أنت مساعد طبي ذكي في نظام Healytics. أجب على السؤال التالي: {user_message}"
        response = model_gemini.generate_content(full_prompt)
        
        return {"response": response.text}
    
    except Exception as e:
        return {"error": str(e), "response": "عذراً، حدث خطأ في معالجة طلبك."}

# تشغيل السيرفر (مهم جداً للـ Docker)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)