import os
import json
import torch
import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 1. أهم سطر: تعريف الـ app لازم يكون فوق خالص
app = FastAPI()

# 2. تفعيل الـ CORS عشان Lovable يقدر يكلم السيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. إعداد Gemini (بيقرأ المفتاح من الـ Secrets أوتوماتيك)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# 4. الـ Endpoints
@app.get("/")
def home():
    return {"status": "MedAssist AI is Online!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            return {"response": "أهلاً بك في Healytics، كيف أساعدك؟"}

        # إرسال الرسالة لـ Gemini
        full_prompt = f"أنت مساعد طبي ذكي. أجب باختصار: {user_message}"
        response = model_gemini.generate_content(full_prompt)
        
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e), "response": "حدث خطأ في النظام."}