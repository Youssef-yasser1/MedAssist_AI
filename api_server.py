from fastapi import FastAPI, UploadFile, File, Form
import google.generativeai as genai
# ... نفس مكتبات Torch و PIL القديمة ...

@app.post("/chat")
async def chat_endpoint(
    text: str = Form(None), 
    file: UploadFile = File(None)
):
    vision_info = ""
    
    # 1. لو في صورة.. شغل موديل الأشعة
    if file:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert('RGB')
        # ... (كود المعالجة والتوقع بتاعك هنا) ...
        vision_info = f"نتيجة الأشعة: {disease}. النصيحة: {info.get('advice')}."

    # 2. ابعت الكلام (سواء معاه أشعة أو لأ) لـ Gemini
    # لو مفيش أشعة، الـ vision_info هتكون فاضية والرد هيكون عام
    full_prompt = f"السياق الطبي: {vision_info}\nسؤال المريض: {text}\nرد بالعامية المصرية كطبيب ودود."
    
    response = gemini_model.generate_content(full_prompt)
    
    return {
        "reply": response.text,
        "has_image": True if file else False,
        "diagnosis": disease if file else None
    }