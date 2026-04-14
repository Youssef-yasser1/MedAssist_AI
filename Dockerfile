# 1. استخدم نسخة بايثون خفيفة
FROM python:3.10

# 2. حدد مكان العمل جوه السيرفر
WORKDIR /code

# 3. انسخ ملف المكتبات ويسطبها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. انسخ باقي ملفات المشروع (الكود والموديل)
COPY . .

# 5. شغل السيرفر على بورت 7860 (ده البورت اللي HF بيحبه)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]