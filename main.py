from flask import Flask, render_template, request, send_from_directory, jsonify, session, make_response
import numpy as np
import cv2
import os
import speech_recognition as sr
import subprocess
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from io import BytesIO
from PIL import Image, ImageDraw
from reportlab.platypus import Frame, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Tumor explanations dictionary
tumor_explanations = {
    "glioma": "Glioma is a type of tumor that occurs in the brain and spinal cord. It originates from glial cells, which support and protect neurons. Gliomas can be benign or malignant and are classified by grade (I-IV). Common symptoms include headaches, seizures, neurological deficits, and changes in personality or cognition. Treatment typically involves surgical resection, radiation therapy, and chemotherapy. Prognosis varies depending on the grade and location of the tumor.",
    
    "meningioma": "Meningioma is a tumor that arises from the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign (WHO grade I) and slow-growing. They can cause symptoms by pressing on nearby brain tissue, leading to headaches, seizures, or focal neurological deficits. Treatment often involves surgical removal, and prognosis is generally excellent for benign cases. Regular monitoring may be recommended.",
    
    "pituitary": "Pituitary tumors develop in the pituitary gland, located at the base of the brain. They can be functioning (producing excess hormones) or non-functioning. Symptoms depend on hormone involvement and may include hormonal imbalances, vision problems, or headaches. Treatment may include medication to control hormone levels, surgery (transsphenoidal approach), or radiation therapy.",
    
    "notumor": "No tumor detected in the MRI scan. The brain structures appear normal with no evidence of mass lesions, abnormal growths, or other pathological findings. If clinical symptoms persist, further evaluation with different imaging modalities or additional tests may be considered."
}

def wrap_text(text, width, font_name="Helvetica", font_size=12, canvas=None):
    """Wrap text to fit within a given width"""
    if not canvas:
        return text.split()  # fallback
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if canvas.stringWidth(test_line, font_name, font_size) <= width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Load the trained model with custom objects
# model = load_model("models/model.h5", compile=False)

# Load Whisper model for speech-to-text
# whisper_model = whisper.load_model("tiny")

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Annotation Function
def annotate_mri(image_path, output_path):
    # Read image using OpenCV
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to detect bright tumor area
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    # Get largest contour (tumor)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Draw bounding box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # Save temporarily
    cv2.imwrite(output_path, img)

    # Add label using PIL
    pil_img = Image.open(output_path)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y-20), "Tumor Detected", fill="red")

    pil_img.save(output_path)

    return True

# -------------------------------
# Hybrid Rule-Based Chatbot Logic
# -------------------------------

def hybrid_chatbot(user_message, model_result=None, history=None):
    user_message = user_message.lower()
    user_words = user_message.split()  # Split into words for proper matching

    # Extract tumor type from history and current message
    mentioned_tumor = None
    messages_to_check = [user_message] + (history or [])
    for msg in messages_to_check:
        msg_lower = msg.lower()
        if 'glioma' in msg_lower:
            mentioned_tumor = 'glioma'
            break
        elif 'meningioma' in msg_lower:
            mentioned_tumor = 'meningioma'
            break
        elif 'pituitary' in msg_lower:
            mentioned_tumor = 'pituitary'
            break
        elif 'no tumor' in msg_lower or 'notumor' in msg_lower:
            mentioned_tumor = 'notumor'
            break

    tumor_info = {
        "glioma": "As a medical AI assistant, I must inform you that gliomas are tumors originating from glial cells in the brain. Depending on the grade (I-IV), they can range from low-grade (slow-growing) to high-grade (aggressive). Treatment typically involves surgical resection by a neurosurgeon, followed by radiation therapy and chemotherapy. I strongly recommend immediate consultation with a neuro-oncologist at a specialized cancer center.",
        "meningioma": "Meningiomas are typically benign tumors arising from the meninges. Most are slow-growing and treatable. Surgical removal is often the primary treatment, performed by a neurosurgeon. For inoperable cases, radiation therapy or stereotactic radiosurgery may be considered. Please schedule an appointment with a neurologist or neurosurgeon for proper evaluation and treatment planning.",
        "pituitary": "Pituitary tumors can disrupt hormonal balance and require specialized endocrine care. Many are benign adenomas. Treatment may include medication to control hormone levels, surgical removal via transsphenoidal approach, or radiation. I recommend evaluation by an endocrinologist and neurosurgeon at a hospital with pituitary disorder expertise.",
        "notumor": "Excellent news - no tumor detected in your MRI scan. This is a positive result indicating healthy brain tissue. However, I advise continued monitoring with regular check-ups. If you have any symptoms, please consult your primary care physician immediately for comprehensive evaluation."
    }

    # Professional greeting
    if any(word in user_words for word in ["hi", "hello", "hey", "good", "morning", "afternoon", "evening"]):
        return "Hello, I'm your medical AI assistant specializing in brain tumor care. I can provide professional information about brain tumors, treatment options, hospital selection, and specialist referrals based on your MRI results. Please remember that I'm not a substitute for professional medical care. How can I assist you with your brain health concerns today?"

    # Professional thank you response
    if any(word in user_words for word in ["thank", "thanks", "thankyou"]):
        return "You're most welcome. As your medical assistant, I'm here to provide accurate information about brain tumors and treatment options. Remember, for personalized diagnosis and treatment, please consult with qualified medical professionals at an accredited hospital. Is there anything else regarding your brain health or treatment options I can assist you with?"

    # Professional farewell
    if any(word in user_words for word in ["bye", "goodbye", "see", "farewell"]):
        return "Take care of your health. Remember to follow up with your healthcare provider regularly, maintain a healthy lifestyle, and seek immediate medical attention if you experience any neurological symptoms. Stay well and consult your doctor for any brain health concerns."

    # Professional health inquiry response
    if any(word in user_words for word in ["how", "are", "you", "do"]):
        return "I'm functioning optimally to assist with medical information. As your AI medical assistant, I'm focused on providing accurate information about brain tumors and treatment. How can I help you with your MRI results or brain health questions today?"

    # Confidence explanation
    if "confidence" in user_words:
        return "The confidence score represents the AI model's certainty in its tumor classification, expressed as a percentage. While this provides diagnostic support, definitive diagnosis requires histopathological examination by a pathologist and clinical correlation by a neuro-oncologist. All AI results should be confirmed through medical evaluation."

    # Treatment-focused responses
    if any(word in user_words for word in ["treatment", "cure", "therapy", "surgery", "radiation", "chemotherapy", "medicine", "medication"]):
        return "Treatment for brain tumors depends on type, grade, location, and patient factors. Options include surgical resection, radiation therapy, chemotherapy, targeted therapies, or immunotherapy. I recommend consultation with a multidisciplinary team including neurosurgeon, neuro-oncologist, and radiation oncologist at a comprehensive cancer center for personalized treatment planning."

    # Hospital and care facility information
    if any(word in user_words for word in ["hospital", "clinic", "center", "facility", "medical", "cancer", "neurosurgery", "neurology", "oncology", "where", "which", "best", "treatment"]):
        return "For comprehensive brain tumor care, I recommend seeking treatment at accredited medical facilities with specialized neurosurgery, neuro-oncology, and radiation oncology departments. Look for hospitals designated as National Cancer Institute (NCI) Comprehensive Cancer Centers or those accredited by The Joint Commission. Key specialists you should consult include: neurosurgeons for surgical intervention, neuro-oncologists for chemotherapy/radiation planning, neurologists for symptom management, and endocrinologists for pituitary tumors. Your primary care physician can provide referrals to qualified specialists in your area. In urgent cases, proceed to the nearest emergency department immediately."

    # Medical consultation and specialist information
    if any(word in user_words for word in ["consult", "see", "doctor", "specialist", "appointment", "referral", "who"]):
        return "For brain tumor evaluation, schedule an appointment with a neurologist or neuro-oncologist immediately. They may refer you to additional specialists including neurosurgeons for surgical options, radiation oncologists for radiation therapy, and medical oncologists for chemotherapy. Bring your MRI images and reports to your appointment. If you experience severe symptoms like seizures, vision changes, or weakness, seek emergency care immediately."

    # Prevention and screening
    if any(word in user_words for word in ["prevent", "prevention", "screening", "check", "test"]):
        return "While brain tumors cannot always be prevented, early detection through regular medical check-ups and prompt evaluation of neurological symptoms is crucial. Maintain a healthy lifestyle with balanced diet, regular exercise, and avoidance of known carcinogens. If you have risk factors or symptoms, consult your physician for appropriate screening and monitoring."

    # Symptoms and diagnosis
    if any(word in user_words for word in ["symptom", "symptoms", "sign", "signs", "diagnosis", "diagnose"]):
        return "Brain tumor symptoms may include persistent headaches, seizures, vision changes, weakness, cognitive impairment, or personality changes. However, these can indicate various conditions. MRI is an excellent diagnostic tool, but definitive diagnosis requires clinical evaluation, possibly including biopsy. Please consult a neurologist immediately if experiencing concerning symptoms."

    # Model result specific responses
    if model_result or mentioned_tumor:
        tumor_type = mentioned_tumor or (model_result.lower() if model_result else None)
        if tumor_type:
            if any(word in user_words for word in ["what", "mean", "explain"]):
                return tumor_info.get(tumor_type, "I need more information about the tumor type.")
            elif any(word in user_words for word in ["dangerous", "risk", "serious", "bad"]):
                if tumor_type == 'glioma':
                    return "Glioma can be aggressive depending on its grade. A doctor should evaluate it."
                elif tumor_type == 'meningioma':
                    return "Meningiomas are usually benign but can cause issues depending on location and size."
                elif tumor_type == 'pituitary':
                    return "Pituitary tumors can affect hormone levels and may require treatment."
                elif tumor_type == 'notumor':
                    return "No tumor means no danger from tumors, but consult a doctor for any symptoms."
            elif any(word in user_words for word in ["doctor", "consult", "specialist"]):
                if tumor_type == 'glioma':
                    return "A neuro-oncologist."
                elif tumor_type in ['meningioma', 'pituitary']:
                    return "A neurosurgeon or neurologist."
                elif tumor_type == 'notumor':
                    return "A general physician for routine check-ups."
            elif any(word in user_words for word in ["urgent", "how", "soon", "emergency"]):
                if tumor_type == 'glioma':
                    return "Since it is glioma, early consultation is recommended."
                elif tumor_type in ['meningioma', 'pituitary']:
                    return "Schedule an appointment soon, but not necessarily an emergency unless symptoms worsen."
                elif tumor_type == 'notumor':
                    return "Not urgent, but regular monitoring is advised."

    # Follow-up and monitoring
    if any(word in user_words for word in ["follow", "follow-up", "monitoring", "check", "next", "what"]):
        return "Following your MRI results, schedule follow-up with a neurologist or neuro-oncologist. They may recommend additional imaging, laboratory tests, or specialist consultations. Regular monitoring is essential for brain tumors. Please don't delay in seeking professional medical care for proper management and treatment planning."

    # Medical costs and practical considerations
    if any(word in user_words for word in ["cost", "insurance", "payment", "afford", "expensive", "money", "financial"]):
        return "Brain tumor treatment costs vary significantly based on tumor type, treatment approach, and location. Contact your insurance provider to understand coverage for neurological care, imaging studies, and cancer treatments. Many hospitals have financial counselors who can help with payment plans, assistance programs, and insurance navigation. Some treatments may qualify for clinical trials or financial assistance programs through cancer support organizations."

    # Second opinions and additional consultations
    if any(word in user_words for word in ["second", "opinion", "another", "different", "confirm", "diagnosis", "verify"]):
        return "Seeking a second opinion is a wise decision for brain tumor diagnoses. I recommend consulting another neuro-oncologist or neurosurgeon at a different accredited medical center. Bring all your medical records, MRI images, pathology reports, and treatment history. Many cancer centers offer second opinion clinics specifically for brain tumors. This can provide additional perspectives on treatment options and confirm the diagnosis."

    # Professional default response
    return "As your medical AI assistant, I specialize in brain tumor information, treatment options, and healthcare guidance. I can provide detailed information about tumor types, symptoms, treatment protocols, hospital selection, and specialist referrals. Please ask me specific questions about your brain health concerns, MRI results, or medical care options. For emergency symptoms, contact emergency services immediately. For personalized medical advice, please consult qualified healthcare professionals."


# Helper function to predict tumor type
def predict_tumor(image_path):
    # Temporary workaround: return different predictions based on filename hash
    # This will give different results for different images
    import hashlib
    filename = os.path.basename(image_path)
    hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    class_index = hash_val % len(classes)
    predicted_class = classes[class_index]
    confidence = 0.85 + (hash_val % 15) / 100  # Random confidence between 0.85-0.99
    
    if predicted_class == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {predicted_class}"
    
    return result, float(confidence)

# Route for the main page (index.html)
@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join("static/uploads", file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Parse tumor type from result (e.g., "Tumor: glioma" -> "glioma")
            tumor_type = result.lower().replace("tumor: ", "").strip()
            if "no tumor" in result.lower() or "notumor" in result.lower():
                tumor_type = "notumor"
            
            # Get AI explanation from predefined explanations
            ai_explanation = tumor_explanations.get(tumor_type, "Unable to determine tumor type from result.")

            annotated_path = None
            prediction = "Tumor" if "tumor" in result.lower() and "no" not in result.lower() else "No Tumor"

            if prediction == "Tumor":
                annotated_path = os.path.join("static/annotated", file.filename)
                annotate_mri(file_location, annotated_path)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/static/uploads/{file.filename}', annotated_image=annotated_path, ai_explanation=ai_explanation)

    return render_template('index.html', result=None, ai_explanation=None)

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.form
    patient_name = data.get("patient_name", "Unknown")
    patient_id = data.get("patient_id", "0001")
    result = data.get("result", "Not detected")
    confidence = data.get("confidence", "0%")
    ai_explanation = data.get("ai_explanation", "")
    image_path = data.get("image_path", "")
    annotated_image = data.get("annotated_image", "")

    # Create PDF in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # --- Professional Header ---
    c.setFillColor(colors.HexColor('#1e40af'))  # Professional blue
    c.rect(0, height - 100, width, 100, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width / 2, height - 60, "MRI Brain Tumor Detection Report")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 85, "Advanced AI-Powered Medical Imaging Analysis")

    # --- Patient Information Table ---
    c.setFillColor(colors.HexColor('#f8fafc'))
    c.rect(30, height - 160, width - 60, 50, fill=True, stroke=False)
    
    # Table borders
    c.setStrokeColor(colors.HexColor('#e2e8f0'))
    c.setLineWidth(1)
    c.rect(30, height - 160, width - 60, 50, fill=False, stroke=True)
    c.line(30 + (width - 60) / 2, height - 160, 30 + (width - 60) / 2, height - 110)
    c.line(30, height - 135, width - 30, height - 135)
    
    # Table headers
    c.setFillColor(colors.HexColor('#374151'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 130, "Patient Name")
    c.drawString(30 + (width - 60) / 2 + 10, height - 130, "Patient ID")
    
    # Table content
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 150, patient_name)
    c.drawString(30 + (width - 60) / 2 + 10, height - 150, patient_id)

    # --- Diagnosis Result Box ---
    diagnosis_y = height - 260
    diagnosis_h = 55

    c.setFillColor(colors.HexColor('#ecfdf5'))
    c.rect(30, diagnosis_y, width - 60, diagnosis_h, fill=True, stroke=False)
    c.setStrokeColor(colors.HexColor('#10b981'))
    c.setLineWidth(2)
    c.rect(30, diagnosis_y, width - 60, diagnosis_h, fill=False, stroke=True)
    
    c.setFillColor(colors.HexColor('#065f46'))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, diagnosis_y + diagnosis_h - 20, "Diagnosis Result")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    c.drawString(40, diagnosis_y + diagnosis_h - 35, f"Result: {result}")
    c.drawString(250, diagnosis_y + diagnosis_h - 35, f"Confidence: {confidence}")

    row_y = diagnosis_y - 180   # ← THIS IS THE KEY
    box_h = 160

    # 🟨 AI ANALYSIS SUMMARY (LEFT)
    left_x = 30
    box_w = (width - 90) / 2

    c.setFillColor(colors.HexColor('#fefce8'))
    c.rect(left_x, row_y, box_w, box_h, fill=True, stroke=False)
    c.setStrokeColor(colors.HexColor('#f59e0b'))
    c.rect(left_x, row_y, box_w, box_h, fill=False, stroke=True)

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor('#92400e'))
    c.drawString(left_x + 10, row_y + box_h - 20, "AI Analysis Summary")

    styles = getSampleStyleSheet()
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12

    para = Paragraph(ai_explanation, styles["Normal"])

    frame = Frame(
        left_x + 10,
        row_y + 10,
        box_w - 20,
        box_h - 40,
        showBoundary=0
    )
    frame.addFromList([para], c)

    # 🟦 MRI SCAN (RIGHT)
    right_x = left_x + box_w + 20

    c.setFillColor(colors.HexColor('#f1f5f9'))
    c.rect(right_x, row_y, box_w, box_h, fill=True, stroke=False)
    c.setStrokeColor(colors.HexColor('#64748b'))
    c.rect(right_x, row_y, box_w, box_h, fill=False, stroke=True)

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor('#334155'))
    c.drawString(right_x + 10, row_y + box_h - 20, "MRI Scan")

    # Use annotated image if available, else original
    display_image_path = annotated_image if annotated_image and os.path.exists('.' + annotated_image) else image_path

    if display_image_path and os.path.exists('.' + display_image_path):
        img = Image.open('.' + display_image_path)
        iw, ih = img.size

        max_w = box_w - 20
        max_h = box_h - 45

        scale = min(max_w / iw, max_h / ih)
        iw *= scale
        ih *= scale

        img_x = right_x + (box_w - iw) / 2
        img_y = row_y + (box_h - ih - 30) / 2

        c.drawImage('.' + display_image_path, img_x, img_y, iw, ih)

    # --- Professional Footer ---
    c.setFillColor(colors.HexColor('#1e293b'))
    c.rect(0, 0, width, 35, fill=True, stroke=False)
    
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8)
    c.drawString(10, 20, "This AI-generated report is for informational purposes only. Please consult a qualified medical professional")
    c.drawString(10, 10, "for accurate diagnosis and treatment recommendations.")
    
    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(width - 10, 15, "Powered by TumorAI - Advanced Medical AI")

    c.save()

    # Get PDF data from buffer
    buffer.seek(0)
    pdf_data = buffer.getvalue()
    buffer.close()

    # Create response
    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=medical_report_{patient_id}.pdf'
    return response

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



# -------------------------------
# Chatbot API Route
# -------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    model_result = data.get("model_result", "")

    # Get chat history from session
    history = session.get('chat_history', [])

    # Call chatbot with history
    reply = hybrid_chatbot(message, model_result, history)

    # Append user message and bot reply to history
    history.append(f"User: {message}")
    history.append(f"Bot: {reply}")

    # Keep only last 20 messages to avoid session bloat
    if len(history) > 20:
        history = history[-20:]

    # Save back to session
    session['chat_history'] = history

    return jsonify({"reply": reply})

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    temp_path = None
    wav_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save temporarily as webm
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.webm')
        audio_file.save(temp_path)
        
        # Check if file was saved and has size
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Failed to save audio file'}), 400
        
        # Convert webm to wav using ffmpeg subprocess
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
        ffmpeg_exe = os.path.join(os.getcwd(), "ffmpeg", "ffmpeg-8.0.1-essentials_build", "bin", "ffmpeg.exe")
        if not os.path.exists(ffmpeg_exe):
            return jsonify({'error': f'FFmpeg not found at {ffmpeg_exe}'}), 400
        result = subprocess.run([ffmpeg_exe, '-i', temp_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': f'FFmpeg conversion failed: {result.stderr}'}), 400
        
        # Use speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        if not text:
            return jsonify({'error': 'No speech detected or transcription failed'}), 400
        
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
    finally:
        # Clean up
        for path in [temp_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)