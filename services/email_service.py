import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

class EmailService:
    @staticmethod
    def send_otp_email(recipient_email, otp_code):
        load_dotenv(override=True)
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        login_user = os.environ.get("SMTP_EMAIL", "").strip()
        sender_password = os.environ.get("SMTP_PASSWORD", "").strip()
        from_email = os.environ.get("SENDER_EMAIL", "").strip() or os.environ.get("MAIL_FROM", "").strip() or os.environ.get("SMTP_FROM", "").strip() or login_user

        if not login_user or not sender_password:
            print("[EmailService Warning] SMTP credentials (SMTP_EMAIL, SMTP_PASSWORD) not configured in .env")
            return False, "SMTP credentials not configured in environment variables. Please set SMTP_EMAIL and SMTP_PASSWORD in .env."

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "TumorAI XAI - Password Reset OTP"
        msg["From"] = f"TumorAI Security <{from_email}>"
        msg["To"] = recipient_email

        text_content = f"""Hello,

You requested a password reset for your TumorAI XAI account.

Your One-Time Password (OTP) code is: {otp_code}

This code will expire in 10 minutes. If you did not request a password reset, please ignore this email.

Best regards,
TumorAI Clinical Platform Security Team
"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            body {{ font-family: 'Plus Jakarta Sans', Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }}
            .card {{ max-width: 500px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 32px; box-shadow: 0 4px 14px rgba(0,0,0,0.05); }}
            .brand {{ font-size: 20px; font-weight: 800; color: #2563eb; margin-bottom: 20px; }}
            .otp-box {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; font-size: 32px; font-weight: 800; color: #2563eb; letter-spacing: 6px; text-align: center; padding: 16px; margin: 24px 0; }}
            .footer {{ font-size: 12px; color: #64748b; margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 16px; }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="brand">🧠 TumorAI XAI Security</div>
            <h2 style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">Password Reset Request</h2>
            <p style="font-size: 14px; color: #475569; line-height: 1.5;">
              Use the following One-Time Password (OTP) to verify your account and reset your password:
            </p>
            <div class="otp-box">{otp_code}</div>
            <p style="font-size: 13px; color: #e11d48; font-weight: 600;">
              ⚠️ This code expires in 10 minutes. Do not share this code with anyone.
            </p>
            <div class="footer">
              If you did not request a password reset, you can safely ignore this email.<br>
              © 2026 TumorAI XAI Clinical Platform.
            </div>
          </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        try:
            print(f"[EmailService] Connecting to {smtp_server}:{smtp_port}... From: {from_email} -> To: {recipient_email}")
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(login_user, sender_password)
                server.sendmail(from_email, recipient_email, msg.as_string())
            print(f"[EmailService] Successfully sent OTP email to {recipient_email}")
            return True, "OTP email sent successfully."
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            print(f"[EmailService Error] {error_msg}")
            return False, error_msg

    @staticmethod
    def send_urgent_triage_alert(recipient_email, scan_info, patient_code="N/A"):
        """
        Sends an automated urgent triage alert to attending clinical staff when a high-risk scan is detected.
        """
        load_dotenv(override=True)
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        login_user = os.environ.get("SMTP_EMAIL", "").strip()
        sender_password = os.environ.get("SMTP_PASSWORD", "").strip()
        from_email = os.environ.get("SENDER_EMAIL", "").strip() or os.environ.get("MAIL_FROM", "").strip() or os.environ.get("SMTP_FROM", "").strip() or login_user

        if not login_user or not sender_password:
            print("[EmailService Notice] Triage Alert logged locally (SMTP not configured).")
            return False, "SMTP not configured."

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 URGENT CLINICAL ALERT — High Priority Scan Detected ({patient_code})"
        msg["From"] = f"TumorAI Triage System <{from_email}>"
        msg["To"] = recipient_email

        pred_class = scan_info.get("predicted_class", "Tumor Detected")
        confidence = round(float(scan_info.get("confidence_score", 0)) * 100, 1)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #fff1f2; padding: 20px;">
          <div style="max-width: 550px; margin: 0 auto; background: #ffffff; border: 2px solid #e11d48; border-radius: 12px; padding: 24px;">
            <h2 style="color: #e11d48; margin-top: 0;">🚨 Priority 1 Clinical Triage Alert</h2>
            <p><strong>Patient Code:</strong> {patient_code}</p>
            <p><strong>AI Diagnosis:</strong> {pred_class.upper()} ({confidence}% Confidence)</p>
            <p><strong>Status:</strong> Marked for immediate radiologist review.</p>
            <hr style="border: 0; border-top: 1px solid #ffe4e6;">
            <p style="font-size: 12px; color: #9f1239;">This is an automated system alert generated by TumorAI XAI Decision Support.</p>
          </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_content, "html"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(login_user, sender_password)
                server.sendmail(from_email, recipient_email, msg.as_string())
            return True, "Triage alert sent."
        except Exception as e:
            return False, str(e)

