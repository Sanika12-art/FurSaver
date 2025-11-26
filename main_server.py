import os
import sys
import subprocess
import json
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from flask import Flask, request, render_template, redirect, url_for, jsonify, session
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pymongo import MongoClient
from fpdf import FPDF
from ultralytics import YOLO
import tensorflow as tf
import flask_mail
from flask_mail import Mail, Message
import sys
import signal
import multiprocessing
import time
from db_config import DatabaseConfig
import logging
import atexit
import sqlite3
import ssl
###############################################################################
# Configure Logging
###############################################################################
# Ensure logs directory exists
###############################################################################
# Configure Logging
###############################################################################
# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

class CustomFormatter(logging.Formatter):
    """Custom formatter that replaces emoji with text alternatives"""
    def format(self, record):
        if hasattr(record, 'msg'):
            record.msg = (str(record.msg)
                         .replace('✅', '[SUCCESS]')
                         .replace('❌', '[ERROR]')
                         .replace('⚠️', '[WARNING]'))
        return super().format(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_server.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Apply custom formatter to all handlers
logger = logging.getLogger('MainServer')
formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logger.handlers + logging.getLogger().handlers:
    handler.setFormatter(formatter)
###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv()  # Load .env file

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    logger.error("MONGO_URI environment variable not set")
    sys.exit(1)
PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com"  # Use live URL for production

###############################################################################
# Flask Configuration
###############################################################################
app = Flask(
    __name__,
    template_folder='frontend/templates',  # Where .html templates reside
    static_folder='frontend/static'        # Where static files reside
)

app.secret_key = os.environ.get("SECRET_KEY", "your-strong-secret-key")

# Define separate upload folders
FULL_ANIMAL_UPLOAD_FOLDER = 'frontend/static/f_upload'
DISEASE_UPLOAD_FOLDER = 'frontend/static/uploads'


# Create directories if they don't exist
for folder in [FULL_ANIMAL_UPLOAD_FOLDER, DISEASE_UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['FULL_ANIMAL_UPLOAD_FOLDER'] = FULL_ANIMAL_UPLOAD_FOLDER
app.config['DISEASE_UPLOAD_FOLDER'] = DISEASE_UPLOAD_FOLDER


###############################################################################
# Allowed File Checker
###############################################################################
def allowed_file(filename):
    """Check file extension to ensure it's PNG, JPG, or JPEG."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
DATABASE_DIR = 'database'
os.makedirs(DATABASE_DIR, exist_ok=True)

DB_PATH = os.path.join(DATABASE_DIR, 'local_storage.db')

def init_sqlite_db():
    """Initialize SQLite database with required tables."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Ensure predictions table exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_name TEXT,
                    pet_name TEXT,
                    pet_gender TEXT,
                    pet_type TEXT,
                    disease TEXT,
                    full_animal_image_path TEXT,
                    disease_image_path TEXT,
                    predicted_image_path TEXT,
                    timestamp TEXT,
                    symptoms TEXT,
                    mongo_id TEXT,
                    is_synced INTEGER DEFAULT 0,
                    error_message TEXT,
                    confidence REAL,
                    is_correct INTEGER DEFAULT 1,
                    correct_label TEXT,
                    corrected_image_path TEXT
                )
            ''')

            # Ensure donations table exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS donations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    donor_name TEXT,
                    donation_email TEXT,
                    paypal_email TEXT,
                    phone TEXT,
                    address TEXT,
                    amount_usd REAL,
                    amount_inr REAL,
                    currency TEXT,
                    exchange_rate REAL,
                    status TEXT,
                    transaction_id TEXT,
                    date TEXT,
                    invoice_path TEXT,
                    mongo_id TEXT,
                    is_synced INTEGER DEFAULT 0,
                    error_message TEXT
                )
            ''')

            conn.commit()
        logger.info("✅ SQLite database initialized successfully.")

    except Exception as e:
        logger.error(f"❌ SQLite database initialization failed: {e}")
        raise SystemExit(1)  # Stop execution if DB initialization fails


# Initialize SQLite
try:
    init_sqlite_db()
except SystemExit:
    exit(1)  # Critical failure, exit application


###############################################################################
# MongoDB Connection
###############################################################################

# Load MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGO_URI", None)

# Initialize database variables
mongo_client = None
db = None
predictions_coll = None
donations_coll = None

try:
    if MONGO_URI:
        # Configure MongoDB with SSL settings
        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # Increased timeout
            ssl=True,
            ssl_cert_reqs=ssl.CERT_NONE,  # Disable certificate verification (development only)
            connect=True,
            retryWrites=True,
            w='majority',
            # Connection Pool Settings
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=50000,
            waitQueueTimeoutMS=5000
        )
        
        db = mongo_client["fur-med"]
        predictions_coll = db["predictions"]
        donations_coll = db["donations"]

        # Test connection
        mongo_client.admin.command('ping')
        logger.info("[SUCCESS] MongoDB connection established successfully.")
    else:
        logger.warning("[WARNING] No MONGO_URI provided. Running in local-only mode.")

except Exception as e:
    logger.error(f"[ERROR] MongoDB connection failed: {e}. Running in local-only mode.")
    mongo_client = None
    db = None
    predictions_coll = None
    donations_coll = None

###############################################################################
# Helper Functions for Database Operations
###############################################################################

def is_mongo_available():
    """Check if MongoDB connection is currently available."""
    return mongo_client is not None and db is not None


def get_db():
    """Get the appropriate database connection based on availability."""
    if is_mongo_available():
        return db
    return sqlite3.connect(DB_PATH)


def cleanup_database_connections():
    """Cleanup database connections on server shutdown."""
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("✅ MongoDB connection closed.")
    except Exception as e:
        logger.error(f"❌ Error during database cleanup: {e}")


# Register cleanup handler
atexit.register(cleanup_database_connections)


###############################################################################
# Save Predictions to Database (Both SQLite & Mongo)
###############################################################################

def save_prediction_to_database(prediction_data):
    """
    Save prediction to both local SQLite and MongoDB if available.
    
    :param prediction_data: Dictionary containing prediction details
    :return: Tuple of (local_id, mongo_id)
    """
    try:
        # Always save to local SQLite
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Prepare data for insertion
            insert_data = (
                prediction_data.get('owner_name', 'Unknown'),
                prediction_data.get('pet_name', 'Unknown'),
                prediction_data.get('pet_gender', 'Unknown'),
                prediction_data.get('pet_type', 'Unknown'),
                prediction_data.get('disease', 'Unknown'),
                prediction_data.get('full_animal_image_path', ''),
                prediction_data.get('disease_image_path', ''),
                prediction_data.get('predicted_image_path', ''),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(prediction_data.get('symptoms', [])),
                None,  # mongo_id
                0,     # is_synced
                None,  # error_message
                prediction_data.get('confidence', 0.0)
            )
            
            cursor.execute('''
                INSERT INTO predictions (
                    owner_name, pet_name, pet_gender, pet_type, disease, 
                    full_animal_image_path, disease_image_path, predicted_image_path,
                    timestamp, symptoms, mongo_id, is_synced, error_message, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            local_id = cursor.lastrowid
            conn.commit()

        # Try to save to MongoDB if available
        mongo_id = None
        if is_mongo_available():
            try:
                # Prepare MongoDB document
                mongo_doc = prediction_data.copy()
                mongo_doc['timestamp'] = datetime.now(timezone.utc)
                
                # Insert to MongoDB
                result = predictions_coll.insert_one(mongo_doc)
                mongo_id = str(result.inserted_id)
                
                # Update local database with MongoDB ID
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE predictions 
                        SET mongo_id = ?, is_synced = 1 
                        WHERE id = ?
                    ''', (mongo_id, local_id))
                    conn.commit()
                
            except Exception as mongo_err:
                logger.warning(f"⚠️ MongoDB save failed: {mongo_err}")
        
        return local_id, mongo_id

    except Exception as e:
        logger.error(f"❌ Error saving prediction: {e}")
        return None, None


###############################################################################
# Verify SQLite Database Integrity & Auto-Fix Missing Tables
###############################################################################

def verify_and_fix_sqlite_tables():
    """Check if required tables exist in SQLite and recreate if missing."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Check and recreate tables if missing
            tables = ['predictions', 'donations']
            for table in tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
                if not cursor.fetchone():
                    logger.warning(f"⚠️ Table '{table}' is missing. Recreating it now...")
                    init_sqlite_db()
                    break  # Exit loop after recreating tables to avoid redundant checks

        logger.info("✅ SQLite tables verified and fixed successfully.")

    except sqlite3.Error as e:
        logger.error(f"❌ SQLite error during table verification: {e}")


# Run table verification and auto-fix missing tables
verify_and_fix_sqlite_tables()


###############################################################################
# mail logic
###############################################################################
# After app initialization
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')  # Your Gmail App Password

mail = Mail(app)


###############################################################################
# Currency Conversion Helper
###############################################################################
def get_usd_to_inr():
    """
    Fetch USD to INR rate from a free API or fallback to 83 if fails.
    """
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        if response.status_code == 200:
            return response.json().get("rates", {}).get("INR", 83.0)
    except Exception as e:
        print(f"⚠️ Exchange rate API failed: {e}")
    return 83.0  # fallback rate

###############################################################################
# Invoice Generator (Same as in your app.py)
###############################################################################
def generate_invoice(transaction_id, donor_name, email, phone, address, amount, currency, date):
    invoice_dir = os.path.join("frontend", "static", "invoices")
    if not os.path.exists(invoice_dir):
        os.makedirs(invoice_dir)

    invoice_path = os.path.join(invoice_dir, f"invoice_{transaction_id}.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # FurMed Logo
    logo_path = os.path.join("frontend", "static", "images", "logo.png")
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=10, w=30)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "FurMed - Donation Invoice", ln=True, align="C")
    pdf.ln(10)

    # Receipt Info
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, "RECEIPT INFORMATION", 1, 1, "L", True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(100, 8, f"Transaction ID: {transaction_id}", ln=True)
    pdf.cell(100, 8, f"Date: {date}", ln=True)
    pdf.ln(5)

    # Donor Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "DONOR INFORMATION", 1, 1, "L", True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(100, 8, f"Name: {donor_name}", ln=True)
    pdf.cell(100, 8, f"Email: {email}", ln=True)
    pdf.cell(100, 8, f"Phone: {phone}", ln=True)
    pdf.cell(100, 8, f"Address: {address}", ln=True)
    pdf.ln(5)

    # Donation Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "DONATION SUMMARY", 1, 1, "L", True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(130, 8, "Description", 1, 0, "C")
    pdf.cell(60, 8, "Amount", 1, 1, "C")

    pdf.cell(130, 8, "Charitable Donation to FurMed", 1, 0, "L")
    pdf.cell(60, 8, f"INR {amount:,.2f}", 1, 1, "R")

    pdf.set_font("Arial", "B", 11)
    pdf.cell(130, 8, "Total Amount Donated", 1, 0, "L")
    pdf.cell(60, 8, f"INR {amount:,.2f}", 1, 1, "R")

    # Thank You
    pdf.ln(10)
    pdf.set_font("Arial", "I", 11)
    pdf.set_text_color(0, 100, 255)
    pdf.multi_cell(0, 8, 
        "Thank you for your generous donation! Your support helps us provide medical care to animals in need.", 
        align="C"
    )

    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(128)
    pdf.cell(0, 5, 
        "This is a computer-generated invoice. For any queries, contact support@furmed.com", 
        ln=True, align="C"
    )

    pdf.output(invoice_path)
    return f"/static/invoices/invoice_{transaction_id}.pdf"

###############################################################################
# Disease Info (for display on result page)
###############################################################################

cat_symptom_mapping = {
    "dermatitis": ["Redness", "Swelling", "Itching", "Inflammation", "Hair Loss"],
    "ringworm": ["Circular Hair Loss", "Scaly Patches", "Redness", "Crusty Skin"],
    "scabies": ["Intense Itching", "Crusty Skin", "Hair Loss", "Redness"],
    "Mange": ["Patchy Hair Loss", "Thickened Skin", "Intense Itching"],
    "Healthy skin": ["No Symptoms"]
}

dog_symptom_mapping = {
    "Cataratas": ["Cloudy Eyes", "Vision Loss", "Eye Discharge"],
    "Conjuntivitis": ["Red Eyes", "Discharge", "Swelling", "Eye Irritation"],
    "Infección Bacteriana": ["Pus", "Swelling", "Wound", "Redness"],
    "PyodermaNasal": ["Nasal Discharge", "Crusty Nose", "Nose Irritation"],
    "Sarna": ["Intense Itching", "Scabs", "Hair Loss", "Redness"],
    "dermatitis": ["Inflammation", "Redness", "Scaling", "Itching"],
    "flea_allergy": ["Constant Scratching", "Hair Loss", "Scabs"],
    "ringworm": ["Circular Lesions", "Hair Loss", "Scaling"],
    "scabies": ["Crusty Skin", "Intense Itching", "Hair Loss"],
    "Healthy skin": ["No Symptoms"]
}

# Dog Disease Information Dictionary
dog_disease_info = {
    "Cataratas": {
        "details": "A condition causing clouding of the eye lens, making it difficult for dogs to see. Like looking through a foggy window. Common in older dogs (usually over 6 years), diabetic dogs, and certain breeds like Cocker Spaniels, Poodles, and Labrador Retrievers. Signs include cloudy/bluish-grey eyes, bumping into furniture, hesitation on stairs, reduced interest in catching toys, and increased clumsiness, especially in dim light. May affect one or both eyes, and can progress at different rates. Early signs often include a slight haziness that gets worse over months or years. Some dogs may show signs of eye discomfort by rubbing their eyes or having increased tear production.",
        "first_aid": "Keep home layout consistent and don't move furniture around. Use clear verbal commands like 'step up' or 'careful'. Place water and food bowls in easily accessible locations. Add night lights to help with evening navigation. Create safe pathways through the house by removing obstacles and using textured mats to mark different areas. Use sound cues like small bells on collars of other pets to help with orientation. Over-the-counter options: Sterile saline eye drops (specifically marked as safe for dogs) can help keep eyes clean - use up to 3 times daily. Artificial tears can provide temporary comfort. Vitamin E supplements may help support eye health (check with vet for dosage). Never use human eye drops or contact lens solutions. Keep the eye area clean by gently wiping with warm water and a soft cloth.",
        "treatment": "Veterinary consultation is essential. Treatment options include surgery to remove and replace clouded lens, which has a high success rate in healthy dogs. Surgery is typically done by a veterinary ophthalmologist using phacoemulsification (similar to human cataract surgery) and artificial lens implantation. Some dogs may need eye drops before and after surgery to reduce inflammation and prevent infection. Regular check-ups needed to monitor progression. Many dogs adapt well with proper care even if surgery isn't an option. Post-surgery care includes wearing a protective collar, administering prescribed eye drops, limiting activity, and attending follow-up appointments. Success rates are typically 80-90% in otherwise healthy dogs. Some dogs may need ongoing eye medications even after successful surgery. Cost and recovery time should be discussed with your vet as they can vary significantly.",
    },
    "Conjuntivitis": {
        "details": "An eye infection causing redness, swelling, and discharge. Similar to pink eye in humans. Can be triggered by allergies, bacteria, viruses, or irritants like dust or shampoo. Signs include red/pink eyes, watery or sticky discharge, frequent blinking, and pawing at eyes. Can affect one or both eyes.",
        "first_aid": "Gently clean around eyes with warm water and clean cotton balls, wiping from inner corner outward. Keep facial fur trimmed around eyes. Apply warm compress for comfort. Over-the-counter options: Saline eye wash made for pets can be used to flush eyes gently. Chamomile tea bags (cooled) can be used as a compress. OTC antihistamine eye drops marked safe for pets may help if allergy-related.",
        "treatment": "Veterinary examination to determine cause is crucial. Treatment usually includes antibiotic or antiviral eye drops/ointment applied several times daily. Typically clears within 7-14 days with proper treatment. May need oral medications if infection is severe."
    },
    "Infección Bacteriana": {
        "details": "A bacterial skin infection causing redness, inflammation, and sores. Often starts in areas already irritated by allergies, scratching, or moisture. Shows up as red, irritated skin with pimple-like bumps, hair loss, and sometimes has an unpleasant odor. Can spread quickly if not treated.",
        "first_aid": "Keep affected areas clean and dry using pet-safe antiseptic wipes. Prevent scratching or licking - consider protective clothing or cone collar. Over-the-counter options: Chlorhexidine solutions (0.5%) can be used to clean affected areas. Betadine solution diluted to tea-color is safe for cleaning. OTC hydrocortisone cream (1%) can be used sparingly for small areas.",
        "treatment": "Requires veterinary prescribed antibiotics, usually for 2-4 weeks. Often includes medicated shampoos or ointments. Deep infections may need longer treatment. Regular rechecks to ensure infection is clearing."
    },
    "PyodermaNasal": {
        "details": "A bacterial infection specifically affecting the nose area, causing crusting, swelling, and discharge. Can be painful and interfere with breathing or eating. Signs include crusty nose, colored discharge, redness, frequent nose rubbing, and possible loss of nose pigmentation.",
        "first_aid": "Clean nose area gently with warm saline solution 2-3 times daily. Pat dry carefully with soft cloth. Over-the-counter options: Saline nasal spray (non-medicated) can help clear nasal passages. Pet-safe antibiotic ointment can be applied thinly if skin isn't broken. Coconut oil can be applied as a natural moisturizer.",
        "treatment": "Veterinary visit needed for proper diagnosis and treatment plan. Usually requires combination of oral antibiotics and topical treatments for 2-3 weeks. May need testing for underlying causes. Regular monitoring of nose healing and breathing function."
    },
    "Sarna": {
        "details": "A severe skin condition caused by microscopic mites that burrow into skin. Extremely itchy and uncomfortable. Causes intense scratching, hair loss, red irritated skin, and scaly or crusty patches. Highly contagious to other dogs and can sometimes affect humans.",
        "first_aid": "Isolate dog from other pets immediately. Wash all bedding, toys, and surfaces in hot water. Over-the-counter options: Anti-itch oatmeal shampoo can provide temporary relief. Apple cider vinegar diluted 50/50 with water can be sprayed on affected areas (avoid open sores). Benadryl for itching (1mg per pound of body weight).",
        "treatment": "Urgent veterinary treatment required. Usually involves special dips, oral medications, or injections to kill mites. Treatment typically lasts 4-6 weeks. Entire environment must be treated to prevent reinfestation. All pets in household need treatment even if not showing symptoms."
    },
    "dermatitis": {
        "details": "Skin inflammation with multiple possible causes including allergies, parasites, or immune system issues. Creates red, itchy patches that can appear anywhere on body. May be seasonal or year-round. Can lead to hair loss, scaly skin, and hot spots if untreated.",
        "first_aid": "Keep affected areas clean and dry. Use cool compresses for relief. Over-the-counter options: Oatmeal-based pet shampoo for bathing. Benadryl for itching (1mg per pound). Zesty Paws Aller-Immune supplement can help. Pet-safe calamine lotion for spot treatment.",
        "treatment": "Veterinary exam needed to determine underlying cause. May include allergy testing. Treatment varies but often includes antihistamines, steroids, or immunosuppressants. Special shampoos or diet changes frequently recommended."
    },
    "fine": {
        "details": "No visible health issues detected. Dog appears healthy and normal. Regular monitoring should continue as part of routine care. Important to maintain current health through preventive measures.",
        "first_aid": "Continue regular grooming and care routine. Monitor for any changes in behavior or appearance. Keep up with regular exercise and healthy diet.",
        "treatment": "No immediate treatment needed. Continue preventive care including regular vet check-ups, vaccinations, and parasite prevention as recommended."
    },
    "flea_allergy": {
        "details": "Severe allergic reaction to flea saliva causing intense itching and discomfort. Even a single flea bite can trigger major reaction. Creates red, irritated skin especially at base of tail, thighs, and belly. Can lead to hair loss and scabs.",
        "first_aid": "Use flea comb to check for and remove fleas/flea dirt. Over-the-counter options: Pet-safe flea shampoo (follow instructions carefully). Oral Benadryl for itching (1mg per pound). Natural options include diatomaceous earth (food grade) for environment treatment.",
        "treatment": "Veterinary treatment plan needed. Usually includes both immediate flea elimination and allergy management. May need steroids or antihistamines for severe reactions. Monthly flea prevention essential."
    },
    "ringworm": {
        "details": "A fungal infection causing circular patches of hair loss and scaling. Not actually a worm - named for its ring-like appearance. Highly contagious to other pets and humans. Most common on face, ears, and legs. Can spread rapidly in multi-pet households.",
        "first_aid": "Isolate infected dog from other pets and limit human contact. Keep affected areas clean and dry. Over-the-counter options: Antifungal pet shampoos containing miconazole. Topical antifungal creams (check with vet first). Apple cider vinegar can be applied diluted to affected areas.",
        "treatment": "Requires 6-8 weeks of antifungal medications from vet. Usually combination of oral medicine and medicated shampoo/ointment. Weekly progress checks recommended. Complete cleaning of environment necessary."
    },
    "scabies": {
        "details": "Highly contagious mite infestation causing severe itching and skin inflammation. Different from regular mange, caused by Sarcoptes scabiei mites. Creates intense discomfort, hair loss, and crusty skin lesions. Can briefly affect humans.",
        "first_aid": "Isolate dog immediately to prevent spread. Use gloves when handling. Over-the-counter options: Oatmeal baths can provide temporary relief. Benadryl for itching (1mg per pound). Anti-itch sprays containing hydrocortisone can help temporarily.",
        "treatment": "Requires immediate veterinary treatment. Special dips or medications to kill mites. Treatment usually lasts 4-6 weeks. All in-contact animals need treatment. Environment must be thoroughly cleaned."
    }
}

# Cat Disease Information Dictionary
cat_disease_info = {
    "dermatitis": {
        "details": "Skin inflammation causing itching, redness, and discomfort. Can be caused by allergies, fleas, food sensitivities, or environmental irritants. Often appears on belly, thighs, or base of tail. May show as small scabs or excessive grooming.",
        "first_aid": "Keep affected areas clean and dry. Over-the-counter options: Zymox enzymatic cream without hydrocortisone. Pet-safe oatmeal shampoo for bathing. Benadryl for itching (1mg per pound). Natural options include coconut oil for mild cases.",
        "treatment": "Veterinary diagnosis needed to identify underlying cause. May include allergy testing or food trials. Treatment often combines oral medications and topical treatments."
    },
    "fine": {
        "details": "No health issues detected. Cat appears healthy with normal skin and coat condition. Regular monitoring should continue as part of routine care.",
        "first_aid": "Continue regular grooming and care routine. Monitor for any changes in behavior or appearance. Maintain current diet and exercise habits.",
        "treatment": "No immediate treatment needed. Continue preventive care including regular vet check-ups, vaccinations, and parasite prevention as recommended."
    },
    "flea_allergy": {
        "details": "Severe allergic reaction to flea saliva causing intense itching and skin problems. Even one flea bite can trigger major reaction. Creates small scabs (miliary dermatitis), especially around neck and base of tail.",
        "first_aid": "Use flea comb daily to check for and remove fleas. Over-the-counter options: Capstar for immediate flea killing (check weight guidelines). Pet-safe flea shampoo if cat tolerates bathing. Benadryl for severe itching (1mg per pound).",
        "treatment": "Requires comprehensive vet treatment plan. Includes flea elimination on cat and in environment. May need steroids or antihistamines for severe reactions. Monthly flea prevention crucial."
    },
    "ringworm": {
        "details": "A fungal skin infection causing circular patches of hair loss. Despite name, not caused by worms. Highly contagious to other pets and humans. Shows as circular lesions, often on head, ears or paws first.",
        "first_aid": "Isolate infected cat immediately. Over-the-counter options: Antifungal shampoos containing miconazole (if cat tolerates bathing). Topical terbinafine cream for small areas (prevent licking). Diluted betadine solution for cleaning lesions.",
        "treatment": "Requires vet-prescribed antifungal medications for 6-8 weeks minimum. Usually includes both oral medication and topical treatments. Weekly progress checks important."
    },
    "Ringworm": {
        "details": "Alternative classification for fungal skin infection. Identical to lowercase 'ringworm' in terms of condition and treatment. Causes circular patches of hair loss and can spread to humans and other pets.",
        "first_aid": "Isolate infected cat immediately. Over-the-counter options: Antifungal shampoos containing miconazole (if cat tolerates bathing). Topical terbinafine cream for small areas (prevent licking). Diluted betadine solution for cleaning lesions.",
        "treatment": "Requires vet-prescribed antifungal medications for 6-8 weeks minimum. Usually includes both oral medication and topical treatments. Weekly progress checks important."
    },
    "scabies": {
        "details": "Intensely itchy skin condition caused by microscopic mites. Creates severe discomfort and can spread quickly. Causes hair loss, red irritated skin, and crusty patches. Most common on ears, face, and edges of ears.",
        "first_aid": "Isolate affected cat from other pets. Over-the-counter options: Benadryl for itching relief (1mg per pound). Sulfur-based or oatmeal shampoos can provide temporary relief. Cool compresses for comfort.",
        "treatment": "Urgent veterinary treatment required. Usually involves special medications to kill mites. Treatment typically lasts 4-6 weeks with regular rechecks. All in-contact pets need evaluation."
    },
    "Healthy skin": {
        "details": "Normal, healthy skin and coat condition. Skin should be pink and supple, coat should be shiny and full. No signs of irritation, parasites, or abnormalities.",
        "first_aid": "Continue regular grooming routine. Brush regularly to distribute natural oils. Monitor for any changes.",
        "treatment": "No treatment needed. Maintain regular preventive care including proper diet, grooming, and annual vet check-ups."
    },
    "Mange": {
        "details": "Severe skin condition caused by mites. Can be caused by different types of mites. Results in intense itching, hair loss, and skin inflammation. More severe than regular scabies.",
        "first_aid": "Isolate cat immediately. Over-the-counter options: Benadryl for itch relief (1mg per pound). Oatmeal baths for temporary comfort. Keep environment extremely clean.",
        "treatment": "Requires immediate veterinary intervention. Special medications needed to kill mites. Treatment usually lasts 4-6 weeks or longer. All in-contact animals need evaluation."
    }
}

def save_to_local_db(record, collection_type='predictions'):
    """Save record to local SQLite database."""
    try:
        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
            cursor = conn.cursor()
            
            if collection_type == 'predictions':
                query = '''
                    INSERT INTO predictions (
                        owner_name, pet_name, pet_gender, pet_type, disease,
                        full_animal_image_path, disease_image_path, predicted_image_path,
                        timestamp, symptoms, mongo_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                values = (
                    record.get('owner_name'),
                    record.get('pet_name'),
                    record.get('pet_gender'),
                    record.get('pet_type'),
                    record.get('disease'),
                    record.get('full_animal_image_path'),
                    record.get('disease_image_path'),
                    record.get('predicted_image_path'),
                    record.get('timestamp').isoformat(),
                    json.dumps(record.get('symptoms', [])),
                    None  # mongo_id will be updated later if available
                )
            else:  # donations
                query = '''
                    INSERT INTO donations (
                        donor_name, donation_email, paypal_email, phone, address,
                        amount_usd, amount_inr, currency, exchange_rate, status,
                        transaction_id, date, invoice_path, mongo_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                values = (
                    record.get('donor_name'),
                    record.get('donation_email'),
                    record.get('paypal_email'),
                    record.get('phone'),
                    record.get('address'),
                    record.get('amount_usd'),
                    record.get('amount_inr'),
                    record.get('currency'),
                    record.get('exchange_rate'),
                    record.get('status'),
                    record.get('transaction_id'),
                    record.get('date'),
                    record.get('invoice_path'),
                    None
                )
            
            cursor.execute(query, values)
            return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return None

def get_local_record(record_id, collection_type='predictions'):
    """Retrieve record from local SQLite database."""
    try:
        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = f'SELECT * FROM {collection_type} WHERE id = ?'
            cursor.execute(query, (record_id,))
            row = cursor.fetchone()
            if row:
                record = dict(row)
                if collection_type == 'predictions' and record.get('symptoms'):
                    record['symptoms'] = json.loads(record['symptoms'])
                return record
            return None
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return None

@app.route('/send_feedback', methods=['POST'])
def send_feedback():
    try:
        data = request.get_json()
        email = data.get('email')
        message = data.get('message')
        
        if not email or not message:
            return jsonify({
                'status': 'error',
                'message': 'Email and message are required'
            }), 400

        msg = Message('Feedback from FurMed Website',
                     sender=email,
                     recipients=['furmed.19@gmail.com'])  # Your receiving email

        msg.body = f"""
        Feedback from: {email}
        Message: {message}
        """

        mail.send(msg)
        return jsonify({
            'status': 'success',
            'message': 'Feedback sent successfully!'
        }), 200
        
    except Exception as e:
        print(f"Error sending feedback: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
###############################################################################
# Flask Routes
###############################################################################
@app.route('/')
def route_home():
    """Home page (Landing)."""
    return render_template('home.html')

###############################################################################
# Admin Login Route with Authentication
###############################################################################
@app.route('/login', methods=['GET', 'POST'])
def route_login():
    if request.method == 'POST':
        data = request.json
        username = data.get("username")
        password = data.get("password")

        # Hardcoded admin credentials (replace with DB check in production)
        ADMIN_USERNAME = "admin"
        ADMIN_PASSWORD = "admin123"

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return jsonify({"success": True, "redirect_url": url_for('admin_dashboard')}), 200
        else:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return render_template('login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('route_login'))  
    return render_template("admin_dashboard.html")

@app.route('/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('route_home'))

@app.route('/api/get_predictions', methods=['GET'])
def get_predictions():
    if not session.get('admin_logged_in'):
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        try:
            # Try MongoDB first
            total_predictions = predictions_coll.count_documents({})
            predictions = list(
                predictions_coll.find({}, {"_id": 0, "owner_name": 1, "pet_name": 1, "pet_type": 1, "disease": 1, "is_correct": 1})
                .sort("timestamp", -1)
                .skip((page - 1) * per_page)
                .limit(per_page)
            )
            
            # Handle missing fields
            for pred in predictions:
                pred.setdefault("owner_name", "Unknown")
                pred.setdefault("pet_type", "Unknown")
            
            source = "mongodb"
            logger.info("Successfully fetched predictions from MongoDB")
            
        except Exception as mongo_error:
            logger.warning(f"Failed to fetch from MongoDB: {mongo_error}, using local storage")
            # Fallback to local storage
            with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute('SELECT COUNT(*) as count FROM predictions')
                total_predictions = cursor.fetchone()['count']
                
                # Get paginated records
                cursor.execute('''
                    SELECT owner_name, pet_name, pet_type, disease, is_correct 
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                ''', (per_page, (page - 1) * per_page))
                
                predictions = []
                for row in cursor.fetchall():
                    pred = dict(row)
                    # Convert JSON string to list for symptoms
                    if pred.get('symptoms'):
                        try:
                            pred['symptoms'] = json.loads(pred['symptoms'])
                        except json.JSONDecodeError:
                            pred['symptoms'] = []
                    # Handle missing fields
                    pred.setdefault("owner_name", "Unknown")
                    pred.setdefault("pet_type", "Unknown")
                    predictions.append(pred)
                source = "local"

        # Add standard fields if missing
        for p in predictions:
            if "is_correct" not in p:
                p["is_correct"] = True
            if "correct_label" not in p:
                p["correct_label"] = None

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "total_pages": (total_predictions + per_page - 1) // per_page,
            "current_page": page,
            "data_source": source
        })

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/get_donations', methods=['GET'])
def get_donations():
    if not session.get('admin_logged_in'):
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        try:
            # Try MongoDB first
            total_donations = donations_coll.count_documents({})
            all_donations = donations_coll.find({}, {"_id": 0, "amount_inr": 1})
            grand_total_donations = sum(d.get('amount_inr', 0) for d in all_donations)
            
            donations = list(
                donations_coll.find({}, {
                    "_id": 0,
                    "donor_name": 1,
                    "donation_email": 1,
                    "amount_inr": 1,
                    "date": 1,
                    "transaction_id": 1,  # Added transaction_id
                    "invoice_path": 1     # Added invoice_path
                })
                .sort("date", -1)
                .skip((page - 1) * per_page)
                .limit(per_page)
            )
            
            # Handle missing fields
            for donation in donations:
                donation.setdefault("donor_name", "Unknown")
                donation.setdefault("donation_email", "Unknown")
                donation.setdefault("transaction_id", None)  # Added default
                donation.setdefault("invoice_path", None)    # Added default
            
            source = "mongodb"
            logger.info("Successfully fetched donations from MongoDB")
            
        except Exception as mongo_error:
            logger.warning(f"Failed to fetch from MongoDB: {mongo_error}, using local storage")
            # Fallback to local storage
            with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get total count and sum
                cursor.execute('''
                    SELECT COUNT(*) as count, COALESCE(SUM(amount_inr), 0) as total 
                    FROM donations
                ''')
                result = cursor.fetchone()
                total_donations = result['count']
                grand_total_donations = result['total']
                
                # Get paginated records
                cursor.execute('''
                    SELECT 
                        donor_name, 
                        donation_email, 
                        amount_inr, 
                        date,
                        transaction_id,    -- Added transaction_id
                        invoice_path       -- Added invoice_path
                    FROM donations
                    ORDER BY date DESC
                    LIMIT ? OFFSET ?
                ''', (per_page, (page - 1) * per_page))
                
                donations = [dict(row) for row in cursor.fetchall()]
                
                # Handle missing fields
                for donation in donations:
                    donation.setdefault("donor_name", "Unknown")
                    donation.setdefault("donation_email", "Unknown")
                    donation.setdefault("transaction_id", None)  # Added default
                    donation.setdefault("invoice_path", None)    # Added default
                
                source = "local"

        return jsonify({
            "status": "success",
            "donations": donations,
            "total_pages": (total_donations + per_page - 1) // per_page,
            "current_page": page,
            "grand_total_donations": grand_total_donations,
            "data_source": source
        })

    except Exception as e:
        logger.error(f"Error fetching donations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/generate-invoice/<transaction_id>')
def generate_missing_invoice(transaction_id):
    try:
        # First try MongoDB
        if is_mongo_available():
            donation = donations_coll.find_one({"transaction_id": transaction_id})
        
        # If not found in MongoDB or MongoDB not available, try SQLite
        if not donation:
            with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM donations WHERE transaction_id = ?', 
                    (transaction_id,)
                )
                donation = cursor.fetchone()
                if donation:
                    donation = dict(donation)

        if not donation:
            return jsonify({"status": "error", "message": "Donation not found"}), 404

        # Generate invoice
        invoice_path = generate_invoice(
            transaction_id=donation['transaction_id'],
            donor_name=donation.get('donor_name', 'Unknown'),
            email=donation.get('donation_email', 'Unknown'),
            phone=donation.get('phone', 'N/A'),
            address=donation.get('address', 'N/A'),
            amount=donation['amount_inr'],
            currency='INR',
            date=donation['date']
        )

        return jsonify({"status": "success", "invoice_path": invoice_path}), 200

    except Exception as e:
        logger.error(f"Error generating invoice: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
###############################################################################
# Disease Route (Refactored to use Proxy Server)
###############################################################################
@app.route('/disease', methods=['GET', 'POST'])
def route_disease():
    if request.method == 'POST':
        try:
            owner_name = request.form.get('ownerName', 'Unknown')
            pet_name = request.form.get('petName', 'Unknown')
            pet_gender = request.form.get('petGender', 'Unknown')
            selected_pet_type = request.form.get('petType', 'Unknown')
            user_symptoms = request.form.getlist('symptoms')

            # Handle Full Animal Image
            full_animal_file = request.files.get('full_animal_image')
            if not full_animal_file or not allowed_file(full_animal_file.filename):
                return render_template('disease.html', error="Invalid full animal image format.")

            full_animal_filename = secure_filename(full_animal_file.filename)
            full_animal_filepath = os.path.join(app.config['FULL_ANIMAL_UPLOAD_FOLDER'], full_animal_filename)
            full_animal_file.save(full_animal_filepath)

            # Handle Disease Image
            disease_file = request.files.get('disease_image')
            if not disease_file or not allowed_file(disease_file.filename):
                return render_template('disease.html', error="Invalid disease image format.")

            disease_filename = secure_filename(disease_file.filename)
            disease_filepath = os.path.join(app.config['DISEASE_UPLOAD_FOLDER'], disease_filename)
            disease_file.save(disease_filepath)

            # POST to proxy server (AI analysis)
            proxy_url = "http://localhost:5001/predict_disease"
            with open(full_animal_filepath, 'rb') as fa, open(disease_filepath, 'rb') as da:
                files_data = {
                    "full_animal_image": (full_animal_filename, fa, full_animal_file.content_type),
                    "disease_image": (disease_filename, da, disease_file.content_type)
                }
                form_data = {
                    "ownerName": owner_name,
                    "petName": pet_name,
                    "petGender": pet_gender,
                    "petType": selected_pet_type,
                    "symptoms": user_symptoms
                }
                response = requests.post(proxy_url, data=form_data, files=files_data)

            if response.status_code != 200:
                error_msg = response.json().get("error", "Error from AI Server.")
                return render_template('disease.html', error=error_msg)

            ai_result = response.json()
            predicted_type = ai_result.get("pet_type", "Other")
            final_disease = ai_result.get("disease", "No disease detected")
            detected_image_url = ai_result.get("detected_image_url", None)

            if predicted_type == "Other":
                return render_template('disease.html', error="Please upload a valid cat or dog image.")

            if not detected_image_url:
                detected_image_url = url_for('static', filename='images/no_prediction.png')

            # Determine disease info for final_disease
            disease_info = (cat_disease_info.get(final_disease, {})
                            if predicted_type.lower() == "cat"
                            else dog_disease_info.get(final_disease, {}))

            # Create record
            record = {
                "owner_name": owner_name,
                "pet_name": pet_name,
                "pet_gender": pet_gender,
                "pet_type": predicted_type,
                "disease": final_disease,
                "full_animal_image_path": url_for('static', filename=f'f_upload/{full_animal_filename}'),
                "disease_image_path": url_for('static', filename=f'uploads/{disease_filename}'),
                "predicted_image_path": detected_image_url,
                "timestamp": datetime.now(timezone.utc),
                "symptoms": user_symptoms
            }

            # Try MongoDB first
            mongo_id = None
            try:
                result = predictions_coll.insert_one(record)
                mongo_id = str(result.inserted_id)
                logger.info(f"Prediction saved to MongoDB with ID: {mongo_id}")
            except Exception as e:
                logger.warning(f"Failed to save prediction to MongoDB: {e}")

            # Always save to local storage
            local_id = save_to_local_db(record, 'predictions')
            if local_id:
                logger.info(f"Prediction saved to local storage with ID: {local_id}")

                # If MongoDB save was successful, update local record with mongo_id
                if mongo_id:
                    with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            'UPDATE predictions SET mongo_id = ? WHERE id = ?',
                            (mongo_id, local_id)
                        )
            else:
                logger.error("Failed to save prediction to local storage")

            # ##################################
            # Distinguish if it's from Mongo or Local
            if mongo_id:
                # e.g., "mongo-646b789..."
                final_prediction_id = f"mongo-{mongo_id}"
            else:
                # e.g., "local-17"
                final_prediction_id = f"local-{local_id}"

            return render_template(
                'result.html',
                prediction_id=final_prediction_id,    # <-- THIS is the key change
                owner_name=owner_name,
                pet_name=pet_name,
                pet_gender=pet_gender,
                pet_type=predicted_type,
                disease_name=final_disease,
                details=disease_info.get("details", "No information available."),
                first_aid=disease_info.get("first_aid", "No first aid available."),
                treatment=disease_info.get("treatment", "No treatment info available."),
                image_path=url_for('static', filename=f'uploads/{disease_filename}'),
                predicted_image_path=detected_image_url,
                symptoms=user_symptoms
            )

        except Exception as ex:
            logger.error(f"Error during disease analysis: {ex}")
            return render_template('disease.html', error="An unexpected error occurred")

    # GET method
    return render_template('disease.html')

###############################################################################
# Leaflet Map Integration for Nearby Veterinary Clinics
###############################################################################
@app.route('/get_nearby_vets', methods=['GET'])
def get_nearby_vets():
    try:
        latitude = float(request.args.get('lat', 0))
        longitude = float(request.args.get('lng', 0))

        # Generate mock or real vet clinic data
        vet_clinics = [
            {
                "name": f"Vet Clinic {i+1}",
                "lat": latitude + (np.random.rand() - 0.5) * 0.02,
                "lng": longitude + (np.random.rand() - 0.5) * 0.02,
                "contact": "1234567890"
            }
            for i in range(5)
        ]

        return jsonify({"status": "success", "clinics": vet_clinics}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/result')
def route_result():
    return redirect(url_for('route_disease'))

###############################################################################
# Feedback for Incorrect Predictions
###############################################################################
def get_local_prediction_by_mongo_id(mongo_id):
    """Retrieve a prediction record from local SQLite by its mongo_id."""
    try:
        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = 'SELECT * FROM predictions WHERE mongo_id = ?'
            cursor.execute(query, (mongo_id,))
            row = cursor.fetchone()
            if row:
                record = dict(row)
                # **Fix: Ensure `symptoms` is properly parsed**
                if record.get('symptoms'):
                    try:
                        record['symptoms'] = json.loads(record['symptoms'])
                    except json.JSONDecodeError:
                        record['symptoms'] = []
                return record
            return None
    except sqlite3.Error as e:
        logger.error(f"SQLite error in get_local_prediction_by_mongo_id: {e}")
        return None


@app.route('/feedback', methods=['POST'])
def save_incorrect_prediction():
    try:
        data = request.json
        full_prediction_id = data.get("prediction_id", "").strip()
        correct_label = data.get("correct_label", "").strip()

        if not full_prediction_id or not correct_label:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        logger.info(f"Processing incorrect prediction for {full_prediction_id} with correct label {correct_label}")

        pred_rec = None

        # **Handle MongoDB-based records**
        if full_prediction_id.startswith("mongo-"):
            mongo_id = full_prediction_id.replace("mongo-", "")
            if is_mongo_available():
                try:
                    pred_rec = predictions_coll.find_one({"_id": ObjectId(mongo_id)})
                except Exception as e:
                    logger.warning(f"MongoDB lookup failed in feedback: {e}")

            # If not found in MongoDB, fallback to SQLite
            if not pred_rec:
                pred_rec = get_local_prediction_by_mongo_id(mongo_id)
                logger.info(f"Fetched from local SQLite: {pred_rec}")

            # If still not found, return error
            if not pred_rec:
                return jsonify({"status": "error", "message": "Prediction not found"}), 404

            # **Ensure `pred_rec` is a dictionary**
            if not isinstance(pred_rec, dict):
                logger.error(f"Expected dict but got {type(pred_rec)}: {pred_rec}")
                return jsonify({"status": "error", "message": "Database returned unexpected format"}), 500

            return update_incorrect_prediction_mongo(mongo_id, pred_rec, correct_label)

        # **Handle Local SQLite-based records**
        elif full_prediction_id.startswith("local-"):
            local_str = full_prediction_id.replace("local-", "")
            if not local_str.isdigit():
                return jsonify({"status": "error", "message": "Invalid local ID format"}), 400

            local_id = int(local_str)
            with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                conn.row_factory = sqlite3.Row  # Fetch results as dictionary-like objects
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM predictions WHERE id = ?", (local_id,))
                row = cursor.fetchone()

                if not row:
                    return jsonify({"status": "error", "message": "Local prediction not found"}), 404
                
                pred_rec = dict(row)  # Convert SQLite Row to Dictionary

                # **Ensure `pred_rec` is a dictionary**
                if not isinstance(pred_rec, dict):
                    logger.error(f"Expected dict but got {type(pred_rec)}: {pred_rec}")
                    return jsonify({"status": "error", "message": "Database returned unexpected format"}), 500

                # **Convert symptoms from JSON string to list**
                if "symptoms" in pred_rec and isinstance(pred_rec["symptoms"], str):
                    try:
                        pred_rec["symptoms"] = json.loads(pred_rec["symptoms"])
                    except json.JSONDecodeError:
                        pred_rec["symptoms"] = []

            return update_incorrect_prediction_local_only(local_id, pred_rec, correct_label)
        else:
            return jsonify({"status": "error", "message": "Invalid prediction_id prefix"}), 400

    except Exception as e:
        logger.error(f"Error in save_incorrect_prediction: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500



def update_incorrect_prediction_mongo(mongo_id, pred_rec, correct_label):
    """Move file, set is_correct=0, etc., update local by mongo_id, and if Mongo is available, update that too."""
    original_image_path = pred_rec.get("disease_image_path")
    if not original_image_path:
        return jsonify({"status": "error", "message": "Original image not found"}), 404

    # Move file to incorrect_predictions
    incorrect_folder = os.path.join(app.config['DISEASE_UPLOAD_FOLDER'], 'incorrect_predictions')
    os.makedirs(incorrect_folder, exist_ok=True)

    new_filename = f"{correct_label}_{os.path.basename(original_image_path)}"
    new_filepath = os.path.join(incorrect_folder, new_filename)

    source_path = os.path.join(app.root_path, 'frontend/static', original_image_path.lstrip('/static/'))
    if not os.path.exists(source_path):
        return jsonify({"status": "error", "message": "Source image not found"}), 404

    try:
        os.rename(source_path, new_filepath)
    except Exception as e:
        return jsonify({"status": "error", "message": f"File operation failed: {str(e)}"}), 500

    corrected_path = f"/static/uploads/incorrect_predictions/{new_filename}"

    # Always update local
    try:
        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions
                SET is_correct = 0,
                    correct_label = ?,
                    corrected_image_path = ?
                WHERE mongo_id = ?
            ''', (correct_label, corrected_path, mongo_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Local DB update failed in feedback: {e}")
        return jsonify({"status": "error", "message": "Local DB update failed"}), 500

    # Update in Mongo if available
    if is_mongo_available():
        try:
            predictions_coll.update_one(
                {"_id": ObjectId(mongo_id)},
                {
                    "$set": {
                        "is_correct": False,
                        "correct_label": correct_label,
                        "corrected_image_path": corrected_path
                    }
                }
            )
        except Exception as e:
            logger.warning(f"Mongo update failed in feedback: {e}")

    return jsonify({
        "status": "success",
        "message": "Feedback saved successfully (mongo-based)"
    }), 200


def update_incorrect_prediction_local_only(local_id, pred_rec, correct_label):
    """Move file, set is_correct=0, correct_label, etc., update local by 'id'."""
    original_image_path = pred_rec.get("disease_image_path")
    if not original_image_path:
        return jsonify({"status": "error", "message": "Original image not found"}), 404

    # Move file
    incorrect_folder = os.path.join(app.config['DISEASE_UPLOAD_FOLDER'], 'incorrect_predictions')
    os.makedirs(incorrect_folder, exist_ok=True)

    new_filename = f"{correct_label}_{os.path.basename(original_image_path)}"
    new_filepath = os.path.join(incorrect_folder, new_filename)

    source_path = os.path.join(app.root_path, 'frontend/static', original_image_path.lstrip('/static/'))
    if not os.path.exists(source_path):
        return jsonify({"status": "error", "message": "Source image not found"}), 404

    try:
        os.rename(source_path, new_filepath)
    except Exception as e:
        return jsonify({"status": "error", "message": f"File operation failed: {str(e)}"}), 500

    corrected_path = f"/static/uploads/incorrect_predictions/{new_filename}"

    # Update local by 'id'
    try:
        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions
                SET is_correct = 0,
                    correct_label = ?,
                    corrected_image_path = ?
                WHERE id = ?
            ''', (correct_label, corrected_path, local_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Local DB update (local-only) failed: {e}")
        return jsonify({"status": "error", "message": "Local DB update failed"}), 500

    return jsonify({
        "status": "success",
        "message": "Feedback saved successfully (local-only)"
    }), 200


###############################################################################
# Services Page
###############################################################################
@app.route('/services')
def route_services():
    return render_template('services.html')

###############################################################################
# Donation Route
###############################################################################
@app.route('/donation', methods=['GET', 'POST'])
def route_donation():
    if request.method == 'POST':
        try:
            donor_name = request.form.get('name', '').strip()
            donor_email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            address = request.form.get('address', '').strip()
            amount_inr = request.form.get('amount', '').strip()

            # Validate
            if not donor_name or not donor_email or not phone or not address or not amount_inr:
                return render_template(
                    "donation.html",
                    error="All fields are required.",
                    paypal_client_id=PAYPAL_CLIENT_ID
                )

            if not phone.isdigit() or len(phone) != 10:
                return render_template(
                    "donation.html",
                    error="Phone must be 10 digits.",
                    paypal_client_id=PAYPAL_CLIENT_ID
                )

            if '@' not in donor_email or '.' not in donor_email.split('@')[-1]:
                return render_template(
                    "donation.html",
                    error="Invalid email address.",
                    paypal_client_id=PAYPAL_CLIENT_ID
                )

            try:
                amount_inr = float(amount_inr)
                if amount_inr <= 0:
                    return render_template(
                        "donation.html",
                        error="Amount must be > 0.",
                        paypal_client_id=PAYPAL_CLIENT_ID
                    )
            except ValueError:
                return render_template(
                    "donation.html",
                    error="Invalid amount.",
                    paypal_client_id=PAYPAL_CLIENT_ID
                )

            # Convert INR to USD
            usd_rate = get_usd_to_inr()
            amount_usd = round(amount_inr / usd_rate, 2)

            # Store email in session for PayPal success
            session["donation_email"] = donor_email

            # Prepare the record
            donation_data = {
                "donor_name": donor_name,
                "donation_email": donor_email,
                "phone": phone,
                "address": address,
                "amount_inr": amount_inr,
                "amount_usd": amount_usd,
                "currency": "INR",
                "status": "Pending",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            }

            #################################################################
            # Try saving to MongoDB if available, fallback to local
            #################################################################
            mongo_id = None
            if is_mongo_available():
                try:
                    result = donations_coll.insert_one(donation_data)
                    mongo_id = str(result.inserted_id)
                    logger.info(f"Donation saved to MongoDB with ID: {mongo_id}")
                except Exception as e:
                    logger.warning(f"Failed to save donation to MongoDB: {e}")
                    # Fallback: We'll rely on local only
                    pass

            # Always save to local
            local_id = save_to_local_db(donation_data, 'donations')
            if local_id:
                logger.info(f"Donation saved to local storage with ID: {local_id}")

                # If Mongo worked, update local record with the mongo_id
                if mongo_id:
                    with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            'UPDATE donations SET mongo_id = ? WHERE id = ?',
                            (mongo_id, local_id)
                        )
            else:
                logger.error("Failed to save donation to local storage")

            # Use MongoDB ID if available, otherwise use local ID
            donation_id = mongo_id if mongo_id else str(local_id)

            return render_template(
                "donation.html",
                thank_you=f"Thank you, {donor_name}, for your donation!",
                donation_id=donation_id,
                paypal_client_id=PAYPAL_CLIENT_ID
            )

        except Exception as e:
            logger.error(f"Error while processing donation: {e}")
            return render_template(
                "donation.html",
                error="An unexpected error occurred.",
                paypal_client_id=PAYPAL_CLIENT_ID
            )

    # GET request:
    return render_template("donation.html", paypal_client_id=PAYPAL_CLIENT_ID)


###############################################################################
# About Page
###############################################################################
@app.route('/about')
def route_about():
    return render_template('about.html')

def get_local_donation_by_txn_id(transaction_id):
    """Retrieve a donation record from local SQLite by transaction_id."""
    try:
        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM donations WHERE transaction_id = ?', (transaction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"SQLite error in get_local_donation_by_txn_id: {e}")
        return None

###############################################################################
# PayPal Success Callback
###############################################################################
@app.route('/paypal-success', methods=['POST'])
def paypal_success():
    """Handle successful PayPal payment and generate invoice."""
    try:
        # Extract data from request
        data = request.json
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data received"
            }), 400

        # Extract required fields
        order_id = data.get("orderID")
        phone = data.get("phone", "").strip()
        address = data.get("address", "").strip()

        # Validate required fields
        if not order_id:
            return jsonify({"status": "error", "message": "Missing order ID"}), 400
        if not phone:
            return jsonify({"status": "error", "message": "Phone number is required"}), 400
        if not address:
            return jsonify({"status": "error", "message": "Address is required"}), 400

        #################################################################
        # 1. Check for Duplicate in Mongo if available, else in local
        #################################################################
        existing_transaction = None
        if is_mongo_available():
            try:
                existing_transaction = donations_coll.find_one({"transaction_id": order_id})
            except Exception as e:
                logger.warning(f"Mongo check for existing transaction failed: {e}")
                existing_transaction = get_local_donation_by_txn_id(order_id)
        else:
            existing_transaction = get_local_donation_by_txn_id(order_id)

        if existing_transaction:
            return jsonify({
                "status": "error",
                "message": "Duplicate transaction detected.",
                "transaction_id": order_id,
                "invoice_url": existing_transaction.get("invoice_path")
            }), 409

        #################################################################
        # 2. Verify transaction with PayPal
        #################################################################
        try:
            auth = (PAYPAL_CLIENT_ID, PAYPAL_SECRET)
            headers = {"Content-Type": "application/json"}
            paypal_response = requests.get(
                f"{PAYPAL_API_BASE}/v2/checkout/orders/{order_id}",
                auth=auth,
                headers=headers,
                timeout=10  # Add timeout
            )
            if paypal_response.status_code != 200:
                return jsonify({
                    "status": "error",
                    "message": f"PayPal API Error: {paypal_response.text}"
                }), 400

            payment_data = paypal_response.json()
            if payment_data.get("status") != "COMPLETED":
                return jsonify({
                    "status": "error",
                    "message": "Transaction not completed"
                }), 400

        except requests.exceptions.RequestException as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to verify with PayPal: {str(e)}"
            }), 500

        #################################################################
        # 3. Extract PayPal Payment Details
        #################################################################
        payer = payment_data["payer"]
        amount_usd = float(payment_data["purchase_units"][0]["amount"]["value"])
        paypal_email = payer["email_address"]
        donor_name = f"{payer['name']['given_name']} {payer['name']['surname']}"

        # Get donor email from session or use PayPal email
        donor_email = session.pop("donation_email", paypal_email)

        # Convert USD to INR
        exchange_rate = get_usd_to_inr()
        amount_inr = round(amount_usd * exchange_rate, 2)

        # Generate timestamp in IST
        utc_now = datetime.now(timezone.utc)
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        ist_time_str = ist_now.strftime("%Y-%m-%d %H:%M:%S")

        #################################################################
        # 4. Generate Invoice
        #################################################################
        try:
            invoice_path = generate_invoice(
                order_id, donor_name, donor_email,
                phone, address, amount_inr, "INR", ist_time_str
            )
            invoice_url = url_for(
                'static',
                filename=f'invoices/invoice_{order_id}.pdf',
                _external=True
            )
        except Exception as e:
            logger.error(f"Error generating invoice: {e}")
            # Continue even if invoice generation fails
            invoice_path = None
            invoice_url = None

        #################################################################
        # 5. Create Donation Record
        #################################################################
        donation_record = {
            "donor_name": donor_name,
            "donation_email": donor_email,
            "paypal_email": paypal_email,
            "phone": phone,
            "address": address,
            "amount_usd": amount_usd,
            "amount_inr": amount_inr,
            "currency": "INR",
            "exchange_rate": exchange_rate,
            "status": "Completed",
            "transaction_id": order_id,
            "date": ist_time_str,
            "invoice_path": invoice_url
        }

        #################################################################
        # 6. Save Donation (Mongo first, fallback local)
        #################################################################
        if is_mongo_available():
            try:
                result = donations_coll.insert_one(donation_record)
                logger.info("Donation saved to MongoDB in paypal-success")

                # Also save to local for redundancy
                local_id = save_to_local_db(donation_record, 'donations')
                if local_id:
                    # If needed, update local DB with the new mongo_id
                    with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_storage.db')) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            'UPDATE donations SET mongo_id = ? WHERE id = ?',
                            (str(result.inserted_id), local_id)
                        )
                else:
                    logger.warning("Failed to save donation to local DB (but saved in Mongo).")

            except Exception as e:
                logger.warning(f"MongoDB insert failed in paypal-success: {e}")
                # fallback - save only to local
                local_id = save_to_local_db(donation_record, 'donations')
                if not local_id:
                    logger.error("Failed to save donation to local storage in fallback scenario.")
                    return jsonify({
                        "status": "error",
                        "message": "Failed to save donation record (Mongo & Local both failed)."
                    }), 500
        else:
            # No Mongo, so we must save only to local
            local_id = save_to_local_db(donation_record, 'donations')
            if not local_id:
                logger.error("Failed to save donation to local storage (Mongo unavailable).")
                return jsonify({
                    "status": "error",
                    "message": "Failed to save donation record (Mongo not available, local insert failed)."
                }), 500

        #################################################################
        # 7. Return Success
        #################################################################
        return jsonify({
            "status": "success",
            "message": "Donation recorded successfully.",
            "transaction_id": order_id,
            "invoice_url": invoice_url
        }), 200

    except Exception as e:
        logger.error(f"❌ Error processing PayPal transaction: {e}")
        return jsonify({
            "status": "error",
            "message": f"Server Error: {str(e)}"
        }), 500

###############################################################################
# Convert INR to USD

###############################################################################
@app.route('/convert-inr-to-usd', methods=['POST'])
def convert_inr_to_usd():
    data = request.json
    amount_inr = float(data.get("amount_inr", 0))

    usd_rate = get_usd_to_inr()
    amount_usd = round(amount_inr / usd_rate, 2)

    return jsonify({"amount_usd": amount_usd})

###############################################################################
# Run
###############################################################################
# In main_server.py
if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    finally:
        # If we started the proxy server above, you can optionally kill it here.
        pass