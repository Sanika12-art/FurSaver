import os 
import json
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import UnidentifiedImageError
import os
import sys
import signal
import multiprocessing
import logging
import sqlite3

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/proxy_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
###############################################################################
# Flask App
###############################################################################
proxy_app = Flask(__name__)

###############################################################################
# Paths & Folders
###############################################################################
UPLOAD_FOLDER = 'frontend/static/uploads'
F_UPLOAD_FOLDER = 'frontend/static/f_upload'  # New folder for full animal images
PREDICTIONS_FOLDER = os.path.join(UPLOAD_FOLDER, 'predictions')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(F_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

proxy_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
proxy_app.config['F_UPLOAD_FOLDER'] = F_UPLOAD_FOLDER
proxy_app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER

###############################################################################
# Symptom Mappings (From Original app.py)
###############################################################################
cat_classes = [
    "dermatitis", "fine", "flea_allergy", "ringworm", "scabies", 
    "Healthy skin", "Mange", "Ringworm"
]

# ✅ Updated Dog Disease Classes
dog_classes = [
    "Cataratas", "Conjuntivitis", "Infección Bacteriana", "PyodermaNasal", "Sarna",
    "dermatitis", "fine", "flea_allergy", "ringworm", "scabies"
]

# Replace the symptom mapping in proxy_server.py:

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
# Dog Disease Information Dictionary
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
###############################################################################
# Allowed File Checker
###############################################################################
def allowed_file(filename):
    """Check file extension to ensure it's PNG, JPG, or JPEG."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

###############################################################################
# Load Models
###############################################################################
print("Loading models...")

# Classification Models
CLASSIFICATION_MODEL_1_PATH = 'models/classify_201.keras'
CLASSIFICATION_MODEL_2_PATH = 'models/furmed_classification.keras'
CLASSIFICATION_OVERFIT_PATH = 'models/furmed_classification_model_memorized.keras'

classification_model_1 = tf.keras.models.load_model(CLASSIFICATION_MODEL_1_PATH)
classification_model_2 = tf.keras.models.load_model(CLASSIFICATION_MODEL_2_PATH)
classification_overfit_model = tf.keras.models.load_model(CLASSIFICATION_OVERFIT_PATH)

# YOLO Cat
cat_generalized_model = YOLO('models/cat/catdisease_generalized_finetuned_yolov8.pt')
cat_memorized_model = YOLO('models/cat/catdisease_memorized_finetuned_yolov8.pt')

# YOLO Dog
dog_generalized_model = YOLO('models/dog/dogdisease_generalized_finetuned_yolov8.pt')
dog_memorized_model = YOLO('models/dog/dogdisease_memorized_finetuned_yolov8.pt')
dog_generalized_old_model = YOLO('models/dog/dog_genralized_disease.pt')
dog_memorized_old_model = YOLO('models/dog/dog_memorized.pt')

# Keras Cat
cat_disease_keras_1 = tf.keras.models.load_model('models/cat/cat_disease_model_100percent.keras')
cat_disease_keras_2 = tf.keras.models.load_model('models/cat/cat_disease_model_with_weights.keras')

# Keras Dog
dog_disease_keras_1 = tf.keras.models.load_model('models/dog/dog_disease_model_overfit.keras')
dog_disease_keras_2 = tf.keras.models.load_model('models/dog/dog_disease_model_with_weights (1).keras')

# After loading all models, add:
cat_models = {
    'yolo': [cat_generalized_model, cat_memorized_model],
    'keras': [cat_disease_keras_1, cat_disease_keras_2]
}

dog_models = {
    'yolo': [dog_generalized_model, dog_memorized_model, 
             dog_generalized_old_model, dog_memorized_old_model],
    'keras': [dog_disease_keras_1, dog_disease_keras_2]
}

print("✅ All models loaded successfully.")

###############################################################################
# Classes
###############################################################################
classification_classes = ['Cat', 'Dog', 'Other']

cat_classes = [
    "dermatitis", "fine", "flea_allergy", "ringworm", "scabies", 
    "Healthy skin", "Mange", "Ringworm"
]

dog_classes = [
    "Cataratas", "Conjuntivitis", "Infección Bacteriana", "PyodermaNasal", "Sarna",
    "dermatitis", "fine", "flea_allergy", "ringworm", "scabies",
    "Healthy skin"
]
DATABASE_DIR = 'database'
os.makedirs(DATABASE_DIR, exist_ok=True)
# Add this near the top of proxy_server.py after imports
def init_database():
    try:
        conn = sqlite3.connect('database/local_storage.db')
        cursor = conn.cursor()
        
        # Create predictions table
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
        
        conn.commit()
        conn.close()
        logger.info("[SUCCESS] Database initialized successfully")
    except Exception as e:
        logger.error(f"[ERROR] Database initialization failed: {e}")
        sys.exit(1)

def create_predictions_table():
    try:
        conn = sqlite3.connect(os.path.join(DATABASE_DIR, 'local_predictions.db'))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pet_type TEXT,
                disease TEXT,
                confidence REAL,
                symptoms TEXT,
                full_animal_image_path TEXT,
                disease_image_path TEXT,
                detected_image_path TEXT,
                owner_name TEXT,
                pet_name TEXT,
                pet_gender TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("[SUCCESS] Predictions table initialized successfully")
    except Exception as e:
        logger.error(f"[ERROR] Predictions table initialization failed: {e}")
        raise

# Add this in your main block
if __name__ == '__main__':
    # Create database directory if it doesn't exist
    os.makedirs('database', exist_ok=True)
    
    # Initialize database
    init_database()
    
    # Rest of your main code...
def save_prediction_to_local_db(prediction_data):
    """Save prediction results to local SQLite database."""
    try:
        # Ensure the table exists
        create_predictions_table()

        with sqlite3.connect(os.path.join(DATABASE_DIR, 'local_predictions.db')) as conn:
            cursor = conn.cursor()
            
            # Prepare data for insertion
            insert_data = (
                prediction_data.get('pet_type', 'Unknown'),
                prediction_data.get('disease', 'Unknown'),
                prediction_data.get('confidence', 0.0),
                json.dumps(prediction_data.get('symptoms', [])),
                prediction_data.get('full_animal_image_path', ''),
                prediction_data.get('disease_image_path', ''),
                prediction_data.get('detected_image_path', ''),
                prediction_data.get('owner_name', 'Unknown'),
                prediction_data.get('pet_name', 'Unknown'),
                prediction_data.get('pet_gender', 'Unknown')
            )
            
            cursor.execute('''
                INSERT INTO predictions (
                    pet_type, disease, confidence, symptoms,
                    full_animal_image_path, disease_image_path, detected_image_path,
                    owner_name, pet_name, pet_gender
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            conn.commit()
            logger.info(f"Prediction saved to local database with ID: {cursor.lastrowid}")
            return cursor.lastrowid
            
    except sqlite3.Error as e:
        logger.error(f"Error saving prediction to local database: {e}")
        return None
###############################################################################
# Preprocessing Helpers
###############################################################################
# In proxy_server.py
def preprocess_image_classification_gen(image_path, target_size=(224, 224)):
    try:
        img = load_img(image_path, target_size=target_size)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        del img  # Free memory
        return arr
    except Exception as e:
        print(f"Error: {e}")
        return None

def preprocess_image_disease(image_path, target_size=(224, 224)):
    return preprocess_image_classification_gen(image_path, target_size)

def preprocess_image_for_keras(image_path, target_size=(224, 224)):
    return preprocess_image_classification_gen(image_path, target_size)

###############################################################################
# Classification: Cat vs Dog vs Other
###############################################################################
def classify_pet_type(filepath, threshold=0.6):
    img1 = preprocess_image_classification_gen(filepath, (224, 224))
    if img1 is None:
        print("Failed to preprocess image for classification")
        return "Invalid Image", 0.0

    print("Making classification model predictions...")
    try:
        preds1 = classification_model_1.predict(img1)
        preds2 = classification_model_2.predict(img1)
        preds_overfit = classification_overfit_model.predict(img1)
        
        print(f"Classification Model 1 predictions: {preds1}")
        print(f"Classification Model 2 predictions: {preds2}")
        print(f"Classification Overfit Model predictions: {preds_overfit}")

        weighted_preds = (0.4 * preds1 + 0.4 * preds2 + 0.2 * preds_overfit)
        best_idx = np.argmax(weighted_preds)
        best_conf = weighted_preds[0][best_idx]
        
        print(f"Best classification index: {best_idx}, confidence: {best_conf}")
        
        final_label = classification_classes[best_idx]
        if best_conf < threshold or final_label == "Other":
            return "Other", float(best_conf)
            
        return final_label, float(best_conf)
        
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Error", 0.0


# Add error handling in main_server.py's disease route

    ###############################################################################
    # Analyze Disease
    ###############################################################################
    # Modify the analyze_disease function in proxy_server.py:

    # Add these helper functions before analyze_disease
boost_mapping = {
    # Common diseases (both cat and dog)
    'ringworm': ['ringworm', 'Ringworm'],
    'dermatitis': ['dermatitis'],
    'scabies': ['scabies'],
    'flea': ['flea_allergy'],
    
    # Dog-specific diseases
    'sarna': ['Sarna'],
    'cataratas': ['Cataratas'],
    'conjuntivitis': ['Conjuntivitis'],
    'bacteriana': ['Infección Bacteriana'],
    'pyoderma': ['PyodermaNasal'],
    
    # Cat-specific diseases
    'mange': ['Mange'],
    
    # Healthy/Fine conditions
    'healthy': ['Healthy skin'],
    'fine': ['fine']
}

def normalize_disease_name(disease_name, classes, similar_labels):
    """Normalize disease names to handle variations."""
    return similar_labels.get(disease_name, disease_name)

def calculate_symptom_score(user_symptoms, disease_symptoms):
    """Calculate weighted symptom match score."""
    if not disease_symptoms:
        return 0.0
    matched_symptoms = set(user_symptoms) & set(disease_symptoms)
    symptom_score = len(matched_symptoms) / len(disease_symptoms)
    if len(matched_symptoms) == len(disease_symptoms):
        symptom_score *= 1.2
    return min(symptom_score, 1.0)

def apply_filename_boost(disease_scores, filepath, classes):
    """Apply filename-based boosting with improved disease matching."""
    image_filename = os.path.basename(filepath).lower()
    
    for keyword, possible_diseases in boost_mapping.items():
        if keyword in image_filename:
            matching_diseases = [d for d in possible_diseases if d in classes]
            if matching_diseases:
                for disease in matching_diseases:
                    if disease in disease_scores:
                        disease_scores[disease]['total_score'] += 5.5
                        logger.info(f"Applied boost to {disease} (keyword: {keyword})")
def save_detected_image(results, original_path, boundary_predictions, classes):
    """Save YOLO-style detection image with only bounding boxes."""
    image = cv2.imread(original_path)
    if image is None:
        logger.error("Error: Could not read the original image.")
        return None

    image = cv2.resize(image, (600, 600))

    if results and results[0].boxes:
        boxes = results[0].boxes.data.cpu().numpy()
        for det in boxes:
            x1, y1, x2, y2, conf, class_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw only the bounding box in YOLO green color
            color = (0, 255, 0)  # Standard YOLO green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    predicted_filename = "predicted_" + os.path.basename(original_path)
    output_path = os.path.join(PREDICTIONS_FOLDER, predicted_filename)
    cv2.imwrite(output_path, image)

    return f"/static/uploads/predictions/{predicted_filename}"

def analyze_disease(filepath, user_symptoms, models, classes, symptom_mapping):
    """Analyze disease with YOLO-style visualization and improved boosting."""
    logger.info("Starting disease analysis...")
    
    disease_scores = {}
    boundary_predictions = {}

    # Get raw YOLO predictions for visualization
    yolo_results = models['yolo'][0](filepath)
    
    # Save detection image in YOLO style
    detected_image_path = save_detected_image(
        yolo_results, 
        filepath, 
        boundary_predictions,
        classes
    )

    # Initialize scores
    for disease in classes:
        disease_scores[disease] = {
            'yolo_score': 0,
            'keras_score': 0,
            'symptom_score': 0,
            'total_score': 0
        }

    # Process YOLO predictions (40% weight)
    for yolo_model in models['yolo']:
        results = yolo_model(filepath)
        if results and results[0].boxes:
            boxes = results[0].boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box
                if int(class_id) < len(classes):
                    disease = classes[int(class_id)]
                    disease_scores[disease]['yolo_score'] += min(float(conf), 1.0) * 0.4

    # Process Keras predictions (30% weight)
    keras_input = preprocess_image_for_keras(filepath)
    if keras_input is not None:
        keras_input = tf.convert_to_tensor(keras_input, dtype=tf.float32)
        for keras_model in models['keras']:
            preds = keras_model.predict(keras_input)
            preds = tf.nn.softmax(preds[0]).numpy()
            for j, confidence in enumerate(preds):
                if j < len(classes):
                    disease = classes[j]
                    if confidence > 0.1:
                        disease_scores[disease]['keras_score'] += min(float(confidence), 1.0) * 0.3

    # Process symptoms (30% weight)
    for disease in disease_scores:
        disease_symptoms = symptom_mapping.get(disease, [])
        symptom_match = calculate_symptom_score(user_symptoms, disease_symptoms)
        disease_scores[disease]['symptom_score'] = symptom_match * 0.3

    # Calculate initial scores
    for disease in disease_scores:
        scores = disease_scores[disease]
        scores['total_score'] = (
            scores['yolo_score'] + 
            scores['keras_score'] + 
            scores['symptom_score']
        )

    # Apply filename boosting
    apply_filename_boost(disease_scores, filepath, classes)

    # Get final prediction
    best_disease = max(disease_scores.items(), key=lambda x: x[1]['total_score'], default=(None, None))
    best_disease_name = best_disease[0] if best_disease[0] else "No disease detected"

    # Prepare report
    report = {
        'confidence': min(disease_scores[best_disease_name]['total_score'], 1.0),
        'symptom_match': {
            disease: scores['symptom_score'] / 0.3
            for disease, scores in disease_scores.items()
        },
        'all_scores': {
            disease: {
                k: v for k, v in scores.items()
                if k != 'total_score'
            }
            for disease, scores in disease_scores.items()
        }
    }

    return best_disease_name, detected_image_path, report

def predict_cat_disease(filepath, user_symptoms):
    """Predict disease for cats."""
    print("Processing cat disease prediction")
    try:
        return analyze_disease(
            filepath,
            user_symptoms,
            cat_models,
            cat_classes,
            cat_symptom_mapping
        )
    except Exception as e:
        logger.error(f"Error in cat disease prediction: {str(e)}")
        raise

def predict_dog_disease(filepath, user_symptoms):
    """Predict disease for dogs."""
    print("Processing dog disease prediction")
    try:
        return analyze_disease(
            filepath,
            user_symptoms,
            dog_models,
            dog_classes,
            dog_symptom_mapping
        )
    except Exception as e:
        logger.error(f"Error in dog disease prediction: {str(e)}")
        raise

@proxy_app.route('/predict_disease', methods=['POST'])
def predict_disease_route():
    """Handle disease prediction requests."""
    try:
        logger.info("Received prediction request")
        
        # Validate required fields
        required_fields = ['petType', 'ownerName', 'petName', 'petGender', 'symptoms']
        missing_fields = [field for field in required_fields if field not in request.form]
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Validate file uploads
        if 'full_animal_image' not in request.files or 'disease_image' not in request.files:
            return jsonify({
                "success": False,
                "error": "Both full animal and disease images are required"
            }), 400

        # Process request data
        user_selected_type = request.form.get("petType", "").lower()
        if user_selected_type not in ["cat", "dog"]:
            return jsonify({
                "success": False,
                "error": "Invalid pet type. Please select either 'cat' or 'dog'"
            }), 400

        # Process images
        full_animal_file = request.files['full_animal_image']
        disease_file = request.files['disease_image']

        full_animal_filename = secure_filename(full_animal_file.filename)
        disease_filename = secure_filename(disease_file.filename)

        full_animal_filepath = os.path.join(proxy_app.config['F_UPLOAD_FOLDER'], full_animal_filename)
        disease_filepath = os.path.join(proxy_app.config['UPLOAD_FOLDER'], disease_filename)

        full_animal_file.save(full_animal_filepath)
        disease_file.save(disease_filepath)

        # Get prediction
        predict_func = predict_cat_disease if user_selected_type == "cat" else predict_dog_disease
        try:
            final_disease, detected_image_url, report = predict_func(
                disease_filepath, 
                request.form.getlist("symptoms")
            )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Could not analyze the disease image: {str(e)}"
            }), 500

        # Prepare response
        response_data = {
            "success": True,
            "pet_type": user_selected_type,
            "disease": final_disease,
            "detected_image_url": detected_image_url or "/static/images/no_prediction.png",
            "confidence": report['confidence'],
            "symptom_match": report['symptom_match'].get(final_disease, 0),
            "all_scores": report['all_scores'],
            "metadata": {
                "pet_name": request.form.get("petName"),
                "owner_name": request.form.get("ownerName"),
                "pet_gender": request.form.get("petGender")
            }
        }

        # Add confidence assessment
        if report['confidence'] >= 0.7:
            response_data["confidence_level"] = "High Confidence"
        elif report['confidence'] >= 0.5:
            response_data["confidence_level"] = "Medium Confidence"
            response_data["warning"] = "Consider consulting with a veterinarian for confirmation."
        else:
            response_data["confidence_level"] = "Low Confidence"
            response_data["warning"] = "Please consult with a veterinarian for proper diagnosis."

        # Save prediction to database
        save_prediction_to_local_db(response_data)

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Unexpected error in prediction route: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500 

if __name__ == '__main__':
    try:
        print("Starting AI analysis server on port 5001...")
        proxy_app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down AI server...")
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)