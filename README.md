# 🐾 FurSaver: AI-Powered Pet & Stray Animal Health Diagnosis System

**FurSaver** is an AI-driven web application designed to diagnose diseases in pets and stray animals using deep learning models. It leverages **YOLOv8 for object detection** and **custom-trained CNN models** for disease classification, offering a fast and scalable **AI-powered veterinary diagnosis system**.

## 🚀 Features

- **Pet Classification:** Identifies if an uploaded image is of a **Cat, Dog, or Other**
- **Disease Detection:** Uses **YOLOv8 & Keras models** to classify diseases in cats and dogs
- **Dual-Server Architecture:**
  - **Main Server:** Handles user interactions, form submissions, and disease reports
  - **Proxy Server:** Performs deep learning inference for disease prediction
- **MongoDB Integration:** Stores **disease reports, user details, and donations**
- **PayPal Payment Gateway:** Enables donations with **INR support & invoice generation**
- **Email Feedback System:** Users can send feedback via email
- **Mobile Responsive UI:** Optimized for **desktop, tablet, and mobile devices**

## 📁 Project Structure

```
FurSaver/
│
├── backend/
│   ├── main_server.py     # Flask main server (User interactions, DB, PayPal)
│   └── proxy_server.py    # Flask proxy server (ML model inference)
│
├── models/                # Disease classification models
│   ├── cat/              # Cat disease models
│   │   ├── cat_disease_model_100percent.keras
│   │   ├── cat_disease_model_with_weights.keras
│   │   ├── catdisease_generalized_finetuned_yolov8.pt
│   │   └── catdisease_memorized_finetuned_yolov8.pt
│   │
│   └── dog/              # Dog disease models
│       ├── dog_disease_model_overfit.keras
│       ├── dog_disease_model_with_weights.keras
│       ├── dogdisease_generalized_finetuned_yolov8.pt
│       └── dogdisease_memorized_finetuned_yolov8.pt
│
├── frontend/
│   ├── static/           # Static files
│   │   ├── css/         # Stylesheets
│   │   ├── js/          # JavaScript files
│   │   ├── images/      # Static images
│   │   ├── uploads/     # Disease image uploads
│   │   ├── f_upload/    # Full animal uploads
│   │   └── invoices/    # Generated donation invoices
│   │
│   └── templates/        # HTML Templates
│       ├── home.html
│       ├── disease.html
│       ├── result.html
│       └── admin_dashboard.html
│
├── .env                  # Environment variables
├── .gitignore           # Git ignore file
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🛠 Installation

### Prerequisites

- Python 3.9+
- MongoDB 5.0+
- Git

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/your-username/FurSaver.git
cd FurSaver
```

### 2️⃣ Create & Activate Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 4️⃣ MongoDB Setup

1. Create a MongoDB database named 'fur-med'
2. Create collections: 'predictions' and 'donations'
3. Set up indexes (see mongodbsetup.txt for detailed instructions)

### 5️⃣ Configure Environment Variables

Create a `.env` file in the root directory:

```ini
# MongoDB
MONGO_URI=your_mongodb_uri

# PayPal Configuration
PAYPAL_CLIENT_ID=your_paypal_client_id
PAYPAL_SECRET=your_paypal_secret
PAYPAL_API_BASE=https://api-m.sandbox.paypal.com

# Mail Configuration
MAIL_USERNAME=FurSaver.19@gmail.com
MAIL_PASSWORD=your_app_password

# App Secret
SECRET_KEY=your-strong-secret-key
```

### 6️⃣ Create Required Directories

```sh
mkdir -p frontend/static/uploads
mkdir -p frontend/static/f_upload
mkdir -p frontend/static/invoices
```

## 🚀 Running the Application

1. Start the Proxy Server (AI Models):
```sh
python proxy_server.py
```

2. Start the Main Server (in a new terminal):
```sh
python main_server.py
```*99999999999999999999999999999999999999999996


The application will be available at: http://localhost:5000

## 🔥 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/disease` | POST | Upload and analyze pet images |
| `/donation` | POST | Process donations |
| `/send_feedback` | POST | Submit user feedback |
| `/api/get_predictions` | GET | Fetch prediction history |
| `/api/get_donations` | GET | Fetch donation history |

## 🤖 Supported Diseases

### Dogs
- Cataratas
- Conjuntivitis
- Infección Bacteriana
- PyodermaNasal
- Sarna
- Dermatitis
- Ringworm

### Cats
- Dermatitis
- Flea Allergy
- Ringworm
- Scabies
- Mange

## 🔒 Security

- All sensitive credentials are stored in environment variables
- PayPal integration uses sandbox for testing
- MongoDB access is protected with authentication
- File uploads are validated and sanitized

## 📧 Contact

For questions or support:
- Email: FurSaver.19@gmail.com
- Website: [FurSaver](https://FurSaver.com)

## 📄 License

This project is licensed under the MIT License.

---
🐾 **FurSaver - Enhancing Animal Healthcare with AI** 🐱🐶
