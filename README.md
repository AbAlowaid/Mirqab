# 🎯 Mirqab - Advanced AI-Powered Camouflage Detection System

A comprehensive AI-powered web application for real-time detection and analysis of camouflaged soldiers using DeepLabV3 deep learning model with automated reporting and intelligent querying capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Firebase](https://img.shields.io/badge/Firebase-Admin-orange.svg)

## ✨ Features

### 🔍 Detection & Analysis
- **Image Upload Analysis**: Upload images for automatic camouflage soldier detection
- **Real-time Camera Detection**: Live webcam feed processing with automatic report generation
- **DeepLabV3 Segmentation**: State-of-the-art semantic segmentation model
- **GPU & CPU Support**: Optimized for both GPU acceleration and CPU processing

### 🤖 AI-Powered Intelligence
- **OpenAI GPT-4 Vision**: Automated detailed analysis of detections
- **Smart Report Generation**: Automatic environment, attire, and equipment analysis
- **Moraqib RAG System**: Natural language querying of historical detection reports
- **Contextual Insights**: AI-powered answers based on your detection database

### 📊 Detection Reports Dashboard
- **SOC-Style Interface**: Professional security operations center dashboard
- **Real-time Statistics**: Live detection counts and trends
- **Interactive Charts**: Visual analytics with Chart.js
- **Advanced Filtering**: Filter by time range, status, severity, and device
- **Status Management**: Track and update detection statuses
- **Team Assignment**: Assign detections to team members

### 📄 Report Management
- **PDF Export**: Professional PDF reports with images and metadata
- **Firebase Storage**: Automatic cloud storage of detection images
- **Location Tracking**: GPS/IP-based location detection
- **Image Evidence**: Both original and segmented images stored
- **Detailed Metadata**: Comprehensive detection information

### 🌐 Real-time Features
- **Live Detection**: Continuous monitoring with automatic reporting
- **2-Minute Cooldown**: Intelligent spam prevention for auto-reports
- **Visual Countdown**: Real-time countdown to next report
- **Multi-Device Support**: Track detections from multiple sources

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API Key
- Firebase Project (for storage and database)
- Model file: `best_deeplabv3_camouflage.pth`

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/AbAlowaid/Mirqab.git
cd Mirqab
```

#### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file in project root
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "FIREBASE_CREDENTIALS_PATH=path/to/firebase-credentials.json" >> .env
echo "DEVICE_ID=Main-Detection-Station" >> .env
```

#### 3. Firebase Setup
1. Create a Firebase project at https://console.firebase.google.com
2. Enable Firestore Database and Firebase Storage
3. Download service account credentials JSON file
4. Place it in your project directory
5. Update `FIREBASE_CREDENTIALS_PATH` in `.env`

#### 4. Start Backend Server
```bash
cd backend
python main.py
```
Backend will be available at `http://localhost:8000`

#### 5. Frontend Setup
```bash
cd frontend
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```
Frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
Mirqab/
├── backend/                          # FastAPI Backend
│   ├── main.py                      # Main API server with all endpoints
│   ├── model_handler.py             # DeepLabV3 model handler
│   ├── llm_handler.py               # OpenAI GPT-4 Vision integration
│   ├── moraqib_rag.py               # RAG system for intelligent queries
│   ├── firestore_handler.py         # Firebase Firestore database
│   ├── firebase_storage_handler.py  # Firebase Storage for images
│   └── utils.py                     # Utility functions
├── frontend/                         # Next.js Frontend
│   ├── src/
│   │   ├── app/                     # Next.js 14 App Router
│   │   │   ├── page.tsx            # Home page
│   │   │   ├── upload/             # Upload & analysis page
│   │   │   ├── detection-reports/  # Reports dashboard
│   │   │   ├── moraqib/            # RAG AI assistant
│   │   │   ├── about/              # About page
│   │   │   └── live/               # Live detection (optional)
│   │   ├── components/              # React components
│   │   │   ├── Navbar.tsx          # Navigation bar
│   │   │   ├── ReportModal.tsx     # Report display modal
│   │   │   ├── LocationMap.tsx     # Map component
│   │   │   └── DetectionReports/   # Dashboard components
│   │   └── utils/
│   │       └── pdfGenerator.ts     # PDF generation utility
│   └── public/                      # Static assets
│       └── Mirqab_white (1).png    # Logo
├── best_deeplabv3_camouflage.pth    # Trained model file
├── segmentation.py                   # Live detection script
├── realtime_camera_detection.py      # Camera detection with auto-report
├── RASPBERRY_PI_5_SETUP_GUIDE.md    # Raspberry Pi deployment guide
├── requirements.txt                  # Python dependencies
├── .env                             # Environment variables
└── README.md                        # This file
```

## 🛠️ Technologies

### Backend
- **FastAPI**: High-performance async API framework
- **PyTorch**: Deep learning framework
- **DeepLabV3**: Semantic segmentation model (ResNet-101 backbone)
- **OpenAI GPT-4 Vision**: Advanced image analysis
- **Firebase Admin SDK**: Cloud database and storage
- **LangChain**: RAG system implementation

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Interactive charts and graphs
- **Leaflet**: Interactive maps
- **jsPDF**: PDF generation

### AI/ML
- **DeepLabV3**: Semantic segmentation
- **OpenAI GPT-4 Vision**: Image understanding and analysis
- **OpenAI Embeddings**: Vector embeddings for RAG
- **FAISS**: Vector similarity search

## 📡 API Endpoints

### Detection & Analysis
- `POST /api/analyze_media` - Analyze uploaded image/video
- `POST /api/report_detection` - Create new detection report
- `POST /api/test_segmentation` - Test segmentation on image

### Reports Management
- `GET /api/detection-reports` - Get all detection reports with filters
- `GET /api/detection-stats` - Get detection statistics
- `GET /api/fetch-image-base64` - Fetch Firebase images for PDF generation

### AI Assistant
- `POST /api/moraqib_query` - Query reports using natural language

### System
- `GET /health` - Health check and system status

## 🎯 Key Features Explained

### 1. Live Detection with Auto-Reporting
Run `segmentation.py` or `realtime_camera_detection.py` for continuous monitoring:
- Detects camouflaged soldiers in real-time
- Automatically generates reports every 2 minutes
- Uploads images to Firebase Storage
- Stores reports in Firestore
- Displays countdown to next report

### 2. Moraqib RAG System
Intelligent AI assistant that can:
- Answer questions about detection history
- Analyze trends and patterns
- Provide insights from past reports
- Query by location, time, environment, etc.

### 3. SOC Dashboard
Professional security operations center interface:
- Real-time KPI cards
- Interactive charts (detections over time, environment breakdown)
- Filterable detection table
- Status and severity management
- Team assignment capabilities

### 4. PDF Report Generation
Professional reports including:
- Report ID and timestamp
- Location with coordinates
- AI analysis (environment, soldier count, attire, equipment)
- Visual evidence (original and segmented images)
- Mirqab branding

## 🔧 Configuration

### Environment Variables

**Root `.env` file:**
```env
OPENAI_API_KEY=your_openai_api_key_here
FIREBASE_CREDENTIALS_PATH=./backend/firebase-credentials.json
DEVICE_ID=Main-Detection-Station
```

**Frontend `.env.local`:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Firebase Setup
1. Create Firestore collection: `detection_reports`
2. Enable Firebase Storage bucket
3. Set up storage rules for public read access
4. Download service account credentials

## 🚦 Usage Examples

### Upload Image Analysis
1. Navigate to Upload page
2. Select image or drag & drop
3. Wait for AI analysis
4. View detailed report
5. Download PDF

### Live Detection
```bash
python segmentation.py
# or
python realtime_camera_detection.py
```

### Query with Moraqib AI
1. Navigate to Moraqib AI page
2. Ask questions like:
   - "How many detections were made today?"
   - "Show me detections in forest environments"
   - "What equipment was most commonly detected?"

## 👥 Development Team

**Data Science & Machine Learning Bootcamp - Tuwaiq Academy 2025**

- **Saif Alotaibi** - Team Leader  
  📧 cssaif.o@gmail.com

- **Fatimah Alsubaie** - Data Scientist  
  📧 fatima.t.alsubaie@gmail.com

- **Abdulrahman Attar** - Data Analyst  
  📧 abdulrahman.att7@gmail.com

- **Mousa Alatewi** - Data Scientist  
  📧 mousa.alatwei.1@gmail.com

- **Abdulelah Alowaid** - Data Scientist  
  📧 ab.alowaid@gmail.com

## 🏢 Project Beneficiaries

This system is designed to benefit key defense and security organizations in Saudi Arabia:

- **SAMI** - Saudi Arabian Military Industries
- **GAMI** - General Authority for Military Industries
- **SAFCSP** - Saudi Federation for Cybersecurity, Programming and Drones

## 📝 License

This project is developed as part of the Data Science & ML Bootcamp at Tuwaiq Academy.

## 🤝 Contributing

This is an educational project developed during the bootcamp. For questions or suggestions, please contact the development team.

## 🙏 Acknowledgments

- Tuwaiq Academy for providing the learning platform
- OpenAI for GPT-4 Vision API
- Firebase for cloud infrastructure
- PyTorch and torchvision teams

---

**Status: ✅ Production Ready** | **Version: 3.0** | **Last Updated: October 2025**

*Built with ❤️ by Team Mirqab at Tuwaiq Academy*
