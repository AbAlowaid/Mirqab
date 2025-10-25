# 🎯 Mirqab - Advanced Camouflage Detection System

An AI-powered web application for real-time detection and analysis of camouflaged soldiers using DeepLabV3 deep learning model.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)

## ✨ Features

- **🔍 Image Analysis**: Upload images for automatic soldier detection
- **📹 Real-time Detection**: Live webcam feed processing
- **🤖 AI Reports**: Automatic detailed analysis with Google Gemini
- **📊 Dashboard**: View and manage detection reports
- **🧠 RAG Assistant**: Natural language querying of reports
- **📄 PDF Export**: Downloadable reports with metadata

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Google Gemini API Key
- Firebase Project (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AbAlowaid/Mirqab.git
cd Mirqab
```

2. **Backend Setup**
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
echo "PROMPTLAYER_API_KEY=your_promptlayer_key_here" >> .env

# Start backend
cd backend
python main.py
```

3. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

4. **Access Application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 📁 Project Structure

```
Mirqab/
├── backend/                 # FastAPI backend
│   ├── main.py             # API server
│   ├── model_handler.py    # DeepLabV3 model
│   ├── llm_handler.py      # Google Gemini integration
│   └── moraqib_rag.py      # RAG system
├── frontend/               # Next.js frontend
│   ├── src/app/            # Pages
│   └── src/components/     # React components
├── pictures/               # Project images
└── requirements.txt        # Python dependencies
```

## 🛠️ Technologies

- **Backend**: FastAPI, PyTorch, DeepLabV3, Google Gemini
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **AI/ML**: DeepLabV3 (ResNet-101), Google Gemini 2.5 Flash
- **Database**: Firebase Firestore (optional)

## 📡 API Endpoints

- `POST /api/analyze_media` - Analyze uploaded images
- `POST /api/moraqib_query` - Query reports with natural language
- `GET /api/detection-reports` - Get detection reports
- `GET /health` - Health check

## 👥 Development Team

**Data Science & Machine Learning Bootcamp at Tuwaiq Academy**

- **Saif Alotaibi** - Team Leader  
  📧 cssaif.o@gmail.com
- **Fatimah Alsubaie** - Data Scientist  
  📧 fatima.t.alsubaie@gmail.com
- **Abdulrahman Attar** - Data Analyst  
  📧 abdulrahman.att7@gmail.com
- **Mousa Alatwei** - Data Scientist  
  📧 mousa.alatwei.1@gmail.com
- **Abdulelah Alowaid** - Data Scientist  
  📧 ab.alowaid@gmail.com

## 🏢 Project Beneficiaries

This system benefits key defense organizations in Saudi Arabia:
- **SAMI** (Saudi Arabian Military Industries)
- **GAMI** (General Authority for Military Industries)  
- **SAFCSP** (Saudi Federation for Cybersecurity, Programming and Drones)


---

**Status: ✅ Production Ready** | **Version: 2.0** | **Last Updated: October 2025**