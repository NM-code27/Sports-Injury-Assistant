PhysioAI – Sports Injury Recovery Chatbot

PhysioAI is a sports injury recovery and prevention chatbot that uses Retrieval-Augmented Generation (RAG) with Ollama and FastAPI.
The system retrieves relevant information from sports medicine documents and generates evidence-based responses.

System Architecture

PhysioAI works using the following pipeline:

User asks a question about a sports injury

Relevant text chunks are retrieved from PDF documents

Embeddings are generated using nomic-embed-text

The llama3.2 model generates the final response

The response is streamed back to the user interface

This allows the chatbot to provide context-aware answers based on medical literature.

Project Folder Structure
physioai
│
├── backend        # FastAPI backend and RAG pipeline
├── frontend       # Chat interface
├── data           # PDF knowledge documents
├── chroma_db      # Vector database (generated automatically)
├── requirements.txt
└── README.md
Prerequisites

## 📥 Download Sample Data

The original PDF documents used in this project are not included due to size limitations.

You can download similar or sample documents from the links below:

- Sports Injury Guide: https://bodyinmovementphysio.wordpress.com/wp-content/uploads/2017/01/the-ioc-manual-of-sports-injuries-2012.pdf
- Athlete Nutrition: https://example.com/nutrition-for-athletes.pdf  
- Injury Prevention Guide: https://stillmed.olympics.com/media/Document%20Library/OlympicOrg/IOC/Who-We-Are/Commissions/Medical-and-Scientific-Commission/Handbooks/2009_Bahr.pdf
-Post-Workout Recovery Nutrition: https://bjsm.bmj.com/content/52/7/439
-Injury Recovery Nutrition: https://www.nata.org/sites/default/files/2025-08/nutrition-for-injury-recovery-and-rehabilitation.pdf
-Pre-Workout Nutrition: https://www.researchgate.net/publication/391923200_The_Influence_of_Pre-and_Post-Workout_Nutrition_on_Enhancing_Physical_Fitness_A_Comprehensive_Review

After downloading, place all PDFs inside:

/data

Then restart the backend.

Before running the project, install:

Python 3.12

Ollama

VS Code or any terminal

Step 1 — Open the Project Folder

Open the PhysioAI project folder in VS Code
or navigate using terminal.

cd physioai

Step 2 — Create a Virtual Environment
py -3.12 -m venv physioenv

Step 3 — Activate the Environment

Windows:

.\physioenv\Scripts\Activate.ps1

After activation you should see:

(physioenv)

Step 4 — Install Dependencies

Install required libraries:

pip install -r requirements.txt
Step 5 — Install Ollama Models

Download the required models:

ollama pull llama3.2
ollama pull nomic-embed-text
Step 6 — Verify Ollama Models
ollama list

Example output:

NAME
llama3.2
nomic-embed-text

Step 7 — Start the Backend Server

Navigate to backend:

cd backend

Run the FastAPI server:

uvicorn main:app --reload

You should see:

Uvicorn running on http://127.0.0.1:8000
Chroma RAG ready
Step 8 — Open the Chatbot

Open the browser and go to:

http://127.0.0.1:8000

The PhysioAI interface will appear.

Knowledge Base

The chatbot retrieves knowledge from sports medicine PDFs placed inside:

/data

These documents include:

Sports injury recovery guides

Injury prevention research

Athlete nutrition for injury recovery

When the server runs for the first time, a Chroma vector database is automatically created in:

/chroma_db

Example Questions

Users can ask questions such as:

How to recover from an ACL injury?

Exercises for ankle sprain recovery

How to prevent hamstring injuries in football?

Nutrition tips for muscle recovery

Notes

The first run may take a few minutes while the Chroma vector database is built.

After the first run, the system loads much faster.

Ensure Ollama is installed and models are downloaded before starting the backend.

Technologies Used

FastAPI – backend API

Ollama – local LLM inference

ChromaDB – vector database

RAG (Retrieval Augmented Generation)

Python