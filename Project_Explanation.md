# 📌 Resume Category Predictor & Career Agent (ARIA) - Project Documentation

## 1. Project Overview

**What is the project about?**
The Resume Category Predictor is an intelligent, dual-engine web application. It combines traditional Machine Learning (to classify resumes into specific job categories) with an advanced Agentic AI system called ARIA (to provide deep career insights, Applicant Tracking System (ATS) scoring, and resume optimization).

**Problem Statement:**
1. **For HRs/Recruiters:** Manually screening hundreds of resumes is time-consuming, prone to bias, and inefficient.
2. **For Candidates:** Job seekers struggle to understand why their resumes are rejected by ATS algorithms and lack personalized guidance on how to tailor their skills for specific Job Descriptions (JDs).

**Objective of the project:**
To bridge the gap between candidates and recruiters by automating resume classification and providing a personal AI career agent that actively analyzes skill gaps, scores resumes against JDs, and generates optimized, ATS-friendly documents.

**Target Users:**
- **Job Seekers:** To analyze and improve their resumes.
- **Recruiters/HR Professionals:** To rapidly screen and categorize applicant data.

---

## 🏗️ 2. System Architecture

**Overall Workflow:**
1. **Input Stage:** The user uploads a resume (PDF, DOCX, or TXT) and optionally provides a target Job Description.
2. **Text Processing:** The backend extracts raw text from the file and cleans it using Regular Expressions (Regex) to remove noise (URLs, special characters, etc.).
3. **ML Classification Engine:** The cleaned text is vectorized using TF-IDF and passed to a pre-trained Support Vector Machine (SVM) model to predict the candidate's professional domain.
4. **Agentic AI Engine (Groq):** The raw text, the predicted role, and the JD are sent to the Groq API. The LLM acts as an Agent, processing the context and returning highly structured JSON data.
5. **Output Stage:** The Streamlit frontend parses the JSON to dynamically render interactive charts, match scores, and dynamically generates a downloadable, optimized `.docx` file.

**Component Interaction:**
The architecture is monolithic but highly modularized. The **Frontend** (Streamlit UI) directly calls Python backend functions. The **Backend** interacts with local storage for ML weights (`.pkl` files) and communicates via HTTP requests to the external **Groq Cloud API** for LLM generation. 

---

## 💻 3. Technologies Used

**Tools, Frameworks, and Languages:**
- **Language:** Python (Core logic and backend)
- **Frontend Framework:** Streamlit (For rapid, interactive UI development)
- **Machine Learning:** Scikit-learn (SVM, TF-IDF, LabelEncoder), Pandas (Data manipulation)
- **Agentic AI / LLM:** Groq API (specifically using the `llama3-8b-8192` model)
- **File Processing:** `PyPDF2` (for PDF text extraction), `python-docx` (for reading and writing Word documents)

**Why each technology was chosen:**
- *Streamlit:* Allows building beautiful, data-driven web apps entirely in Python without needing separate HTML/JS code.
- *SVM & TF-IDF:* SVM is highly effective for high-dimensional text classification tasks (like NLP), and TF-IDF accurately represents word importance.
- *Groq (Llama 3):* Chosen for its ultra-fast inference speeds, which is crucial for Agentic workflows that require real-time JSON generation.

**Role of AntiGravity in the Project:**
AntiGravity acted as your advanced coding pair-programmer. My role involved integrating the Groq API, structuring the strict JSON-enforced prompts, building the interactive Streamlit UI (tabs, columns, metrics), debugging file parsing errors, and implementing the dynamic `.docx` generation feature.

---

## 💾 4. Data Handling

**Type of Data Used:**
- **Static Data:** Pre-trained ML weights (`clf.pkl`, `tfidf.pkl`, `encoder.pkl`) trained on a dataset of 900+ resumes (`UpdatedResumeDataSet.csv`).
- **User-Generated Data:** Resumes uploaded by the user and custom Job Descriptions.
- **Real-Time Data:** The JSON insights generated on-the-fly by the Groq AI model.

**Data Storage & Structure:**
Currently, the application is stateless. User data is processed in-memory (RAM) during the session and is not stored in a persistent database (like SQL or MongoDB), ensuring total data privacy for the candidate. The ML models are stored locally as serialized Pickle objects.

**Data Flow from Input to Output:**
`File Upload -> Byte Stream -> Extracted String -> Regex Cleaned String -> TF-IDF Vector Array -> SVM Prediction -> Extracted String + Prediction + JD -> Groq API Prompt -> JSON String -> Python Dictionary -> Streamlit UI / DOCX Buffer.`

---

## ✨ 5. Features Breakdown

**1. Basic ML Classification**
- *What it does:* Predicts the professional category of the uploaded resume (e.g., Data Scientist, Java Developer).
- *How it works:* Cleans the text, converts it to numbers via TF-IDF, and runs it through the trained SVM model.

**2. ARIA Interactive Dashboard**
- *What it does:* Provides a high-level overview of the candidate, showing a career score, top skills, skill gaps, and a visual skills distribution chart.
- *How it works:* The Groq LLM evaluates the text and outputs structured metrics. Pandas and Streamlit render the charts.

**3. JD Match & ATS Report**
- *What it does:* Compares the resume to a specific Job Description and calculates an ATS score. It highlights exactly which keywords are missing and what formatting issues exist.
- *How it works:* The LLM acts as a recruiter, cross-referencing the JD with the resume, outputting strengths and weaknesses into specific JSON arrays.

**4. AI Resume Optimizer & Exporter**
- *What it does:* Tells the user exactly what to *add*, *replace*, or *remove*, suggests specific projects to build, and generates a brand-new, formatted Microsoft Word (`.docx`) file.
- *How it works:* `python-docx` creates a new document buffer in memory, loops through the LLM's suggested JSON structure, applies formatting (headings, bullet points), and serves it as a download button.

---

## 🤖 6. Agentic AI Usage

**Where and What it does:**
The AI Agent (named ARIA) powers the entire right side of the application. Instead of just "chatting," it performs analytical tasks: scoring resumes, identifying domain-specific skill gaps, and restructuring document formatting.

**How it makes decisions (Logic/Flow):**
The Agent is constrained by **System Prompting**. It is fed the resume text and the ML-predicted role (to ground its context and prevent hallucinations). It is instructed strictly to evaluate the data as an ATS system would.

**Tools & Frameworks:**
- **Groq API:** The engine running the LLM.
- **Llama-3-8b-8192:** The specific open-source model used for its fast logic processing.
- **JSON Mode:** The Groq API is forced to return `{"type": "json_object"}`, ensuring the output is heavily structured and never breaks the Python application.

**Step-by-Step Working of AI Feature:**
1. User clicks "Render Dashboard".
2. Python crafts a massive prompt containing the Resume, JD, and a blank "Schema" of how the JSON must look.
3. The prompt is sent to Groq.
4. Groq replies with a JSON string.
5. Python uses `json.loads()` to convert it to a dictionary.
6. The app loops through the dictionary keys to populate Streamlit UI columns and metrics.

---

## 🛠️ 7. Implementation Details

**Development Process:**
1. **Phase 1 (Data Science):** Cleaned the Kaggle dataset, applied TF-IDF, trained the SVM classifier, and saved the models using `pickle`.
2. **Phase 2 (Core App):** Created `app.py`, built the file upload logic handling PDF/DOCX encodings.
3. **Phase 3 (Agentic Integration):** Wrote the `generate_interactive_ui_json` functions, integrating Groq and handling API keys dynamically.
4. **Phase 4 (UI/UX):** Used Streamlit tabs, columns, and metric components to make the JSON data visually stunning. Added the `.docx` generation pipeline.

**Algorithms Used:**
- **Support Vector Machine (SVM):** Used for finding the optimal hyperplane to separate text data into distinct job categories.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Used to convert text into mathematical vectors by evaluating how relevant a word is to a document in a collection.

---

## 🚶 8. User Flow

1. **Launch:** User opens the application.
2. **Auth:** User inputs their Groq API Key in the sidebar.
3. **Upload:** User uploads their current Resume.
4. **Prediction:** System instantly displays their predicted job role.
5. **Context:** User pastes a target Job Description.
6. **Analyze:** User clicks "Render ARIA Dashboard" to view their ATS score, missing skills, and JD Match verdict.
7. **Optimize:** User clicks "Auto Optimize Resume" to read suggested changes.
8. **Action:** User downloads the newly generated `ARIA_Optimized_Resume.docx` to apply for the job.

---

## 🚧 9. Challenges & Solutions

**Challenge 1: LLM Hallucinations and Bad Formatting**
- *Problem:* Standard LLMs return conversational text (e.g., "Here is your analysis..."), which breaks frontend rendering logic.
- *Solution:* Implemented strict prompt engineering and utilized Groq's `response_format={"type": "json_object"}`. Added a robust `try-except` block with a hardcoded fallback JSON in case of API failure.

**Challenge 2: Multi-format File Parsing**
- *Problem:* Users upload different file types, and text extraction libraries often mess up Unicode characters.
- *Solution:* Implemented conditional parsing logic using `PyPDF2` for PDFs and `python-docx` for Word files, followed by a Regex cleaner to strip unreadable ASCII characters.

---

## 🚀 10. Future Scope

1. **Scalable Backend Integration:** Migrating the frontend to Next.js/React and the backend to FastAPI/Node.js for deployment to millions of users.
2. **Database Integration:** Adding PostgreSQL to allow users to create accounts, save their resume history, and track their ATS scores over time.
3. **Automated JD Scraping:** Integrating a web scraper so users can just paste a LinkedIn URL instead of copying and pasting the whole Job Description.
4. **Live Interview Prep:** Using the extracted resume data to generate a mock AI voice interview based on their weaknesses.

---

## 🎓 11. Conclusion

**Final Outcome:**
The project successfully evolved from a basic ML classification script into a full-fledged, Agentic AI Career Dashboard. It accurately categorizes candidates while actively assisting them in improving their employability.

**Impact and Usefulness:**
By combining traditional deterministic ML with generative AI, the system provides both speed and intelligence. It empowers job seekers to bypass rigid ATS filters and allows HRs to make data-driven, skill-based hiring decisions rapidly.
