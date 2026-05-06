# you need to install all these in your terminal
# pip install streamlit scikit-learn python-docx PyPDF2 google-generativeai

import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import json
import google.generativeai as genai

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
try:
    svc_model = pickle.load(open('clf.pkl', 'rb'))  
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
    le = pickle.load(open('encoder.pkl', 'rb'))  
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure clf.pkl, tfidf.pkl, and encoder.pkl exist.")


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# --- AGENTIC AI FUNCTIONS ---
def get_model():
    # Automatically find an available model for the user's API key
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if not available_models:
        return genai.GenerativeModel('gemini-1.5-flash')
    
    # Prefer ANY flash model (e.g., gemini-1.5-flash, gemini-2.0-flash) as they have generous free tiers
    for name in available_models:
        if 'flash' in name: return genai.GenerativeModel(name)
    
    # Fallback to older pro models, avoid newer ones with 0 free tier
    for name in available_models:
        if 'pro' in name and '2.5' not in name: return genai.GenerativeModel(name)
    
    return genai.GenerativeModel(available_models[0])

def generate_comprehensive_report_json(resume_text, job_description=""):
    model = get_model()
    prompt = f"""
You are ARIA — an Intelligent AI Recruitment & Career Analysis Agent.

Your task is to analyze a resume and optionally compare it with a Job Description (JD).

IMPORTANT:
- Your output MUST be in clean, structured JSON format
- This JSON will be used to render an interactive dashboard UI
- Keep text concise, impactful, and UI-friendly
- Avoid long paragraphs (use bullet-style insights)

----------------------------------------
🔹 OUTPUT FORMAT (STRICT JSON)
----------------------------------------

{{
  "sections": [
    
    {{
      "id": "jd_matching",
      "title": "JD Matching Analysis",
      "visible": true,
      "data": {{
        "match_percentage": 75,
        "recommendation": "Hire",
        
        "matching_skills": [],
        "missing_skills": [],
        
        "analysis": {{
          "experience": "",
          "education": "",
          "projects": ""
        }},

        "strengths": [],
        "weaknesses": [],
        "role_fit_summary": ""
      }}
    }},

    {{
      "id": "ats_report",
      "title": "ATS Report",
      "visible": true,
      "data": {{
        "ats_score": 96,

        "breakdown": {{
          "keywords": 18,
          "formatting": 14,
          "content": 20,
          "skills": 19,
          "experience": 15,
          "grammar": 10
        }},

        "issues": [],
        "improvements": []
      }}
    }},

    {{
      "id": "dashboard",
      "title": "Career Dashboard",
      "visible": true,
      "data": {{
        "profile_summary": "",
        "experience_level": "",

        "skills": {{
          "technical": [],
          "tools": [],
          "soft": []
        }},

        "top_skills": [],
        "skill_gaps": [],

        "scores": {{
          "career_readiness": 0,
          "project_quality": 0,
          "resume_quality": 0
        }},

        "domain": "",
        "career_paths": [],
        "next_steps": {{
          "skills": [],
          "certifications": [],
          "projects": []
        }}
      }}
    }}

  ]
}}

----------------------------------------
🔹 UI INSTRUCTIONS (VERY IMPORTANT)
----------------------------------------

- Keep each list short (5–8 items max)
- Use short phrases instead of long explanations
- Scores must be integers (0–100)
- Avoid redundancy
- Make insights crisp for card-based UI

----------------------------------------
🔹 BEHAVIOR RULES
----------------------------------------

- If JD is NOT provided or is empty/None:
  → Skip JD Matching OR mark "visible": false

- If JD is provided:
  → Fill all sections properly

----------------------------------------
🔹 INPUTS
----------------------------------------

Resume:
{resume_text}

Job Description:
{job_description if job_description else "None provided."}
    """
    # Force Gemini to return strictly JSON
    response = model.generate_content(
        prompt, 
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json"
        )
    )
    return response.text


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction & Agent", page_icon="📄", layout="wide")

    # Sidebar for Agentic AI Configuration
    st.sidebar.title("🤖 ARIA Agent Settings")
    st.sidebar.markdown("Enter your Google Gemini API key to unlock advanced agentic features.")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key, transport='rest')
        st.sidebar.success("API Key configured successfully!")
    else:
        st.sidebar.warning("Please enter your API Key to use the Agentic features.")

    st.title("Resume Category Prediction & Career Agent")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Successfully extracted the text from the uploaded resume.")

            # Create tabs for Basic Prediction vs Advanced Agent
            tab1, tab2 = st.tabs(["📊 Basic ML Classification", "🤖 Interactive Career Dashboard"])

            with tab1:
                st.subheader("Predicted Category")
                category = pred(resume_text)
                st.write(f"The predicted category of the uploaded resume is: **{category}**")
                
                if st.checkbox("Show extracted text", False):
                    st.text_area("Extracted Resume Text", resume_text, height=300)

            with tab2:
                st.subheader("Interactive ARIA Dashboard")
                if not api_key:
                    st.info("Please enter your Gemini API Key in the sidebar to unlock these features.")
                else:
                    st.markdown("Provide an optional Job Description to get a tailored JD Matching Analysis.")
                    job_desc = st.text_area("Job Description (Optional)", height=150, placeholder="Paste job description here...")
                    
                    if st.button("🚀 Generate Interactive Dashboard", use_container_width=True):
                        with st.spinner("ARIA is generating your structured JSON dashboard data..."):
                            report_json_str = generate_comprehensive_report_json(resume_text, job_desc)
                            
                            try:
                                report_data = json.loads(report_json_str)
                                
                                # Render the JSON data as a beautiful UI
                                for section in report_data.get("sections", []):
                                    if not section.get("visible", True):
                                        continue
                                        
                                    st.markdown("---")
                                    st.header(section["title"])
                                    data = section["data"]
                                    
                                    if section["id"] == "jd_matching":
                                        col1, col2 = st.columns(2)
                                        col1.metric("Match Percentage", f"{data.get('match_percentage', 0)}%")
                                        col2.metric("Recommendation", data.get('recommendation', 'N/A'))
                                        
                                        col_skills1, col_skills2 = st.columns(2)
                                        with col_skills1:
                                            st.markdown("**✅ Matching Skills**")
                                            for s in data.get('matching_skills', []): st.caption(f"- {s}")
                                        with col_skills2:
                                            st.markdown("**❌ Missing Skills**")
                                            for s in data.get('missing_skills', []): st.caption(f"- {s}")
                                        
                                        st.markdown("**Detailed Analysis**")
                                        analysis = data.get('analysis', {})
                                        st.write("**Experience:** ", analysis.get('experience', ''))
                                        st.write("**Education:** ", analysis.get('education', ''))
                                        st.write("**Projects:** ", analysis.get('projects', ''))
                                        
                                        col_sw1, col_sw2 = st.columns(2)
                                        with col_sw1:
                                            st.markdown("**💪 Strengths:**")
                                            for s in data.get('strengths', []): st.write(f"- {s}")
                                        with col_sw2:
                                            st.markdown("**⚠️ Weaknesses:**")
                                            for w in data.get('weaknesses', []): st.write(f"- {w}")
                                        
                                        st.info(data.get('role_fit_summary', ''))
                                        
                                    elif section["id"] == "ats_report":
                                        st.metric("Total ATS Score", f"{data.get('ats_score', 0)}/100")
                                        
                                        st.markdown("**Score Breakdown**")
                                        breakdown = data.get("breakdown", {})
                                        cols = st.columns(len(breakdown))
                                        for i, (k, v) in enumerate(breakdown.items()):
                                            cols[i].metric(k.capitalize(), v)
                                            
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("**🔴 Key Issues Detected:**")
                                            for iss in data.get('issues', []): st.write(f"- {iss}")
                                        with col2:
                                            st.markdown("**🟢 ATS Improvement Suggestions:**")
                                            for imp in data.get('improvements', []): st.write(f"- {imp}")
                                            
                                    elif section["id"] == "dashboard":
                                        st.markdown(f"*{data.get('profile_summary', '')}*")
                                        col_l, col_d = st.columns(2)
                                        col_l.markdown(f"**Experience Level:** {data.get('experience_level', '')}")
                                        col_d.markdown(f"**Domain:** {data.get('domain', '')}")
                                        
                                        # scores
                                        scores = data.get('scores', {})
                                        sc1, sc2, sc3 = st.columns(3)
                                        sc1.metric("Career Readiness", f"{scores.get('career_readiness', 0)}/100")
                                        sc2.metric("Project Quality", f"{scores.get('project_quality', 0)}/100")
                                        sc3.metric("Resume Quality", f"{scores.get('resume_quality', 0)}/100")
                                        
                                        st.markdown("**Skills Overview**")
                                        skills = data.get('skills', {})
                                        st.write("**Technical:** ", ", ".join(skills.get('technical', [])))
                                        st.write("**Tools:** ", ", ".join(skills.get('tools', [])))
                                        st.write("**Soft Skills:** ", ", ".join(skills.get('soft', [])))
                                        
                                        st.markdown("**Top Skills & Gaps**")
                                        st.write("**Top Skills:** ", ", ".join(data.get('top_skills', [])))
                                        st.write("**Skill Gaps:** ", ", ".join(data.get('skill_gaps', [])))
                                        
                                        st.markdown("**Career Progression**")
                                        st.write("**Suggested Paths:** ", ", ".join(data.get('career_paths', [])))
                                        
                                        st.markdown("**Next Steps:**")
                                        next_steps = data.get('next_steps', {})
                                        st.write("**Skills to learn:** ", ", ".join(next_steps.get('skills', [])))
                                        st.write("**Certifications:** ", ", ".join(next_steps.get('certifications', [])))
                                        st.write("**Projects:** ", ", ".join(next_steps.get('projects', [])))

                                # Add download button for raw JSON
                                st.markdown("---")
                                st.download_button(
                                    label="📥 Download Raw JSON Report",
                                    data=json.dumps(report_data, indent=2),
                                    file_name="ARIA_Interactive_Report.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                                
                            except json.JSONDecodeError:
                                st.error("Failed to parse JSON from ARIA. Please try again.")
                                st.code(report_json_str)

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
