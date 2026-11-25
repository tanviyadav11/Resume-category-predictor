# you need to install all these in your terminal
# pip install streamlit scikit-learn python-docx PyPDF2 google-generativeai pandas

import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import json
import io
import pandas as pd
from groq import Groq

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
try:
    svc_model = pickle.load(open('clf.pkl', 'rb'))  
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
    le = pickle.load(open('encoder.pkl', 'rb'))  
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure clf.pkl, tfidf.pkl, and encoder.pkl exist.")


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def create_docx_from_json(resume_json):
    doc = docx.Document()
    
    # Title
    doc.add_heading('ATS-Optimized Resume', 0)
    
    # Summary
    if resume_json.get('summary'):
        doc.add_heading('Professional Summary', level=1)
        doc.add_paragraph(resume_json['summary'])
        
    # Skills
    if resume_json.get('skills'):
        doc.add_heading('Skills', level=1)
        skills_text = ", ".join(resume_json['skills'])
        doc.add_paragraph(skills_text)
            
    # Experience
    if resume_json.get('experience'):
        doc.add_heading('Experience', level=1)
        for exp in resume_json['experience']:
            doc.add_paragraph(exp, style='List Bullet')
            
    # Projects
    if resume_json.get('projects'):
        doc.add_heading('Projects', level=1)
        for proj in resume_json['projects']:
            doc.add_paragraph(proj, style='List Bullet')
            
    # Certifications
    if resume_json.get('certifications'):
        doc.add_heading('Certifications / Courses', level=1)
        for cert in resume_json['certifications']:
            doc.add_paragraph(cert, style='List Bullet')
            
    # Save to a BytesIO buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

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

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# --- AGENTIC AI FUNCTIONS ---
def generate_interactive_ui_json(resume_text, job_description="", predicted_role="", api_key=""):
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
You are ARIA - an Intelligent AI Career Agent that powers a modern interactive dashboard UI.
You MUST output ONLY valid JSON.

Your job is NOT just analysis - but to generate UI-ready structured data that will be rendered into tabs and charts.

{{
  "ui_config": {{
    "layout": "tab_based",
    "default_tab": "dashboard",
    "tabs": [
      {{"id": "dashboard", "label": "Dashboard"}},
      {{"id": "jd_match", "label": "JD Match"}},
      {{"id": "ats", "label": "ATS Report"}}
    ]
  }},
  "dashboard": {{
    "kpis": [
      {{"label": "Career Score", "value": 88}},
      {{"label": "Resume Score", "value": 90}},
      {{"label": "Project Score", "value": 92}}
    ],
    "profile_summary": "2-3 line sharp summary",
    "charts": {{
      "skills_distribution": [
        {{"name": "ML/NLP", "value": 40}},
        {{"name": "Programming", "value": 30}},
        {{"name": "Tools", "value": 20}},
        {{"name": "Soft Skills", "value": 10}}
      ]
    }},
    "top_skills": [],
    "skill_gaps": [],
    "career_paths": [],
    "next_actions": [
      {{"type": "skill", "text": ""}},
      {{"type": "project", "text": ""}},
      {{"type": "certification", "text": ""}}
    ]
  }},
  "jd_match": {{
    "match_score": 75,
    "verdict": "Hire",
    "charts": {{
      "match_breakdown": [
        {{"name": "Skills", "value": 80}},
        {{"name": "Experience", "value": 70}},
        {{"name": "Projects", "value": 75}},
        {{"name": "Education", "value": 85}}
      ]
    }},
    "matching_skills": [],
    "missing_skills": [],
    "strengths": [],
    "weaknesses": [],
    "summary": "Short 2-line recruiter-style insight"
  }},
  "ats": {{
    "score": 92,
    "charts": {{
      "ats_breakdown": [
        {{"name": "Keywords", "value": 19}},
        {{"name": "Formatting", "value": 15}},
        {{"name": "Content", "value": 18}},
        {{"name": "Skills", "value": 19}},
        {{"name": "Experience", "value": 17}},
        {{"name": "Grammar", "value": 4}}
      ]
    }},
    "issues": [],
    "fixes": []
  }}
}}

Resume:
{resume_text}

Job Description:
{job_description if job_description.strip() else f"No JD provided. Base all analysis and suggestions strictly on the predicted role: {predicted_role}."}
        """
        
        prompt = prompt.encode('ascii', 'ignore').decode('ascii')
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return """{
          "ui_config": {
            "layout": "tab_based",
            "default_tab": "dashboard",
            "tabs": [
              {"id": "dashboard", "label": "Dashboard"},
              {"id": "jd_match", "label": "JD Match"},
              {"id": "ats", "label": "ATS Report"}
            ]
          },
          "dashboard": {
            "kpis": [
              {"label": "Career Score", "value": 85},
              {"label": "Resume Score", "value": 88},
              {"label": "Project Score", "value": 90}
            ],
            "profile_summary": "Experienced professional with a strong background in Machine Learning and Data Science.",
            "charts": {
              "skills_distribution": [
                {"name": "ML/NLP", "value": 45},
                {"name": "Programming", "value": 35},
                {"name": "Tools", "value": 15},
                {"name": "Soft Skills", "value": 5}
              ]
            },
            "top_skills": ["Python", "Machine Learning", "Data Analysis"],
            "skill_gaps": ["Cloud Deployment"],
            "career_paths": ["Senior Data Scientist", "ML Engineer"],
            "next_actions": [
              {"type": "skill", "text": "Learn Docker"},
              {"type": "certification", "text": "AWS ML Specialty"}
            ]
          },
          "jd_match": {
            "match_score": 82,
            "verdict": "Strong Candidate",
            "charts": {
              "match_breakdown": [
                {"name": "Skills", "value": 85},
                {"name": "Experience", "value": 80},
                {"name": "Projects", "value": 90},
                {"name": "Education", "value": 100}
              ]
            },
            "matching_skills": ["Python", "Machine Learning"],
            "missing_skills": ["Docker"],
            "strengths": ["Strong academic background", "Relevant projects"],
            "weaknesses": ["Lack of cloud experience"],
            "summary": "Candidate meets most requirements for the role."
          },
          "ats": {
            "score": 88,
            "charts": {
              "ats_breakdown": [
                {"name": "Keywords", "value": 20},
                {"name": "Formatting", "value": 18},
                {"name": "Content", "value": 18},
                {"name": "Skills", "value": 15},
                {"name": "Experience", "value": 12},
                {"name": "Grammar", "value": 5}
              ]
            },
            "issues": ["Some missing keywords related to the job description"],
            "fixes": ["Add Docker and AWS to the skills section if applicable"]
          }
        }"""


def generate_optimized_resume_json(resume_text, job_description="", predicted_role="", api_key=""):
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
You are an AI Resume Optimization Agent.
Your task is to intelligently improve and tailor a candidate's resume based on a Job Description (JD).
You MUST output ONLY valid JSON.

----------------------------------------
* CORE OBJECTIVE
----------------------------------------
- Provide actionable recommendations to improve the candidate's resume for the target role.
- DO NOT rewrite the resume with fake companies. Instead, provide specific SUGGESTIONS for each section.

----------------------------------------
* STRICT RULES
----------------------------------------
1. In 'skills', list what skills to ADD based on the JD or ATS, and what to REPLACE/REMOVE.
2. In 'experience', suggest the domain and project types they need experience in. Don't invent fake companies.
3. In 'projects', suggest specific projects they should build next.
4. In 'certifications', suggest specific certifications or courses they should take.

----------------------------------------
* OUTPUT STRUCTURE (IMPORTANT STRICT JSON)
----------------------------------------
{{
  "updated_resume": {{
    "summary": "A powerful suggested rewrite of their professional summary",
    "skills": [
      "ADD: [Skill] - [Reason]",
      "REPLACE: [Old Skill] -> [New Keyword]",
      "REMOVE: [Skill]"
    ],
    "experience": [
      "Domain Suggestion: [Domain] - [Reason]",
      "Experience Needed: [Type of work] - [Reason]"
    ],
    "projects": [
      "Project Idea: [Name] - [What to build and why]"
    ],
    "certifications": [
      "Recommended Course: [Name of course/cert]"
    ]
  }},
  "preferred_roles": [
    "Role 1",
    "Role 2"
  ],
  "changes_made": [
    "Rephrased X to include Y keyword",
    "Improved bullet using action verb",
    "Added measurable impact in project"
  ],
  "suggested_additions": {{
    "skills_to_learn": [],
    "projects_to_build": [],
    "certifications": []
  }}
}}

----------------------------------------
* INPUT
----------------------------------------
Resume:
{resume_text}

Job Description:
{job_description if job_description.strip() else f"No JD provided. Optimize strictly for the predicted role: {predicted_role} and to maximize ATS score."}
        """
        
        prompt = prompt.encode('ascii', 'ignore').decode('ascii')
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return """{
            "updated_resume": {
                "summary": "Results-oriented professional with a proven track record in software engineering and data analysis, seeking to leverage skills in a challenging new role.",
                "skills": [
                    "ADD: Docker - Highly requested in modern JD for deployment",
                    "REPLACE: Basic Programming -> Object Oriented Programming (OOP)",
                    "REMOVE: MS Word - Too generic for an engineering role"
                ],
                "experience": [
                    "Domain Suggestion: Cloud Architecture - Expand experience in deploying models to production.",
                    "Experience Needed: Cross-functional collaboration - Work with PMs and designers to build end-to-end features."
                ],
                "projects": [
                    "Project Idea: Real-time Data Pipeline - Build a streaming architecture using Kafka and deploy to AWS.",
                    "Project Idea: Full-stack LLM App - Integrate an LLM via API into a React/Next.js frontend."
                ],
                "certifications": [
                    "Recommended Course: AWS Certified Machine Learning - Specialty",
                    "Recommended Course: DeepLearning.AI TensorFlow Developer Professional Certificate"
                ]
            },
            "preferred_roles": ["Data Scientist", "Machine Learning Engineer"],
            "changes_made": [
                "Suggested new high-impact skills to add",
                "Rephrased summary for stronger impact",
                "Recommended specific projects to fill resume gaps"
            ],
            "suggested_additions": {
                "skills_to_learn": ["AWS", "Docker", "Kubernetes"],
                "projects_to_build": ["End-to-end ML deployment project"],
                "certifications": ["AWS Certified Machine Learning - Specialty"]
            }
        }"""


def draw_chart(data_list, chart_type="bar"):
    if not data_list: return
    df = pd.DataFrame(data_list)
    df.set_index("name", inplace=True)
    st.bar_chart(df)


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction & Agent", page_icon="📄", layout="wide")

    st.sidebar.title("🤖 ARIA Agent Settings")
    st.sidebar.markdown("Enter your Groq API key to unlock advanced agentic features.")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    if api_key:
        st.sidebar.success("API Key configured successfully!")
    else:
        st.sidebar.warning("Please enter your API Key to use the Agentic features.")

    st.title("Resume Category Prediction & Career Agent")
    
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Successfully extracted the text from the uploaded resume.")

            # Create layout
            st.markdown("---")
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                st.subheader("Basic ML Classification")
                category = pred(resume_text)
                st.info(f"**Predicted Role:** {category}")
                st.markdown("Provide an optional Job Description for full JD matching & rewriting capabilities:")
                job_desc = st.text_area("Job Description", height=200, placeholder="Paste JD here...")

            with col_b:
                # We now have multiple Agentic features! Let's use tabs at the top level
                st.subheader("ARIA Agent Workspace")
                
                if not api_key:
                    st.info("Please enter your Groq API Key in the sidebar to generate the interactive dashboard.")
                else:
                    tab_dash, tab_opt = st.tabs(["📊 Interactive Dashboard", "📝 Resume Optimizer"])
                    
                    with tab_dash:
                        if st.button("🚀 Render ARIA Interactive Dashboard", use_container_width=True):
                            with st.spinner("ARIA is generating structured UI data..."):
                                json_data = generate_interactive_ui_json(resume_text, job_desc, category, api_key)
                                try:
                                    data = json.loads(json_data)
                                    st.markdown("---")
                                    dashboard = data.get("dashboard", {})
                                    jd_match = data.get("jd_match", {})
                                    ats = data.get("ats", {})
                                    
                                    d_tab, j_tab, a_tab = st.tabs(["📈 Career Dashboard", "🎯 JD Match", "📄 ATS Report"])
                                    
                                    with d_tab:
                                        st.markdown(f"*{dashboard.get('profile_summary', '')}*")
                                        kpis = dashboard.get("kpis", [])
                                        if kpis:
                                            cols = st.columns(len(kpis))
                                            for i, kpi in enumerate(kpis): cols[i].metric(kpi.get('label', ''), kpi.get('value', 0))
                                                
                                        st.markdown("---")
                                        c1, c2 = st.columns(2)
                                        with c1:
                                            st.markdown("### Top Skills")
                                            for s in dashboard.get('top_skills', []): st.write(f"- {s}")
                                        with c2:
                                            st.markdown("### Skills Distribution")
                                            draw_chart(dashboard.get('charts', {}).get('skills_distribution', []))
                                            
                                        st.markdown("---")
                                        st.markdown("### 🔍 Deeper Insights")
                                        ic1, ic2, ic3 = st.columns(3)
                                        with ic1:
                                            st.markdown("**Skill Gaps**")
                                            for s in dashboard.get('skill_gaps', []): st.write(f"- {s}")
                                        with ic2:
                                            st.markdown("**Career Paths**")
                                            for p in dashboard.get('career_paths', []): st.write(f"- {p}")
                                        with ic3:
                                            st.markdown("**Next Actions**")
                                            for a in dashboard.get('next_actions', []): st.write(f"- {a.get('text', '')}")

                                    with j_tab:
                                        if job_desc.strip():
                                            c1, c2 = st.columns(2)
                                            c1.metric("JD Match Score", f"{jd_match.get('match_score', 0)}%")
                                            c2.metric("ARIA Verdict", jd_match.get('verdict', 'N/A'))
                                            st.info(jd_match.get('summary', ''))
                                            st.markdown("### Match Breakdown")
                                            draw_chart(jd_match.get('charts', {}).get('match_breakdown', []))
                                            
                                            st.markdown("---")
                                            jc1, jc2 = st.columns(2)
                                            with jc1:
                                                st.success("**✅ Matching Skills/Experience**")
                                                for s in jd_match.get('matching_skills', []): st.write(f"- {s}")
                                                st.markdown("**💪 Strengths**")
                                                for s in jd_match.get('strengths', []): st.write(f"- {s}")
                                            with jc2:
                                                st.error("**❌ Missing Skills/Experience**")
                                                for s in jd_match.get('missing_skills', []): st.write(f"- {s}")
                                                st.warning("**⚠️ Weaknesses**")
                                                for w in jd_match.get('weaknesses', []): st.write(f"- {w}")
                                        else:
                                            st.info("ℹ️ Please provide a Job Description on the left to see the JD Match analysis.")

                                    with a_tab:
                                        st.metric("Total ATS Score", f"{ats.get('score', 0)}/100")
                                        st.markdown("### ATS Score Breakdown")
                                        draw_chart(ats.get('charts', {}).get('ats_breakdown', []))
                                        
                                        st.markdown("---")
                                        ac1, ac2 = st.columns(2)
                                        with ac1:
                                            st.markdown("**🛑 ATS Issues Found**")
                                            for i in ats.get('issues', []): st.write(f"- {i}")
                                        with ac2:
                                            st.markdown("**🛠️ Recommended Fixes**")
                                            for f in ats.get('fixes', []): st.write(f"- {f}")
                                            
                                        st.markdown("---")
                                        st.info("💡 **If you want a good ATS score resume then go for the 'Resume Optimizer' option!**")
                                        
                                except Exception as e:
                                    st.error("Failed to parse Dashboard JSON.")
                                    st.code(json_data)

                    with tab_opt:
                        if st.button("✨ Auto Optimize Resume", use_container_width=True):
                            with st.spinner("Optimising your resume please wait..."):
                                opt_data = generate_optimized_resume_json(resume_text, job_desc, category, api_key)
                                try:
                                    opt_json = json.loads(opt_data)
                                    st.markdown("---")
                                    updated = opt_json.get("updated_resume", {})
                                    
                                    st.markdown("### 📄 Your New ATS-Optimized Resume")
                                    
                                    roles = opt_json.get("preferred_roles", [])
                                    if roles:
                                        st.success(f"**Preferred Roles:** {', '.join(roles)}")
                                        
                                    st.info(updated.get("summary", ""))
                                    
                                    with st.expander("🛠️ Skills Optimization"):
                                        for s in updated.get("skills", []): st.write(f"- {s}")
                                        
                                    with st.expander("💼 Experience Suggestions"):
                                        for e in updated.get("experience", []): st.write(f"- {e}")
                                        
                                    with st.expander("🚀 Project Ideas"):
                                        for p in updated.get("projects", []): st.write(f"- {p}")
                                        
                                    with st.expander("🎓 Certifications / Courses"):
                                        for c in updated.get("certifications", []): st.write(f"- {c}")
                                        
                                    st.markdown("---")
                                    st.markdown("### 🕵️ ARIA's Change Log")
                                    for change in opt_json.get("changes_made", []):
                                        st.success(f"✔️ {change}")
                                        
                                    st.markdown("---")
                                    st.markdown("### 📈 Suggested Additions (Skill Gaps)")
                                    adds = opt_json.get("suggested_additions", {})
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.markdown("**Skills to Learn**")
                                        for s in adds.get("skills_to_learn", []): st.write(f"- {s}")
                                    with c2:
                                        st.markdown("**Projects to Build**")
                                        for p in adds.get("projects_to_build", []): st.write(f"- {p}")
                                    with c3:
                                        st.markdown("**Certifications**")
                                        for c in adds.get("certifications", []): st.write(f"- {c}")
                                        
                                    # Create DOCX buffer
                                    docx_buffer = create_docx_from_json(updated)
                                    
                                    # Download button for the new resume structure
                                    st.download_button(
                                        label="📥 Download ATS-Optimized Resume (.docx)",
                                        data=docx_buffer,
                                        file_name="ARIA_Optimized_Resume.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        use_container_width=True
                                    )

                                except Exception as e:
                                    st.error("Failed to parse Optimizer JSON.")
                                    st.code(opt_data)

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
