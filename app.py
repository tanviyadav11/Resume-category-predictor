# Resume Category Predictor + ARIA Career Agent (Streamlit)
# Run locally:  pip install -r requirements.txt && streamlit run app.py
import io
import json
import os
import pickle
import re

import docx
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

try:
    from groq import Groq
except ImportError:  # app still works (ML + local JD match) without Groq
    Groq = None

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # llama3-8b-8192 is retired
MAX_CHARS = 6000

st.set_page_config(page_title="Resume Category Predictor", page_icon="📄", layout="wide")


# ---------------------------------------------------------------- model loading
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "clf.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(base, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(base, "encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return clf, tfidf, le


try:
    clf, tfidf, le = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load clf.pkl / tfidf.pkl / encoder.pkl: {e}\n\n"
        "Make sure the three files are in the same folder as app.py and that "
        "scikit-learn in requirements.txt matches the version used to train them."
    )
    st.stop()


# ---------------------------------------------------------------- text helpers
def clean_resume(txt: str) -> str:
    t = re.sub(r"http\S+\s*", " ", txt)
    t = re.sub(r"\bRT\b|\bcc\b", " ", t)
    t = re.sub(r"#\S+\s*", " ", t)
    t = re.sub(r"@\S+", " ", t)
    t = re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), " ", t)
    t = re.sub(r"[^\x00-\x7f]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def read_docx(file) -> str:
    return "\n".join(p.text for p in docx.Document(file).paragraphs)


def read_txt(file) -> str:
    raw = file.read()  # read bytes once, then try encodings
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def extract_text(uploaded) -> str:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    readers = {"pdf": read_pdf, "docx": read_docx, "txt": read_txt}
    if ext not in readers:
        raise ValueError("Unsupported file type. Upload PDF, DOCX or TXT.")
    return readers[ext](uploaded)


# ---------------------------------------------------------------- ML prediction
def predict_category(text: str):
    X = tfidf.transform([clean_resume(text)])
    try:
        pred = clf.predict(X)
    except Exception:  # some models were trained on dense arrays
        pred = clf.predict(X.toarray())
    label = le.inverse_transform(pred)[0]
    confidence = None
    try:
        proba = clf.predict_proba(X if hasattr(X, "shape") else X.toarray())
        confidence = float(proba.max()) * 100
    except Exception:
        pass
    return label, confidence


# ---------------------------------------------------------------- local JD match (no API needed)
def local_jd_match(resume: str, jd: str):
    r, j = clean_resume(resume), clean_resume(jd)
    vecs = tfidf.transform([r, j])
    score = float(cosine_similarity(vecs[0], vecs[1])[0][0]) * 100

    stop = set(tfidf.get_stop_words() or [])
    jd_terms = {w for w in re.findall(r"[a-zA-Z+#.]{3,}", j.lower()) if w not in stop}
    res_terms = set(re.findall(r"[a-zA-Z+#.]{3,}", r.lower()))
    # rank JD terms by tf-idf weight so "missing skills" are meaningful
    vocab = tfidf.vocabulary_
    weights = vecs[1].toarray()[0]
    ranked = sorted(
        (w for w in jd_terms if w in vocab), key=lambda w: weights[vocab[w]], reverse=True
    )
    matched = [w for w in ranked if w in res_terms][:15]
    missing = [w for w in ranked if w not in res_terms][:15]
    return round(score, 1), matched, missing


# ---------------------------------------------------------------- Groq agent
def call_llm(prompt: str, api_key: str) -> dict:
    if Groq is None:
        raise RuntimeError("groq package not installed. Add `groq` to requirements.txt.")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You output ONLY valid JSON. No prose, no markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def dashboard_prompt(resume, jd, role):
    target = jd.strip() or f"No JD provided. Base analysis on the predicted role: {role}."
    return f"""Analyse the resume and return JSON with exactly this shape:
{{
 "dashboard": {{"kpis":[{{"label":"Career Score","value":0}},{{"label":"Resume Score","value":0}},{{"label":"Project Score","value":0}}],
   "profile_summary":"", "charts":{{"skills_distribution":[{{"name":"","value":0}}]}},
   "top_skills":[], "skill_gaps":[], "career_paths":[], "next_actions":[{{"type":"skill","text":""}}]}},
 "jd_match": {{"match_score":0,"verdict":"","charts":{{"match_breakdown":[{{"name":"Skills","value":0}},{{"name":"Experience","value":0}},{{"name":"Projects","value":0}},{{"name":"Education","value":0}}]}},
   "matching_skills":[], "missing_skills":[], "strengths":[], "weaknesses":[], "summary":""}},
 "ats": {{"score":0,"charts":{{"ats_breakdown":[{{"name":"Keywords","value":0}},{{"name":"Formatting","value":0}},{{"name":"Content","value":0}},{{"name":"Skills","value":0}},{{"name":"Experience","value":0}},{{"name":"Grammar","value":0}}]}},
   "issues":[], "fixes":[]}}
}}
Fill every field from the actual resume (scores 0-100; ats_breakdown values sum to the ATS score out of 100). Be honest, do not inflate.

Resume:
{resume[:MAX_CHARS]}

Target / Job Description:
{target[:3000]}"""


def optimizer_prompt(resume, jd, role):
    target = jd.strip() or f"No JD provided. Optimise for the predicted role: {role} and for ATS."
    return f"""Suggest improvements to this resume for the target role. Do NOT invent companies or fake experience.
Return JSON with exactly this shape:
{{"updated_resume":{{"summary":"","skills":["ADD: skill - reason","REMOVE: skill"],"experience":[],"projects":[],"certifications":[]}},
 "preferred_roles":[], "changes_made":[],
 "suggested_additions":{{"skills_to_learn":[],"projects_to_build":[],"certifications":[]}}}}

Resume:
{resume[:MAX_CHARS]}

Target / Job Description:
{target[:3000]}"""


def build_docx(updated: dict) -> io.BytesIO:
    d = docx.Document()
    d.add_heading("ATS-Optimized Resume Suggestions", 0)
    if updated.get("summary"):
        d.add_heading("Professional Summary", 1)
        d.add_paragraph(updated["summary"])
    for key, title in [("skills", "Skills"), ("experience", "Experience"),
                       ("projects", "Projects"), ("certifications", "Certifications / Courses")]:
        if updated.get(key):
            d.add_heading(title, 1)
            for item in updated[key]:
                d.add_paragraph(str(item), style="List Bullet")
    buf = io.BytesIO()
    d.save(buf)
    buf.seek(0)
    return buf


def chart(data):
    if data:
        df = pd.DataFrame(data)
        if {"name", "value"} <= set(df.columns):
            st.bar_chart(df.set_index("name"))


def bullets(items):
    for i in items or []:
        st.write(f"- {i}")


# ---------------------------------------------------------------- UI
def get_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    key = key or os.getenv("GROQ_API_KEY", "")
    return st.sidebar.text_input("Groq API Key (optional)", value=key, type="password")


def main():
    st.sidebar.title("🤖 ARIA Agent")
    st.sidebar.caption("ML prediction and local JD match work without a key. "
                       "The AI dashboard and optimizer need a free Groq key.")
    api_key = get_api_key()

    st.title("Resume Category Predictor & Career Agent")
    uploaded = st.file_uploader("Upload a resume", type=["pdf", "docx", "txt"])
    if not uploaded:
        st.info("Upload a PDF, DOCX or TXT resume to begin.")
        return

    try:
        resume_text = extract_text(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    if len(resume_text.strip()) < 30:
        st.warning("Very little text was extracted. If this is a scanned PDF, try a text-based one.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Category prediction")
        category, conf = predict_category(resume_text)
        st.success(f"**Predicted category:** {category}")
        if conf is not None:
            st.caption(f"Model confidence: {conf:.1f}%")
        job_desc = st.text_area("Job description (optional)", height=220, placeholder="Paste JD here...")

    with right:
        t_local, t_dash, t_opt = st.tabs(["🎯 JD Match (offline)", "📊 AI Dashboard", "📝 Resume Optimizer"])

        with t_local:
            if job_desc.strip():
                score, matched, missing = local_jd_match(resume_text, job_desc)
                st.metric("JD match (TF-IDF cosine similarity)", f"{score}%")
                c1, c2 = st.columns(2)
                with c1:
                    st.success("**Matching keywords**")
                    bullets(matched)
                with c2:
                    st.error("**Skill gap: JD keywords missing from resume**")
                    bullets(missing)
            else:
                st.info("Paste a job description on the left to see the match and skill gap.")

        with t_dash:
            if not api_key:
                st.info("Add a Groq API key in the sidebar to enable this tab.")
            elif st.button("🚀 Generate dashboard", use_container_width=True):
                try:
                    with st.spinner("Analysing..."):
                        data = call_llm(dashboard_prompt(resume_text, job_desc, category), api_key)
                except Exception as e:
                    st.error(f"AI request failed: {e}")
                else:
                    dash, jd, ats = data.get("dashboard", {}), data.get("jd_match", {}), data.get("ats", {})
                    d_tab, j_tab, a_tab = st.tabs(["Career", "JD Match", "ATS Report"])
                    with d_tab:
                        st.write(dash.get("profile_summary", ""))
                        kpis = dash.get("kpis", [])
                        if kpis:
                            for col, k in zip(st.columns(len(kpis)), kpis):
                                col.metric(k.get("label", ""), k.get("value", 0))
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Top skills**"); bullets(dash.get("top_skills"))
                        with c2:
                            st.markdown("**Skills distribution**")
                            chart(dash.get("charts", {}).get("skills_distribution"))
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**Skill gaps**"); bullets(dash.get("skill_gaps"))
                        with c2:
                            st.markdown("**Career paths**"); bullets(dash.get("career_paths"))
                        with c3:
                            st.markdown("**Next actions**")
                            bullets([a.get("text", "") for a in dash.get("next_actions", [])])
                    with j_tab:
                        c1, c2 = st.columns(2)
                        c1.metric("Match score", f"{jd.get('match_score', 0)}%")
                        c2.metric("Verdict", jd.get("verdict", "N/A"))
                        st.info(jd.get("summary", ""))
                        chart(jd.get("charts", {}).get("match_breakdown"))
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Matching**"); bullets(jd.get("matching_skills"))
                            st.markdown("**Strengths**"); bullets(jd.get("strengths"))
                        with c2:
                            st.markdown("**Missing**"); bullets(jd.get("missing_skills"))
                            st.markdown("**Weaknesses**"); bullets(jd.get("weaknesses"))
                    with a_tab:
                        st.metric("ATS score", f"{ats.get('score', 0)}/100")
                        chart(ats.get("charts", {}).get("ats_breakdown"))
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Issues**"); bullets(ats.get("issues"))
                        with c2:
                            st.markdown("**Fixes**"); bullets(ats.get("fixes"))

        with t_opt:
            if not api_key:
                st.info("Add a Groq API key in the sidebar to enable this tab.")
            elif st.button("✨ Optimize resume", use_container_width=True):
                try:
                    with st.spinner("Optimizing..."):
                        opt = call_llm(optimizer_prompt(resume_text, job_desc, category), api_key)
                except Exception as e:
                    st.error(f"AI request failed: {e}")
                else:
                    up = opt.get("updated_resume", {})
                    if opt.get("preferred_roles"):
                        st.success("**Preferred roles:** " + ", ".join(opt["preferred_roles"]))
                    st.info(up.get("summary", ""))
                    for key, title in [("skills", "🛠️ Skills"), ("experience", "💼 Experience"),
                                       ("projects", "🚀 Projects"), ("certifications", "🎓 Certifications")]:
                        with st.expander(title):
                            bullets(up.get(key))
                    st.markdown("**Change log**"); bullets(opt.get("changes_made"))
                    st.download_button(
                        "📥 Download suggestions (.docx)", build_docx(up),
                        file_name="ARIA_Optimized_Resume.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()