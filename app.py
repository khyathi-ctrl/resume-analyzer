import streamlit as st
from pypdf import PdfReader
import docx
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills import SKILLS

nlp = spacy.load("en_core_web_sm")


# -------------------------
# Extract Text from PDF
# -------------------------
def extract_pdf_text(pdf_file):
    text = ""

    pdf_reader = PdfReader(pdf_file)

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text.lower()


# -------------------------
# Extract Text from DOCX
# -------------------------
def extract_docx_text(docx_file):
    doc = docx.Document(docx_file)

    text = ""

    for para in doc.paragraphs:
        text += para.text

    return text.lower()


# -------------------------
# Extract Skills
# -------------------------
def extract_skills(text):

    found_skills = []

    for skill in SKILLS:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills


# -------------------------
# ATS Score
# -------------------------
def calculate_score(resume_text, job_description):

    documents = [resume_text, job_description]

    cv = CountVectorizer()

    matrix = cv.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    return round(similarity * 100, 2)


# -------------------------
# Streamlit UI
# -------------------------
st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

job_description = st.text_area("Paste Job Description")


if uploaded_file and job_description:

    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        resume_text = extract_pdf_text(uploaded_file)

    else:
        resume_text = extract_docx_text(uploaded_file)

    resume_skills = extract_skills(resume_text)

    jd_skills = extract_skills(job_description.lower())

    matched_skills = list(set(resume_skills) & set(jd_skills))

    missing_skills = list(set(jd_skills) - set(resume_skills))

    score = calculate_score(resume_text, job_description)

    st.subheader("ATS Match Score")
    st.write(f"{score}%")

    st.subheader("Matched Skills")
    st.write(matched_skills)

    st.subheader("Missing Skills")
    st.write(missing_skills)

    st.subheader("Resume Skills")
    st.write(resume_skills)