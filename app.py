import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile

from fpdf import FPDF

from resume_utils import preprocess, extract_resume_text
from analysis import compute_similarity, extract_top_keywords, find_keyword_matches


# ---------------------- Streamlit Page Config ---------------------- #
st.set_page_config(page_title="CV Analyzer", layout="wide")
st.title("📄 AI-Based Resume Analyzer")
st.write("Compare multiple resumes against a job description using NLP and similarity scoring.")


# ---------------------- Sidebar Settings ---------------------- #
st.sidebar.header("⚙️ Settings")

top_n_keywords = st.sidebar.slider(
    "Number of JD Keywords to Extract",
    min_value=5,
    max_value=40,
    value=20,
    step=1,
    help="Top important terms from the Job Description (using TF-IDF)."
)


# ---------------------- File Upload Section ---------------------- #
st.subheader("📂 Upload Files")

st.info("Upload a Job Description (PDF) and multiple resumes (PDF or DOCX).")

jd_file = st.file_uploader("📌 Upload Job Description (PDF)", type=["pdf"])

resume_files = st.file_uploader(
    "📥 Upload Resumes (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)


# ---------------------- Helper: Generate PDF Report ---------------------- #
def generate_pdf_report(resume_name, similarity, matched_keywords, missing_keywords):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(0, 10, "Resume Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # Basic info
    pdf.cell(0, 10, f"Resume: {resume_name}", ln=True)
    pdf.cell(0, 10, f"Similarity Score: {similarity:.2f}%", ln=True)
    pdf.ln(5)

    # Matched keywords
    matched_text = ", ".join(matched_keywords) if matched_keywords else "None"
    pdf.multi_cell(0, 8, f"Matched Keywords:\n{matched_text}")
    pdf.ln(3)

    # Missing keywords
    missing_text = ", ".join(missing_keywords) if missing_keywords else "None"
    pdf.multi_cell(0, 8, f"Missing Keywords:\n{missing_text}")

    # Return as BytesIO
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    buffer = BytesIO()
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer


# ---------------------- Helper: Generate Excel File ---------------------- #
def generate_excel_file(dataframe: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Results")
    output.seek(0)
    return output


# ---------------------- Main Logic ---------------------- #
if jd_file and resume_files:
    with st.spinner("🔍 Analyzing resumes..."):

        # 1. Extract and preprocess JD text
        jd_text = extract_resume_text(jd_file)
        if not jd_text.strip():
            st.error("Could not extract text from the Job Description. Please upload a valid PDF.")
            st.stop()

        jd_clean = preprocess(jd_text)

        # 2. Auto-extract top keywords from JD (TF-IDF)
        jd_keywords = extract_top_keywords(jd_clean, n=top_n_keywords)

        results = []
        pdf_buffers = {}

        # 3. Process each resume
        for resume in resume_files:
            resume_name = resume.name
            resume_text = extract_resume_text(resume)

            if not resume_text.strip():
                st.warning(f"Could not extract text from {resume_name}. Skipping this file.")
                continue

            resume_clean = preprocess(resume_text)

            # Similarity score
            similarity = compute_similarity(jd_clean, resume_clean)

            # Keyword matches / missing
            matched, missing = find_keyword_matches(jd_clean, resume_clean, jd_keywords)

            # Store result
            results.append({
                "Resume Name": resume_name,
                "Similarity (%)": round(similarity, 2),
                "Matched Keywords": ", ".join(matched) if matched else "None",
                "Missing Keywords": ", ".join(missing) if missing else "None"
            })

            # Individual PDF report
            pdf_buffers[resume_name] = generate_pdf_report(
                resume_name, similarity, matched, missing
            )

    if not results:
        st.error("No valid resumes were processed.")
        st.stop()

    # 4. Put results into DataFrame and sort by similarity
    df_results = pd.DataFrame(results).sort_values(
        "Similarity (%)", ascending=False
    )

    st.subheader("📊 Analysis Results")
    st.dataframe(df_results, use_container_width=True)

    # 5. Bar Chart
    st.subheader("📈 Similarity Chart")
    fig, ax = plt.subplots()
    ax.bar(df_results["Resume Name"], df_results["Similarity (%)"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Similarity (%)")
    plt.title("Resume vs JD Similarity")
    st.pyplot(fig)

    # 6. Excel Download
    st.subheader("⬇️ Download Results as Excel")
    excel_file = generate_excel_file(df_results)
    st.download_button(
        label="Download Excel File",
        data=excel_file,
        file_name="resume_analysis_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # 7. PDF Reports ZIP Download
    st.subheader("⬇️ Download Individual PDF Reports")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for name, buffer in pdf_buffers.items():
            filename = f"{os.path.splitext(name)[0]}_report.pdf"
            zipf.writestr(filename, buffer.getvalue())
    zip_buffer.seek(0)

    st.download_button(
        label="Download All Reports (ZIP)",
        data=zip_buffer,
        file_name="resume_reports.zip",
        mime="application/zip"
    )

elif jd_file or resume_files:
    st.info("Please upload **both** a Job Description file and at least one resume.")
else:
    st.info("Start by uploading a Job Description and some resumes.")


