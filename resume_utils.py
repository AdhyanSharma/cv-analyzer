"""
AI Resume Screening Platform
Resume Utility Engine

Handles:
    - PDF extraction
    - PDF hyperlink extraction
    - DOCX extraction
    - DOCX hyperlink extraction
    - Text cleaning
    - NLP preprocessing
    - Resume section detection
    - Job Description section detection
"""

import os
import re
import zipfile
from io import BytesIO
from typing import Dict, List
import xml.etree.ElementTree as ET

import nltk
import PyPDF2
import docx2txt
from nltk.stem import WordNetLemmatizer


def _ensure_nltk() -> None:
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as error:
                print(f"NLTK warning: {error}")


_ensure_nltk()

try:
    lemmatizer = WordNetLemmatizer()
except Exception:
    lemmatizer = None


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\xa0", " ")

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        text,
    )

    text = text.replace("https ://", "https://")
    text = text.replace("http ://", "http://")

    # Repair protocols with spaces introduced by extraction.
    text = re.sub(
        r"https?\s*:\s*/\s*/",
        lambda match: (
            "https://"
            if match.group(0).lower().startswith("https")
            else "http://"
        ),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preprocess(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower()

    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9+#./\-\s@]", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"\s*#\s*", "#", text)

    processed_words = []

    for word in text.split():
        if any(symbol in word for symbol in ("+", "#", "/", ".", "-", "@")):
            processed_words.append(word)
        elif lemmatizer:
            try:
                processed_words.append(lemmatizer.lemmatize(word))
            except Exception:
                processed_words.append(word)
        else:
            processed_words.append(word)

    return re.sub(r"\s+", " ", " ".join(processed_words)).strip()


def extract_pdf_hyperlinks(uploaded_file) -> List[str]:
    """
    Extract hyperlink targets from PDF annotations.

    This helps when a PDF visually shows only:
        GitHub
        LinkedIn
        Email

    while the actual URL is stored in the PDF hyperlink.
    """
    if uploaded_file is None:
        return []

    links = []

    try:
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)

        for page in reader.pages:
            try:
                annotations = page.get("/Annots")
                if not annotations:
                    continue

                for annotation_ref in annotations:
                    try:
                        annotation = annotation_ref.get_object()
                    except Exception:
                        continue

                    action = annotation.get("/A")
                    if not action:
                        continue

                    try:
                        action = action.get_object()
                    except Exception:
                        pass

                    uri = action.get("/URI")
                    if uri:
                        links.append(str(uri).strip())

            except Exception:
                continue

    except Exception as error:
        print(f"PDF hyperlink extraction warning: {error}")

    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    return list(dict.fromkeys(links))


def extract_pdf_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    text_parts = []

    try:
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as error:
                print(
                    f"PDF page {page_number} warning: {error}"
                )

        visible_text = "\n".join(text_parts)

        # Recover hidden hyperlink targets.
        hyperlinks = extract_pdf_hyperlinks(uploaded_file)

        extra_links = []
        for link in hyperlinks:
            link = link.strip()
            if not link:
                continue

            if link.lower().startswith("mailto:"):
                email = link[7:].strip()
                if email:
                    extra_links.append(email)
            else:
                extra_links.append(link)

        if extra_links:
            visible_text += "\n" + "\n".join(extra_links)

        return clean_extracted_text(visible_text)

    except Exception as error:
        print(f"PDF extraction error: {error}")
        return ""


def extract_docx_hyperlinks(uploaded_file) -> List[str]:
    """
    Extract actual hyperlink targets from a DOCX package.

    Handles:
        https://github.com/user
        https://linkedin.com/in/user
        mailto:user@example.com
    """
    if uploaded_file is None:
        return []

    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        if not file_bytes:
            return []

        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            document_xml = archive.read("word/document.xml")
            relationship_xml = archive.read(
                "word/_rels/document.xml.rels"
            )

        document_root = ET.fromstring(document_xml)
        relationship_root = ET.fromstring(relationship_xml)

        relationship_targets = {}

        for relationship in relationship_root:
            relationship_id = relationship.attrib.get("Id")
            target = relationship.attrib.get("Target")
            target_mode = relationship.attrib.get("TargetMode", "")

            if not relationship_id or not target:
                continue

            target = target.strip()

            if (
                target_mode.lower() == "external"
                or target.lower().startswith(
                    ("http://", "https://", "mailto:")
                )
            ):
                relationship_targets[relationship_id] = target

        hyperlink_tag = (
            "{http://schemas.openxmlformats.org/"
            "wordprocessingml/2006/main}hyperlink"
        )

        relationship_attribute = (
            "{http://schemas.openxmlformats.org/"
            "officeDocument/2006/relationships}id"
        )

        hyperlinks = []

        for hyperlink in document_root.iter(hyperlink_tag):
            relationship_id = hyperlink.attrib.get(
                relationship_attribute
            )

            if not relationship_id:
                continue

            target = relationship_targets.get(relationship_id)

            if target:
                hyperlinks.append(target)

        # Fallback for some DOCX generators.
        if not hyperlinks:
            for target in relationship_targets.values():
                if target.lower().startswith(
                    ("http://", "https://", "mailto:")
                ):
                    hyperlinks.append(target)

        return list(dict.fromkeys(hyperlinks))

    except Exception as error:
        print(f"DOCX hyperlink extraction warning: {error}")
        return []

    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass


def extract_docx_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    try:
        uploaded_file.seek(0)
        visible_text = docx2txt.process(uploaded_file)

        hyperlinks = extract_docx_hyperlinks(uploaded_file)

        extra_links = []
        for link in hyperlinks:
            link = link.strip()
            if not link:
                continue

            if link.lower().startswith("mailto:"):
                email = link[7:].strip()
                if email:
                    extra_links.append(email)
            else:
                extra_links.append(link)

        if extra_links:
            visible_text += "\n" + "\n".join(extra_links)

        return clean_extracted_text(visible_text)

    except Exception as error:
        print(f"DOCX extraction error: {error}")
        return ""


def extract_resume_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    filename = getattr(uploaded_file, "name", "")
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    if lower_name.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    return ""


# ============================================================
# RESUME / JD SECTIONS
# ============================================================

SECTION_ALIASES = {
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career summary",
        "career objective",
        "objective",
        "about me",
    ],
    "skills": [
        "skills",
        "technical skills",
        "technical skill",
        "core skills",
        "key skills",
        "professional skills",
        "technologies",
        "technical expertise",
        "skills and technologies",
        "skills & technologies",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "work history",
        "professional history",
        "internship",
        "internships",
        "internship experience",
    ],
    "education": [
        "education",
        "academic background",
        "academic qualification",
        "academic qualifications",
        "qualifications",
        "educational background",
        "educational qualification",
    ],
    "projects": [
        "projects",
        "academic projects",
        "personal projects",
        "key projects",
        "project experience",
        "project",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "certification",
        "licenses",
        "licenses and certifications",
    ],
    "achievements": [
        "achievements",
        "awards",
        "honors",
        "honours",
        "accomplishments",
    ],
    "responsibilities": [
        "responsibilities",
        "job responsibilities",
        "key responsibilities",
    ],
    "publications": [
        "publications",
        "research publications",
        "papers",
        "research papers",
    ],
    "interests": [
        "interests",
        "hobbies",
        "hobbies and interests",
    ],
}


def normalize_heading(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower().strip()
    text = text.replace("&", "and")
    text = re.sub(r"[:|•●▪◦\-_=]+", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_section_heading(line: str) -> str:
    normalized = normalize_heading(line)

    if not normalized:
        return ""

    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if normalized == normalize_heading(alias):
                return section

    return ""


def extract_resume_sections(text: str) -> Dict[str, str]:
    if not text:
        return {}

    sections = {}
    current_section = None
    buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        detected = detect_section_heading(line)

        if detected:
            if current_section and buffer:
                content = "\n".join(buffer).strip()
                if content:
                    if current_section in sections:
                        sections[current_section] += "\n" + content
                    else:
                        sections[current_section] = content

            current_section = detected
            buffer = []
            continue

        if current_section:
            buffer.append(line)

    if current_section and buffer:
        content = "\n".join(buffer).strip()
        if content:
            if current_section in sections:
                sections[current_section] += "\n" + content
            else:
                sections[current_section] = content

    return sections


def extract_jd_sections(text: str) -> Dict[str, str]:
    if not text:
        return {}

    sections = extract_resume_sections(text)

    if sections:
        return sections

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return {}

    skill_lines = []
    responsibility_lines = []
    requirement_lines = []
    preferred_lines = []
    general_lines = []

    skill_terms = [
        "skills", "python", "java", "sql",
        "machine learning", "deep learning", "tensorflow",
        "pytorch", "aws", "docker", "kubernetes",
        "react", "fastapi", "flask", "scikit", "nlp",
    ]

    responsibility_terms = [
        "responsible", "responsibilities", "develop",
        "design", "build", "implement", "maintain",
        "analyze", "collaborate", "create", "developing",
        "deployment", "deploy",
    ]

    requirement_terms = [
        "requirement", "required", "qualification",
        "experience", "degree", "years", "bachelor",
        "master", "minimum",
    ]

    preferred_terms = [
        "preferred", "nice to have", "nice-to-have",
        "desired", "plus", "bonus",
    ]

    for line in lines:
        lower_line = line.lower()

        if any(term in lower_line for term in preferred_terms):
            preferred_lines.append(line)
        elif any(term in lower_line for term in requirement_terms):
            requirement_lines.append(line)
        elif any(term in lower_line for term in skill_terms):
            skill_lines.append(line)
        elif any(
            term in lower_line
            for term in responsibility_terms
        ):
            responsibility_lines.append(line)
        else:
            general_lines.append(line)

    result = {}

    if skill_lines:
        result["skills"] = "\n".join(skill_lines)

    if responsibility_lines:
        result["responsibilities"] = "\n".join(
            responsibility_lines
        )

    if requirement_lines:
        result["requirements"] = "\n".join(requirement_lines)

    if preferred_lines:
        result["preferred"] = "\n".join(preferred_lines)

    if general_lines:
        result["general"] = "\n".join(general_lines)

    if not result:
        result["general"] = text

    return result


def get_document_info(uploaded_file) -> Dict[str, str]:
    if uploaded_file is None:
        return {}

    name = getattr(uploaded_file, "name", "")
    extension = os.path.splitext(name)[1] if name else ""

    if extension.lower() == ".pdf":
        file_type = "PDF"
    elif extension.lower() == ".docx":
        file_type = "DOCX"
    else:
        file_type = "Unknown"

    return {
        "filename": name,
        "file_type": file_type,
        "extension": extension,
    }


def is_extractable_text(
    text: str,
    minimum_characters: int = 20,
) -> bool:
    if not text:
        return False

    cleaned = re.sub(r"\s+", "", text)
    return len(cleaned) >= minimum_characters


if __name__ == "__main__":
    print("resume_utils.py loaded successfully.")
