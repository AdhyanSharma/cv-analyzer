"""
AI Resume Screening Platform
Resume Intelligence Engine

Extracts:
    - Candidate name
    - Email
    - Phone
    - LinkedIn
    - GitHub
    - Skills
    - Education
    - Experience
    - Projects
    - Certifications
    - Achievements
    - Experience years
    - Resume sections
    - Resume statistics

This module is intentionally independent and does not import itself.
"""

import os
import re
from typing import Dict, List


# ============================================================
# TEXT
# ============================================================

def normalize_resume_text(text: str) -> str:
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

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CONTACT INFORMATION
# ============================================================

def extract_email(text: str) -> str:
    if not text:
        return ""

    pattern = (
        r"\b"
        r"[A-Za-z0-9._%+-]+"
        r"@"
        r"[A-Za-z0-9.-]+"
        r"\."
        r"[A-Za-z]{2,}"
        r"\b"
    )

    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


def extract_phone(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"\+91[\s.-]?[6-9]\d{4}[\s.-]?\d{5}",
        r"\b[6-9]\d{4}[\s.-]?\d{5}\b",
        r"\+\d{1,3}[\s.-]?\d{7,14}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue

        phone = match.group(0).strip()
        digits = re.sub(r"\D", "", phone)

        if len(digits) >= 10:
            return phone

    return ""


def extract_linkedin(text: str) -> str:
    if not text:
        return ""

    pattern = (
        r"(?:https?://)?"
        r"(?:www\.)?"
        r"linkedin\.com/"
        r"(?:in|pub)/"
        r"[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    url = match.group(0).strip()
    url = url.rstrip(".,;:)]}>\"'")

    if not url.lower().startswith("http"):
        url = "https://" + url

    return url


def extract_github(text: str) -> str:
    if not text:
        return ""

    # --------------------------------------------------------
    # Full URL
    # --------------------------------------------------------
    url_pattern = (
        r"(?:https?://)?"
        r"(?:www\.)?"
        r"github\.com/"
        r"[A-Za-z0-9][A-Za-z0-9._-]*"
    )

    match = re.search(
        url_pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match:
        url = match.group(0).strip()
        url = url.rstrip(".,;:)]}>\"'")

        if not url.lower().startswith("http"):
            url = "https://" + url

        return url

    # --------------------------------------------------------
    # GitHub label + username
    # --------------------------------------------------------
    patterns = [
        r"github\s*[:|>\-]\s*@?([A-Za-z0-9][A-Za-z0-9._-]{1,38})",
        r"github\s+@?([A-Za-z0-9][A-Za-z0-9._-]{1,38})",
    ]

    blocked = {
        "github",
        "profile",
        "account",
        "repository",
        "repositories",
        "link",
        "url",
    }

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        username = match.group(1).strip()

        if username.lower() in blocked:
            continue

        return "https://github.com/" + username

    return ""


# ============================================================
# NAME EXTRACTION
# ============================================================

NAME_REJECT_EXACT = {
    "resume",
    "curriculum vitae",
    "cv",
    "email",
    "phone",
    "mobile",
    "contact",
    "linkedin",
    "github",
    "objective",
    "summary",
    "profile",
    "address",
    "skills",
    "technical skills",
    "technical skill",
    "education",
    "experience",
    "work experience",
    "professional experience",
    "projects",
    "project",
    "academic projects",
    "personal projects",
    "certifications",
    "certification",
    "achievements",
    "awards",
    "responsibilities",
    "references",
    "publications",
    "interests",
    "hobbies",
}

NAME_REJECT_WORDS = {
    "used",
    "using",
    "use",
    "experience",
    "experienced",
    "developed",
    "developing",
    "build",
    "built",
    "create",
    "created",
    "creating",
    "implemented",
    "implement",
    "worked",
    "working",
    "responsible",
    "responsibilities",
    "technical",
    "technologies",
    "technology",
    "developer",
    "engineer",
    "analyst",
    "manager",
    "designer",
    "consultant",
    "intern",
    "student",
    "professional",
    "specialist",
    "python",
    "javascript",
    "typescript",
    "java",
    "sql",
    "html",
    "css",
    "react",
    "node",
    "nodejs",
    "fastapi",
    "flask",
    "django",
    "streamlit",
    "pytorch",
    "tensorflow",
    "keras",
    "opencv",
    "machine",
    "learning",
    "deep",
    "nlp",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "pandas",
    "numpy",
    "matplotlib",
    "github",
    "linkedin",
    "academic",
    "projects",
    "education",
    "certifications",
    "achievements",
    "publications",
    "objective",
    "summary",
}


def clean_name_candidate(line: str) -> str:
    if not line:
        return ""

    line = str(line).strip()
    line = re.sub(r"^[•●▪◦>*=_-]+", "", line)
    line = line.strip(" :|-_=•●▪◦")
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def is_probable_name(line: str) -> bool:
    """
    Strict validation intended to avoid false names such as:
        Used: Javascript, Python, SQL, HTML/CSS
        Academic Projects
        Technical Skills
    """
    if not line:
        return False

    line = clean_name_candidate(line)
    lower = line.lower()

    if len(line) < 3 or len(line) > 50:
        return False

    if lower in NAME_REJECT_EXACT:
        return False

    if "@" in line:
        return False

    for token in (
        "http://",
        "https://",
        "www.",
        "linkedin.com",
        "github.com",
    ):
        if token in lower:
            return False

    # Technical lists and labelled fields commonly contain
    # these characters; normal names do not.
    if any(
        char in line
        for char in (":", ",", "/", "|", ";", "[", "]", "{", "}")
    ):
        return False

    if re.search(r"\d", line):
        return False

    words = line.split()

    if len(words) < 2 or len(words) > 5:
        return False

    word_lowers = {
        word.lower().strip(".")
        for word in words
    }

    if word_lowers & NAME_REJECT_WORDS:
        return False

    sentence_words = {
        "with",
        "and",
        "the",
        "for",
        "from",
        "into",
        "using",
        "that",
        "this",
        "which",
        "where",
        "who",
        "have",
        "has",
        "been",
    }

    if any(word.lower() in sentence_words for word in words):
        return False

    # Allow letters plus standard name punctuation only.
    allowed = re.sub(r"[^A-Za-z .'-]", "", line).strip()

    if allowed != line:
        return False

    for word in words:
        alpha = re.sub(r"[^A-Za-z]", "", word)
        if not alpha or len(alpha) == 1:
            return False

    return True


def extract_name_from_filename(filename: str) -> str:
    if not filename:
        return ""

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    name = re.sub(r"[_-]+", " ", name)

    name = re.sub(
        r"\b(resume|cv|curriculum vitae)\b",
        " ",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(r"\s+", " ", name).strip()

    return name if is_probable_name(name) else ""


def extract_name(
    text: str,
    source_filename: str = "",
) -> str:
    """
    Extract the candidate name only from the header area.

    This intentionally does not search the entire document.
    That prevents project names and section headings from being
    mistaken for candidate names.
    """
    filename_candidate = extract_name_from_filename(
        source_filename
    )

    if not text:
        return filename_candidate

    lines = [
        clean_name_candidate(line)
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return filename_candidate

    # ONLY the header area participates in name detection.
    header_lines = lines[:15]

    contact_indexes = []

    for index, line in enumerate(header_lines):
        lower = line.lower()

        has_email = bool(re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            line,
        ))

        has_linkedin = "linkedin.com" in lower
        has_github = "github.com" in lower

        has_phone = bool(re.search(
            r"(?:\+91[\s.-]?)?[6-9]\d{4}[\s.-]?\d{5}",
            line,
        ))

        if has_email or has_linkedin or has_github or has_phone:
            contact_indexes.append(index)

    candidates = []

    for index, line in enumerate(header_lines):
        if not is_probable_name(line):
            continue

        score = 0

        if index == 0:
            score += 8
        elif index < 3:
            score += 6
        elif index < 6:
            score += 3
        else:
            score += 1

        if contact_indexes:
            first_contact = min(contact_indexes)
            distance = first_contact - index

            if distance == 1:
                score += 8
            elif distance == 2:
                score += 5
            elif 0 < distance <= 4:
                score += 3

        words = line.split()

        capitalized = sum(
            1
            for word in words
            if word[:1].isupper() or word.isupper()
        )

        if capitalized == len(words):
            score += 3

        if len(words) in {2, 3}:
            score += 2

        candidates.append((score, index, line))

    if candidates:
        candidates.sort(
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        best_score, _, best_name = candidates[0]

        if best_score >= 5:
            return best_name

    return filename_candidate


# ============================================================
# SECTIONS
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


def extract_sections(text: str) -> Dict[str, str]:
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


# ============================================================
# SKILLS
# ============================================================

SKILL_DATABASE = {
    "programming": [
        "python", "c++", "c#", "java", "javascript", "typescript",
        "sql", "golang", "go", "rust", "kotlin", "swift", "php",
    ],
    "machine_learning": [
        "machine learning", "supervised learning", "unsupervised learning",
        "reinforcement learning", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "random forest", "logistic regression",
        "linear regression", "decision tree", "support vector machine",
        "svm", "clustering", "k-means", "feature engineering",
    ],
    "deep_learning": [
        "deep learning", "tensorflow", "pytorch", "keras", "cnn",
        "convolutional neural network", "rnn", "lstm", "gru", "transformer",
        "transformers", "vision transformer", "vit", "gan",
        "generative adversarial network", "autoencoder", "diffusion models",
    ],
    "nlp": [
        "nlp", "natural language processing", "nltk", "spacy", "bert", "gpt",
        "llm", "large language model", "word2vec", "fasttext", "tf-idf",
        "text classification", "sentiment analysis", "named entity recognition",
        "ner", "tokenization", "text generation",
    ],
    "computer_vision": [
        "opencv", "computer vision", "image processing", "object detection",
        "face detection", "yolo", "resnet", "mtcnn", "deepface",
        "image classification", "image segmentation", "ocr",
    ],
    "data_science": [
        "pandas", "numpy", "matplotlib", "seaborn", "data analysis",
        "data visualization", "statistics", "data cleaning",
        "exploratory data analysis", "eda",
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "sql server", "redis", "sqlite",
        "oracle", "firebase", "database",
    ],
    "cloud": [
        "aws", "amazon web services", "azure", "google cloud", "gcp",
        "ec2", "s3", "lambda", "cloud computing",
    ],
    "devops": [
        "docker", "kubernetes", "jenkins", "ci/cd", "github actions",
        "terraform", "ansible", "continuous integration", "continuous deployment",
    ],
    "web": [
        "html", "css", "react", "react.js", "reactjs", "node.js", "nodejs",
        "fastapi", "flask", "django", "streamlit", "next.js", "nextjs",
        "rest api", "restful api",
    ],
    "tools": [
        "git", "github", "jira", "linux", "vscode", "visual studio code",
        "postman", "jupyter", "google colab",
    ],
}

SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "ml": "machine learning",
    "dl": "deep learning",
    "tf idf": "tf-idf",
    "nodejs": "node.js",
    "node js": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
    "next js": "next.js",
    "opencv-python": "opencv",
    "cv2": "opencv",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
}


def normalize_skill(skill: str) -> str:
    if not skill:
        return ""

    skill = skill.lower().strip()
    return SKILL_ALIASES.get(skill, skill)


def skill_exists(
    skill: str,
    text: str,
) -> bool:
    if not skill or not text:
        return False

    skill = normalize_skill(skill)

    # Avoid noisy short standalone matches.
    if skill in {"r", "go"}:
        return False

    text = text.lower()

    replacements = {
        "nodejs": "node.js",
        "node js": "node.js",
        "reactjs": "react",
        "react.js": "react",
        "nextjs": "next.js",
        "next js": "next.js",
        "sklearn": "scikit-learn",
        "scikit learn": "scikit-learn",
        "cv2": "opencv",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    pattern = (
        rf"(?<![a-z0-9])"
        rf"{re.escape(skill)}"
        rf"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def extract_skills_from_resume(
    text: str,
) -> Dict[str, List[str]]:
    if not text:
        return {}

    detected = {}

    for category, skills in SKILL_DATABASE.items():
        found = []

        for skill in skills:
            canonical = normalize_skill(skill)

            if skill_exists(canonical, text):
                found.append(canonical)

        if found:
            detected[category] = sorted(set(found))

    return detected


def flatten_skills(
    skills: Dict[str, List[str]]
) -> List[str]:
    if not skills:
        return []

    result = []

    for values in skills.values():
        result.extend(values)

    return sorted(set(result))


# ============================================================
# EDUCATION / EXPERIENCE / PROJECTS
# ============================================================

DEGREE_PATTERNS = [
    r"\bB\.?\s*Tech\b",
    r"\bM\.?\s*Tech\b",
    r"\bB\.?\s*E\.?\b",
    r"\bM\.?\s*E\.?\b",
    r"\bB\.?\s*Sc\.?\b",
    r"\bM\.?\s*Sc\.?\b",
    r"\bB\.?\s*CA\b",
    r"\bM\.?\s*CA\b",
    r"\bB\.?\s*BA\b",
    r"\bM\.?\s*BA\b",
    r"\bMBA\b",
    r"\bMCA\b",
    r"\bBCA\b",
    r"\bPh\.?\s*D\.?\b",
    r"\bBachelor(?:'s)?\b",
    r"\bMaster(?:'s)?\b",
    r"\bDiploma\b",
]


def extract_education(
    text: str,
    sections: Dict[str, str],
) -> str:
    education = sections.get("education", "").strip()

    if education:
        return education

    matches = []

    for pattern in DEGREE_PATTERNS:
        matches.extend(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )

    if not matches:
        return ""

    return ", ".join(
        sorted(set(matches), key=str.lower)
    )


def extract_experience(
    sections: Dict[str, str]
) -> str:
    return sections.get("experience", "").strip()


def extract_projects(
    sections: Dict[str, str]
) -> str:
    return sections.get("projects", "").strip()


def extract_certifications(
    sections: Dict[str, str]
) -> str:
    return sections.get("certifications", "").strip()


def extract_achievements(
    sections: Dict[str, str]
) -> str:
    return sections.get("achievements", "").strip()


def extract_experience_years(
    text: str
) -> float:
    if not text:
        return 0.0

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)"
        r"(?:\s+of)?\s+"
        r"(?:professional\s+|work\s+|relevant\s+)?experience",
        r"experience\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    ]

    values = []

    for pattern in patterns:
        for value in re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            try:
                number = float(value)
                if 0 <= number <= 50:
                    values.append(number)
            except (ValueError, TypeError):
                continue

    return round(max(values), 2) if values else 0.0


def calculate_text_statistics(
    text: str
) -> Dict[str, int]:
    if not text:
        return {
            "characters": 0,
            "words": 0,
            "lines": 0,
        }

    words = re.findall(r"\b\w+\b", text)
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
    }


def calculate_section_completeness(
    sections: Dict[str, str]
) -> float:
    important_sections = [
        "summary",
        "skills",
        "experience",
        "education",
        "projects",
        "certifications",
    ]

    present = sum(
        1
        for section in important_sections
        if sections.get(section, "").strip()
    )

    return round(
        (present / len(important_sections)) * 100,
        2,
    )


def calculate_contact_completeness(
    email: str,
    phone: str,
    linkedin: str,
    github: str,
) -> float:
    fields = [
        email,
        phone,
        linkedin,
        github,
    ]

    present = sum(
        1
        for field in fields
        if field
    )

    return round(
        (present / len(fields)) * 100,
        2,
    )


def generate_quality_flags(
    text: str,
    sections: Dict[str, str],
    email: str,
    phone: str,
) -> List[str]:
    flags = []

    if not text:
        return [
            "No extractable resume text detected."
        ]

    if not email:
        flags.append(
            "Email address was not detected."
        )

    if not phone:
        flags.append(
            "Phone number was not detected."
        )

    if not sections.get("skills"):
        flags.append(
            "A dedicated Skills section was not detected."
        )

    if not sections.get("education"):
        flags.append(
            "A dedicated Education section was not detected."
        )

    if not sections.get("experience"):
        flags.append(
            "A dedicated Experience section was not detected."
        )

    if not sections.get("projects"):
        flags.append(
            "A dedicated Projects section was not detected."
        )

    statistics = calculate_text_statistics(text)

    if statistics["words"] < 100:
        flags.append(
            "Very little extractable text was detected."
        )

    if statistics["words"] > 2500:
        flags.append(
            "The resume contains a large amount of text."
        )

    if not flags:
        flags.append(
            "No major basic resume-quality issues were detected."
        )

    return flags


# ============================================================
# MASTER
# ============================================================

def analyze_resume_intelligence(
    text: str,
    source_filename: str = "",
) -> Dict:
    if not text:
        return {
            "name": "",
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
            "skills": {},
            "all_skills": [],
            "education": "",
            "experience": "",
            "projects": "",
            "certifications": "",
            "achievements": "",
            "experience_years": 0.0,
            "sections": {},
            "section_completeness": 0.0,
            "contact_completeness": 0.0,
            "quality_flags": [
                "No resume text was provided."
            ],
            "statistics": {
                "characters": 0,
                "words": 0,
                "lines": 0,
            },
        }

    cleaned_text = normalize_resume_text(text)

    email = extract_email(cleaned_text)
    phone = extract_phone(cleaned_text)
    linkedin = extract_linkedin(cleaned_text)
    github = extract_github(cleaned_text)

    sections = extract_sections(cleaned_text)

    name = extract_name(
        cleaned_text,
        source_filename=source_filename,
    )

    skills = extract_skills_from_resume(cleaned_text)
    all_skills = flatten_skills(skills)

    education = extract_education(
        cleaned_text,
        sections,
    )

    experience = extract_experience(sections)
    projects = extract_projects(sections)
    certifications = extract_certifications(sections)
    achievements = extract_achievements(sections)

    experience_years = extract_experience_years(
        cleaned_text
    )

    section_completeness = calculate_section_completeness(
        sections
    )

    contact_completeness = calculate_contact_completeness(
        email,
        phone,
        linkedin,
        github,
    )

    quality_flags = generate_quality_flags(
        cleaned_text,
        sections,
        email,
        phone,
    )

    statistics = calculate_text_statistics(
        cleaned_text
    )

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "skills": skills,
        "all_skills": all_skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
        "achievements": achievements,
        "experience_years": experience_years,
        "sections": sections,
        "section_completeness": section_completeness,
        "contact_completeness": contact_completeness,
        "quality_flags": quality_flags,
        "statistics": statistics,
    }


if __name__ == "__main__":
    sample = """
    Adhyan Sharma
    adhyan.sharma@gmail.com
    +91 98765 43210
    GitHub: AdhyanSharma
    LinkedIn: linkedin.com/in/adhyansharma

    PROFESSIONAL SUMMARY
    Artificial Intelligence undergraduate.

    TECHNICAL SKILLS
    Python, SQL, Machine Learning, PyTorch, Docker, Git.

    EDUCATION
    B.Tech in Artificial Intelligence

    PROJECTS
    AI Resume Analyzer
    Mood Places Recommender
    """

    result = analyze_resume_intelligence(
        sample,
        "Adhyan Sharma Resume.pdf",
    )

    print("Name:", result["name"])
    print("Email:", result["email"])
    print("Phone:", result["phone"])
    print("LinkedIn:", result["linkedin"])
    print("GitHub:", result["github"])
