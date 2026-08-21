# ============================================================
# AI CAREER RECOMMENDATION SYSTEM
# Streamlit Application
# ============================================================

import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from urllib.parse import quote_plus
from collections import Counter

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GENERAL
       ====================================================== */

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* ======================================================
       CAREER SCORE
       ====================================================== */

    .score {
        font-size: 35px;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
    }

    .score-label {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        background: #0d6efd;
        color: white;
        font-size: 13px;
    }


    /* ======================================================
       CAREER CARD
       ====================================================== */

    .career-card-content {
        min-height: 0;
    }

    .career-info-title {
        font-size: 16px;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 7px;
    }

    .career-info-text {
        color: #6c757d;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 12px;
    }


    /* ======================================================
       SKILL BADGES
       IMPORTANT:
       inline-flex / inline-block means badges stay
       side-by-side until the available width is full.
       ====================================================== */

    .skill-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        width: 100%;
        margin-top: 5px;
        margin-bottom: 10px;
    }

    .skill-badge {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        max-width: 100%;
        padding: 6px 10px;
        border-radius: 18px;
        font-size: 12px;
        line-height: 1.2;
        white-space: nowrap;
        box-sizing: border-box;
    }

    .matched {
        background: #d1e7dd;
        color: #146c43;
        border: 1px solid #badbcc;
    }

    .develop {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffecb5;
    }


    /* ======================================================
       ROADMAP
       ====================================================== */

    .roadmap-stage {
        border-left: 4px solid #0d6efd;
        padding: 14px 16px;
        margin-bottom: 12px;
        background: #f8f9fa;
        border-radius: 0 12px 12px 0;
    }

    .roadmap-skill-completed {
        background: #d1e7dd;
        color: #146c43;
        padding: 7px 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        font-size: 13px;
    }

    .roadmap-skill-next {
        background: #fff3cd;
        color: #856404;
        padding: 7px 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        font-size: 13px;
    }


    /* ======================================================
       JOB BOX
       ====================================================== */

    .job-box {
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 8px;
        background: #ffffff;
    }

    .job-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .job-description {
        color: #6c757d;
        font-size: 13px;
        line-height: 1.4;
    }


    /* ======================================================
       WELCOME MODAL
       ====================================================== */

    .welcome-highlight {
        background: #eef5ff;
        border: 1px solid #cfe2ff;
        border-radius: 12px;
        padding: 14px;
        margin: 10px 0;
    }

    .welcome-tip {
        background: #fff8e1;
        border: 1px solid #ffe69c;
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
    }


    /* ======================================================
       RESPONSIVE
       ====================================================== */

    @media (max-width: 768px) {

        .main-title {
            font-size: 30px;
        }

        .sub-title {
            font-size: 15px;
        }

        .skill-badge {
            font-size: 11px;
            padding: 5px 8px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = "models"


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    "career_data.pkl",
    "career_embeddings.npy",
    "career_mapping.pkl",
    "hdbscan_model.pkl",
]

missing_files = []

for file in required_files:

    path = os.path.join(
        MODEL_DIR,
        file
    )

    if not os.path.exists(path):

        missing_files.append(path)


sbert_path = os.path.join(
    MODEL_DIR,
    "sbert_model"
)


if not os.path.exists(sbert_path):

    missing_files.append(sbert_path)


if missing_files:

    st.error("Some model files are missing.")

    for file in missing_files:

        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    sbert_model = SentenceTransformer(
        os.path.join(
            MODEL_DIR,
            "sbert_model"
        )
    )

    hdbscan_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "hdbscan_model.pkl"
        )
    )

    career_mapping = joblib.load(
        os.path.join(
            MODEL_DIR,
            "career_mapping.pkl"
        )
    )

    career_data = pd.read_pickle(
        os.path.join(
            MODEL_DIR,
            "career_data.pkl"
        )
    )

    career_embeddings = np.load(
        os.path.join(
            MODEL_DIR,
            "career_embeddings.npy"
        )
    )

    return (
        sbert_model,
        hdbscan_model,
        career_mapping,
        career_data,
        career_embeddings
    )


(
    sbert_model,
    hdbscan_model,
    career_mapping,
    career_data,
    career_embeddings
) = load_models()


# ============================================================
# DATASET CHECK
# ============================================================

required_columns = [
    "Skills",
    "Interests",
    "Education",
    "Recommended_Career"
]

for column in required_columns:

    if column not in career_data.columns:

        st.error(
            f"The dataset does not contain "
            f"a '{column}' column."
        )

        st.stop()


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if value is None:

        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    "js": "javascript",
    "javascript": "javascript",

    "py": "python",
    "python": "python",

    "reactjs": "react",
    "react js": "react",
    "react.js": "react",

    "vuejs": "vue",
    "vue js": "vue",
    "vue.js": "vue",

    "angularjs": "angular",
    "angular js": "angular",
    "angular.js": "angular",

    "nodejs": "node js",
    "node js": "node js",
    "node.js": "node js",

    "sql server": "sql",
    "ms sql": "sql",
    "mssql": "sql",

    "postgresql": "postgres",
    "postgres": "postgres",

    "mongodb": "mongo db",
    "mongo": "mongo db",

    "c sharp": "c#",
    "c-sharp": "c#",

    "dotnet": ".net",
    "dot net": ".net",

    "asp net": "asp.net",
    "aspnet": "asp.net",

    "github": "github",
    "git hub": "github",

    "ms office": "microsoft office",
    "office": "microsoft office"
}


def normalize_skill(skill):

    value = normalize_text(skill)

    return SKILL_ALIASES.get(
        value,
        value
    )


# ============================================================
# GET UNIQUE SKILLS
# ============================================================

@st.cache_data
def get_skills(df):

    skills = set()

    for value in df["Skills"].dropna():

        for skill in str(value).split(";"):

            skill = skill.strip()

            if skill:

                skills.add(skill)

    return sorted(
        skills,
        key=str.lower
    )


# ============================================================
# GET UNIQUE INTERESTS
# ============================================================

@st.cache_data
def get_interests(df):

    interests = set()

    for value in df["Interests"].dropna():

        for interest in str(value).split(";"):

            interest = interest.strip()

            if interest:

                interests.add(interest)

    return sorted(
        interests,
        key=str.lower
    )


# ============================================================
# GET EDUCATION LIST
# ============================================================

@st.cache_data
def get_education_list(df):

    values = set()

    for value in df["Education"].dropna():

        value = str(value).strip()

        if value:

            values.add(value)

    return sorted(
        values,
        key=str.lower
    )


skills_list = get_skills(
    career_data
)

interests_list = get_interests(
    career_data
)

education_list = get_education_list(
    career_data
)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {

    "selected_skills": [],

    "selected_interests": [],

    "show_skill_other": False,

    "show_interest_other": False,

    "skill_selector_value": "",

    "interest_selector_value": "",

    "recommendations": [],

    "selected_career": None,

    "job_career": None,

    "roadmap_career": None,

    "welcome_shown": False
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# WELCOME MODAL
# ============================================================

@st.dialog(
    "🎯 Welcome to Career Recommendation System",
    width="medium"
)
def show_welcome_modal():

    st.markdown(
        "### 👋 Welcome!"
    )

    st.write(
        "This system recommends suitable IT career paths "
        "based on your skills, interests and education."
    )

    st.markdown(
        f"""
        <div class="welcome-highlight">

        <b>📊 Dataset Information</b><br><br>

        • Dataset Records: <b>{len(career_data):,}</b><br>
        • Available Skills: <b>{len(skills_list)}</b><br>
        • Available Interests: <b>{len(interests_list)}</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📝 How to get better recommendations"
    )

    st.write(
        "Please provide your information as accurately "
        "as possible."
    )

    st.markdown(
        """
        - 💻 Select all skills that you actually know.
        - ❤️ Select interests that genuinely match you.
        - ➕ Use **Other** when your skill or interest is not listed.
        - 🎓 Select the education level that best represents you.
        - 🎯 More accurate profile information can help the system provide more relevant recommendations.
        """
    )

    st.markdown(
        """
        <div class="welcome-tip">

        <b>💡 Tip</b><br>
        The recommendation quality depends on the quality
        of the information you provide. Try to include
        relevant skills and interests instead of leaving
        them empty.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "✓ Got it, Let's Start",
        type="primary",
        use_container_width=True
    ):

        st.session_state.welcome_shown = True

        st.rerun()


# ============================================================
# SHOW WELCOME MODAL ONLY ON FIRST VISIT
# ============================================================

if not st.session_state.welcome_shown:

    show_welcome_modal()


# ============================================================
# ADD / REMOVE SKILL
# ============================================================

def add_skill(skill):

    skill = str(skill).strip()

    if not skill:

        return

    existing = [
        normalize_skill(x)
        for x in st.session_state.selected_skills
    ]

    if normalize_skill(skill) not in existing:

        st.session_state.selected_skills.append(
            skill
        )


def remove_skill(skill):

    if skill in st.session_state.selected_skills:

        st.session_state.selected_skills.remove(
            skill
        )


# ============================================================
# ADD / REMOVE INTEREST
# ============================================================

def add_interest(interest):

    interest = str(interest).strip()

    if not interest:

        return

    existing = [
        normalize_text(x)
        for x in st.session_state.selected_interests
    ]

    if normalize_text(interest) not in existing:

        st.session_state.selected_interests.append(
            interest
        )


def remove_interest(interest):

    if interest in st.session_state.selected_interests:

        st.session_state.selected_interests.remove(
            interest
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Career Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Discover suitable IT career paths based on your profile, '
    'skills and interests.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👤 Personal Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=22,
        step=1,
        help="Age must be between 18 and 65."
    )


with col2:

    education = st.selectbox(
        "Education",
        options=education_list,
        index=None,
        placeholder="Select your education"
    )
# ============================================================
# SKILLS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '💻 Skills'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SKILL SELECTOR STATE
# ============================================================

if "skill_selector_id" not in st.session_state:
    st.session_state.skill_selector_id = 0


# ============================================================
# SKILL SELECT CALLBACK
# ============================================================

def handle_skill_selection():

    key = (
        f"skill_selector_"
        f"{st.session_state.skill_selector_id}"
    )

    selected = st.session_state.get(
        key,
        ""
    )

    if not selected:
        return


    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    if selected == "Other":

        st.session_state.show_skill_other = True


    # --------------------------------------------------------
    # DATASET SKILL
    # --------------------------------------------------------

    else:

        add_skill(selected)


    # --------------------------------------------------------
    # CREATE NEW SELECTBOX
    #
    # New key means the old selected value disappears
    # from the input box.
    # --------------------------------------------------------

    st.session_state.skill_selector_id += 1


# ============================================================
# REMOVE ALREADY SELECTED SKILLS
# ============================================================

selected_skill_normalized = {
    normalize_skill(skill)
    for skill in st.session_state.selected_skills
}


available_skills = [
    skill
    for skill in skills_list
    if normalize_skill(skill)
    not in selected_skill_normalized
]


# ============================================================
# SKILL OPTIONS
# ============================================================

skill_options = [
    ""
] + available_skills + [
    "Other"
]


# ============================================================
# SKILL SELECTBOX
# ============================================================

skill_key = (
    f"skill_selector_"
    f"{st.session_state.skill_selector_id}"
)


st.selectbox(
    "Search or select skills",
    options=skill_options,
    format_func=lambda x: (
        "Search or select skills"
        if x == ""
        else x
    ),
    key=skill_key,
    on_change=handle_skill_selection
)


# ============================================================
# OTHER SKILL
# ============================================================

if st.session_state.show_skill_other:

    st.markdown(
        "**➕ Add Other Skill**"
    )

    with st.form(
        "other_skill_form",
        clear_on_submit=True
    ):

        other_skill = st.text_input(
            "Enter another skill",
            placeholder="Type your skill and press Enter"
        )

        submitted_skill = st.form_submit_button(
            "➕ Add Skill",
            type="primary"
        )

        if submitted_skill:

            value = other_skill.strip()

            if value:

                add_skill(value)

                st.session_state.show_skill_other = False

                # Create a fresh selectbox
                st.session_state.skill_selector_id += 1

            else:

                st.warning(
                    "Please enter a skill."
                )


# ============================================================
# SELECTED SKILLS
# ============================================================

if st.session_state.selected_skills:

    st.markdown(
        "**Selected Skills**"
    )

    skill_cols = st.columns(
        min(
            4,
            len(
                st.session_state.selected_skills
            )
        )
    )

    for index, skill in enumerate(
        st.session_state.selected_skills
    ):

        with skill_cols[
            index % len(skill_cols)
        ]:

            if st.button(
                f"❌ {skill}",
                key=f"remove_skill_{index}",
                use_container_width=True
            ):

                remove_skill(skill)

                st.rerun()


# ============================================================
# INTERESTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '❤️ Interests'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INTEREST SELECTOR STATE
# ============================================================

if "interest_selector_id" not in st.session_state:

    st.session_state.interest_selector_id = 0


# ============================================================
# INTEREST SELECT CALLBACK
# ============================================================

def handle_interest_selection():

    key = (
        f"interest_selector_"
        f"{st.session_state.interest_selector_id}"
    )

    selected = st.session_state.get(
        key,
        ""
    )

    if not selected:
        return


    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    if selected == "Other":

        st.session_state.show_interest_other = True


    # --------------------------------------------------------
    # DATASET INTEREST
    # --------------------------------------------------------

    else:

        add_interest(selected)


    # --------------------------------------------------------
    # CREATE NEW SELECTBOX
    #
    # New key means the old selected value disappears
    # from the input box.
    # --------------------------------------------------------

    st.session_state.interest_selector_id += 1


# ============================================================
# REMOVE ALREADY SELECTED INTERESTS
# ============================================================

selected_interest_normalized = {
    normalize_text(interest)
    for interest in st.session_state.selected_interests
}


available_interests = [
    interest
    for interest in interests_list
    if normalize_text(interest)
    not in selected_interest_normalized
]


# ============================================================
# INTEREST OPTIONS
# ============================================================

interest_options = [
    ""
] + available_interests + [
    "Other"
]


# ============================================================
# INTEREST SELECTBOX
# ============================================================

interest_key = (
    f"interest_selector_"
    f"{st.session_state.interest_selector_id}"
)


st.selectbox(
    "Search or select interests",
    options=interest_options,
    format_func=lambda x: (
        "Search or select interests"
        if x == ""
        else x
    ),
    key=interest_key,
    on_change=handle_interest_selection
)


# ============================================================
# OTHER INTEREST
# ============================================================

if st.session_state.show_interest_other:

    st.markdown(
        "**➕ Add Other Interest**"
    )

    with st.form(
        "other_interest_form",
        clear_on_submit=True
    ):

        other_interest = st.text_input(
            "Enter another interest",
            placeholder="Type your interest and press Enter"
        )

        submitted_interest = st.form_submit_button(
            "➕ Add Interest",
            type="primary"
        )

        if submitted_interest:

            value = other_interest.strip()

            if value:

                add_interest(value)

                st.session_state.show_interest_other = False

                # Create a fresh selectbox
                st.session_state.interest_selector_id += 1

            else:

                st.warning(
                    "Please enter an interest."
                )


# ============================================================
# SELECTED INTERESTS
# ============================================================

if st.session_state.selected_interests:

    st.markdown(
        "**Selected Interests**"
    )

    interest_cols = st.columns(
        min(
            4,
            len(
                st.session_state.selected_interests
            )
        )
    )

    for index, interest in enumerate(
        st.session_state.selected_interests
    ):

        with interest_cols[
            index % len(interest_cols)
        ]:

            if st.button(
                f"❌ {interest}",
                key=f"remove_interest_{index}",
                use_container_width=True
            ):

                remove_interest(
                    interest
                )

                st.rerun()


# ============================================================
# VALIDATION
# ============================================================

st.divider()


recommend_button = st.button(
    "🚀 Recommend Career",
    type="primary",
    use_container_width=True
)


# ============================================================
# SKILL MATCHING
# ============================================================

def skill_matches(
    user_skill,
    dataset_skill
):

    user = normalize_text(
        user_skill
    )

    dataset = normalize_text(
        dataset_skill
    )

    if not user or not dataset:
        return False


    # ========================================================
    # EXACT MATCH
    # ========================================================

    if user == dataset:
        return True


    # ========================================================
    # COMMON ABBREVIATIONS / EQUIVALENT NAMES
    # ========================================================

    aliases = {

        "js": "javascript",

        "javascript": "javascript",

        "py": "python",

        "reactjs": "react",

        "react js": "react",

        "nodejs": "node js",

        "node js": "node js",

        "sql server": "sql",

        "ms sql": "sql",

        "mssql": "sql"
    }


    user_normalized = aliases.get(
        user,
        user
    )

    dataset_normalized = aliases.get(
        dataset,
        dataset
    )


    # ========================================================
    # NORMALIZED EXACT MATCH
    # ========================================================

    if user_normalized == dataset_normalized:
        return True


    # ========================================================
    # PHRASE MATCHING
    # ========================================================

    if (
        dataset_normalized in user_normalized
        or
        user_normalized in dataset_normalized
    ):
        return True


    return False


# ============================================================
# GET CAREER REQUIRED SKILLS
#
# CHANGED:
# Previously 10%.
# Now 5% so less common but still relevant skills
# are not unnecessarily removed.
# ============================================================

def get_career_required_skills(
    career,
    minimum_ratio=0.05,
    max_skills=40
):

    career_rows = career_data[
        career_data[
            "Recommended_Career"
        ] == career
    ]

    counter = Counter()

    total_rows = len(
        career_rows
    )

    if total_rows == 0:

        return []


    # --------------------------------------------------------
    # Count each skill once per dataset row
    # --------------------------------------------------------

    for value in career_rows[
        "Skills"
    ].dropna():

        row_skills = set()

        for skill in str(value).split(";"):

            skill = skill.strip()

            if skill:

                row_skills.add(
                    normalize_skill(skill)
                )

        for skill in row_skills:

            counter[skill] += 1


    # --------------------------------------------------------
    # Lower threshold so more valid skills are retained
    # --------------------------------------------------------

    threshold = max(
        1,
        int(
            np.ceil(
                total_rows
                * minimum_ratio
            )
        )
    )


    selected = [
        skill
        for skill, count in counter.most_common()
        if count >= threshold
    ]


    # --------------------------------------------------------
    # Convert normalized skill to readable dataset value
    # --------------------------------------------------------

    readable = {}

    for value in career_rows[
        "Skills"
    ].dropna():

        for skill in str(value).split(";"):

            skill = skill.strip()

            if skill:

                key = normalize_skill(
                    skill
                )

                if key not in readable:

                    readable[key] = skill


    return [
        readable.get(
            skill,
            skill
        )
        for skill in selected[:max_skills]
    ]


# ============================================================
# MATCH USER SKILLS
# ============================================================

def calculate_skill_matches(
    selected_skills,
    required_skills
):

    matched = []

    for required_skill in required_skills:

        for user_skill in selected_skills:

            if skill_matches(
                user_skill,
                required_skill
            ):

                # Show the user's actual skill name
                # rather than duplicating dataset skill
                if not any(
                    normalize_skill(x)
                    == normalize_skill(user_skill)
                    for x in matched
                ):

                    matched.append(
                        user_skill
                    )

                break

    return matched


# ============================================================
# SKILLS TO DEVELOP
# ============================================================

def calculate_skills_to_develop(
    selected_skills,
    required_skills
):

    missing = []

    for required_skill in required_skills:

        found = False

        for user_skill in selected_skills:

            if skill_matches(
                user_skill,
                required_skill
            ):

                found = True

                break

        if not found:

            missing.append(
                required_skill
            )

    return missing


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_career(
    age,
    education,
    selected_skills,
    selected_interests
):

    profile_text = (

        " ".join(
            selected_skills
        )

        + " "

        + " ".join(
            selected_interests
        )

        + " "

        + str(
            education
        )
    )


    # ========================================================
    # CREATE EMBEDDING
    # ========================================================

    user_embedding = sbert_model.encode(
        [profile_text],
        normalize_embeddings=True
    )


    # ========================================================
    # HDBSCAN PREDICTION
    # ========================================================

    try:

        predicted_labels, strengths = (
            hdbscan_model.approximate_predict(
                user_embedding
            )
        )

        predicted_cluster = int(
            predicted_labels[0]
        )

    except Exception:

        predicted_cluster = -1


    cluster_career = career_mapping.get(
        predicted_cluster
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    user_vector = user_embedding[0]

    dataset_vectors = career_embeddings


    user_norm = np.linalg.norm(
        user_vector
    )

    dataset_norms = np.linalg.norm(
        dataset_vectors,
        axis=1
    )


    denominator = (
        dataset_norms
        * user_norm
    )


    denominator[
        denominator == 0
    ] = 1


    similarities = np.dot(
        dataset_vectors,
        user_vector
    ) / denominator


    # ========================================================
    # CAREER SCORE
    # ========================================================

    career_scores = {}


    for index in np.argsort(
        similarities
    )[::-1]:

        career = career_data.iloc[
            index
        ]["Recommended_Career"]

        score = float(
            similarities[index]
        )

        if career not in career_scores:

            career_scores[career] = []

        career_scores[career].append(
            score
        )


    career_final_scores = {}


    for career, scores in career_scores.items():

        top_scores = sorted(
            scores,
            reverse=True
        )[:10]

        career_final_scores[career] = (
            np.mean(top_scores)
        )


    # ========================================================
    # CLUSTER BOOST
    # ========================================================

    if cluster_career:

        career_final_scores[
            cluster_career
        ] = career_final_scores.get(
            cluster_career,
            0
        ) + 0.05


    # ========================================================
    # SORT
    # ========================================================

    sorted_careers = sorted(
        career_final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # TOP 3
    # ========================================================
    recommendations = []

    for career, raw_score in sorted_careers:

        required_skills = get_career_required_skills(
            career
        )

        matched_skills = calculate_skill_matches(
            selected_skills,
            required_skills
        )

        # No matched skill = do not recommend
        if not matched_skills:
            continue

        skills_to_develop = calculate_skills_to_develop(
            selected_skills,
            required_skills
        )

        score = max(
            0,
            min(
                1,
                float(raw_score)
            )
        )

        recommendations.append(
            {
                "career": career,

                "score": score,

                "matched_skills":
                    matched_skills,

                "skills_to_develop":
                    skills_to_develop,

                "required_skills":
                    required_skills
            }
        )

        # Stop after 3 valid careers
        if len(recommendations) >= 3:
            break

    return recommendations

# ============================================================
# RECOMMEND BUTTON ACTION
# ============================================================

if recommend_button:

    # --------------------------------------------------------
    # AGE VALIDATION
    # --------------------------------------------------------

    if age < 18 or age > 65:

        st.error(
            "Age must be between 18 and 65."
        )

        st.stop()


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    if not education:

        st.warning(
            "Please select your education."
        )

        st.stop()


    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    if not st.session_state.selected_skills:

        st.warning(
            "Please select at least one skill."
        )

        st.stop()


    # --------------------------------------------------------
    # INTERESTS
    # --------------------------------------------------------

    if not st.session_state.selected_interests:

        st.warning(
            "Please select at least one interest."
        )

        st.stop()


    # --------------------------------------------------------
    # RECOMMEND
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing your profile..."
    ):

        recommendations = recommend_career(

            age,

            education,

            st.session_state.selected_skills,

            st.session_state.selected_interests

        )


    # ========================================================
    # SAVE RECOMMENDATIONS
    # ========================================================

    st.session_state.recommendations = (
        recommendations
    )

    st.session_state.selected_career = None


    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if recommendations:

        st.success(
            "Career recommendations generated successfully!"
        )

    else:

        st.warning(
            "⚠️ No Matching Career Found"
        )

        st.info(
            "We couldn't find a suitable career based on "
            "the skills you selected. Please add more relevant "
            "skills or use 'Other' to add additional skills."
        )


# ============================================================
# CAREER ICON
# ============================================================

def get_career_icon(career):

    career_lower = career.lower()

    if "cyber" in career_lower:

        return "🛡️"

    if (
        "data" in career_lower
        or "ai" in career_lower
    ):

        return "🧠"

    if (
        "cloud" in career_lower
        or "devops" in career_lower
    ):

        return "☁️"

    if "frontend" in career_lower:

        return "💻"

    if "backend" in career_lower:

        return "🖥️"

    if "full stack" in career_lower:

        return "🌐"

    if (
        "ui" in career_lower
        or "ux" in career_lower
    ):

        return "🎨"

    if "marketing" in career_lower:

        return "📢"

    if (
        "project" in career_lower
        or "product" in career_lower
    ):

        return "📋"

    return "💼"


# ============================================================
# ROADMAP DATA
# ============================================================

career_roadmaps = {

    "Backend Developer": {

        "description":
            "Build server-side applications, APIs, databases and backend systems.",

        "stages": [

            {
                "title": "Programming Foundation",

                "skills": [
                    "Programming Fundamentals",
                    "Problem Solving",
                    "Object-Oriented Programming",
                    "Git",
                    "GitHub"
                ]
            },

            {
                "title": "Backend Fundamentals",

                "skills": [
                    "HTTP",
                    "Client-Server Architecture",
                    "Database Fundamentals",
                    "SQL",
                    "REST API"
                ]
            },

            {
                "title": "Backend Development",

                "skills": [
                    "MVC Architecture",
                    "API Development",
                    "Database Integration",
                    "Authentication",
                    "Authorization"
                ]
            },

            {
                "title": "Advanced Backend",

                "skills": [
                    "API Security",
                    "Error Handling",
                    "Testing",
                    "Caching",
                    "Performance Optimization"
                ]
            },

            {
                "title": "Projects & Deployment",

                "skills": [
                    "Backend Project",
                    "GitHub Portfolio",
                    "Documentation",
                    "Deployment",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "Frontend Developer": {

        "description":
            "Create responsive, interactive and user-friendly web interfaces.",

        "stages": [

            {
                "title": "Web Foundation",

                "skills": [
                    "HTML",
                    "CSS",
                    "Responsive Design",
                    "Git",
                    "GitHub"
                ]
            },

            {
                "title": "JavaScript Fundamentals",

                "skills": [
                    "JavaScript",
                    "DOM",
                    "ES6",
                    "API Integration"
                ]
            },

            {
                "title": "Frontend Development",

                "skills": [
                    "Component-Based Development",
                    "State Management",
                    "API Integration",
                    "Form Handling",
                    "Client-Side Routing"
                ]
            },

            {
                "title": "Frontend Quality",

                "skills": [
                    "Frontend Testing",
                    "Accessibility",
                    "Performance Optimization",
                    "Browser Compatibility",
                    "Debugging"
                ]
            },

            {
                "title": "Projects & Deployment",

                "skills": [
                    "Frontend Project",
                    "GitHub Portfolio",
                    "Deployment",
                    "Responsive Web Application",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "Cloud & DevOps": {

        "description":
            "Build, automate and maintain scalable cloud infrastructure and deployment pipelines.",

        "stages": [

            {
                "title": "Foundation",

                "skills": [
                    "Linux",
                    "Networking Fundamentals",
                    "Git",
                    "GitHub",
                    "Bash"
                ]
            },

            {
                "title": "Cloud Fundamentals",

                "skills": [
                    "Cloud Computing",
                    "AWS",
                    "Azure",
                    "Cloud Storage",
                    "Virtual Machines"
                ]
            },

            {
                "title": "DevOps Tools",

                "skills": [
                    "Docker",
                    "CI/CD",
                    "Jenkins",
                    "GitHub Actions"
                ]
            },

            {
                "title": "Infrastructure",

                "skills": [
                    "Kubernetes",
                    "Terraform",
                    "Infrastructure as Code",
                    "Monitoring"
                ]
            },

            {
                "title": "Career Ready",

                "skills": [
                    "Cloud Project",
                    "Deploy Application",
                    "Build DevOps Portfolio",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "Data Science & AI": {

        "description":
            "Analyze data and build machine learning and artificial intelligence solutions.",

        "stages": [

            {
                "title": "Foundation",

                "skills": [
                    "Python",
                    "Programming Fundamentals",
                    "Statistics",
                    "Mathematics"
                ]
            },

            {
                "title": "Data Analysis",

                "skills": [
                    "NumPy",
                    "Pandas",
                    "Data Cleaning",
                    "Data Visualization",
                    "Matplotlib"
                ]
            },

            {
                "title": "Machine Learning",

                "skills": [
                    "Scikit-learn",
                    "Supervised Learning",
                    "Unsupervised Learning",
                    "Model Evaluation"
                ]
            },

            {
                "title": "AI",

                "skills": [
                    "Deep Learning",
                    "Neural Networks",
                    "NLP",
                    "Computer Vision"
                ]
            },

            {
                "title": "Career Ready",

                "skills": [
                    "Machine Learning Project",
                    "AI Project",
                    "GitHub Portfolio",
                    "Deploy Model",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "Cybersecurity": {

        "description":
            "Protect applications, networks, systems and data from security threats.",

        "stages": [

            {
                "title": "IT Foundation",

                "skills": [
                    "Computer Fundamentals",
                    "Networking",
                    "Linux",
                    "Operating Systems"
                ]
            },

            {
                "title": "Security Fundamentals",

                "skills": [
                    "Cybersecurity Fundamentals",
                    "CIA Triad",
                    "Authentication",
                    "Access Control"
                ]
            },

            {
                "title": "Network Security",

                "skills": [
                    "Firewalls",
                    "VPN",
                    "Network Security",
                    "Wireshark"
                ]
            },

            {
                "title": "Security Testing",

                "skills": [
                    "Ethical Hacking",
                    "Penetration Testing",
                    "Vulnerability Assessment",
                    "OWASP"
                ]
            },

            {
                "title": "Career Ready",

                "skills": [
                    "Security Lab Projects",
                    "Security Portfolio",
                    "Security Certifications",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "UI/UX Designer": {

        "description":
            "Design intuitive, accessible and engaging digital experiences.",

        "stages": [

            {
                "title": "Design Foundation",

                "skills": [
                    "Design Principles",
                    "Color Theory",
                    "Typography",
                    "Layout"
                ]
            },

            {
                "title": "UX Research",

                "skills": [
                    "User Research",
                    "User Personas",
                    "User Journey",
                    "Information Architecture"
                ]
            },

            {
                "title": "Wireframing",

                "skills": [
                    "Wireframing",
                    "User Flow",
                    "Prototyping"
                ]
            },

            {
                "title": "UI Design",

                "skills": [
                    "Figma",
                    "UI Design",
                    "Design System",
                    "Responsive Design"
                ]
            },

            {
                "title": "Career Ready",

                "skills": [
                    "Portfolio",
                    "Case Studies",
                    "Prototype Projects",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "Project/Product Management": {

        "description":
            "Plan, coordinate and deliver successful technology products and projects.",

        "stages": [

            {
                "title": "Management Foundation",

                "skills": [
                    "Communication",
                    "Problem Solving",
                    "Leadership",
                    "Teamwork"
                ]
            },

            {
                "title": "Project Management",

                "skills": [
                    "Project Planning",
                    "Requirements Gathering",
                    "Risk Management",
                    "Time Management"
                ]
            },

            {
                "title": "Agile & Scrum",

                "skills": [
                    "Agile",
                    "Scrum",
                    "Sprint Planning",
                    "Jira"
                ]
            },

            {
                "title": "Product Management",

                "skills": [
                    "Product Strategy",
                    "Product Roadmap",
                    "User Stories",
                    "Product Analytics"
                ]
            },

            {
                "title": "Career Ready",

                "skills": [
                    "Project Portfolio",
                    "Product Case Study",
                    "Leadership Experience",
                    "Apply for Jobs"
                ]
            }
        ]
    },


    "Digital Marketing": {

        "description":
            "Use digital channels, content and analytics to grow brands and businesses.",

        "stages": [

            {
                "title": "Marketing Foundation",

                "skills": [
                    "Marketing Fundamentals",
                    "Communication",
                    "Content Writing",
                    "Branding"
                ]
            },

            {
                "title": "Social Media",

                "skills": [
                    "Social Media Marketing",
                    "Facebook Marketing",
                    "Instagram Marketing",
                    "Content Strategy"
                ]
            },

            {
                "title": "SEO & Content",

                "skills": [
                    "SEO",
                    "Keyword Research",
                    "Content Marketing",
                    "Google Analytics"
                ]
            },

            {
                "title": "Advertising",

                "skills": [
                    "Google Ads",
                    "Facebook Ads",
                    "Campaign Management",
                    "Conversion Optimization"
                ]
            },

            {
                "title": "Career Ready",

                "skills": [
                    "Marketing Campaign",
                    "Marketing Portfolio",
                    "Analytics Report",
                    "Apply for Jobs"
                ]
            }
        ]
    }
}


# ============================================================
# FULL STACK ROADMAP
# ============================================================

career_roadmaps[
    "Full Stack Developer"
] = {

    "description":
        "Develop complete web applications from frontend to backend and deployment.",

    "stages": [

        {
            "title": "Frontend Track",

            "skills": [
                "HTML",
                "CSS",
                "JavaScript",
                "Responsive Design",
                "React / Vue / Angular",
                "API Integration"
            ]
        },

        {
            "title": "Backend Track",

            "skills": [
                "Backend Programming",
                "Database",
                "SQL",
                "REST API",
                "Authentication",
                "Authorization"
            ]
        },

        {
            "title": "Technology Path",

            "skills": [
                "PHP / Laravel",
                "Python / Django",
                "Node.js",
                "Java / Spring Boot",
                "C# / ASP.NET"
            ]
        },

        {
            "title": "Deployment",

            "skills": [
                "Git",
                "GitHub",
                "Linux",
                "Docker",
                "Cloud Deployment"
            ]
        },

        {
            "title": "Career Ready",

            "skills": [
                "Full Stack Project",
                "GitHub Portfolio",
                "Deploy Application",
                "CV / Resume",
                "Apply for Jobs"
            ]
        }
    ]
}


# ============================================================
# TECHNOLOGY OPTIONS
# ============================================================

backend_technologies = [

    (
        "PHP / Laravel",
        "Build modern web applications and REST APIs using PHP and Laravel."
    ),

    (
        "Python Backend",
        "Build backend applications and APIs using Python, Django or FastAPI."
    ),

    (
        "Node.js",
        "Develop scalable backend services using Node.js and Express.js."
    ),

    (
        "Java / Spring Boot",
        "Build enterprise-level backend applications using Java and Spring Boot."
    ),

    (
        "C# / ASP.NET",
        "Build enterprise applications and APIs using C# and ASP.NET."
    )
]


frontend_technologies = [

    (
        "React",
        "Build modern and interactive user interfaces with React."
    ),

    (
        "Vue.js",
        "Build flexible and progressive web interfaces using Vue.js."
    ),

    (
        "Angular",
        "Build structured and scalable web applications using Angular."
    )
]


# ============================================================
# ROADMAP SKILL MATCH
# ============================================================

def is_skill_matched(
    skill,
    user_skills
):

    for user_skill in user_skills:

        if skill_matches(
            user_skill,
            skill
        ):
            return True

    return False


# ============================================================
# ROADMAP MODAL
# ============================================================

@st.dialog(
    "🛣️ Career Roadmap",
    width="large"
)
def show_roadmap_modal(career):

    roadmap = career_roadmaps.get(
        career
    )

    if not roadmap:

        st.info(
            "Roadmap information is not available yet "
            "for this career."
        )

        return


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.subheader(
        f"{get_career_icon(career)} {career}"
    )

    st.info(
        roadmap["description"]
    )

    # --------------------------------------------------------
    # FIND USER MATCHED SKILLS
    # --------------------------------------------------------

    selected_item = None

    for item in st.session_state.recommendations:

        if item["career"] == career:

            selected_item = item

            break


    # User's actual selected skills
    # Includes skills entered through "Other"
    roadmap_matched = (
        st.session_state.selected_skills
    )

    # --------------------------------------------------------
    # ROADMAP STAGES
    # --------------------------------------------------------

    for index, stage in enumerate(
        roadmap["stages"]
    ):

        st.markdown(
            f"### {index + 1}. {stage['title']}"
        )

        stage_col1, stage_col2 = st.columns(2)

        with stage_col1:

            st.markdown(
                "**Skills**"
            )

            for skill in stage["skills"]:

                matched = is_skill_matched(
                    skill,
                    roadmap_matched
                )

                if matched:

                    st.success(
                        f"{skill} — Completed",
                        icon="✅"
                    )

                else:

                    st.warning(
                        f"{skill} — Next Step",
                        icon="📌"
                    )


        with stage_col2:

            st.markdown(
                "**Progress**"
            )

            completed_count = sum(
                is_skill_matched(
                    skill,
                    roadmap_matched
                )
                for skill in stage["skills"]
            )

            total_count = len(
                stage["skills"]
            )

            progress = (
                completed_count / total_count
                if total_count
                else 0
            )

            st.progress(
                progress
            )

            st.caption(
                f"{completed_count} of "
                f"{total_count} skills matched"
            )


        st.divider()


    # --------------------------------------------------------
    # BACKEND TECHNOLOGY
    # --------------------------------------------------------

    if career == "Backend Developer":

        st.subheader(
            "🔀 Choose Your Backend Technology"
        )

        st.caption(
            "Choose the technology that best matches "
            "your interests and career goals."
        )


        for title, description in backend_technologies:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    description
                )


    # --------------------------------------------------------
    # FRONTEND TECHNOLOGY
    # --------------------------------------------------------

    if career == "Frontend Developer":

        st.subheader(
            "🔀 Choose Your Frontend Technology"
        )

        st.caption(
            "Choose the frontend framework that best "
            "matches your interests and career goals."
        )


        for title, description in frontend_technologies:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    description
                )

# ============================================================
# JOB SEARCH DATA
# ============================================================

def get_job_url(career, market):

    keyword = quote_plus(
        career
    )

    if market == "JobNet Myanmar":

        return (
            f"https://www.jobnet.com.mm/jobs?kw={keyword}"
        )

    elif market == "LinkedIn":

        return (
            "https://www.linkedin.com/jobs/search/"
            f"?keywords={keyword}"
        )

    elif market == "MyJobs":

        return (
            f"https://myjobs.com.mm/jobs?title={keyword}"
        )

    return "#"


# ============================================================
# AVAILABLE JOB MARKETS
# ============================================================

JOB_MARKETS = [

    "JobNet Myanmar",

    "LinkedIn",

    "MyJobs"

]


# ============================================================
# JOB MODAL
# ============================================================

@st.dialog(
    "🔎 Find Jobs",
    width="small"
)
def show_job_modal(career):

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:18px;
            font-weight:700;
            margin-bottom:15px;
        ">
            Find Jobs
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # JOB MARKET DROPDOWN
    # --------------------------------------------------------

    selected_market = st.selectbox(
        "Select Job Market",
        JOB_MARKETS,
        key="job_market_select"
    )


    # --------------------------------------------------------
    # FIND JOB BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔎 Find Jobs",
        type="primary",
        use_container_width=True,
        key="find_job_modal_button"
    ):

        # Generate URL immediately using
        # the CURRENT dropdown value.
        job_url = get_job_url(
            career,
            selected_market
        )

        # Open the selected job market directly.
        st.link_button(
            "Open Job Search",
            job_url,
            use_container_width=True
        )

# ============================================================
# SHOW CAREER RECOMMENDATIONS
# ============================================================
if st.session_state.recommendations:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🎯 Recommended Careers'
        '</div>',
        unsafe_allow_html=True
    )

    recommendations = (
        st.session_state.recommendations
    )

    columns = st.columns(
        len(recommendations)
    )

    # Existing recommendation card code continues here


    recommendations = (
        st.session_state.recommendations
    )


    columns = st.columns(
        len(recommendations)
    )


    for index, item in enumerate(
        recommendations
    ):

        career = item[
            "career"
        ]

        score = item[
            "score"
        ]

        matched_skills = item[
            "matched_skills"
        ]

        skills_to_develop = item[
            "skills_to_develop"
        ]


        percentage = round(
            score * 100
        )


        icon = get_career_icon(
            career
        )


        # ====================================================
        # CARD
        # ====================================================

        with columns[index]:

            with st.container(
                border=True
            ):

                # ------------------------------------------------
                # ICON
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        font-size:40px;
                        margin-bottom:4px;
                    ">
                        {icon}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # CAREER NAME
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <h3 style="
                        text-align:center;
                        margin-top:0;
                        margin-bottom:5px;
                    ">
                        {html.escape(career)}
                    </h3>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # SCORE
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="score">
                        {percentage}%
                    </div>

                    <div style="
                        text-align:center;
                        margin-bottom:10px;
                    ">
                        <span class="score-label">
                            AI Match Score
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =================================================
                # RECOMMENDED BECAUSE
                # =================================================

                st.markdown(
                    "**💡 Recommended Because**"
                )

                st.caption(
                    "Your skills, interests and profile "
                    "show a strong alignment with this "
                    "career path."
                )


                # =================================================
                # MATCHED SKILLS
                # =================================================

                st.markdown(
                    "**✅ Matched Skills**"
                )


                if matched_skills:

                    badges_html = ""

                    for skill in matched_skills:

                        badges_html += (
                            '<span class="skill-badge matched">'
                            f'{html.escape(str(skill))}'
                            '</span>'
                        )


                    st.markdown(
                        f"""
                        <div class="skill-badge-container">
                            {badges_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.caption(
                        "No direct skill match found."
                    )


                # =================================================
                # SKILLS TO DEVELOP
                # =================================================

                st.markdown(
                    "**📈 Skills to Develop**"
                )


                if skills_to_develop:

                    badges_html = ""

                    for skill in skills_to_develop:

                        badges_html += (
                            '<span class="skill-badge develop">'
                            f'{html.escape(str(skill))}'
                            '</span>'
                        )


                    st.markdown(
                        f"""
                        <div class="skill-badge-container">
                            {badges_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.success(
                        "You already have the key skills."
                    )


                st.caption(
                    "Skills are identified from the "
                    "training data for this career."
                )


                # =================================================
                # BUTTONS
                # =================================================

                st.write("")


                button_col1, button_col2 = (
                    st.columns(2)
                )


                with button_col1:

                    if st.button(
                        "🛣️ Roadmap",
                        key=f"roadmap_{index}",
                        use_container_width=True
                    ):

                        st.session_state.roadmap_career = (
                            career
                        )

                        st.rerun()


                with button_col2:

                    if st.button(
                        "🔎 Find Jobs",
                        key=f"jobs_{index}",
                        use_container_width=True
                    ):

                        st.session_state.job_career = (
                            career
                        )

                        st.rerun()


# ============================================================
# OPEN ROADMAP MODAL
# ============================================================

if st.session_state.roadmap_career:

    career_to_show = (
        st.session_state.roadmap_career
    )

    st.session_state.roadmap_career = None

    show_roadmap_modal(
        career_to_show
    )


# ============================================================
# OPEN JOB MODAL
# ============================================================

if st.session_state.job_career:

    career_to_show = (
        st.session_state.job_career
    )

    st.session_state.job_career = None

    show_job_modal(
        career_to_show
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Career Recommendation System"
)
