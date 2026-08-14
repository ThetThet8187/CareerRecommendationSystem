import joblib
import numpy as np
import pandas as pd
import math

from collections import Counter

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# LOAD ONNX MODEL
# =====================================================

MODEL_PATH = "thetthettun/career-sbert-onnx"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

onnx_model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_PATH
)

print("ONNX model loaded successfully!")


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

embeddings = np.load(
    "models/career_embeddings.npy"
)


# =====================================================
# LOAD CAREER DATA
# =====================================================

career_df = pd.read_pickle(
    "models/career_data.pkl"
)


# =====================================================
# LOAD CAREER MAP
# =====================================================

career_map = joblib.load(
    "models/career_mapping.pkl"
)

# =====================================================
# ONNX SBERT ENCODING
# =====================================================

def encode_text(text):

    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=256
    )

    outputs = onnx_model(
        **inputs
    )

    token_embeddings = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]

    mask = attention_mask[..., None]

    masked_embeddings = (
        token_embeddings * mask
    )

    summed = masked_embeddings.sum(
        axis=1
    )

    counts = np.clip(
        mask.sum(axis=1),
        a_min=1e-9,
        a_max=None
    )

    embeddings_output = (
        summed / counts
    )

    # Normalize
    norms = np.linalg.norm(
        embeddings_output,
        axis=1,
        keepdims=True
    )

    embeddings_output = (
        embeddings_output
        /
        np.clip(
            norms,
            a_min=1e-12,
            a_max=None
        )
    )

    return embeddings_output

# =====================================================
# HELPER
# =====================================================

def clean_skill_list(value):

    if pd.isna(value):
        return []

    value = str(value)

    value = (
        value
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    value = value.replace(";", ",")

    skills = []

    for skill in value.split(","):

        skill = skill.strip()

        if skill:
            skills.append(skill)

    return skills


# =====================================================
# NORMALIZE SKILL
# =====================================================

def normalize_skill(skill):

    return (
        str(skill)
        .strip()
        .lower()
    )


# =====================================================
# SKILL MATCH
# =====================================================

def skills_match(
    required_skill,
    candidate_skill
):

    required = normalize_skill(
        required_skill
    )

    candidate = normalize_skill(
        candidate_skill
    )

    # Exact match
    if required == candidate:
        return True

    # One contains the other
    if (
        required in candidate
        or
        candidate in required
    ):
        return True

    return False


# =====================================================
# CREATE CAREER SKILL MAP
# =====================================================

career_skill_map = {}

MAX_REQUIRED_SKILLS = 15


for career in career_df[
    "Recommended_Career"
].dropna().unique():

    career_rows = career_df[
        career_df["Recommended_Career"] == career
    ]

    skill_counter = Counter()

    for value in career_rows["Skills"]:

        for skill in clean_skill_list(value):

            skill = skill.strip()

            if skill:
                skill_counter[skill] += 1

    # -------------------------------------------------
    # Most Common Skills
    # -------------------------------------------------

    top_skills = [
        skill
        for skill, count
        in skill_counter.most_common(
            MAX_REQUIRED_SKILLS
        )
    ]

    career_skill_map[career] = top_skills


# =====================================================
# RECOMMEND CAREER
# =====================================================

def recommend_career(
    education,
    skills,
    interests,
    top_n=3
):

    # =================================================
    # SAFETY
    # =================================================

    if not skills:
        skills = []

    if not interests:
        interests = []

    if not education:
        education = ""


    # =================================================
    # CANDIDATE TEXT
    # =================================================

    text = (
        " ".join(skills)
        + " "
        + " ".join(interests)
        + " "
        + education
    )


    # =================================================
    # SBERT ENCODE
    # =================================================

    vector = encode_text(text)


    # =================================================
    # COSINE SIMILARITY
    # =================================================

    similarity = cosine_similarity(
        vector,
        embeddings
    )[0]


    # =================================================
    # GET TOP NEAREST PROFILES
    # =================================================

    top_indices = (
        similarity
        .argsort()[-30:][::-1]
    )


    # =================================================
    # CLUSTER SCORES
    # =================================================

    cluster_scores = {}

    cluster_counts = {}


    for i in top_indices:

        cluster = career_df.iloc[i][
            "HDBSCAN_Cluster"
        ]

        # Ignore HDBSCAN noise
        if cluster == -1:
            continue

        score = float(
            similarity[i]
        )

        cluster_scores[cluster] = (
            cluster_scores.get(
                cluster,
                0
            )
            + score
        )

        cluster_counts[cluster] = (
            cluster_counts.get(
                cluster,
                0
            )
            + 1
        )


    # =================================================
    # AVERAGE CLUSTER SIMILARITY
    # =================================================

    for cluster in cluster_scores:

        cluster_scores[cluster] = (
            cluster_scores[cluster]
            /
            cluster_counts[cluster]
        )


    # =================================================
    # SORT CLUSTERS
    # =================================================

    sorted_clusters = sorted(
        cluster_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # =================================================
    # CAREER SCORES
    # =================================================

    career_scores = {}


    for cluster, score in sorted_clusters:

        cluster_careers = career_map.get(
            cluster,
            {}
        )

        for career, probability in cluster_careers.items():

            career_scores[career] = (
                career_scores.get(
                    career,
                    0
                )
                +
                score
                *
                (
                    probability
                    /
                    100
                )
            )


    # =================================================
    # NO CAREER FOUND
    # =================================================

    if not career_scores:

        return []


    # =================================================
    # SORT CAREERS BY AI SCORE
    # =================================================

    sorted_careers = sorted(
        career_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # =================================================
    # CANDIDATE SKILLS
    # =================================================

    candidate_skills = [
        normalize_skill(skill)
        for skill in skills
    ]


    # =================================================
    # VALID CAREERS
    #
    # Only careers with at least one
    # matched skill will continue.
    # =================================================

    valid_careers = []


    for career, semantic_score in sorted_careers:

        # -------------------------------------------------
        # Required Skills
        # -------------------------------------------------

        required_skills = (
            career_skill_map.get(
                career,
                []
            )
        )


        # -------------------------------------------------
        # Matched Skills
        # -------------------------------------------------

        matched_skills = []


        for required_skill in required_skills:

            for candidate_skill in skills:

                if skills_match(
                    required_skill,
                    candidate_skill
                ):

                    matched_skills.append(
                        required_skill
                    )

                    break


        # -------------------------------------------------
        # Remove Duplicates
        # -------------------------------------------------

        matched_skills = list(
            dict.fromkeys(
                matched_skills
            )
        )


        # =================================================
        # FILTER
        #
        # No matched skill = no recommendation
        # =================================================

        if len(matched_skills) == 0:

            continue


        # =================================================
        # SKILL MATCH RATIO
        # =================================================

        if required_skills:

            skill_match_ratio = (
                len(matched_skills)
                /
                len(required_skills)
            )

        else:

            skill_match_ratio = 0


        # =================================================
        # FINAL SCORE
        #
        # AI Match       = 70%
        # Skill Match    = 30%
        # =================================================

        ai_weight = 0.70

        skill_weight = 0.30


        final_score = (
            float(semantic_score)
            *
            ai_weight
        ) + (
            float(skill_match_ratio)
            *
            skill_weight
        )


        # =================================================
        # SKILLS TO DEVELOP
        # =================================================

        skills_to_develop = [

            skill

            for skill in required_skills

            if not any(

                skills_match(
                    skill,
                    candidate_skill
                )

                for candidate_skill in skills

            )

        ]


        # -------------------------------------------------
        # Remove duplicates
        # -------------------------------------------------

        skills_to_develop = list(
            dict.fromkeys(
                skills_to_develop
            )
        )


        # =================================================
        # SAVE VALID CAREER
        # =================================================

        valid_careers.append({

            "career":
                career,

            "semantic_score":
                float(semantic_score),

            "skill_match_ratio":
                float(skill_match_ratio),

            "final_score":
                float(final_score),

            "required_skills":
                required_skills,

            "matched_skills":
                matched_skills,

            "skills_to_develop":
                skills_to_develop

        })


    # =================================================
    # NO VALID CAREER
    # =================================================

    if not valid_careers:

        return []


    # =================================================
    # SORT BY FINAL SCORE
    # =================================================

    valid_careers.sort(

        key=lambda x:
        x["final_score"],

        reverse=True

    )


    # =================================================
    # TOP 3 VALID CAREERS
    # =================================================

    valid_careers = valid_careers[
        :top_n
    ]


    # =================================================
    # AI MATCH SCORE CALIBRATION
    # =================================================
    #
    # Convert final ranking scores
    # into percentage-style confidence.
    #
    # =================================================

    raw_scores = [

        float(
            item["final_score"]
        )

        for item in valid_careers

    ]


    temperature = 0.20


    max_raw_score = max(
        raw_scores
    )


    exp_scores = [

        math.exp(

            (
                score
                -
                max_raw_score
            )
            /
            temperature

        )

        for score in raw_scores

    ]


    total_exp = sum(
        exp_scores
    )


    # =================================================
    # PREVENT DIVISION BY ZERO
    # =================================================

    if total_exp == 0:

        ai_match_scores = [

            100
            /
            len(valid_careers)

            for _ in valid_careers

        ]

    else:

        ai_match_scores = [

            (
                value
                /
                total_exp
            )
            *
            100

            for value in exp_scores

        ]


    # =================================================
    # BUILD FINAL RESPONSE
    # =================================================

    response = []


    for index, item in enumerate(
        valid_careers
    ):

        # -------------------------------------------------
        # Rank
        # -------------------------------------------------

        rank = index + 1


        # -------------------------------------------------
        # Career
        # -------------------------------------------------

        career = item[
            "career"
        ]


        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = round(
            ai_match_scores[index],
            2
        )


        # -------------------------------------------------
        # Required Skills
        # -------------------------------------------------

        required_skills = item[
            "required_skills"
        ]


        # -------------------------------------------------
        # Matched Skills
        # -------------------------------------------------

        matched_skills = item[
            "matched_skills"
        ]


        # -------------------------------------------------
        # Skills To Develop
        # -------------------------------------------------

        skills_to_develop = item[
            "skills_to_develop"
        ]


        # =================================================
        # FINAL RESPONSE
        # =================================================

        response.append({

            "rank":
                rank,

            "career":
                career,

            "confidence":
                confidence,

            "required_skills":
                required_skills[:10],

            "matched_skills":
                matched_skills,

            "skills_to_develop":
                skills_to_develop[:10]

        })


    # =================================================
    # RETURN
    # =================================================

    return response