from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from recommendation import recommend_career

import numpy as np
import pandas as pd
import os


# =====================================================
# APP
# =====================================================

app = Flask(__name__)

CORS(app)


# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Trained dataset",
    "career_dataset.csv"
)


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    DATASET_PATH
)


# =====================================================
# CONVERT NUMPY
# =====================================================

def convert_numpy(obj):

    if isinstance(
        obj,
        (
            np.float16,
            np.float32,
            np.float64
        )
    ):

        return float(obj)


    if isinstance(
        obj,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64
        )
    ):

        return int(obj)


    if isinstance(obj, list):

        return [
            convert_numpy(item)
            for item in obj
        ]


    if isinstance(obj, dict):

        return {

            key: convert_numpy(value)

            for key, value in obj.items()

        }


    return obj


# =====================================================
# GET UNIQUE VALUES
# =====================================================

def get_unique_values(column_name):

    values = set()


    for item in df[
        column_name
    ].dropna():

        item = str(item)


        # Remove list symbols

        item = (
            item
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )


        # Support comma and semicolon

        item = item.replace(
            ";",
            ","
        )


        parts = item.split(",")


        for value in parts:

            value = value.strip()


            if value:

                value = value.lower()

                value = value.title()

                values.add(value)


    return sorted(
        list(values)
    )


# =====================================================
# LOAD SKILLS / INTERESTS
# =====================================================

ALL_SKILLS = get_unique_values(
    "Skills"
)


ALL_INTERESTS = get_unique_values(
    "Interests"
)


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return render_template(

        "index.html",

        skills=ALL_SKILLS,

        interests=ALL_INTERESTS

    )


# =====================================================
# RECOMMEND API
# =====================================================

@app.route(
    "/recommend",
    methods=["POST"]
)

def recommend():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "success": False,

            "message":
            "Invalid request data"

        }), 400


    # =================================================
    # AGE
    # =================================================

    try:

        age = int(
            data.get(
                "age",
                0
            )
        )

    except (
        ValueError,
        TypeError
    ):

        age = 0


    if age < 18 or age > 60:

        return jsonify({

            "success": False,

            "message":
            "Age must be between 18 and 60"

        }), 400


    # =================================================
    # INPUT
    # =================================================

    education = data.get(
        "education",
        ""
    )


    skills = data.get(
        "skills",
        []
    )


    interests = data.get(
        "interests",
        []
    )


    # =================================================
    # SAFETY
    # =================================================

    if not isinstance(
        skills,
        list
    ):

        skills = []


    if not isinstance(
        interests,
        list
    ):

        interests = []


    # =================================================
    # RECOMMEND
    # =================================================

    try:

        result = recommend_career(

            education,

            skills,

            interests

        )


    except Exception as e:

        print(
            "Recommendation error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "message":
            "Recommendation failed"

        }), 500


    # =================================================
    # RESPONSE
    # =================================================

    return jsonify({

        "success": True,

        "recommendations":
        convert_numpy(result)

    })


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route(
    "/health",
    methods=["GET"]
)

def health():

    return jsonify({

        "status": "ok"

    })


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )