from flask import Flask, request, jsonify, render_template

from flask_cors import CORS

from recommendation import recommend_career

import numpy as np
import pandas as pd



app = Flask(__name__)

CORS(app)



# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "career_dataset.csv"
)



def convert_numpy(obj):

    if isinstance(obj, np.float32):

        return float(obj)


    if isinstance(obj, np.int64):

        return int(obj)


    if isinstance(obj, list):

        return [
            convert_numpy(item)
            for item in obj
        ]


    if isinstance(obj, dict):

        return {

            key: convert_numpy(value)

            for key,value in obj.items()

        }


    return obj




def get_unique_values(column_name):

    values = set()


    for item in df[column_name].dropna():

        item = str(item)


        # remove list symbols
        item = (
            item
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )


        # support comma and semicolon
        item = item.replace(";", ",")


        parts = item.split(",")


        for value in parts:


            value = value.strip()


            if value:


                value = value.lower()

                value = value.title()


                values.add(value)



    return sorted(list(values))


ALL_SKILLS = get_unique_values(
    "Skills"
)


ALL_INTERESTS = get_unique_values(
    "Interests"
)






# ==========================================
# HOME
# ==========================================


@app.route("/")
def home():

    return render_template(

        "index.html",

        skills=ALL_SKILLS,

        interests=ALL_INTERESTS

    )







# ==========================================
# RECOMMEND API
# ==========================================


@app.route(
    "/recommend",
    methods=["POST"]
)

def recommend():


    data=request.json



    age=data.get(
        "age",
        0
    )



    if age < 18 or age > 60:


        return jsonify({

            "success":False,

            "message":
            "Age must be between 18 and 60"

        }),400






    education=data.get(

        "education",

        ""

    )



    skills=data.get(

        "skills",

        []

    )



    interests=data.get(

        "interests",

        []

    )




    result=recommend_career(

        education,

        skills,

        interests

    )





    return jsonify({

        "success":True,

        "recommendations":

            convert_numpy(result)

    })







if __name__=="__main__":


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
