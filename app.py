
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "academic_advisor_model.pkl"

# Load the trained model when the application starts
model = joblib.load(MODEL_PATH)

FEATURES = [
    "cgpa",
    "attendance",
    "assessment_score",
    "course_load",
    "failed_courses"
]


def get_advice(category, cgpa, attendance, score, failed):
    """Generate practical advice based on the prediction."""

    advice = []

    if category == "High Support":
        advice.append(
            "Arrange a meeting with your academic adviser "
            "to review your academic progress."
        )
    elif category == "Moderate Support":
        advice.append(
            "Create a consistent weekly study timetable "
            "and review difficult topics regularly."
        )
    else:
        advice.append(
            "Maintain your study routine and continue "
            "monitoring your academic progress."
        )

    if attendance < 75:
        advice.append(
            "Work on improving your class attendance."
        )

    if score < 60:
        advice.append(
            "Practise more assessment questions and ask "
            "for help with topics you find difficult."
        )

    if cgpa < 3.0:
        advice.append(
            "Set realistic academic targets and review "
            "your performance in each course."
        )

    if failed > 0:
        advice.append(
            "Discuss a recovery plan for previously "
            "failed courses with your academic adviser."
        )

    return advice


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            cgpa = float(request.form["cgpa"])
            attendance = float(request.form["attendance"])
            score = float(request.form["assessment_score"])
            course_load = int(request.form["course_load"])
            failed = int(request.form["failed_courses"])

            # Validate user inputs
            if not 0.0 <= cgpa <= 5.0:
                raise ValueError("CGPA must be between 0 and 5.")

            if not 0 <= attendance <= 100:
                raise ValueError(
                    "Attendance must be between 0 and 100."
                )

            if not 0 <= score <= 100:
                raise ValueError(
                    "Assessment score must be between 0 and 100."
                )

            if not 1 <= course_load <= 20:
                raise ValueError(
                    "Course load must be between 1 and 20."
                )

            if not 0 <= failed <= course_load:
                raise ValueError(
                    "Failed courses must be between 0 "
                    "and the course load."
                )

            student = pd.DataFrame([{
                "cgpa": cgpa,
                "attendance": attendance,
                "assessment_score": score,
                "course_load": course_load,
                "failed_courses": failed
            }], columns=FEATURES)

            # Ask the trained model to make a prediction
            prediction = model.predict(student)[0]

            advice = get_advice(
                prediction, cgpa, attendance, score, failed
            )

            result = {
                "category": prediction,
                "advice": advice,
                "cgpa": cgpa,
                "attendance": attendance,
                "score": score
            }

        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc) or "Please enter valid values."

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
