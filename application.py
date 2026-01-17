from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import  CustomData,Predictionpipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race/ethnicity'),
            parental_level_of_education=request.form.get('parental level of education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test preparation course'),
            math_score=int(request.form.get('math score')),
            reading_score=int(request.form.get('reading score')),
            writing_score=int(request.form.get('writing score'))
        )
        final_df = data.get_data_as_dataframe()

        predict_pipeline = Predictionpipeline()
        result = predict_pipeline.prediction(final_df)

        return render_template(
            'home.html',
            result=f"Predicted Score: {round(result[0], 2)}"
        )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
