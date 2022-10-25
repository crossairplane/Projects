## Project 2: Disaster Response Pipeline Project
- Motivation:
This project make a web app using both an ETL and Machine Learning Pipelines to create a model that will send messages to a specific disaster relief organization.
- Instructions:
  1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  2. Run the following command in the app's directory to run your web app.
    `python run.py`

  3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com

- Results:
Check pictures in "images/main_page" and "images/message_to_predict_genres".
