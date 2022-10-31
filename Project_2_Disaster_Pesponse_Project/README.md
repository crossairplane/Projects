## Project 2: Disaster Response Pipeline Project

- Summary:

In recent years, whenever a disaster occurs, the information released through various platforms such as social media, news reports and so on is dramatically large. We are always overwhelmed by the massive amount of information. So how to dig out effective information for further relief has become the important topic.

In this project, we built an ETL pipeline and an Machine Learning pipeline to analyze disaster data from Figure Eight company to classify disaster messages. We send those categorized events to different appropriate disaster relief agency more efficiently.

- File structure:
  - app

    | - templates

    | |- master.html # main page of web app

    | |- go.html # classification result page of web app

    |- run.py # Flask file that runs app
  - data

    |- disaster_categories.csv # data to process

    |- disaster_messages.csv # data to process

    |- process_data.py

    |- DisasterResponse.db # database to save clean data to
  - models

    |- train_classifier.py

    |- classifier.pkl # saved model

  - images # not used in program, just a demo.
    |- main_page.png
    |- message_to_predict_genres.png
  - README.md

- Instructions:
  1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  2. Run the following command in the app's directory to run your web app.
  `python run.py`

  3. Go to http://your_localhost:3001 to vist this web app.
    eg. http://192.168.0.108:3001

- Results:
Check pictures ![main_page](https://github.com/crossairplane/Projects/blob/main/Project_2_Disaster_Pesponse_Project/images/main_page.png)
![message_to_predict](https://github.com/crossairplane/Projects/blob/main/Project_2_Disaster_Pesponse_Project/images/message_to_predict_genres.png).
