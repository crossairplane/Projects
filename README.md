# Projects
## Project 1
We anaylse tennis matches from 2011 till 2019.
- Libraries used: numpy, pandas, matplotlib, sklearn ad seaborn.
- Motivation for the project: undestanding the CRISP-DM process.
- A summary of the results of the analysis:
  - The best player for the surfac 1 is Caroline Wozniacki who won 129 matches from 2011 to 2019.
  - 37.56% matches end by the score of 2-0 for player 1.
  - The 1st serves won is more relative to the 2nd serves won. And If a tennis player has better performance of 1st serves, then she had lager probability to win.
- Necessary acknowledgments: be familiar with the rules of tennis.

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
Check pictures [main_page](https://github.com/crossairplane/Projects/blob/main/main_page.png) and [messgae](https://github.com/crossairplane/Projects/blob/main/message_to_predict_genres.png).
