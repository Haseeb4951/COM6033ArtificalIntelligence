Diabetes Prediction Model
Project Overview|:
This project is designed to predict diabetes progression using a machine learning model. The goal is to predict how diabetes progresses based on various health factors.

We used several machine learning models and evaluated their performances. After testing different models, we focus on Gradient Boosting for the best results.

Dataset:
Pregnancies: Number of pregnancies
Glucose: Glucose concentration
BloodPressure: Blood pressure
SkinThickness: Skin thickness
Insulin: Insulin level
BMI: Body mass index
DiabetesPedigreeFunction: Family history of diabetes
Age: Age of the patient
Outcome: Diabetes progression score (the target variable)

Steps to Run the Code:
1. Clone or Download the Repository
Download the project to your local machine:
bash
Copy code
git clone <repository_link>

2. Install Dependencies
To run the code, you’ll need the following Python libraries:
pandas: For data manipulation
numpy: For numerical operations
scikit-learn: For building machine learning models
joblib: To save and load models
Install the dependencies by running this in your terminal:
bash
Copy code
pip install -r requirements.txt

3. Running the Code
The code is divided into different steps, starting with data loading and cleaning, followed by model training and evaluation.
You can open the Jupyter notebook (notebook.ipynb) in Visual Studio Code to see the code and run each cell step by step.

4. Model Evaluation
We test several models like Linear Regression, Random Forest, and Gradient Boosting. We then evaluated the models using Mean Squared Error (MSE) and R-squared to see how well the model predicts diabetes progression.

5. Save the Model
Once the best model is chosen, it is saved using joblib to make predictions later without retraining the model.
joblib.dump(best_model, 'diabetes_model.pkl')

Resources used for this project were:
W3School
https://www.w3schools.com/

