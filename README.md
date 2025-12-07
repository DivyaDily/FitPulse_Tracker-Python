FitPulse_Tracker-Python

Description:
FitPulse Tracker is an interactive Personal Fitness Tracker built with Python and Streamlit. It helps users track their fitness metrics, calculate calories burned, monitor vital signs, and get personalized dietary and workout recommendations. The application uses machine learning models to predict calorie expenditure based on user inputs and historical exercise data.

Features:

User Profile Management: Save and view user profiles including name, age, weight, height, and BMI.

Calorie Prediction: Predict calories burned using Random Forest, Linear Regression, and Decision Tree models.

Visualizations: Graphical representation of calorie distribution and user-specific predictions.

Dietary Advice: Personalized nutrition guidance based on BMI and activity level.

Workout Recommendations: Suggested exercises based on fitness goals and current fitness level.

Interactive Dashboard: Built with Streamlit, providing a clean, user-friendly interface.

Downloadable Data: Export saved user profiles as CSV for offline tracking.

Technologies Used:

Python

Streamlit

Pandas & Numpy

Matplotlib

Scikit-learn

SQLite (for storing user profiles)

How to Run:

Clone the repository:

git clone https://github.com/YourUsername/FitPulse_Tracker-Python.git


Navigate to the project folder:

cd FitPulse_Tracker-Python


Create and activate a virtual environment (optional but recommended).

Install dependencies:

pip install streamlit pandas numpy matplotlib scikit-learn


Run the app:

streamlit run app.py


Use Cases:

Track and analyze fitness progress over time.

Predict calories burned during different workouts.

Get personalized advice for diet and exercises.

Maintain an organized database of user fitness profiles.

Future Improvements:

Add integration with wearable devices (Fitbit, Apple Watch).

Include more advanced ML models for better calorie prediction.

Add a progress tracker with graphs over time.

Mobile-friendly version with responsive design.
