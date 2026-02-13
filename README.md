# intro-ml-projects-

Small machine learning projects in Python as preparation for an MSc in Artificial Intelligence.

## Projects

1. `california_housing_regression.ipynb`  
   - Dataset: built-in California housing dataset from scikit-learn.  
   - Task: predict median house value from features such as median income, house age, average rooms, population and latitude/longitude.  
   - Methods: train/test split, feature standardisation, Linear Regression model, evaluation with mean squared error (MSE), mean absolute error (MAE), R-squared, and a scatter plot of actual vs predicted values.

2. `salary_regression.ipynb`  
   - Dataset: Kaggle Salary Dataset – Simple Linear Regression, with YearsExperience and Salary columns [web:174].  
   - Task: predict salary from years of professional experience.  
   - Methods: removing an extra index column, exploratory data analysis and scatter plot, train/test split, Linear Regression model, evaluation with mean squared error (MSE), mean absolute error (MAE), R-squared (≈ 0.90), and visualisation of actual vs predicted salaries.

3. `twitter_sentiment_classification.ipynb`  
   - Dataset: Kaggle Twitter entity-level sentiment dataset (tweet ID, entity, sentiment label, tweet text) [web:170].  
   - Task: classify tweets into four sentiment classes (Irrelevant, Negative, Neutral, Positive) based on the tweet content.  
   - Methods: data cleaning and column renaming, handling missing values, train/test split with stratification, TF-IDF text vectorisation, Logistic Regression classifier, evaluation with accuracy, precision/recall/F1-score per class, and confusion matrix heatmap (test accuracy ≈ 74%).

4. `wine_classification.ipynb`  
   - Dataset: built-in wine dataset from scikit-learn.  
   - Task: classify wine samples into one of three cultivars based on their chemical properties.  
   - Methods: train/test split, feature standardisation, Logistic Regression classifier, accuracy, confusion matrix and classification report.
