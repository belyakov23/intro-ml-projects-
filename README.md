# intro-ml-projects-

Small machine learning projects in Python as preparation for an MSc in Artificial Intelligence.

## Projects

1. `california_housing_regression.ipynb`  
   - Dataset: built-in California housing dataset from scikit-learn.  
   - Task: predict median house value from features such as median income, house age, average rooms, population and latitude/longitude.  
   - Methods: train/test split, feature standardisation, Linear Regression model, evaluation with mean squared error (MSE), mean absolute error (MAE), R-squared, and a scatter plot of actual vs predicted values.
   - Key results: Linear Regression achieves a strong R-squared on the test set with reasonable MSE and MAE, and predictions align closely with actual median house values in the scatter plot.
  
2. `cats_dogs_classification.ipynb`  
   - Dataset: Kaggle "Cats and Dogs image classification" dataset with separate train/test folders for cat and dog JPEG images [web:268].  
   - Task: build a convolutional neural network to classify each image as either a cat or a dog.  
   - Methods: Keras `ImageDataGenerator` for preprocessing and train/validation split, small CNN with multiple conv–max pooling blocks, training with binary cross-entropy and Adam, evaluation on validation and test sets (test accuracy ≈ 69%).
   - Key results: CNN reaches about 85% training accuracy, around 66% validation accuracy, and roughly 69% accuracy on the held-out test set.

3. `salary_regression.ipynb`  
   - Dataset: Kaggle Salary Dataset – Simple Linear Regression, with YearsExperience and Salary columns [web:174].  
   - Task: predict salary from years of professional experience.  
   - Methods: removing an extra index column, exploratory data analysis and scatter plot, train/test split, Linear Regression model, evaluation with mean squared error (MSE), mean absolute error (MAE), R-squared (≈ 0.90), and visualisation of actual vs predicted salaries.
   - Key results: Model attains R-squared ≈ 0.90 on the test set, with mean absolute error around 6,300 salary units and a clear linear relationship between experience and salary.

4. `twitter_sentiment_classification.ipynb`  
   - Dataset: Kaggle Twitter entity-level sentiment dataset (tweet ID, entity, sentiment label, tweet text) [web:170].  
   - Task: classify tweets into four sentiment classes (Irrelevant, Negative, Neutral, Positive) based on the tweet content.  
   - Methods: data cleaning and column renaming, handling missing values, train/test split with stratification, TF-IDF text vectorisation, Logistic Regression classifier, evaluation with accuracy, precision/recall/F1-score per class, and confusion matrix heatmap (test accuracy ≈ 74%).
   - Key results: Logistic Regression with TF-IDF features achieves about 74% accuracy on the test set, with per-class F1-scores between roughly 0.67 and 0.78.

5. `wine_classification.ipynb`  
   - Dataset: built-in wine dataset from scikit-learn.  
   - Task: classify wine samples into one of three cultivars based on their chemical properties.  
   - Methods: train/test split, feature standardisation, Logistic Regression classifier, accuracy, confusion matrix and classification report.
   - Key results: Model reaches high test accuracy on the 3-class wine task and shows balanced performance across classes in the confusion matrix and classification report.
