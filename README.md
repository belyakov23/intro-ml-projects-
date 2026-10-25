Machine Learning Projects

A collection of machine learning projects developed in Python, covering regression, classification, computer vision and natural language processing.

The projects explore different ML algorithms, data preprocessing techniques, model training and evaluation.

Projects

1. Twitter Sentiment Classification

File: twitter_sentiment_classification.ipynb

* Task: Classify tweets into four sentiment categories: Positive, Negative, Neutral and Irrelevant.
* Methods: Text preprocessing, TF-IDF vectorisation, Logistic Regression, stratified train/test split and model evaluation.
* Results: Achieved approximately 74% test accuracy, with macro F1-score around 0.73.
* Tools: Python, scikit-learn, pandas.

2. Cats vs Dogs Image Classification (CNN)

File: cats_dogs_classification.ipynb

* Task: Train a convolutional neural network to classify images of cats and dogs.
* Methods: Image preprocessing, convolutional and max-pooling layers, binary cross-entropy loss and Adam optimiser.
* Results: Achieved approximately 89% training accuracy and 67% validation accuracy after 10 epochs. The difference suggests overfitting.
* Tools: Python, TensorFlow/Keras.

3. Housing Price Prediction — Data Preprocessing and Model Comparison

File: messy_housing_regression.ipynb

* Task: Predict housing prices using a small dataset containing missing values and categorical features.
* Methods: Data cleaning, one-hot encoding, train/test split, Linear Regression and Random Forest models.
* Evaluation: Compared models using MAE, RMSE and R².
* Results: Random Forest achieved MAE ≈ 0.88 and R² ≈ 0.97 in the experiment where incomplete rows were removed.
* Additional experiment: Compared removing incomplete observations with mean imputation. Missing values were calculated using training data only to avoid data leakage. The approaches used different test sets, so their results are not directly comparable.
* Tools: Python, pandas, scikit-learn.

4. California Housing Price Regression

File: california_housing_regression.ipynb

* Task: Predict median housing values using the California Housing dataset.
* Methods: Feature standardisation, Linear Regression, train/test split and evaluation using MAE, MSE and R².
* Visualisation: Compared actual and predicted housing values using scatter plots.
* Tools: Python, scikit-learn, pandas.

5. Wine Classification

File: wine_classification.ipynb

* Task: Classify wine samples into three categories using chemical measurements.
* Methods: Feature standardisation, Logistic Regression and multiclass classification.
* Evaluation: Accuracy, confusion matrix and classification report.
* Tools: Python, scikit-learn.

6. Salary Prediction

File: salary_regression.ipynb

* Task: Predict salaries based on years of professional experience.
* Methods: Exploratory data analysis, Linear Regression and train/test split.
* Results: Achieved approximately R² = 0.90 and MAE = 6,300 salary units on the test set.
* Tools: Python, pandas, scikit-learn.

Technical Skills Demonstrated

* Programming: Python, NumPy, pandas
* Machine Learning: Linear Regression, Logistic Regression, Random Forest, classification and regression
* Deep Learning: Convolutional Neural Networks, TensorFlow/Keras
* Natural Language Processing: TF-IDF, text classification
* Data Preprocessing: Missing-value handling, categorical encoding, feature standardisation
* Model Evaluation: Accuracy, precision, recall, F1-score, MAE, MSE, RMSE, R² and confusion matrices

Running the Projects

The projects are provided as Jupyter notebooks and can be opened in Google Colab.

Some notebooks require external datasets, with their sources described in the individual projects.
