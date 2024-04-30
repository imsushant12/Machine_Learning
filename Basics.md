### What is Machine Learning?
Machine learning is a subfield of Artificial Intelligence (AI) that allows computers to learn from data without explicit programming. Instead of writing out every instruction, machines can improve at a specific task by analyzing data. This data can be used to predict future values, classify information, or find hidden patterns.

#### Types of Machine Learning
There are two main categories of machine learning: **supervised learning** and **unsupervised learning**. These categories are determined by the type of data and the learning objective.
* **Supervised Learning**: In supervised learning, the data is labeled. This means each data point has a corresponding label or target value that the model is trying to predict. The model learns the relationship between the features (inputs) and the target variables (labels) during training. Then, it can use this learned relationship to predict the target variable for new, unseen data points. 
    * Common supervised learning tasks include classification (e.g., spam detection, image recognition) and regression (e.g., weather forecasting, stock price prediction).

* **Unsupervised Learning**: In unsupervised learning, the data is unlabeled. The model does not have any predefined labels or target values. The goal of unsupervised learning is to uncover hidden patterns or structures within the data itself. These patterns can be used for tasks like anomaly detection, customer segmentation, or dimensionality reduction.
    * Common unsupervised learning tasks include clustering (e.g., grouping similar data points together) and dimensionality reduction (e.g., reducing the number of features without losing important information).

#### Common Terms in Machine Learning
* **Data** (Input): The raw information used to train a machine learning model. Data can come in many forms like numbers, text, or images.
* **Model**: The representation of the learned knowledge from the data. This can be a mathematical equation, a computer program, or a set of rules. 
* **Training** (Learning): The process of feeding data to the model and adjusting its internal parameters to improve performance on a specific task. 
* **Prediction** (Output): The model's output after being trained on data. Predictions can be used for various purposes like forecasting future values or classifying new data points.
* **Features** (Attributes): The individual attributes or measurements used to describe a data point. These features are fed into the model to make predictions.
* **Target Variable (Label)**: The value the model is trying to predict. This can be a numerical value or a category label. In essence, it's the **desired outcome** the model is trying to learn and predict based on the features. For example, in an image classification task, the features might be pixel values of an image, and the label could be the category of the object in the image (e.g., cat, dog, car).

#### Commonly Used Machine Learning Algorithms (Basic Definitions)
* **Linear Regression**: This algorithm finds a linear relationship between features and a target variable. It's often used for prediction tasks like forecasting future sales or stock prices.
* **Logistic Regression**: Similar to linear regression, but used for predicting binary outcomes (Yes/No, True/False). 
* **K-Nearest Neighbors (KNN)**: This algorithm classifies data points based on the similarity to their nearest neighbors in the training data.
* **Decision Trees**: This algorithm makes predictions by following a tree-like structure of questions based on the features. It's good for interpretable models where you want to understand the reasoning behind the prediction.
* **Support Vector Machines (SVM)**: This algorithm creates a hyperplane that best separates data points of different categories. It's powerful for classification tasks with high-dimensional data.
* **Random Forest**: This ensemble method combines multiple decision trees for improved prediction accuracy and to prevent overfitting the training data.

---
### Linear Regression
Linear regression is a fundamental statistical technique used for modeling the relationship between a dependent variable (often denoted as \( y \)) and one or more independent variables (often denoted as \( x \)). It assumes that there is a linear relationship between the independent variable(s) and the dependent variable. The goal of linear regression is to find the best-fitting straight line that describes this relationship.

The equation of a simple linear regression model with one independent variable can be expressed as:

\[ y = mx + c \]

Where:
- \( y \) is the dependent variable.
- \( x \) is the independent variable.
- \( m \) is the slope of the line (also known as the coefficient of the independent variable).
- \( c \) is the intercept of the line (the value of \( y \) when \( x \) is 0).

The coefficients \( m \) and \( c \) are estimated from the data to minimize the error in predicting \( y \) from \( x \).

#### Best Fit Line
The best fit line is the straight line that best represents the relationship between the independent and dependent variables. It minimizes the difference between the observed values and the values predicted by the line. In other words, it is the line that minimizes the sum of the squared differences between the actual values of the dependent variable and the values predicted by the line.

#### Types of Linear Regression
1. **Simple Linear Regression**: In simple linear regression, there is only one independent variable. The equation of the line is \( y = mx + c \), where \( m \) is the slope and \( c \) is the intercept.
2. **Multiple Linear Regression**: In multiple linear regression, there are multiple independent variables. The equation of the linear model is extended to accommodate multiple predictors:
![multiple linear regression](https://imgur.com/Vloky0u.png)

### Important Algorithms and Terminologies related to Linear Regression and Machine Learning
#### Cost or loss function
The cost or loss function, also known as the objective function, is a mathematical function that quantifies the difference between the predicted values of a model and the actual values of the target variable (dependent variable) in a dataset. It represents how well the model is performing on the given data.

In the context of linear regression, the most commonly used cost or loss function is the Mean Squared Error (MSE). The MSE measures the average squared difference between the predicted values and the actual values of the target variable.

#### Mean Squared Error
MSE stands for Mean Squared Error. It is a commonly used metric for evaluating the performance of a regression model, including linear regression.

Mathematically, Mean Squared Error (MSE) is calculated as the average of the squared differences between the actual values (observed values) and the predicted values of the dependent variable. It provides a measure of the average squared deviation of the predictions from the actual values.

#### Gradient Descent
Gradient Descent is an iterative optimization algorithm used to find the minimum of a function. In the context of linear regression, it can be used to minimize the cost function associated with the model parameters (e.g., MSE). It iteratively updates the parameters in the direction of the steepest descent of the cost function until convergence is reached.

#### Stochastic Gradient Descent 
Stochastic Gradient Descent (SGD) is a variant of the gradient descent algorithm commonly used in training machine learning models, especially when dealing with large datasets. Unlike regular gradient descent, which computes the gradient of the cost function using the entire dataset, SGD updates the model parameters using only a single data point (or a small subset) at a time. This makes SGD faster and more scalable, especially for large datasets.

#### Classifier
It refers to a type of model used for classification tasks. A classifier is a supervised learning algorithm that assigns a class label to input data based on its features. The goal of classification is to learn a mapping from input features to class labels.

#### K-Nearest Neighbors
K-Nearest Neighbors (KNN) is a ML algorithm used for classification and regression tasks. It determines the class or value of a new data point by examining the `k` nearest neighbors in the feature space and using their information. 

The choice of `k` influences model performance, with larger `k` values providing smoother decision boundaries but potentially overlooking local patterns.
- KNN is simple to understand and implement.
- KNN memorizes the entire training dataset and uses it during prediction.
- KNN measures the similarity between data points using distance metrics.
- KNN's computational complexity grows with dataset size due to distance calculations. 

#### One Hot Encoding
One-hot encoding is a technique used in machine learning to convert categorical variables into a numerical format that can be provided to machine learning algorithms to improve performance. It involves representing each categorical variable as a binary vector.

**Example**: A list of colors: [`red`, `blue`, `green`] is given, with one-hot encoding, it becomes:
- `red` becomes `[1, 0, 0]`
- `blue` becomes `[0, 1, 0]`
- `green` becomes `[0, 0, 1]`

Each position in the vector corresponds to a unique category, and only one position is "hot" (i.e., set to 1) to indicate the presence of that category. This encoding ensures that machine learning algorithms can effectively interpret and utilize categorical data in their calculations.

#### Overfitting
It occurs when a model learns the training data too well, capturing noise and random fluctuations in the data rather than the underlying pattern. As a result, an overfitted model performs well on the training data but poorly on unseen data. It essentially memorizes the training data instead of learning the generalizable patterns.

**Analogy**: Imagine a student who memorizes the answers to a specific set of practice questions without understanding the underlying concepts. When faced with similar but slightly different questions in the exam, the student struggles to answer correctly because they haven't truly learned the material.

#### Underfitting
It happens when a model is too simple to capture the underlying structure of the data. It fails to learn the patterns present in the training data and performs poorly both on the training data and unseen data. An underfitted model lacks the complexity to represent the relationship between the features and the target variable adequately.

**Analogy**: Consider a student who only studies the first chapter of a textbook before taking an exam that covers the entire book. Without learning the full breadth of the material, the student is ill-prepared to answer questions on topics beyond the first chapter.

> **Note**: Techniques such as cross-validation, regularization, and feature engineering can help mitigate overfitting and underfitting, leading to more robust and generalizable models.

---

### Classification
Classification in machine learning is a type of supervised learning where the goal is to predict the category or class of a given input data point. It involves learning a mapping from input features to discrete output labels. The output labels can represent different classes, categories, or groups to which the input data points belong. Classification is widely used in various applications such as spam detection, sentiment analysis, image recognition, medical diagnosis, and fraud detection.

#### Logistic Regression
Logistic Regression is a popular algorithm used for binary classification tasks, where the target variable has only two possible outcomes (e.g., yes/no, spam/not spam, positive/negative). Despite its name, logistic regression is a linear model for classification rather than regression. It predicts the probability that a given input belongs to a particular class.

There are different types of classification algorithms, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Neural Networks
- SVM

#### Working of Logistic Regression
1. **Maps linear relationship to probabilities**: It takes in features (independent variables) from data and uses a linear regression model internally. However, instead of directly predicting the target variable, it transforms the linear relationship into probabilities between 0 and 1 using a mathematical function called the sigmoid function.
2. **Sigmoid function**: This S-shaped function squashes the output values from the linear model to a range between 0 and 1. A value closer to 1 represents a higher probability of belonging to the positive class, and a value closer to 0 represents a higher probability of belonging to the negative class.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

where:
* `x` is the input value to the function.
* `e` is the base of the natural logarithm (approximately 2.71828).
