### What is Machine Learning?
Machine learning is a subfield of Artificial Intelligence (AI) that allows computers to learn from data without explicit programming. Instead of writing out every instruction, machines can improve at a specific task by analysing data. This data can predict future values, classify information, or find hidden patterns.

#### Types of Machine Learning
There are two main categories of machine learning: **supervised learning** and **unsupervised learning**. These categories are determined by the type of data and the learning objective.
* **Supervised Learning**: The data is labelled in supervised learning. This means each data point has a corresponding label or target value that the model tries to predict. During training, the model learns the relationship between the features (inputs) and the target variables (labels). Then, it can use this learned relationship to predict the target variable for new, unseen data points. 
    * Common supervised learning tasks include classification (e.g., spam detection, image recognition) and regression (e.g., weather forecasting, stock price prediction).

* **Unsupervised Learning**: The data is unlabeled in unsupervised learning. The model does not have any predefined labels or target values. Unsupervised learning aims to uncover hidden patterns or structures within the data itself. These patterns can be used for anomaly detection, customer segmentation, or dimensionality reduction.
    * Common unsupervised learning tasks include clustering (e.g., grouping similar data points) and dimensionality reduction (e.g., reducing the number of features without losing essential information).

#### Common Terms in Machine Learning
* **Data** (Input): The raw information used to train a machine learning model. Data can take many forms, such as numbers, text, or images.
* **Model**: A representation of the knowledge learned from the data. This can be a mathematical equation, a computer program, or a set of rules. 
* **Training** (Learning): The process of feeding data to the model and adjusting its internal parameters to improve performance on a specific task. 
* **Prediction** (Output): The model's output after being trained on data. Predictions can be used for various purposes, like forecasting future values or classifying new data points.
* **Features** (Attributes): The individual attributes or measurements used to describe a data point. These features are fed into the model to make predictions.
* **Target Variable (Label)**: The value the model is trying to predict. This can be a numerical value or a category label. In essence, it's the **desired outcome** the model is trying to learn and predict based on the features. For example, in an image classification task, the features might be pixel values of an image, and the label could be the category of the object in the image (e.g., cat, dog, car).

#### Commonly Used Machine Learning Algorithms (Basic Definitions)
* **Linear Regression**: This algorithm finds a linear relationship between features and a target variable. It's often used to predict future sales or stock prices.
* **Logistic Regression**: This is similar to linear regression but is used to predict binary outcomes (Yes/No, True/False). 
* **K-Nearest Neighbors (KNN)**: This algorithm classifies data points based on the similarity to their nearest neighbours in the training data.
* **Decision Trees**: This algorithm makes predictions by following a tree-like structure of questions based on the features. It's suitable for interpretable models where you want to understand the reasoning behind the prediction.
* **Support Vector Machines (SVM)**: This algorithm creates a hyperplane that best separates data points of different categories. It's powerful for classification tasks with high-dimensional data.
* **Random Forest**: This ensemble method combines multiple decision trees to improve prediction accuracy and prevent overfitting the training data.

---

### Linear Regression
Linear regression is a fundamental statistical technique used for modelling the relationship between a dependent variable (often denoted as \(y\)) and one or more independent variables (often denoted as \(x\)). It assumes a linear relationship exists between the independent variable(s) and the dependent variable. Linear regression aims to find the best-fitting straight line that describes this relationship.

The equation of a simple linear regression model with one independent variable can be expressed as:

\[ y = mx + c \]

Where:
- \(y\) is the dependent variable.
- \(x\) is the independent variable.
- \(m\) is the slope of the line (also known as the coefficient of the independent variable).
- \(c\) is the intercept of the line (the value of \(y\) when \(x\) is 0).

The coefficients \(m\) and \(c\) are estimated from the data to minimise the error in predicting \(y\) from \(x\).

#### Best Fit Line
The best-fit line is the straight line that best represents the relationship between the independent and dependent variables. It minimises the difference between the observed values and the values predicted by the line. In other words, the line reduces the sum of the squared differences between the actual values of the dependent variable and the values predicted by the line.

#### Types of Linear Regression
1. **Simple Linear Regression**: There is only one independent variable in simple linear regression. The equation of the line is \( y = mx + c \), where \(m\) is the slope and \(c\) is the intercept.
2. **Multiple Linear Regression**: Multiple linear regression has numerous independent variables. The equation of the linear model is extended to accommodate various predictors:
![multiple linear regression](https://imgur.com/Vloky0u.png)

---

### Important Algorithms and Terminologies related to Linear Regression and Machine Learning
#### Cost or Loss function
The cost or loss function, also known as the objective function, is a mathematical function that quantifies the difference between the predicted values of a model and the actual values of the target variable (dependent variable) in a dataset. It represents how well the model is performing on the given data.

The most commonly used cost or loss function in linear regression is the Mean Squared Error (MSE). The MSE measures the average squared difference between the predicted values and the actual values of the target variable.

#### Mean Squared Error
MSE stands for Mean Squared Error. It is a commonly used metric for evaluating the performance of a regression model, including linear regression.

Mathematically, Mean Squared Error (MSE) is calculated as the average of the squared differences between the actual values (observed values) and the predicted values of the dependent variable. It measures the average squared deviation of the predictions from the actual values.

#### Gradient Descent
Gradient Descent is an iterative optimisation algorithm used to find the minimum of a function. In the context of linear regression, it can minimise the cost function associated with the model parameters (e.g., MSE). It iteratively updates the parameters in the direction of the steepest descent of the cost function until convergence is reached.

#### Stochastic Gradient Descent 
Stochastic Gradient Descent (SGD) is a variant of the gradient descent algorithm commonly used in training machine learning models, especially when dealing with large datasets. Unlike regular gradient descent, which computes the gradient of the cost function using the entire dataset, SGD updates the model parameters using only a single data point (or a small subset) at a time. This makes SGD faster and more scalable, especially for large datasets.

#### Classifier
It refers to a type of model used for classification tasks. A classifier is a supervised learning algorithm that assigns a class label to input data based on its features. Classification aims to learn a mapping from input features to class labels.

#### K-Nearest Neighbors
K-Nearest Neighbors (KNN) is an ML algorithm for classification and regression tasks. It determines the class or value of a new data point by examining the `k` nearest neighbours in the feature space and using their information. 

The choice of `k` influences model performance, with larger `k` values providing smoother decision boundaries but potentially overlooking local patterns.
- KNN is simple to understand and implement.
- KNN memorises the entire training dataset and uses it during prediction.
- KNN measures the similarity between data points using distance metrics.
- KNN's computational complexity grows with dataset size due to distance calculations. 

#### One Hot Encoding
One-hot encoding is a technique used in machine learning to convert categorical variables into a numerical format that can be provided to machine learning algorithms to improve performance. It involves representing each categorical variable as a binary vector.

**Example**: A list of colours: [`red`, `blue`, `green`] is given, with one-hot encoding, it becomes:
- `red` becomes `[1, 0, 0]`
- `blue` becomes `[0, 1, 0]`
- `green` becomes `[0, 0, 1]`

Each position in the vector corresponds to a unique category, and only one position is "hot" (i.e., set to 1) to indicate the presence of that category. This encoding ensures that machine learning algorithms can effectively interpret and utilise categorical data in their calculations.

#### Overfitting
It occurs when a model learns the training data too well, capturing noise and random fluctuations rather than the underlying pattern. As a result, an overfitted model performs well on the training data but needs to improve on unseen data. It memorises the training data instead of learning the generalisable patterns.

**Analogy**: Imagine a student who memorises the answers to a specific set of practice questions and needs help understanding the underlying concepts. When faced with similar but slightly different questions in the exam, the student struggles to answer correctly because they have yet to learn the material.

#### Underfitting
It happens when a model is too simple to capture the underlying structure of the data. It needs to learn the patterns in the training data and perform better on the training and unseen data. 

An underfitted model needs more complexity to adequately represent the relationship between the features and the target variable.

**Analogy**: Consider a student who only studies the first chapter of a textbook before taking an exam that covers the entire book. The student needs to learn the full breadth of the material to prepare to answer questions on topics beyond the first chapter.

> **Note**: Techniques such as cross-validation, regularisation, and feature engineering can help mitigate overfitting and underfitting, leading to more robust and generalisable models.

---

### Classification
Classification in machine learning is a type of supervised learning where the goal is to predict the category or class of a given input data point. It involves learning to map from input features to discrete output labels. The output labels can represent different classes, categories, or groups to which the input data points belong. Classification is widely used in various applications such as spam detection, sentiment analysis, image recognition, medical diagnosis, and fraud detection.

#### Logistic Regression
Logistic Regression is a popular algorithm used for binary classification tasks, where the target variable has only two possible outcomes (e.g., yes/no, spam/not spam, positive/negative). Despite its name, logistic regression is a linear classification model rather than regression. It predicts the probability that a given input belongs to a particular class.

There are different types of classification algorithms, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Neural Networks
- SVM

#### Working on Logistic Regression
1. **Maps linear relationship to probabilities**: It takes features (independent variables) from data and uses a linear regression model internally. However, instead of directly predicting the target variable, it transforms the linear relationship into probabilities between `0` and `1` using a mathematical function called the sigmoid function.
2. **Sigmoid function**: This S-shaped function squashes the output values from the linear model to a range between `0` and `1`. A value closer to `1` represents a higher probability of belonging to the positive class, and a value closer to `0` represents a higher probability of belonging to the negative class.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Where:
* `x` is the input value to the function.
* `e` is the base of the natural logarithm (approximately 2.71828).

#### Decision Tree
A decision tree is a supervised machine-learning algorithm for classification and regression tasks. It is a tree-like structure where each internal node represents a "decision" based on the value of a feature, each branch represents the outcome of that decision, and each leaf node represents the final prediction or outcome.

**When to Use**:
- Decision trees are versatile and can be used for classification and regression tasks.
- They are instrumental when the relationship between features and target variables is non-linear or complex.
- Decision trees are easy to interpret and visualise, making them suitable for explaining the reasoning behind predictions.

However, here are some things to consider:
- **Overfitting**: Decision trees can be prone to overfitting if not carefully controlled. Techniques like pruning or setting a maximum depth can help mitigate this.
- **Not ideal for high-dimensional data**: Decision trees with many features can become complex and computationally expensive.
- **Feature importance**: While interpretable, the importance of features in a decision tree can be biased towards features that appear earlier in the splitting process.

**Choosing the Splitting Column**:
- Choosing the feature to split on at each node is crucial for building an effective decision tree.
- Different algorithms use various criteria to select the best splitting feature, such as information gain, Gini impurity, or entropy.
The goal is to select the feature that maximises the purity of the resulting child nodes, leading to better class separation or a reduction of variance.

**Entropy and Information Gain**:
- Entropy is a measure of impurity or disorder in a dataset. In the context of decision trees, entropy is used to quantify the randomness or unpredictability of the target variable's distribution at a node.
- Low entropy indicates that the samples at a node belong predominantly to one class, while high entropy indicates a more evenly distributed mix of classes.
- Information gain measures the reduction in entropy achieved by splitting a dataset on a particular feature. It represents the amount of information gained about the target variable by knowing the value of the feature.
A high information gain suggests that splitting on a particular feature results in better class separation or increased homogeneity within classes.

```
Information Gain(Feature) = Entropy(Parent) - (Weighted Average of Entropy(Children))
```

**Gini Impurity**:
- Gini impurity is another measure of impurity used in decision trees.
- It measures the probability of incorrectly classifying a randomly chosen element if it were randomly labelled according to the distribution of labels in the node.
- A lower Gini impurity indicates greater purity or homogeneity of classes at a node, while a higher Gini impurity indicates greater impurity or mixed courses.

```
Gini Impurity = 1 - Σ(pi)^2
```
Here, `pi` is the proportion of data points belonging to class `i`.

#### High Entropy/Gini Impurity
It represents a high level of uncertainty or class imbalance. Splitting on this feature will likely improve the purity of the child nodes.

#### Low Entropy/Gini Impurity
It represents a low level of uncertainty or balanced classes. Splitting on this feature might be less beneficial for further separating the data.

### Confusion Matrix
A confusion matrix is a table often used to describe the performance of a classification model on a set of test data for which the actual values are known. It allows visualisation of an algorithm's performance. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa).
- **True Positives (TP)**: These are the cases when the actual class is positive, and the model correctly predicted it.
- **True Negatives (TN)**: These are the cases when the actual class is negative, and the model predicted it correctly as unfavourable.
- **False Positives (FP)**: These are the cases when the actual class is negative, but the model predicted it as positive (Type I error).
- **False Negatives (FN)**: These are the cases when the actual class is positive, but the model predicted it as unfavourable (Type II error).

Example:
```XML
                        Actual Negative   |   Actual Positive
Predicted Negative   |        850         |        50
Predicted Positive   |        20          |        100
```

---

### Stratified shuffling 
It is a technique used in machine learning to split a dataset into training and test sets while maintaining the proportion of classes in both sets.

**Example**: Imagine we have a dataset of images:
- **Class Imbalance**: The dataset has many pictures of cats (the majority class) and very few images of birds (the minority class).
- **Traditional Shuffling**: If we simply shuffle the entire dataset randomly, the training set might consist mainly of cat pictures, neglecting the birds. This can lead to a model that performs well in classifying cats but struggles with birds.
- **Stratified Shuffling**: By shuffling each class (cats and birds) independently before splitting, we ensure both training and testing sets have a similar proportion of cats and birds, allowing the model to learn effectively from both classes.

**Code**:
```python
from sklearn.model_selection import StratifiedShuffleSplit

# Sample data (assuming you have features 'X' and target variable 'y')
X = ...  # Features
y = ...  # Target labels

# Define the split ratio for training and testing data (e.g., 80% for training)
test_size = 0.2

# Create a StratifiedShuffleSplit object
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# Split the data into training and testing sets while maintaining class distribution
# The split method of the object takes the features and target variable as input and returns indices for training and testing sets. 
for train_index, test_index in strat_split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

--- 

## Train Test Split Code
We can also code the train-test split function easily:
```python
def split(data, test_ratio):
    test_size = int(len(data) * test_ratio)
    train_size = len(data)-test_size
    return data.iloc[:train_size], data.iloc[train_size:]

train_data, test_data = split(data, 0.2)
```

We can also shuffle data using the ``random.shuffle()`` method of the ``NumPy`` module:
```python
def split(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_data, test_data = split(data, 0.2)
```

---

### Imputing
Imputing refers to filling in missing values in a dataset with estimated or calculated values. This is important because many machine learning algorithms need help to handle missing data and require complete datasets to function correctly.

Here are the common types of imputation techniques:
* **Mean/Median/Mode Imputation**: This approach replaces missing values with the mean (average), median (middle value), or mode (most frequent value) for the specific feature (column). It is simple and fast but can introduce bias if the missing data isn't randomly distributed.
* **K-Nearest Neighbors (KNN) Imputation**: This method estimates the missing value using the values of the `k` nearest neighbours (data points most similar to the one with missing values). This method can be more accurate than mean/median/mode imputation, but it's computationally expensive and requires choosing an appropriate value for `k`.
* **Interpolation**: Techniques like linear interpolation or spline interpolation estimate missing values based on existing data points. These methods work well for continuous features that have a smooth underlying trend.
* **Model-Based Imputation**: Trains a separate model (e.g., decision tree) to predict missing values based on other features in the dataset. This can be effective, but it requires careful model selection and evaluation.

### `Scikit-learn` Object Types
`Scikit-learn` provides various object types for different machine learning tasks:

* **Estimators**: These objects learn a model from your data. Examples include classification algorithms like `LogisticRegression`, regression algorithms like `LinearRegression`, clustering algorithms like `KMeans`, and dimensionality reduction algorithms like `PCA`.
* **Transformers**: These objects pre-process data, such as scaling features or encoding categorical variables. Examples include `StandardScaler` for feature scaling, `OneHotEncoder` for categorical encoding, and `Imputer` for handling missing data.
* **Pipelines**: These objects combine multiple transformers and estimators to create a complete machine-learning workflow. Pipelines can streamline your machine-learning process and improve code organisation.
* **Validation Objects**: These objects are used for model selection and evaluation. Examples include `StratifiedShuffleSplit` for stratified data splitting, `GridSearchCV` for hyperparameter tuning, and `cross_val_score` for cross-validation.
* **Metrics**: These functions calculate performance metrics for your model. Examples include `accuracy_score`, `precision_score`, `recall_score`, `f1_score` for classification, `mean_squared_error`, and `mean_absolute_error` for regression.

### Feature Scaling and Standardization
Feature scaling or standardisation is a preprocessing step that transforms features to a standard range. This is often necessary for machine learning algorithms that are sensitive to the scale of the data. Different algorithms have different scaling requirements:

* **Distance-based algorithms** (e.g., K-Nearest Neighbors, Support Vector Machines) rely on the Euclidean distance between data points. Feature scaling ensures that all features contribute equally to the distance calculation, preventing features with larger scales from dominating the distance metric.
* **Gradient-based algorithms** (e.g., linear regression and Neural Networks) use the gradients of the cost function during training. Scaling features can improve the convergence rate of gradient-based algorithms.

Here are two standard scaling methods:
* **StandardScaler**: This method transforms features by subtracting the mean and dividing by the standard deviation. It scales features with a mean of 0 and a standard deviation 1. This is a widely used scaling method that works well for most algorithms.
* **MinMaxScaler**: This method scales features to a range between a minimum value (often 0) and a maximum value (usually 1). This can be useful if your features' distribution is skewed or you have outliers.

---

### Accuracy, Precision, Recall, and F-1 score
1. **Accuracy**: Accuracy measures the proportion of correct predictions from the total number of predictions. It's calculated as the sum of true positives and negatives divided by the total number of samples.
2. **Precision**: Precision measures the proportion of accurate positive predictions out of all optimistic predictions made by the model. It indicates how many of the predicted positive instances are positive.
3. **Recall (Sensitivity)**: Recall measures the proportion of accurate optimistic predictions from all actual positive instances in the data. It indicates how many actual positive instances the model correctly identified.
4. **F1 Score**: The F1 score is the harmonic mean of precision and recall. It balances precision and recall, especially when the classes are imbalanced. 

**Example**:
![image](https://imgur.com/TKUN2v8.png)

#### Precision-Recall Tradeoff
The precision-recall tradeoff balances two critical measures in classification: precision and recall. 

Adjusting the model's decision threshold can impact these metrics: increasing the threshold improves precision but decreases recall, and vice versa. Finding the right balance depends on the specific needs of the problem: sometimes, it's more important to be precise, while other times, it's crucial to catch all relevant instances.

--- 

### SVM
Support Vector Machine (SVM) is a robust supervised machine learning algorithm for classification and regression tasks. It's beneficial when dealing with complex datasets with a clear margin of separation between classes. 

Imagine an airport security checkpoint. The security officers need to separate passengers who pose a security risk from those who don't. SVM can help identify these two groups by finding the best line or boundary (called a hyperplane) that maximises the margin between them. This margin represents the distance between the nearest data points of each class, ensuring a clear separation.

Overall, SVM is ideal for scenarios where precise classification is needed in complex datasets with clear class boundaries. For example, if we have numerous decision boundaries (lines separating classes), SVM helps us decide on the most optimal boundary.

Example: ![SVM-image](https://imgur.com/QhL5OTl.png)

In real life, SVM can be used in various applications such as:
1. **Text classification**: distinguishing between spam and non-spam emails.
2. **Image classification**: to identify objects like cats and dogs in images.
3. **Medical diagnosis**: to predict whether a patient has a particular disease based on symptoms.

**Gama** is a parameter that defines the influence of a single training example, also known as a support vector. It determines the reach or extent of influence of a single training example. 

- **High gamma**: A high gamma value means a training example has a high influence, and the decision boundary is more constrained to the training data. This can lead to a more complex and nonlinear decision boundary, potentially resulting in overfitting, especially when the training dataset is small. 
- **Low gamma**: Conversely, a low gamma value means that a training example has less influence, and the decision boundary is smoother and more linear. This can result in underfitting, where the model fails to capture the complexity of the data.

Example: ![Gamma-SVM-image](https://imgur.com/ly3YZah.png)

---

### Random Forest
Random Forest is a machine learning algorithm that's like a group decision-maker. Imagine a big decision to make, like choosing a restaurant. Instead of asking just one friend for their opinion, you ask many friends. Each friend has preferences and experiences, so that they might suggest different restaurants. You collect all their suggestions and choose the one that gets the most votes.

Similarly, in Random Forest, instead of relying on just one **decision tree** (like one friend's opinion), we create a bunch of decision trees (like asking many friends). Each decision tree is trained on a random data subset and makes predictions. Then, when we need to make a prediction, we gather all the predictions from each tree and choose the one that gets the most votes.

Random Forest is helpful when we want to make accurate predictions for things like:
- Predicting whether an email is spam or not.
- Predicting whether a customer will buy a product or not.
- Predicting the likelihood of a disease based on specific symptoms.

We can use the `RandomForestClassifier` from `sklearn.ensemble` for the random forest.

---

### Important Notes
In `matplotlib`, interpolation refers to the method used to resample the image when displayed at a size different from its original resolution. 

When you resize an image, you may encounter situations where the number of pixels in the original image does not match the number of pixels in the display. In such cases, interpolation is used to estimate the colour of the pixels in the resized image based on the colours of nearby pixels in the original image.

There are various interpolation methods available, each with its characteristics. The `interpolation` parameter in `show()` allows you to specify the process. Standard interpolation methods include a nearest neighbour, bilinear, bicubic, and more. 

For example, setting `interpolation='nearest'` will use the nearest neighbour interpolation method, which simply picks the colour of the closest pixel in the original image. This method can result in pixelated images when upscaling, but it preserves sharp edges. 

On the other hand, using `interpolation='bilinear'` or `interpolation='bicubic'` will produce smoother results by considering multiple nearby pixels during resampling.

**Example**:
```python
plt.imshow(data, cmap='Grays', interpolation='nearest')
plt.imshow(data, cmap='plt.cm.gray_r', interpolation='nearest')
plt.imshow(data, cmap='Oranges', interpolation='nearest')
```
