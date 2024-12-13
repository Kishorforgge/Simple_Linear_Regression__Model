
---

# Simple Linear Regression Model for Salary Prediction

This aims to implements a **Simple Linear Regression** model to predict an employee's salary based on their years of experience. The goal of the project is to develop a model from scratch using core Python libraries such as **Numpy**, **Pandas**, and **Matplotlib**. It provides an introduction to machine learning by showcasing the linear regression technique, gradient descent optimization, and how to visualize results.

## Problem Statement

The objective is to predict an employee's salary based on their years of experience. The relationship between **Years of Experience** (independent variable) and **Salary** (dependent variable) is assumed to be linear. We will build a simple linear regression model to fit a line to the data and predict salary values.

## Project Overview

1. **Linear Regression Model**  
   - We implement a simple linear regression model using the equation:  
     \[
     Y = mX + b
     \]
     where:
     - \(Y\) is the predicted salary
     - \(X\) is the years of experience
     - \(m\) is the slope (regression coefficient)
     - \(b\) is the intercept (constant term)

2. **Gradient Descent Optimization**  
   - We use gradient descent to minimize the cost function (Mean Squared Error), adjusting the model parameters \(m\) and \(b\) iteratively to fit the best line.

3. **Visualization**  
   - The data is visualized with a scatter plot showing the years of experience against the salary. The fitted regression line is overlaid to illustrate the model's predictions.

4. **Model Evaluation**  
   - After training the model, we evaluate its performance using metrics such as **R-squared** and **Mean Squared Error (MSE)** to understand how well the model fits the data.

## Libraries Used

- **Numpy**: Used for performing mathematical calculations, matrix manipulations, and implementing gradient descent.
- **Pandas**: Used for data manipulation, loading the dataset, and handling data structures.
- **Matplotlib**: Used for visualizing the data, plotting the regression line, and interpreting results.

## Dataset

The dataset used in this project contains two columns:
- **Years of Experience**: The number of years an employee has worked.
- **Salary**: The corresponding salary for each employee.

### Sample Data

| Years of Experience | Salary |
|---------------------|--------|
| 1.1                 | 39343  |
| 1.3                 | 46205  |
| 1.5                 | 37731  |
| 2.0                 | 43543  |
| 3.0                 | 57212  |

## How to Run the Code

1. **Clone this repository**  
   First, clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/Simple_Linear_Regression__Model.git
   cd Simple_Linear_Regression__Model
   ```

2. **Install dependencies**  
   Ensure that you have Python installed on your system. Install the required libraries using pip:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Run the script**  
   Run the Python script to train the model and visualize the results:
   ```bash
   python simple_linear_regression.py
   ```

   This will train the model, plot the data and regression line, and display the results.

## Project Structure

The project directory contains the following files:

```
Simple_Linear_Regression__Model/
│
├── simple_linear_regression.py     # Main Python script for model training
├── Salary_Data.csv                 # Dataset file
├── README.md                      # Project documentation
└── requirements.txt                # List of required Python packages
```

## Results and Evaluation

After running the code, you should see a plot with the following:
- **Scatter plot**: Representing the training data (Years of Experience vs Salary).
- **Regression Line**: A line that best fits the data, predicted by the model.

The code also prints the model parameters (\(m\) and \(b\)), the cost function value, and the evaluation metrics, which help in determining the accuracy and performance of the model.

### Example output (after training):

```
Slope (m): 9449.96
Intercept (b): 25792.2
Mean Squared Error (MSE): 153147045.7
R-squared: 0.956
```

## Conclusion

This project demonstrates how to implement a simple linear regression model from scratch using Python. By training the model using gradient descent and evaluating its performance, we gain insights into the relationship between years of experience and salary. The model can be expanded or improved by including additional features or experimenting with other machine learning algorithms.

---
