Stock Market Prediction and Strategy Simulation using Logistic Regression and Q-Learning - Tarang Sonkusare






1. Introduction
Stock market prediction remains a challenging problem due to the complex and volatile nature of financial markets. In recent years, machine learning has emerged as a promising approach to address these challenges by predicting price movements and optimizing trading strategies. This paper explores the use of Logistic Regression for predicting stock price directions and Q-Learning for simulating an intelligent trading agent. Logistic Regression, a classical supervised learning algorithm, is employed to predict whether the stock price will increase or decrease on the next trading day based on historical price data. Q-Learning, a reinforcement learning technique, is then used to simulate a trading agent that interacts with the stock market environment, learning to make optimal decisions through trial and error.
The primary dataset used for this research is the historical stock data of Honda Motors from 1980 to 2024, sourced from Kaggle. The aim is to investigate the effectiveness of these two algorithms for stock market analysis and decision-making. The remainder of this paper is organized as follows: Section 2 introduces the dataset, Section 3 describes the preprocessing steps, Section 4 presents the machine learning methods, Section 5 outlines the results, and Section 6 concludes with insights and potential future work.
2. Data
The dataset for this study consists of Honda Motors' stock prices from 1980 to 2024, containing multiple features such as opening price, highest price, lowest price, closing price, adjusted closing price, and volume of stocks traded. The data spans over four decades, providing a rich temporal context for financial analysis. The primary focus of this study is on the closing price, as it is a widely used indicator for stock performance.
2.1 Data Exploration and Visualization
Initial exploration of the dataset revealed that stock prices, as expected, exhibit significant fluctuations, with occasional sharp increases and declines. Visualization of the data (e.g., time-series plots) showed overall upward trends in stock prices, interspersed with periods of decline during major economic crises such as the 2008 financial crisis and the COVID-19 pandemic. We also examined the correlation between different features, finding that the opening price, highest price, and closing price were highly correlated, as is typically the case in stock market data.
2.2 Target Variable Construction
To frame the problem as a classification task, we created a binary target variable called TomorrowUp, where:
1 indicates that the stock price will increase the next day.
0 indicates that the stock price will decrease the next day.
This transformation allowed us to apply machine learning algorithms for binary classification, predicting whether the stock price will go up or down based on historical data.
2.3 Preprocessing
Data preprocessing is crucial to ensure the quality of the model. The following steps were performed:
Datetime Conversion: The date column was converted to a datetime format to facilitate time-based analysis and feature engineering.
Feature Engineering: The TomorrowUp variable was derived by comparing the closing price of the current day with the next day’s closing price. Features such as opening price, closing price, and trading volume were kept for training the models.
Data Splitting: The dataset was divided into training (80%) and testing (20%) sets to evaluate the performance of the models.
Normalization: All numerical features were standardized to have a mean of 0 and a standard deviation of 1. This step ensures that each feature contributes equally to the model's performance.
3. Methods
3.1 Logistic Regression
Logistic Regression is a simple yet powerful algorithm for binary classification problems. The model assumes that the probability of an event (in this case, stock price movement) follows a logistic distribution. Given the dataset, Logistic Regression predicts the probability that the stock price will go up or down based on the historical features. The model was trained using the training dataset, with the TomorrowUp variable as the target. Model evaluation was carried out using accuracy, precision, recall, and the confusion matrix to assess its performance in predicting stock price direction.
3.2 Q-Learning
Q-Learning is a model-free reinforcement learning algorithm used to find optimal action policies in Markov Decision Processes. In this case, the Q-Learning agent interacts with the stock market environment by taking actions such as buy, sell, or hold, and receiving rewards based on the changes in the portfolio value after each action. The goal is for the agent to learn a strategy that maximizes its total reward (portfolio value) over time.
The state space in Q-Learning was represented by whether the stock price was rising or falling, and whether the agent was holding stocks or not. The reward function was defined as the change in portfolio value after each trade. The Q-Learning algorithm iterated through episodes, each representing a series of trading days, updating its policy based on observed rewards. The final model was evaluated by its final portfolio value and the total number of trades executed.
4. Results
4.1 Logistic Regression Performance
The Logistic Regression model achieved an accuracy of 51%, suggesting that it was able to predict stock price movement correctly slightly more than half of the time. The model’s confusion matrix revealed that it performed better at predicting upward price movements (with a recall of 0.60) than downward movements (recall of 0.42). However, given the volatility and noise inherent in financial markets, the model’s performance was not as high as expected. The results indicated that Logistic Regression, while simple and interpretable, may not be robust enough to capture the complexities of stock market dynamics.




4.2 Q-Learning Performance
The Q-Learning model, after extensive training, resulted in a final portfolio value of $23644.14, starting from an initial portfolio value of $10,000. This significant increase in portfolio value demonstrates the model’s ability to make profitable decisions over time. The model executed 11,291 trades over the training period, showing the agent's active participation in the market. The last few trades in the simulation showed a mixture of buy, sell, and hold actions, with the agent adjusting its strategy based on past performance. Despite the impressive return, there is still room for improvement, especially by incorporating transaction costs and refining the action space.

5. Conclusion
This study demonstrates the feasibility of using machine learning techniques like Logistic Regression and Q-Learning for stock market prediction and intelligent trading. While Logistic Regression provides a basic approach to forecasting stock price direction, Q-Learning offers a more adaptive and profitable strategy for simulating trading decisions. The results suggest that while machine learning holds promise for financial applications, further research and model refinement are necessary to overcome the inherent challenges of stock market prediction.



