We use ensembling techniques where we take predictions of several regression models and pass it to our final model(Meta-Regressor). Each regression models parameters are optimized specifically for the coin. 
![](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor_files/stackingregression_overview.png)

We use MAE(Mean Absolute Error) loss function to optimize the models for every coin. 

**Note**:
- Price predictions would have same error(or accuracy) as past 12 months backtesting results but not true always.

- If there is a new pattern in the coin models can take upto 4-5 days to learn new patterns.

- Models may not perform well in instant voltality.

- We are using coin's price data from the time it launched.

- The time series model is purely dependent on the idea that past behavior and price patterns can be used to predict future price behavior.
