# Linear Regression on Bike Sharing dataset
> This assignment is a programming assignment to build a multiple linear regression model for the prediction of demand for shared bikes.

## Table of contents
1. EDA
2. data preparation
3. import linear regression libraries
4. train test split
5. scaling
6. correlation heatmap
7. modelling
8. Inferences

## Conclusions
* Temperature seems to have a good positive correlation with bike sharing. It’s possible that the USA being a cold country, people would like to use more bike sharing when the temperature goes higher
* Rise in snowfall or rain seems to have a moderately negative correlation with bike sharing. It makes sense that when it snows or rains, people wouldn’t want to ride a bike
* Year by year bike sharing is increasing. It's possible that after the pandemic, people are moving out more and using bikes.
* Since cnt was scaled, we should use `scaler.inverse_transform('cnt')` to get the unscaled value of cnt
* m12 is our final model.
