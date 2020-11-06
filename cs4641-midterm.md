---
layout: page
title: Predicting Wildfires
permalink: /cs4641/midterm
---

# Introduction

In North America, at the end of September 2020, acres burned by wildfires exceeded the 10-year average by 1.3 million. Wildfires present many risks, including fatalities, property damage, secondary explosions at industrial plants, and environmental damage. As global warming accelerates, causing temperatures to rise and extreme weather events to occur more often, wildfires are becoming more frequent, intense, and costly. A pressing need exists to enhance risk assessment capabilities and predict attributes of wildfires to achieve better containment. Remote sensing data, as well as information about past wildfires, meteorological conditions, and vegetation, can be used to monitor, contain, and prevent wildfires as well as predict occurrences and probable paths.

# Problem Definition 

Given a particular US location and a future time (month and year), we will (i) predict whether or not a wildfire will occur, and (ii) if it will occur, predict the damage it will cause by land area. 

# Data Collection and Cleaning

## Fire Occurrence Prediction Task 

For this prediction task, we chose to use data for a fixed location that captures information on the state of vegetation and weather. The state of vegetation and weather is more frequently and arguably accurately captured by satellite imagery data. That way, given a location and future date, we can obtain location conditions on that date as true inputs into the model and obtain a yes or no prediction. 

For training, we began with a pre-cleaned and labelled [dataset](https://github.com/ouladsayadyounes/WildFires/blob/master/WildFires_DataSet.csv) that contains parameters extracted from [MODIS](https://modis.gsfc.nasa.gov/) satellite data for a central region of Canada. This original dataset contained three features (described more below) and labels that categorized datapoints as either ‘fire’ or ‘no_fire’. Of the three parameters in the dataset, we chose two: 

(i) Normalized Difference Vegetation Index (NDVI) originally extracted from dataset [MOD13Q1](https://lpdaac.usgs.gov/products/mod13q1v006/), a vegetation index for crops derived from infrared remote sensing that estimates the depth of surface vegetation (canopies). Resultant values lie between 0 and 1, where 0 indicates very sparse vegetation and 1 indicates dense vegetation. We chose NDVI specifically because it also indicates whether a crop is dry or wet, which affects the flammability.

(ii) LST (Land Surface Temperature) originally extracted from dataset [MOD11A1](https://lpdaac.usgs.gov/products/mod11a1v006/), capturing the radiative temperature of the ground and is reflective of soil and vegetation temperature. It is highly indicative of surface-atmosphere energy conditions, and higher temperatures also indicate crops being more water-stressed. Values lie between 7500 and 65535 for this dataset.

We chose to drop the third parameter, TA (Thermal Anomalies) data from the dataset, as it is a direct indication of fire presence that is harder to access for future predictive tasks given an unseen datapoint.

To obtain this training data, original satellite image data was translated from MODIS using the GDAL library, and was made free of damaged images using radiometric correction methods. Inverse Distance Weighting (an interpolation method) was used to normalize the clipped data over the ‘daily’ timestamp. To obtain NDVI and LST inputs given a location & time for our final product, we plan to use NDVI and weather forecast APIs. 


## Fire Risk Assessment Task 

Data collection for the fire risk assessment task was more involved. This task involves estimating the overall damage of the fire, if our first model has predicted there will be a fire. For this, we found a second [dataset](https://www.kaggle.com/elikplim/forest-fires-data-set)  with which we trained a model to learn the relationship between features like month, temperature, humidity, rain, wind speed, etc (input) and the area of burned land (output). 

To clean the data, we used Min-Max scaling to bring the variables to the same scale, which prevented the algorithm from being affected by the magnitude of the different features. In order to improve symmetry and reduce skewness, we [applied](http://www3.dsi.uminho.pt/pcortez/fires.pdf) the logarithm model y = ln(x+1) on the burned area column to improve results for right skewed targets.

# Methods

## Fire Occurrence Prediction Task 

**Dataset balancing:**
 Firstly, our dataset was largely unbalanced. It contained 1327 (77%) no_fire and 386 (23%) fire examples, which faultily led to our models obtaining a high accuracy just by defaulting to labelling all test examples as no_fire. Referencing Sayad et al, we chose to deal with this issue by dropping about 60% of the no_fire data points, leaving us with 418 (52%) no_fire and 386 (48%) fire examples. 

**Train-test split:** We trained our models on 70% of the examples and tested on 30%. 

**Model Experimentation:** In the time to this report, we experimented with and tuned three separate binary classification models in sklearn. 

(i) Gaussian Naive Bayes: We chose this because GNB classifiers are easy to construct and interpret, and it does not require complicated iterative hyperparameters.

(ii) Multi-layer perceptron classifier: We chose to use a Neural Network because of their widespread use in classification problems. NNs are also able to work well even with incomplete knowledge, and can parallelize work. 

(iii) Support Vector Machine: We chose this because it typically is regarded as one of the most accurate classification methods.
Furthermore, Sayad et al obtained high accuracy values using MLP and SVMs using the original dataset.

**Model Assessment:** The classification metrics we used to validate our models were precision, recall, and f1-score. We kept track of true/false positive/negative rates using a confusion matrix. We also evaluated our models statistically using shuffle-split cross validation. 

**Model Hyperparameter Tuning:** We tuned hyperparameters manually, iterating values in both directions and monitoring the listed evaluation metrics for improvement. 

## Fire Risk Assessment Task 

**Dataset balancing:** The dataset is heavily skewed due to the many instances where the area burned equals 0 hectares. We removed these 50% of instances for all the models to improve performance, as well as the outliers for the neural network model, which were rows with burned area greater than 200. However, removing instances from the dataset where 0 hectares were burned does not appear to improve any of the models. In fact, the R2 score for the regression models decreased when removing data points where ‘area’ = 0. 

**Feature Selection:**
 
**(i) Visual Analysis:** We began by creating basic scatter plots and histograms to visualize the relationship between each independent variable and our dependent variable, area burned (in hectares). Seaborn was utilized to observe the covariances between the features. In order to create the covariance matrix, scikit-learn’s StandardScaler preprocessing to subtract the mean from each feature and scale to the unit variance of that feature. These visualizations (seen below in Fig 0(a)) helped us better understand the dataset so we could determine which features to consider and which models might work best with our data. 

As a result, we observed several useful correlations in our data.

{% include image.html url="\images\covariance_heatmap.png" description="Fig 0 (a). Covariance heatmap" %}

From above, temperature appears highly correlated with FFMC, FMC, DC, ISI. DMC and DC appear highly correlated, as well as ISI and FFMC. 

{% include image.html url="\images\freq_fire_by_month.png" description="Fig 0 (b). Month appears to be correlated with the frequency of fires occurring and the acreage burned." %}

Month appears to be correlated with the frequency of fires occurring and the acreage burned (Fig 0 (b)). The majority of the fires in the dataset occur during the summer months of August and September. These months also appear to produce fires that burn greater areas of land than the fires that occur in all other months. 

{% include image.html url="\images\freq_fires_day.png" description="Fig 0 (c). Day appears to be correlated with the frequency of fires occurring and the acreage burned." %}

Frequency of fires and damage, assessed by area burned, do not appear to vary significantly by day of the week. 

{% include image.html url="\images\ffmc_dmc.png" description="Fig 0 (d). FFMC and DMC." %}

Higher values of Fine Fuel Moisture Code (FFMC) appear to be correlated to an increase in the occurrence of forest fires as well as an increase in the number of hectares burned. Duff Moisture Code (DMC) does not appear correlated to the occurrence or damage of forest fires, although the scatter plot is slightly skewed to the left, which might indicate that lower DMC values correlate to more forest fire occurrences.

{% include image.html url="\images\dc_isi.png" description="Fig 0 (e). DC and ISI." %}

The scatter plot of Drought Code (DC) versus area burned appears to be skewed to the right, with a slight correlation between the occurrences of fires and a higher DC value as well as a slight correlation between the area burned by fires and a higher DC value. The scatter plot for Initial Spread Index (ISI) is heavily skewed to the left with one outlier value above ISI = 50 and most fires occurring when ISI < 25. The area burned might be slightly increased when the ISI value is smaller, indicating a slight negative correlation.

{% include image.html url="\images\temp_relativehum.png" description="Fig 0 (f). Temperature and relative humidity." %}

Temperature appears positively correlated with area burned. As temperature increases, the hectares burned appears to gradually increase. Relative humidity appears negatively correlated with area burned. Smaller RH values appear to correspond to data points that have higher amounts of hectares burned. 

{% include image.html url="\images\wind_rain.png" description="Fig 0 (g). Wind and rain." %}

Fires with the greatest amount of hectares burned had wind speeds in the middle of the range of recorded wind speeds in this dataset.
Occurrences of fires and acres burned both appear correlated to rainfall. Most fires occur when there is little to no rain, measured in mm/m2.
 
Most fires burned less than 400 hectares. This dataset includes two outlier fires (data points) that both burned more than 700 hectares.

**(ii) Backward Elimination:** We used Backward Elimination to select the best subsets of features in the linear regression model, as explained further in the Results section. 

**(iii) LassoCV Feature Selection:** For the KNN model we used the LassoCV feature selection model which returned the 5 most prominent features as DC, X, Y, month, and day. We then tried to predict the importance of these features by using the SelectKBest model to calculate the p-values for the features and then convert the p-values to scores by taking their negative log. Below you can see the graph of features and how their scores faired:

{% include image.html url="\images\lasso_cv.png" description="Fig 1. Scores of each selected feature using LassoCV feature selection." %}

**Train-test split:** We trained our model on 70% of the data points and tested it on 30% of the data points. 

**Model Experimentation:** We used 3 different models - linear regression, KNN, and neural net to predict the burned area from the aforementioned features that are described below:

(i) Regression: Using R’s leaps package and Python’s scikit-learn package, we experimented with feature selection for the regression model using matplotlib visual analysis and R’s Backward Elimination best subset selection method. We trained three different linear regression models of different sets of features as well, (a) full features, (b) reduced features using visual analysis, and (c) reduced features using both visual analysis & binary rainfall values. 

(ii) K-NN Algorithm.

(iii) Sequential Neural Network. For this midterm report, we arbitrarily chose a single type of network to train.

**Model Assessment:** The regression metrics we used to validate the regression model were mean squared error (MSE), the R2 coefficient, and the adjusted R2 coefficient. The K-NN and Sequential NN models were judged using RMSE values. Additionally, sample user inputs were given to determine what would be the predicted value.

**Model Hyperparameter Tuning:** For K-NN, we used Sklearn’s GridSearchCV to determine the best K value to implement and additionally plotted the RMSE values for different K’s to validate the accuracy of our K value. For the Sequential NN, we manually experimented with varying number of hidden layers and number of neurons in each layer, as well as changing the activation functions

# Results

## Fire Occurrence Prediction Task 

The results obtained for each classification metric is as follows.

{% include image.html url="\images\prediction_table.png" description="Table 1. Results obtained for each model used in the fire prediction task." %}

In terms of true/false positives/negative rates, we noticed that the MLP and SVM classifiers disproportionately predicted more data points as no_fire than as fire (~90% vs. ~10%), whereas the GNB classifier showed more balanced assignments, slightly favoring fire.

Below are the graphs and results obtained from each of the models:

# GNB Classifer

Performance of the GNB model with tuned hyperparameters is shown in the figure below, and discussed later.
{% include image.html url="\images\gnb.png" description="Fig 2. GNB: Learning curve, scalability, and model performance." %}

# MLP Classifier

Performance of the MLP classifier model with tuned hyperparameters is shown in the figure below, and discussed later.
{% include image.html url="\images\mlp.png" description="Fig 2. MLP Classifer: Learning curve, scalability, and model performance." %}

# SVM Classifier

Performance of the SVM classifier model with tuned hyperparameters is shown in the figure below, and discussed later.
{% include image.html url="\images\svm.png" description="Fig 2. SVM Classifer: Learning curve, scalability, and model performance." %}

## Fire Risk Assessment Task 

The RMSE scores obtained for each model is as follows. These are comparable because they result from models trained on the same dataset.

{% include image.html url="\images\risk_table.png" description="Table 2. Results obtained for each model used in the fire risk assessment task." %}

Below are the graphs and results obtained from each of the models:

# Regression

The following metrics were found for each model used in our regression analysis. 

{% include image.html url="\images\regression_table.png" description="Table 3. Results obtained for each regression model that used different features as used in the fire risk assessment task." %}

{% include image.html url="\images\regression_results.png" description="Fig 5. Regression Model Analysis Table using Backward Elimination for subset selection. In the table, a star under a feature means it is included in the model on that row. The adjusted R2 statistic for each model is included in the table to aid evaluation." %}

# K-NN

Performance of the K-NN model with tuned hyperparameters is shown in the figure below, and discussed later.

{% include image.html url="\images\knn.png" description="Fig 6. Graph of RMSE vs K: when all features were used vs. when 5 selected features were used." %}

{% include image.html url="\images\knn_train.png" description="Fig 7. Testing and Training Accuracy vs. K." %}

# Neural Network

Performance of the NN model with tuned hyperparameters is shown in the figure below, and discussed later.

{% include image.html url="\images\nn_rmse.png" description="Fig 8. Sequential NN: Epoch vs MSE of training and testing set." %}

{% include image.html url="\images\nn_scatter.png" description="Fig 9. Sequential NN: Actual burned area vs predicted burned area." %}

# Discussion

## Fire Occurrence Prediction Task

We have some difficulty with the accuracy of our models and its predictions. Our GNB classifier yields a result with 64.876% accuracy but both the MLP Classifier and SVM have less than a 60% accuracy rate. By altering the different parameters we will try to bring up the accuracy and precision of our model. By adding a more rigorous method to determine the parameters, we may be able to create a set of parameters that produce better results. We decided to drop one of the parameters from our dataset, Thermal Anomalies, because we cannot simply use a historical dataset and produce the wanted results. Instead we would need to create a separate model that would output a usable dataset that produces the thermal anomalies of any particular location. Although taking out this parameter has slightly reduced the accuracy of our model, we believe that altering the other parameters will be able to make up for the difference.

## Fire Risk Assessment Task 

We begin with evaluating our linear regression models. All evaluation metrics are listed in a table in the results section. We first built a full multiple linear regression model using scikit-learn and all features in the dataset. This model performed very poorly. Next, we used the visual analysis, explained above, to perform feature selection and create a model using a subset of features in the dataset. The scatter plots showed us that the features appearing most correlated with the hectares burned are ‘month’, ‘temp’, ‘rain’, ‘wind speed’, and ‘relative humidity.’ Since ‘temp’ has a high covariance with ‘FFMC’, ‘FMC’, ‘DC’, and ‘ISI’, these features were omitted from the revised model. The multiple linear regression model using fewer features resulted in slightly better metrics than the ones for the full model, but still show poor results. When ‘rain’ is treated as a binary variable – 0 if no rainfall occurred and 1 if any rainfall occurred – then the metrics improve very slightly again.
 
The leaps package in R was used to deploy Backward Elimination to select the best subsets of features when the number of features used equals 1 through 8. During each step, R picks the most statistically significant features to include in the model. Each additional model includes one more feature than the previous model. Using Backward Elimination, the most statistically significant model uses only temperature as a predictor (see table in results section). 

Finally, we experimented with combinations of features based on the visualizations and the table provided by Backward Elimination subset selection. In a model using ‘temp’ as its only feature, squaring the ‘temp’ feature results in a slightly better adjusted R2 statistic of 0.00902. Squaring other features does not appear to enhance the model.

None of the models have a large enough adjusted R2 statistic to be considered beneficial, so we decided to focus on other selected models. 

For K-NN, we saw (from Fig 6) that the RMSE is lower in the one where only 5 selected features were used when compared to the initial 12 based on the LassoCV Feature Selection. However, despite selecting the best features the KNN model is unable to predict accurate values for the burned area.

Using the Sequential NN, we successfully lowered the RMSE, but the model severely overfit, as displayed in Figure 8. The neural network was configured by manually testing different numbers of hidden layers and neurons per layer to minimize the RMSE and reduce overfitting. The final configuration contained 1 hidden layer with 3 neurons, using a ReLU activation function. As we continue to experiment and settle on the best model to use, we will tune parameters to prevent this from occurring.  

For this particular task we saw that the linear regression model and KNN regression returned poor R-squared and RMSE values respectively. Thus we also implemented a sequential neural network which provided marginal improvements. For the KNN regression, scaling the data and taking the log of the burned area as part of the data preprocessing was efficient in producing lower RMSE values. Additionally, by determining the prominent features we were able to drop 7 out of the 12 features in the dataset which produced better results. Despite these efforts, the model failed to yield a high accuracy, leading us to the conclusion that there are too many unknown and random variables to yield a high accuracy model.

# References

Coffield, S. R., Graff, C. A., Chen, Y., Smyth, P., Foufoula-Georgiou, E., & Randerson, J. T.(2019). Machine learning to predict final fire size at the time of ignition. International Journal of Wildland Fire, 28(11), 861. doi:10.1071/wf19023

Khakzad, N. (2019). Modeling wildfire spread in wildland-industrial interfaces using dynamic Bayesian network. Reliability Engineering & System Safety, 189, 165-176. doi:10.1016/j.ress.2019.04.006

Sayad, Y. O., Mousannif, H., & Moatassime, H. A. (2019). Predictive modeling of wildfires: A new dataset and machine learning approach. Fire Safety Journal, 104, 130-146. doi:10.1016/j.firesaf.2019.01.006

Scott, J. H., Thompson, M. P., & Calkin, D. E. (2013). A wildfire risk assessment framework for land and resource management. Fort Collins, CO: U.S. Dept. of Agriculture, Forest Service, Rocky Mountain Research Station.

2020 North American Wildfire Season. (2020, September 30). Retrieved September 30, 2020, from https://disasterphilanthropy.org/disaster/2020-california-wildfires/

***The first 3 sources are from peer-reviewed journals