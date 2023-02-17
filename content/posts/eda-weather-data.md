---
title: Experience Analysis through Weather Data - Exploratory Data Analysis
date: 2021-11-01
tags: [eda, analysis, visualization]
social_image: /media/eda/weatherdata.png
draft: true
---

Knowing accurate weather conditions is an important element for individuals as well as organizations. Many businesses rely on weather conditions. It is necessary to have the correct data to get accurate decisions. One type of data that’s easier to find on the internet is Weather data. Many sites provide historical data on many meteorological parameters.

![weathercover](/media/eda/weatherdata.png)

**Exploratory Data Analysis** is an approach to analyze data, to summarize the main characteristics of data, and better understand the data set. It also allows us to quickly interpret the data and adjust different variables to see their effect. The three main steps to get a perfect EDA are extracting the data from an authorized source, *cleaning and processing* the data, and performing *data visualization* on the cleaned data set.

---
Here, I will work through out a practical exploratory data analysis which was done a part of my data analytics internship at Suven Consultants & Technology.

---

## Objective:
The main focus of our project was to perform analysis for testing the Influences of Global Warming and finally put forth a conclusion.

### Hypothesis:
A hypothesis is an **assumption**, an idea that is proposed for the sake of argument so that it can be tested to see if it might be true.

The Null Hypothesis H0 is *“Has the Apparent temperature and humidity compared monthly across 10 years of the data indicate an increase due to Global warming”* — That means we need to find whether the average Apparent temperature for the month of a month says April starting from 2006 to 2016 and the average humidity for the same period have increased or not.

> So, What is this Apparent Temperature and Humidity mentioned in the Null Hypothesis (H0)?

These are called **Terminologies**, or rather say the column names or criteria used to constrain the data we have to different specification or class. In order to know that we must look up for basic terminologies used in the data we are working on.

### Terminologies:
**Meteorological Data** refers to data consisting of physical parameters that are measured directly by instrumentation, and include temperature, dew point, wind direction, wind speed, cloud cover, cloud layer(s), ceiling height, visibility, current weather, and precipitation amount.

***Apparent temperature*** is the temperature equivalent perceived by humans, caused by the combined effects of air temperature, relative humidity, and wind speed. The measure is most commonly applied to the perceived outdoor temperature.

***Humidity*** is the amount of water vapor in the air. If there is a lot of water vapor in the air, the humidity will be high. The higher the humidity, the wetter it feels outside.

> You can check out more weather terminologies from [Kestrelmeter’s Glossary](https://kestrelmeters.com/pages/weather-glossary).

### Dataset:
The dataset currently using, can be obtained from [Kaggle](https://www.kaggle.com/muthuj7/weather-dataset). The dataset has hourly temperature recorded for the last 10 years starting from 2006–04–01 00:00:00.000 +0200 to 2016–09–09 23:00:00.000 +0200. It corresponds to Finland, a country in Northern Europe.

> Now, we have our Objective, Dataset, and basic understanding of the Terminologies. So let’s start of journey to analyze the data!

## Data Preprocessing
Here, i’m using Anaconda Environment with Visual Studio Code. You can also set up such a system which enable a faster git and pipeline integration.

> Check out this [article](https://medium.com/@akhilsai831/setting-up-anaconda-environment-with-visual-studio-code-in-windows-10-ac3f9afd80e0) by Sai to set up the environment easier.

### Importing required libraries:
We will be using Python libraries such as Pandas, Numpy, Matplotlib and Seaborn.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Loading dataset:
Load the dataset using **read_csv()** function as the dataset is in CSV form and read the first 5 rows from data using **head()** function.

```python
data = pd.read_csv('weatherHistory.csv')
data.head()
```

![dataset](/media/eda/dataset.png)

**Dimensions** of the dataframe refers to the overall data sample, i.e, the total number of rows and columns in the data. It can be obtained using data.shape function as follows

