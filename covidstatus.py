# -*- coding: utf-8 -*-

__author__ = "Jeremy Charlier"
__date__ = "23 March 2020"


import io
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt


def downloadCSV(url):
    return pd.read_csv(url,parse_dates=[0])


def buildTimeSeries(cdf, cntry, prvnc=None):
    if prvnc == None:
        df = cdf[cdf["Country/Region"] == cntry].drop(columns=['Lat', 'Long'])
    else:
        df = cdf[(cdf["Country/Region"] == cntry) &
                 ((cdf["Province/State"] == prvnc))].drop(columns=['Lat', 'Long'])
    y = df.sum()[2:].values
    x = np.arange(0, len(y)).reshape(-1,1)

    # predict
    decay = 2
    reg = LR().fit(x[-decay:], y[-decay:])
    xpred = np.arange(len(y)-1, len(y)+10).reshape(-1,1)
    ypred = reg.predict(xpred)
    return x, y, xpred, ypred


def plotNorthAmerica(df, pltTitle):
    xCan, yCan, xpredCan, ypredCan = buildTimeSeries(df, "Canada")
    xQue, yQue, xpredQue, ypredQue = buildTimeSeries(df, "Canada", "Quebec")
    xUS, yUS, xpredUS, ypredUS = buildTimeSeries(df, "US")
    xNY, yNY, xpredNY, ypredNY = buildTimeSeries(df, "US", "New York")

    plt.figure()
    plt.plot(xCan, yCan, 'b-', label = 'Canada')
    plt.plot(xpredCan, ypredCan, 'b-'+'-')
    plt.plot(xQue, yQue, 'k-', label = 'Quebec')
    plt.plot(xpredQue, ypredQue, 'k-'+'-')
    plt.plot(
        buildTimeSeries(df, "Canada", "Ontario")[0],
        buildTimeSeries(df, "Canada", "Ontario")[1], 'm-', label = 'Ontario')
    plt.plot(
        buildTimeSeries(df, "Canada", "Ontario")[2],
        buildTimeSeries(df, "Canada", "Ontario")[3], 'm-'+'-')
    plt.plot(xUS, yUS, 'r-', label = 'US')
    plt.plot(xpredUS, ypredUS, 'r-'+'-')
    plt.title(pltTitle)
    xticksLabel = df.drop(columns=['Lat', 'Long']).columns[2:]
    plt.xticks([0, 10, 20, 30, 40, 50, 60],
                  [xticksLabel[0], xticksLabel[10], xticksLabel[20],
                    xticksLabel[30], xticksLabel[40], xticksLabel[50],
                    xticksLabel[60]], rotation=40)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Logarithmic Scale")
    plt.show()


def plotEurope(df, pltTitle):
    plt.figure()
    plt.plot(
        buildTimeSeries(df, "Portugal")[0],
        buildTimeSeries(df, "Portugal")[1],
        'b-', label = "Portugal")
    plt.plot(
        buildTimeSeries(df, "Portugal")[2],
        buildTimeSeries(df, "Portugal")[3],
        'b-'+'-')
    plt.plot(
        buildTimeSeries(df, "Spain")[0],
        buildTimeSeries(df, "Spain")[1],
        'k-', label = "Spain")
    plt.plot(
        buildTimeSeries(df, "Spain")[2],
        buildTimeSeries(df, "Spain")[3],
        'k-'+'-')
    plt.plot(
        buildTimeSeries(df, "Italy")[0],
        buildTimeSeries(df, "Italy")[1],
        'c-', label = "Italy")
    plt.plot(
        buildTimeSeries(df, "Italy")[2],
        buildTimeSeries(df, "Italy")[3],
        'c-'+'-')
    plt.plot(
        buildTimeSeries(df, "France")[0],
        buildTimeSeries(df, "France")[1],
        'g-', label = "France")
    plt.plot(
        buildTimeSeries(df, "France")[2],
        buildTimeSeries(df, "France")[3],
        'g-'+'-')
    plt.plot(
        buildTimeSeries(df, "Belgium")[0],
        buildTimeSeries(df, "Belgium")[1],
        'y-', label = "Belgium")
    plt.plot(
        buildTimeSeries(df, "Belgium")[2],
        buildTimeSeries(df, "Belgium")[3],
        'y-'+'-')
    plt.plot(
        buildTimeSeries(df, "Luxembourg")[0],
        buildTimeSeries(df, "Luxembourg")[1],
        'm-', label = "Luxembourg")
    plt.plot(
        buildTimeSeries(df, "Luxembourg")[2],
        buildTimeSeries(df, "Luxembourg")[3],
        'm-'+'-')
    plt.plot(
        buildTimeSeries(df, "Germany")[0],
        buildTimeSeries(df, "Germany")[1],
        'darkgrey', label = "Germany")
    plt.plot(
        buildTimeSeries(df, "Germany")[2],
        buildTimeSeries(df, "Germany")[3],
        'darkgrey', linestyle='dashed')
    plt.plot(
        buildTimeSeries(df, "United Kingdom")[0],
        buildTimeSeries(df, "United Kingdom")[1],
        'darkorange', label = "UK")
    plt.plot(
        buildTimeSeries(df, "United Kingdom")[2],
        buildTimeSeries(df, "United Kingdom")[3],
        'darkorange', linestyle='dashed')

    plt.title(pltTitle)
    xticksLabel = df.drop(columns=['Lat', 'Long']).columns[2:]
    plt.xticks([0, 10, 20, 30, 40, 50, 60],
                  [xticksLabel[0], xticksLabel[10], xticksLabel[20],
                    xticksLabel[30], xticksLabel[40], xticksLabel[50],
                    xticksLabel[60]], rotation=40)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Logarithmic Scale")
    plt.show()


def plotAsia(df, pltTitle):
    plt.figure()
    plt.plot(
        buildTimeSeries(df, "China")[0],
        buildTimeSeries(df, "China")[1],
        'b-', label = "China")
    plt.plot(
        buildTimeSeries(df, "China")[2],
        buildTimeSeries(df, "China")[3],
        'b-'+'-')
    plt.plot(
        buildTimeSeries(df, "Singapore")[0],
        buildTimeSeries(df, "Singapore")[1],
        'k-', label = "Singapore")
    plt.plot(
        buildTimeSeries(df, "Singapore")[2],
        buildTimeSeries(df, "Singapore")[3],
        'k-'+'-')
    plt.plot(
        buildTimeSeries(df, "Taiwan*")[0],
        buildTimeSeries(df, "Taiwan*")[1],
        'g-', label = "Taiwan")
    plt.plot(
        buildTimeSeries(df, "Taiwan*")[2],
        buildTimeSeries(df, "Taiwan*")[3],
        'g-'+'-')

    plt.title(pltTitle)
    xticksLabel = df.drop(columns=['Lat', 'Long']).columns[2:]
    plt.xticks([0, 10, 20, 30, 40, 50, 60],
                  [xticksLabel[0], xticksLabel[10], xticksLabel[20],
                    xticksLabel[30], xticksLabel[40], xticksLabel[50],
                    xticksLabel[60]], rotation=40)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Logarithmic Scale")
    plt.show()


urlLk = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
urlLk += 'csse_covid_19_data/csse_covid_19_time_series/'
urlTmp = urlLk + 'time_series_covid19_confirmed_global.csv'
dfConf = downloadCSV(urlTmp)

urlTmp = urlLk + 'time_series_covid19_deaths_global.csv'
dfDea = downloadCSV(urlTmp)

plotNorthAmerica(dfConf, 'Confirmed Cases')
plotNorthAmerica(dfDea, 'Death Cases')
plotEurope(dfConf, 'Confirmed Cases')
plotEurope(dfDea, 'Death Cases')
plotAsia(dfConf, 'Confirmed Cases')
plotAsia(dfDea, 'Death Cases')
