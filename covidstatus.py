# -*- coding: utf-8 -*-
"""covidStatus.ipynb
"""

import io
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt


def downloadCSV(url):
    return pd.read_csv(url,parse_dates=[0])


def plotCntry(cdf, cntry, plotTitle, prvnc=None):
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
    xpred = np.arange(len(y)-1, len(y)+3).reshape(-1,1)
    ypred = reg.predict(xpred)

    # plot historic data (continous) + pred (dash)
    plt.figure()
    if plotTitle == 'Confirmed Cases': colorLinePlot = 'b-'
    if plotTitle == 'Death Cases': colorLinePlot = 'r-'
    if plotTitle == 'Recovered Cases': colorLinePlot = 'g-'
    plt.plot(x, y, colorLinePlot)
    plt.plot(xpred, ypred, colorLinePlot+'-')
    if cntry == "Canada": plt.axvline(x=51, color='k', alpha=0.3)
    if cntry == "France": plt.axvline(x=53, color='k', alpha=0.3)
    plt.title(plotTitle)
    plt.xlabel('Number of days since Jan-26 outbreak ')
    plt.legend(['Historical Data', 'Predicted Data', 'Quarantine'])
    plt.show()


urlLk = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
urlLk += 'csse_covid_19_data/csse_covid_19_time_series/'
urlTmp = urlLk + 'time_series_19-covid-Confirmed.csv'
dfConf = downloadCSV(urlTmp)

urlTmp = urlLk + 'time_series_19-covid-Deaths.csv'
dfDea = downloadCSV(urlTmp)

urlTmp = urlLk + 'time_series_19-covid-Recovered.csv'
dfRec = downloadCSV(urlTmp)

plotCntry(dfConf, "Canada", 'Confirmed Cases')
plotCntry(dfDea, "Canada", 'Death Cases')
plotCntry(dfRec, "Canada", 'Recovered Cases')

plotCntry(dfConf, "Canada", 'Confirmed Cases', "Quebec")
plotCntry(dfDea, "Canada", 'Death Cases', "Quebec")
plotCntry(dfRec, "Canada", 'Recovered Cases', "Quebec")

plotCntry(dfConf, "France", 'Confirmed Cases')
plotCntry(dfDea, "France", 'Death Cases')
plotCntry(dfRec, "France", 'Recovered Cases')

plotCntry(dfConf, "Belgium", 'Confirmed Cases')
plotCntry(dfDea, "Belgium", 'Death Cases')
plotCntry(dfRec, "Belgium", 'Recovered Cases')

plotCntry(dfConf, "Luxembourg", 'Confirmed Cases')
plotCntry(dfDea, "Luxembourg", 'Death Cases')
plotCntry(dfRec, "Luxembourg", 'Recovered Cases')

plotCntry(dfConf, "Germany", 'Confirmed Cases')
plotCntry(dfDea, "Germany", 'Death Cases')
plotCntry(dfRec, "Germany", 'Recovered Cases')

plotCntry(dfConf, "Italy", 'Confirmed Cases')
plotCntry(dfDea, "Italy", 'Death Cases')
plotCntry(dfRec, "Italy", 'Recovered Cases')

plotCntry(dfConf, "Spain", 'Confirmed Cases')
plotCntry(dfDea, "Spain", 'Death Cases')
plotCntry(dfRec, "Spain", 'Recovered Cases')

plotCntry(dfConf, "Portugal", 'Confirmed Cases')
plotCntry(dfDea, "Portugal", 'Death Cases')
plotCntry(dfRec, "Portugal", 'Recovered Cases')

plotCntry(dfConf, "US", 'Confirmed Cases')
plotCntry(dfDea, "US", 'Death Cases')
plotCntry(dfRec, "US", 'Recovered Cases')

plotCntry(dfConf, "US", 'Confirmed Cases', "New York")
plotCntry(dfDea, "US", 'Death Cases', "New York")
plotCntry(dfRec, "US", 'Recovered Cases', "New York")

plotCntry(dfConf, "Singapore", 'Confirmed Cases')
plotCntry(dfDea, "Singapore", 'Death Cases')
plotCntry(dfRec, "Singapore", 'Recovered Cases')

plotCntry(dfConf, "Taiwan*", 'Confirmed Cases')
plotCntry(dfDea, "Taiwan*", 'Death Cases')
plotCntry(dfRec, "Taiwan*", 'Recovered Cases')

plotCntry(dfConf, "China", 'Confirmed Cases')
plotCntry(dfDea, "China", 'Death Cases')
plotCntry(dfRec, "China", 'Recovered Cases')