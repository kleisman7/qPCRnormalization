import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import datetime
import os
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy as sp
from scipy import stats
from scipy.stats.mstats import gmean
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

font = {'size'   : 20}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
pd.set_option('display.max_rows', None)
mycolors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
marks = ['o','^','s']
msize = [5,6,5]

base = plt.cm.get_cmap('gist_rainbow')
color_list = base(np.array([0,0,0,.55,.8,.8,.8]))
newmap = ListedColormap(color_list,'newmap')


def CalculateModelFitsFromParams(wbe,paramdf,ww_lod,wwtp,lag,useprev_single,scaleparam=True):
    if ('bcovref' in wbe.columns.tolist()) & ('bcov' in wbe.columns.tolist()):
        wbe['bcov_recovery'] = wbe.bcov/wbe.bcovref
    if 'bcov_recovery' in wbe.columns.tolist():
        wbe['orig-only_bcov_norm'] = wbe.raw_data/wbe.bcov_recovery
        if 'flow' in wbe.columns.tolist():
            wbe['orig-bcov_flow_norm'] = wbe.raw_data*wbe.flow/wbe.bcov_recovery
    if 'pmmov' in wbe.columns.tolist():
        wbe['orig-pmmov_norm'] = wbe.raw_data/wbe.pmmov
        if 'bcov_recovery' in wbe.columns.tolist():
            wbe['orig-pmmov_bcov_norm'] = wbe.raw_data/wbe.pmmov/wbe.bcov_recovery
            if 'flow' in wbe.columns.tolist():
                wbe['orig-pmmov_bcov_flow_norm'] = wbe.raw_data*wbe.flow/wbe.pmmov/wbe.bcov_recovery
    if 'flow' in wbe.columns.tolist():
        wbe['orig-only_flow_norm'] = wbe.raw_data*wbe.flow
    wbe['orig-raw_data'] = wbe.raw_data
    if 'orig_raw' not in wbe.columns.tolist():
        wbe['orig_raw'] = wbe.raw_data
    wbe.loc[(wbe.orig_raw<ww_lod),'orig_raw'] = wbe[wbe.orig_raw<ww_lod].orig_raw.iloc[0]/2
#     print(wwtp,paramdf.feature.unique())
    for fit in paramdf[(paramdf.lag==lag) & (paramdf.prevind==useprev_single)].fit.unique():
        if wwtp=='comb':
            mask = (paramdf.lag==lag) & (paramdf.prevind==useprev_single) & (paramdf.wwtp=='comb') & (paramdf.fit==fit)
            if 'orig-' in fit:
                # Note: for orig fits, we used log(correction/prev) as the target.  This is why the constant multiple powers are all negated here.
                for catch in wbe.catchment.unique():
#                     if (catch=='Stickney Full') & (fit=='orig-raw_data'):
#                         plt.plot(wbe[wbe.catchment==catch].date,wbe[wbe.catchment==catch][fit])
#                         plt.show()
                    if catch in paramdf[mask].feature.unique(): # loop over catchments that have features in the combination.
                        if scaleparam:
                            wbe.loc[(wbe.catchment==catch),fit]=wbe.loc[(wbe.catchment==catch),fit]*10**(-paramdf[
                                mask & (paramdf.feature==catch)].params.iloc[0])
                if 'neronly' in paramdf[mask].feature.unique():
                    wbe.loc[(wbe.neronly),fit] = wbe.loc[(wbe.neronly),fit]*10**(-paramdf[
                        mask & (paramdf.feature=='neronly')].params.iloc[0])
#                     if (catch=='Stickney Full') & (fit=='orig-raw_data'):
#                         plt.plot(wbe[wbe.catchment==catch].date,wbe[wbe.catchment==catch][fit])
#                         plt.show()
            else:
                wbe[fit] = 1 # if catch not in features, default to 1 for constant.
                if scaleparam:
                    for catch in wbe.catchment.unique():
                        if catch in paramdf[mask].feature.unique(): # loop over catchments that have features in the combination.
                            wbe.loc[(wbe.catchment==catch),fit]=10**(paramdf[
                                mask & (paramdf.feature==catch)].params.iloc[0])
                if 'neronly' in paramdf[mask].feature.unique():
                    wbe.loc[(wbe.neronly),fit] = wbe.loc[(wbe.neronly),fit]*10**(paramdf[
                        mask & (paramdf.feature=='neronly')].params.iloc[0])
                for feature in paramdf[mask].feature.unique(): 
                    if ((feature not in wbe.catchment.unique()) and 
                        (feature!='neronly') and 
                        (feature.replace('raw_data','orig_raw') in wbe.columns)): # Now loop over the other features.
                        wbe[fit]=wbe[fit]*wbe[feature.replace('raw_data','orig_raw')]**(paramdf[
                            mask & (paramdf.feature==feature)].params.iloc[0])
        elif wwtp=='catch':
            mask = (paramdf.lag==lag) & (paramdf.prevind==useprev_single) & (paramdf.fit==fit)
            if 'orig-' in fit:
                for catch in wbe.catchment.unique():
                    if scaleparam:
                        if catch in paramdf[mask].wwtp.unique(): # loop over catchments that have been fit.
#                         print(catch,paramdf[mask & (paramdf.wwtp==catch) & (paramdf.feature==fit.replace('orig-',''))].params)
                            wbe.loc[(wbe.catchment==catch),fit]=wbe.loc[(wbe.catchment==catch),fit]*10**(-paramdf[
                                mask & (paramdf.wwtp==catch) & (paramdf.feature==fit.replace('orig-',''))].params.iloc[0]) # I added this neg later... could be wrong?
                        mask2 = (paramdf.wwtp==catch)
                    else:
                        mask2 = (paramdf.wwtp=='comb')
                    if 'neronly' in paramdf[mask & mask2].feature.unique():
#                         print('orig',fit,wwtp,catch,(catch in wbe.catchment.unique()))
#                         wbe.loc[(wbe.catchment==catch),fit] = wbe.loc[
#                             (wbe.catchment==catch),fit]*10**(-paramdf[
#                             mask & mask2 & (paramdf.feature=='neronly')].params.iloc[0])
                        wbe.loc[(wbe.catchment==catch) & (wbe.neronly),fit] = wbe.loc[
                            (wbe.catchment==catch) & (wbe.neronly),fit]*10**(-paramdf[
                            mask & mask2 & (paramdf.feature=='neronly')].params.iloc[0])
            else:
                wbe[fit] = 1 # if catchment not fit, default to 1 for constant.
                for catch in wbe.catchment.unique():
                    if catch in paramdf[mask].wwtp.unique(): # Loop over catchments that have been fit.
                        if scaleparam:
                            wbe.loc[(wbe.catchment==catch),fit] = 10**(paramdf[
                                mask & (paramdf.wwtp==catch) & (paramdf.feature=='const')].params.iloc[0])
                        mask2 = (paramdf.wwtp==catch)
                    else:
                        mask2 = (paramdf.wwtp=='comb')
                    if scaleparam:
                        if 'neronly' in paramdf[mask & mask2].feature.unique():
    #                         print('dim',fit,wwtp,catch,(catch in wbe.catchment.unique()))
    #                         wbe.loc[(wbe.catchment==catch),fit] = wbe.loc[
    #                             (wbe.catchment==catch),fit]*10**(paramdf[
    #                             mask & mask2 & (paramdf.feature=='neronly')].params.iloc[0])
                            wbe.loc[(wbe.catchment==catch) & (wbe.neronly),fit] = wbe.loc[
                                (wbe.catchment==catch) & (wbe.neronly),fit]*10**(paramdf[
                                mask & mask2 & (paramdf.feature=='neronly')].params.iloc[0])
                    for feature in paramdf[mask & mask2].feature.unique(): 
                        if (feature not in wbe.catchment.unique()) and (feature!='const') and (feature!='neronly'): # Now loop over other features.
                            wbe.loc[(wbe.catchment==catch),fit]=wbe.loc[(wbe.catchment==catch),fit]*wbe.loc[
                                (wbe.catchment==catch),feature.replace('raw_data','orig_raw')]**(paramdf[
                                mask & mask2 & (paramdf.feature==feature)].params.iloc[0])
        else: # a specific wwtp should have been specified here that exists in paramdf.
            mask = (paramdf.lag==lag) & (paramdf.prevind==useprev_single) & (paramdf.fit==fit) & (paramdf.wwtp==wwtp)
            if 'orig-' in fit:
                if 'neronly' in paramdf[mask].feature.unique():
                    wbe.loc[(wbe.neronly),fit] = wbe.loc[(wbe.neronly),fit]*10**(-paramdf[
                        mask & (paramdf.feature=='neronly')].params.iloc[0])
            else:
                wbe[fit] = 1
                for catch in wbe.catchment.unique():
                    if catch==wwtp:
                        wbe.loc[(wbe.catchment==catch),fit] = 10**(paramdf[
                            mask & (paramdf.feature=='const')].params.iloc[0])
                    if 'neronly' in paramdf[mask].feature.unique():
                        wbe.loc[(wbe.neronly),fit] = wbe.loc[(wbe.neronly),fit]*10**(paramdf[
                            mask & (paramdf.feature=='neronly')].params.iloc[0])
                    for feature in paramdf[mask].feature.unique(): 
                        if (feature not in wbe.catchment.unique()) and (feature!='const') and (feature!='neronly'): # Now loop over other features.
                            wbe.loc[(wbe.catchment==catch),fit]=wbe.loc[(wbe.catchment==catch),fit]*wbe.loc[
                                (wbe.catchment==catch),feature.replace('raw_data','orig_raw')]**(paramdf[
                                mask & (paramdf.feature==feature)].params.iloc[0])
    return wbe


#classifying the trends based on CI (increase or decrease, with what CI)
def classification(trend, conf):
    if (trend > 0) and (conf > .99):
        cls = 'Almost Certainly Increase'
    elif (trend > 0) and (conf > .90):
        cls = 'Very Likely Increase'
    elif (trend > 0) and (conf > .66):
        cls = 'Likely Increase'
    elif (trend < 0) and (conf > .99):
        cls = 'Almost Certainly Decrease'
    elif (trend < 0) and (conf > .90):
        cls = 'Very Likely Decrease'
    elif (trend < 0) and (conf > .66):
        cls = 'Likely Decrease'
    else:
        cls = 'Uncertain of Change'
    return cls

#defines trends colors based on CI
def discrete_color(val):
    val = val / 100
    if val < 0.01:
        val1 = 0
    elif val < 0.1:
        val1 = 1
    elif val < 0.34:
        val1 = 2
    elif val <= .66:
        val1 = 3
    elif val <= .9:
        val1 = 4
    elif val <= .99:
        val1 = 5
    else:
        val1 = 6
    return val1


def discrete_cmap(base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, 8))
    cmap_name = base.name + str(8)
    return base.from_list(cmap_name, color_list, 8)

def crosscorrW(datax, datay, lags=[0],mergers=['date'],rmse = False):
    """
    Calculate the cross correlation between two timeseries with given timelag,but this one can't have a catchment column.
    Arguments:
        datax: a dataframe with the following two columns:
            date has datetime type
            the second colmun can have any name, but has float type and contains data for the time-series that we want to shift, here the prevalence indicator
        datay: a dataframe with the following two columns:
            date has datetime type
            the second colmun can have any name, but has float type and contains data for the time-series that we don't want to shift, here the wastewater
        lags: (optional) an integer or list of integers indicating number of days to lag.
    Returns:
        An array of the cross correlations between the two timeseries for the given lags.  
        Note that a lag of -5 means data in datay from 1/5 will be correlated to data in datax from 1/10.  This would mean that datay is early, datax is late.
        If we expect wastewater to be a leading indicator, we expect cases data to be late, so we would be interested in a positive lag.
    """
    mycors = []
    for lag in lags:
        # dataz is the same as datay but laged by lag days.
        dataz = datay.copy()
        dataz.date -= datetime.timedelta(days=int(lag))
        # merge data x and dataz (shifted datay)
        mydf0 = pd.merge(datax,dataz,on=mergers,how='inner')
        mydf0.dropna(inplace=True)
        mydf0.drop(mergers,axis=1,inplace=True)
        if rmse:
            mycors.append(mean_squared_error(mydf0[mydf0.columns[0]],mydf0[mydf0.columns[1]],squared=False))
        else:
            mycors.append(mydf0.corr().values[0][1])
    return np.array(mycors)


def rolling_trend_analysis(wbe,prevdf,catches,useprevs,usewws,firstdate,lastdate,num_weeks,prevalence_num_weeks,showfig=False):
    fulldf = pd.DataFrame()
#     num_weeks = num_days // 7
    num_days = num_weeks * 7
    for n, catch in enumerate(catches):
        if showfig:
            plt.figure(figsize=(20,30))
        mdf = pd.merge(prevdf[prevdf.catchment==catch.replace('Stickney Full','Stickney')][['date']+useprevs],\
                       wbe[(wbe.catchment==catch) & (wbe.orig_raw>0) & ~(wbe.raw_data==0)][['date']+usewws],on='date',how='outer')
        for m, usedata in enumerate(mdf.columns[1:]):
            mdf2 = mdf[(mdf[usedata]>0) & (mdf.date<lastdate) & (mdf.date>firstdate)][['date',usedata]].dropna()
            mdf2[f'Percent Change Weekly over {num_weeks} weeks'] = np.nan
            mdf2[f'Confidence of {num_weeks} week Trend'] = np.nan
            mdf2[f'{num_weeks} week Trend Classification'] = np.nan
            mdf2['datatype'] = usedata
            mdf2['catchment'] = catch
            for enddate in mdf2.date:
                #TODO: Consider changing
                #number of days the trend is calculated using, 28 days uses 8 samples, 14 days is worse, two samples per week, first couple of week only has one sample 
                #Certain number of days / certain sample 
                #Using 28 days before time sensitive 
                
                #Lock in prevalence dates at 3 weeks (21 days)
                if usedata in useprevs:
                    startdate = pd.Timestamp(enddate) - pd.Timedelta(prevalence_num_weeks * 7,'days')
                else:
                    startdate = pd.Timestamp(enddate) - pd.Timedelta(num_days,'days')
                mask4 = (mdf2.date>startdate) & (mdf2.date<=enddate)
                target4 = np.array(np.log10(mdf2[mask4][usedata]))
                features4 = mdates.date2num(mdf2[mask4].date)
                mask1 = (mdf2.date==enddate)
                if (len(features4)>2):
                    res4 = stats.linregress(features4, target4)
                    trend4 = (10**(res4.slope*7)-1)*100
                    unc4 = res4.stderr*7*np.log(10)*10**(7*res4.slope)*100
                    conf04 = t.cdf(trend4/unc4,len(features4)-2)
                    conf4 = t.cdf(np.abs(trend4)/unc4,len(features4)-2)
                    if unc4 > 0.01:
                        mdf2.loc[mask1,f'Percent Change Weekly over {num_weeks} weeks'] = trend4
                        mdf2.loc[mask1,f'Confidence of {num_weeks} week Trend'] = conf04 * 100
                        mdf2.loc[mask1,f'{num_weeks} week Trend Classification'] = classification(trend4,conf4)
            if showfig:
                plt.subplot(13,1,m+1)        
                plt.scatter(mdf2.date,np.minimum(mdf2['Percent Change Weekly over 4 weeks'],200),
                            c=mdf2['Confidence of 4 week Trend'].apply(lambda x:discrete_color(x)),
                            s=30,cmap=newmap)
                for p in [0]:
                    plt.axhline(y=p, xmin=0, xmax=1, color='tab:grey', linestyle='-', linewidth=1)
                plt.ylabel(usedata+'\n4 wk')
                plt.gca().set_xbound(lower=mdates.date2num(datetime.datetime.strptime(firstdate, '%Y-%m-%d')-pd.Timedelta(15,'days')),\
                                     upper=mdates.date2num(datetime.datetime.strptime(lastdate, '%Y-%m-%d')+pd.Timedelta(15,'days')))
                plt.colorbar(ticks=range(7))
                plt.clim(-0.5, 6.5)
            if fulldf.empty:
                fulldf = mdf2.rename(columns={usedata:'data'})
            else:
                fulldf = pd.concat([fulldf,mdf2.rename(columns={usedata:'data'})])
        if showfig:
            plt.suptitle(catch.replace('Stickney Full','Stickney'))
            plt.tight_layout()
            plt.show()
    return fulldf

def trend_trajectory(fulldf,catches,useprevs,usewws,prev_CI,ww_CI,prev_daythresh,ww_daythresh,num_weeks,firstdate,lastdate,showfig=False):
    pink_dots = []
    all_types = []
    datedf = pd.DataFrame(columns={'date'})
    trend4 = f'Confidence of {num_weeks} week Trend'
    for catch in catches:
        alltypes = usewws+useprevs
        types = len(alltypes)
        if showfig:
            plt.figure(figsize=(25,types/2))
        for n,usetype in enumerate(alltypes):
            partdf = fulldf[(fulldf.catchment==catch) & (fulldf.datatype==usetype)].copy()
            #TODO: CONFIDENCE THRESHHOLD. CONSIDER CHANGING
            
            #lock in prevalence CI and daythresh at 66 and 14
            if usetype in useprevs:
                candidatemask4 = (partdf[trend4] > prev_CI) 
                newdf4 = partdf[candidatemask4 & (#partdf[trend4]<66).shift(1) & (
                (partdf[candidatemask4].date.diff() > datetime.timedelta(days=prev_daythresh))) & (
                    partdf.date+pd.Timedelta(30,'days')>firstdate)][['date',trend4]]
            else:
                candidatemask4 = (partdf[trend4] > ww_CI) 
                newdf4 = partdf[candidatemask4 & (#partdf[trend4]<66).shift(1) & (
                (partdf[candidatemask4].date.diff() > datetime.timedelta(days=ww_daythresh))) & (
                    partdf.date+pd.Timedelta(30,'days')>firstdate)][['date',trend4]]
            if showfig:
                plt.plot(partdf[candidatemask4].date,
                         (types-1-n-.05)*np.ones(len(partdf[candidatemask4].date)),'.',markersize=3,color='tab:pink')
            pink_dots.append([usetype, partdf[candidatemask4 & (partdf.date+pd.Timedelta(30,'days')>firstdate)][['date',trend4]]])
            #TODO: Has it been at least 15 days since the last date that has an increase
            #15 is arbitary 
            
            if showfig:
                if n == 0:
                    plt.plot(partdf.date,2.5*np.ones(len(partdf.date)),'.',markersize=2,color='grey')
                plt.plot(newdf4.date,(types-1-n-.05)*np.ones(len(newdf4.date)),'*',markersize=12,color=mycolors[np.mod(n,10)])
            bothdf = pd.DataFrame({'date':newdf4.date,catch + ' '+usetype+f' {num_weeks}wk':newdf4.date}).sort_values(by='date')
            datedf = pd.merge(datedf,bothdf,on='date',how='outer').sort_values(by='date')
        if showfig:
            plt.title(catch.replace('Stickney Full','Stickney'))
            plt.gca().set_yticks(range(types))
            plt.gca().set_yticklabels(alltypes[::-1])
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.gca().set_xbound(lower=mdates.date2num(datetime.datetime.strptime(firstdate, '%Y-%m-%d')),\
                            upper=mdates.date2num(datetime.datetime.strptime(lastdate, '%Y-%m-%d')))
            plt.show()
    return datedf


def FormatDatesNicely(surgedf2):
    # Need to fix if want to allow for something with sources other than wastewater, cases, admissions, beds.
    newsurgedf = surgedf2.pivot(index=['catchment','surge'],columns=['source'],values=['date'])['date'].reset_index()
    newsurgedf.sort_values(by=['catchment','surge'],inplace=True)
    newsurgedf = newsurgedf[['catchment','wastewater','cases','admissions','bedsused','surge']]
    newsurgedf['w-a'] = newsurgedf.admissions-newsurgedf.wastewater
    newsurgedf['c-a'] = newsurgedf.admissions-newsurgedf.cases
    newsurgedf['b-a'] = newsurgedf.admissions-newsurgedf.bedsused
    newsurgedf['w-c'] = newsurgedf.cases-newsurgedf.wastewater
    newsurgedf['w-b'] = newsurgedf.bedsused-newsurgedf.wastewater
#     display(newsurgedf)

    print('Wastewater leads admissions in',len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['w-a']>datetime.timedelta(days=0))]),
          'surges by up to',max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-a']).days,'days, and lags admissions in',
          len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['w-a']<=datetime.timedelta(days=0))]),'surges by up to',
         -min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-a']).days,'days.')
    print('Cases leads admissions in',len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['c-a']>datetime.timedelta(days=0))]),
          'surges by up to',max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['c-a']).days,'days, and lags admissions in',
          len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['c-a']<=datetime.timedelta(days=0))]),'surges by up to',
         -min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['c-a']).days,'days.')
    # print('Beds-in-use leads admissions in',len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['b-a']>datetime.timedelta(days=0))]),
    #       'surges by up to',max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['b-a']).days,'days, and lags admissions in',
    #       len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['b-a']<=datetime.timedelta(days=0))]),'surges by up to',
    #      -min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['b-a']).days,'days.')
    print('Beds-in-use lags admissions in all surges by',-max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['b-a']).days,'to',
         -min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['b-a']).days,'days.')
    print('\nWastewater leads cases in',len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['w-c']>datetime.timedelta(days=0))]),
          'surges by up to',max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-c']).days,'days, and lags cases in',
          len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['w-c']<=datetime.timedelta(days=0))]),'surges by up to',
         -min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-c']).days,'days.')
    # print('Wastewater leads beds-in-use in',len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['w-b']>datetime.timedelta(days=0))]),
    #       'surges by up to',max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-b']).days,'days, and lags beds-in-use in',
    #       len(newsurgedf[(newsurgedf.surge.str.contains('surge')) & (newsurgedf['w-b']<=datetime.timedelta(days=0))]),'surges by up to',
    #      -min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-b']).days,'days.')
    print('Wastewater leads beds-in-use in all surges by',min(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-b']).days,'to',
         max(newsurgedf[(newsurgedf.surge.str.contains('surge'))]['w-b']).days,'days.')
    return(newsurgedf)

