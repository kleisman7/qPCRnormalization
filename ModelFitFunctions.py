import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import datetime
import os
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from matplotlib import gridspec
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy as sp
from scipy import stats
from scipy.stats.mstats import gmean

import seaborn as sns
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

def crosscorrC(datax, datay, lags=[0]):
    """
    Calculate the cross correlation between two timeseries with given timelag.  
    Arguments:
        datax: a dataframe with the following three columns:
            date has datetime type
            catchment has object type and is a column that separates the data into groups that should be considered when merging the dataframes. 
            the third colmun can have any name, but has float type and contains the data for the first timeseries.
        datay: a dataframe with the following three columns:
            date has datetime type
            catchment has object type and is a column that separates the data into groups that should be considered when merging the dataframes. 
            the third colmun can have any name, but has float type and contains the data for the second timeseries.
        lags: (optional) an integer or list of integers indicating number of days to lag.
    Returns:
        An array of the cross correlations between the two timeseries for the given lags.  
        Note that a lag of -5 means data in datay from 1/5 will be correlated to data in datax from 1/10.  This would mean that datay is early, datax is late.
    """
    mycors = []
    for lag in lags:
        # dataz is the same as datay but laged by lag days.
        dataz = datay.copy()
        dataz.date -= datetime.timedelta(days=int(lag))
        # merge datax and dataz (lagged datay) together.
        mydf0 = pd.merge(datax,dataz,on=['date','catchment'],how='inner')
        mydf0.dropna(inplace=True)
        mydf0.drop(['date','catchment'],axis=1,inplace=True)
        mycors.append(mydf0.corr().values[0][1])
    return np.array(mycors)


def mymreg(features,target,addconst=1,train_share=1,test_share=None,random_state=1):
    if addconst:
        x2 = sm.add_constant(features)
    else:
        x2 = features.copy()
    if train_share<1:
        if test_share==None:
            test_share = 1-train_share
        X_train, X_test, Y_train, Y_test = train_test_split(x2,target,train_size=train_share,test_size=test_share,random_state=random_state)
        x2 = [x2,X_train,Y_train,X_test,Y_test]
    else:
        X_train = x2.copy()
        Y_train = target
    est = sm.OLS(Y_train,X_train)
    result = est.fit()
    return x2,result


def discrete_color_aic(val):
    if val==0:
        val1 = 0
    elif val<4:
        val1 = 1
    elif val<10:
        val1 = 2
    elif val<20:
        val1 = 3
    elif val<50:
        val1 = 4
    else:
        val1 = 5
    return val1

def aic_colormap():
    base = plt.cm.get_cmap('cividis')
    color_list = base(np.array([1,.85,.55,.28,0]))
    color_list = np.vstack([color_list, [0,0,0,1]])
    newmapaic = ListedColormap(color_list,'newmapaic')
    return newmapaic

def Fix_wbe(wbe):
    wbe.sort_values(by=['date','catchment'],inplace=True,ignore_index=True)
    wbe.pmmov = wbe.pmmov.replace(0,np.nan)
    wbe['pmmov_norm'] = wbe.raw_data/wbe.pmmov
    wbe['only_bcov_norm'] = wbe.raw_data/wbe.bcov_recovery
    wbe['pmmov_bcov_norm'] = wbe.raw_data/wbe.pmmov/wbe.bcov_recovery
    wbe['pmmov_bcov_flow_norm'] = wbe.raw_data*wbe.flow/wbe.pmmov/wbe.bcov_recovery
    wbe['bcov_flow_norm'] = wbe.raw_data*wbe.flow/wbe.bcov_recovery
    wbe['only_flow_norm'] = wbe.raw_data*wbe.flow
    return wbe

def get_prevlags(prevdf,catches,allprev,prevenddate,startdate,enddate,corthresh=0.002,wlags=[-2,5],plotcors=False):
    # updates: write this so it loops over some sort of while statement, updating corthresh to get an optimal threshold - the largest threshold that returns exactly one list of lags. 
#     corthresh = 0.002
#     allprev = ['sensitive cases','regional tpr','public tpr','regional cases','public cases','sensitive admissions','regional admissions','regional bedsused']
    lags = np.arange(-20,21,1) # lags to consider (-20 to 20)
    mask = (prevdf.date>startdate) & (prevdf.date<enddate) & (prevdf.catchment.isin([catch.replace(' Full','') for catch in catches]))
    wlags = [wlags] 
    for n,prev1 in enumerate(allprev[1:]): # First, loop over all prevalence except the first.
        for m,prev2 in enumerate(allprev[:n+1]): # Then also loop over all from start up to prev1...
            print('Correlating:',prev2,prev1)
            # Calculate correlation for different lags, determine lag with max correlation, and save as otherlags all lags that have correlation within corthresh of max.
            mycors = crosscorrC(prevdf[mask & (prevdf.date<prevenddate[prev2])][['date','catchment',prev2]],prevdf[mask & (prevdf.date<prevenddate[prev1])][['date','catchment',prev1]],lags=list(lags))
            maxlag = lags[np.argmax(mycors)]
            otherlags = lags[(mycors[maxlag+20]-mycors<corthresh)]
            print('\tmax lag =',maxlag,'\tother lags = ',otherlags)
            # Now, if it's the first round (n==0), save the otherlags as x.
            # Otherwise, if it's the first comparison of the round, it's comparing the new indicator with the original indicator (sensitive cases).  Save this as y.
            # Otherwise, if it's neither the first round (n==0) nor the first comparison of subsequent rounds (m==0), make a list of otherlags lists, called d.
            # x will always be the list of possible offsets so far.  y will be the possible offsets of the new indicator with the original indicator.
            # d will be offsets between other indicators that need to be checked on.
            if n==0:
                x = [[c] for c in otherlags]
                newlag = [wlags[0][0]+otherlags[0]-1,wlags[0][-1]+otherlags[-1]+1]
            elif (n>0) & (m==0):
                y = [c for c in otherlags]
                z = []
                d = []
                newlag = [wlags[0][0]+otherlags[0]-1,wlags[0][-1]+otherlags[-1]+1]
            elif (n>0) & (m>0):
                d.append([c for c in otherlags])
    #             newlag = [min(newlag[0],wlags[m][0]+otherlags[0]),max(newlag[-1],wlags[m][-1]+otherlags[-1])]
            print('\t',newlag)
        # Now, we've finished a round.  If it's the first round, we already created the first x (list of possible offsets between two indicators).
        # If it's not the first round, though, we'll first append each possible list in x with possible extensions in y, then check these with lists in d. 
        # If the new list passes all tests in d, we will keep it, saving these lists in z for now.  
        # When we're done with this, we'll make z the new x, and go back for the next round to add a new indicator.
        if n==0:
            print(' Possible offsets:',x)
            wlags.append(newlag)
            print(' wlags =',wlags)
        else:
            for xs in x:
                for ys in y:
                    new = xs+[ys]
                    useit = 1
                    for ns,ds in zip(new[:-1],d):
                        if ((ys-ns) not in ds):
                            useit = 0
                            break
                    if useit:
                        z.append(xs+[ys])
            x = z
            print(' Possible offsets:',x)
            wlags.append(newlag)
    #         wlags.append([wlags[0][0]+x[0][0],wlags[0][-1]+x[-1][-1]])
            print(' wlags =',wlags)
        print('')
    # Finally, having created an optimal x, we can create the prevalence lag dictionary:
    # Note: if x is empty, redo with larger corthresh.  If x has two or more lists, redo with smaller corthresh.
    prevlagdict = {allprev[0]:0}
    lower = [wlags[0][0]]
    upper = [wlags[0][-1]]
    for ps,xs,ws in zip(allprev[1:],x[0],wlags[1:]):
        prevlagdict[ps] = xs
        lower.append(ws[0]-xs)
        upper.append(ws[-1]-xs)
    print(prevlagdict)
    prevbounds = pd.DataFrame({'prevind':allprev,'lower':lower,'upper':upper})
    display(prevbounds)
    if plotcors:
        # Visualize these chosen lags
        plt.figure(figsize=(20,30))
        j = 0
        for n,prev1 in enumerate(allprev[1:]): # First, loop over all prevalence except the first.
            for m,prev2 in enumerate(allprev[:n+1]): # Then also loop over all from start up to prev1...
                j+=1
                # Update: make this work well for different number of prev indicators...
                plt.subplot(10,3,j)
                mycors = crosscorrC(prevdf[mask & (prevdf.date<prevenddate[prev2])][['date','catchment',prev2]],prevdf[mask & (prevdf.date<prevenddate[prev1])][['date','catchment',prev1]],lags=list(lags))
                maxlag = lags[np.argmax(mycors)]
                otherlags = lags[(mycors[maxlag+20]-mycors<corthresh)]
                inds = range(otherlags[0]+16,otherlags[-1]+24)
                plt.plot(lags[inds],mycors[inds])
                plt.plot(prevlagdict[prev1]-prevlagdict[prev2],mycors[prevlagdict[prev1]-prevlagdict[prev2]+20],'*',markersize=20)
                plt.title(str(j)+': '+prev2+', '+prev1+', '+str(prevlagdict[prev1]-prevlagdict[prev2]))
                plt.xlabel(prev1+' early... '+prev2+' early... \n'+prev2+' late... '+prev1+' late...')
        plt.tight_layout()
        plt.show()
    return prevlagdict,prevbounds


def FitModelParameters(wbe,prevdf,fitwwtps,useprevs,trymodels,whatiscomb,myend,lags,ww_lod,prevlagdict,prevenddate,
                       altend=None,addparam=False,boolfeat='neronly'):
    # Updates: fix it so it is actually using the same number of sample days even with different lags, when near the edge of available data... I think I actually fixed this already, but should double check later. 
    if altend==None:
        altend=myend
    pdf = pd.DataFrame(columns=['wwtp','prevind','fit','null hyp','lag'])
    paramdf = pd.DataFrame(columns = ['wwtp','prevind','fit','feature','lag'])
    aicdf = pd.DataFrame(columns = ['wwtp','prevind','fit','lag'])
    aicfulldf = pd.DataFrame(columns = ['wwtp','prevind','fit','lag'])
    newcomb = [catch.replace(' Full','') for catch in whatiscomb]
    if addparam:
        newcomb += [boolfeat]
    maxprevlag = max([prevlagdict[p] for p in prevlagdict.keys()])
    # Loop over lags, catchments or combinations, and correction models.  
    # Set the wastewater data mask.  
    # Then also loop over prevalence indicators.
#     print(altend,myend)
    for lagc in lags: 
        for n,catch in enumerate(fitwwtps):
            print('lag:',lagc,', catch: ',catch)
            for p,correction in enumerate(trymodels):
                if catch=='comb':
                    mask = (wbe.catchment.isin(whatiscomb)) & (wbe.raw_data>ww_lod) & (wbe.date<altend)
                else:
                    mask = (wbe.catchment == catch)  & (wbe.raw_data>ww_lod) & (wbe.date<altend)
#                 print(wbe[mask].catchment.unique(),whatiscomb,ww_lod)
#                 wbe[(wbe.catchment.isin(whatiscomb))].info()
#                 wbe[(wbe.catchment.isin(whatiscomb)) & (wbe.date<altend)].info()
#                 wbe[(wbe.raw_data>ww_lod)].info()
                if 'pmmov_norm' in trymodels: # or any other model that uses pmmov...
                    mask = mask & (wbe.pmmov==wbe.pmmov)
                for m,useprev in enumerate(useprevs):
                    lag0 = prevlagdict[useprev] # delaging between prevalence indicators to simplify appropriate range
                    mydate = prevenddate[useprev] # end of reliable data for this prevalence indicator
                    for model in ['orig','new']:
                        # For each model, we do an original/basic/non-power-law version, and a new power-law version. 
                        # Here we specify which features of wbe are needed for each model formulation.  
                        # raw_ci is actually not used, but taking it out now might break something so leave it for now. 
                        # ext is a list of significance tests that I at one point thought would be useful. 
                        # For now I'm not really using this, but it does still return pdf, a df of p-values for those tests. 
                        if correction == 'raw_data':
                            comb = ['date','raw_data','catchment']
                            ext = ''
                        elif correction == 'pmmov_bcov_norm':
                            comb = ['date','raw_data','bcov_recovery','pmmov','catchment']
                            ext = ', pmmov = -raw_data'
                        elif correction == 'pmmov_norm':
                            comb = ['date','raw_data','pmmov','catchment']
                            ext = ', pmmov = -raw_data'
                        elif correction == 'bcov_flow_norm':
                            comb = ['date','raw_data','bcov_recovery','flow','catchment']
                            ext = ', flow = raw_data, flow = -bcov_recovery, bcov_recovery = -raw_data'
                        elif correction == 'pmmov_bcov_flow_norm':
                            comb = ['date','raw_data','flow','bcov_recovery','pmmov','catchment']
                            ext = ', pmmov = -raw_data'
                        elif correction == 'only_flow_norm':
                            comb = ['date','raw_data','flow','catchment']
                            ext = ', flow = raw_data'
                        elif correction == 'only_bcov_norm':
                            comb = ['date','raw_data','bcov_recovery','catchment']
                            ext = ', bcov_recovery = -raw_data'
                        if model=='orig':
                            comb = ['date',correction,'catchment']
                            ext = ''
                            combfull = ['date','catchment','flow','bcov_recovery','pmmov',correction]
                            if correction!='raw_data':
                                combfull = combfull + ['raw_data']
                            if (addparam):# and (catch=='comb'):
                                combfull += [boolfeat]
                        # Create the dataframe mydf2 which has prevalence and wastewater data merged together 
                        # with the appropriate time delay. 
                        if catch=='comb':
                            datax = wbe[mask & (wbe.date+datetime.timedelta(days=int(max(lags)+maxprevlag))<mydate)][combfull].copy()
                            dataz = prevdf[(prevdf.catchment.isin(newcomb)) & (prevdf.date<mydate)][['date','catchment',useprev]].copy()
                            datax.catchment = datax.catchment.replace('Stickney Full','Stickney')
                            dataz.date -= datetime.timedelta(days=int(lagc+lag0))
                            mydf2 = pd.merge(datax,dataz,on=['date','catchment'],how='inner')
                            # When combining catches, we need to create booleans for the different catches 
                            # so the multiplicative constant can vary. 
                            mydf2 = pd.concat([mydf2,pd.get_dummies(mydf2.catchment)],axis=1)
                            if addparam:
                                mydf2[boolfeat] = mydf2[boolfeat].astype(int)
                            mydf2.replace(0,100,inplace=True)
                            mydf2.replace(1,10,inplace=True)
                            comb = comb[:-1]+newcomb
                        else:
                            datax = wbe[mask & (wbe.date+datetime.timedelta(days=int(max(lags)+lag0))<mydate)][combfull]
                            dataz = prevdf[(prevdf.catchment==catch.replace(' Full','')) & 
                                           (prevdf.date<mydate)][['date','catchment',useprev]].copy()
                            dataz.date -= datetime.timedelta(days=int(lagc+lag0))
                            mydf2 = pd.merge(datax,dataz,on=['date','catchment'],how='inner')
                            if addparam:
                                mydf2[boolfeat] = mydf2[boolfeat].astype(int)
                                mydf2.replace(0,100,inplace=True)
                                mydf2.replace(1,10,inplace=True)
                                comb = comb[:-1]+[boolfeat]
                            else:
                                comb = comb[:-1]
#                         print(comb)
#                         print(mydf2.columns)
#                         print(np.min(mydf2.date),np.max(mydf2.date))
                        mydf2.dropna(subset=comb[1:]+[useprev]+[correction],how='any',inplace=True)
                        mydf2 = mydf2[~(mydf2[useprev]==0) & ((mydf2[comb[1:]]!=0).all(axis=1))]
                        # Now get the features and target as log values.  For some models we need to fix this a little bit.  
                        # But then we run the regression fit function mymreg. 
                        mdf3 = mydf2[(mydf2.date<myend)].copy()
#                         features = np.log10(mydf2[(mydf2.date<myend)][comb[1:]]).reset_index(drop=True)
#                         target = np.log10(mydf2[(mydf2.date<myend)][useprev]).reset_index(drop=True)
                        features = np.log10(mdf3[comb[1:]])
                        target = np.log10(mdf3[useprev])
                        if model=='orig':
#                             target = np.log10(mydf2[correction]/mydf2[useprev])
                            target = np.log10(mdf3[correction]/mdf3[useprev])
                            if catch=='comb':
                                features.replace(2,0,inplace=True)
                                comb = ['date'] + newcomb
                                features = features[comb[1:]].astype('float64')
                            else:
                                features = sm.add_constant(features)
                                if addparam:
                                    features.replace(2,0,inplace=True)
                                    features = features[['const',boolfeat]]
                                else:
                                    features = features[['const']]
                            f2,reg = mymreg(features,target,addconst=0)
                            mo = 'orig-'
                        else:
                            if catch=='comb':
                                features.replace(2,0,inplace=True)
                                f2,reg = mymreg(features,target,addconst=0)
                                mytest = reg.t_test((', '.join([c+' = 0' for c in comb[1:]]))+ext)
                            else:
                                if addparam:
                                    features.replace(2,0,inplace=True)
                                f2,reg = mymreg(features,target,addconst=1)
                                mytest = reg.t_test('const = 0, '+(', '.join([c+' = 0' for c in comb[1:]]))+ext)
                            mo = ''
#                         if (m==0) & (lagc==-10) & (p<2):
#                             print(model,catch,comb,correction)
#                             print(features.columns)
#                             print(f2.columns)
#                             display(f2)
                        # Get the p-values
#                         print(comb)
#                         display(features.head(5))
#                         display(features.tail(5))
#                         print(len(features))
                        if model=='new':
                            pvals = mytest.pvalue
                        # Get the parameter values
                        params = reg.params.values
                        paramlb = reg.conf_int()[0].values
                        paramub = reg.conf_int()[1].values 
                        # Save everything to these different dfs that will be returned from this function. 
                        if (catch=='comb') | (model=='orig'):
                            if model=='new':
                                pdf = pd.concat([pdf,pd.DataFrame({
                                    'wwtp':catch,'prevind':useprev,'fit':mo+correction,\
                                    'null hyp':[' = '.join((c+' = 0').split('=')[:2]) for c in comb[1:]+ext.split(', ')[1:]],\
                                    'pvals':pvals,'lag':lagc})],ignore_index=True)
                            paramdf = pd.concat([paramdf,pd.DataFrame({
                                'wwtp':catch,'prevind':useprev,'fit':mo+correction,\
                                'feature':[c for c in comb[1:]],\
                                'params':params,'lb':paramlb,'ub':paramub,'lag':lagc})],ignore_index=True)
                        else:
                            if model =='new':
                                pdf = pd.concat([pdf,pd.DataFrame({
                                    'wwtp':catch,'prevind':useprev,'fit':mo+correction,\
                                    'null hyp':[' = '.join((c.replace('date','const')+' = 0').split('=')[:2]) 
                                                for c in comb+ext.split(', ')[1:]],\
                                    'pvals':pvals,'lag':lagc})],ignore_index=True)
                            paramdf = pd.concat([paramdf,pd.DataFrame({
                                'wwtp':catch,'prevind':useprev,'fit':mo+correction,\
                                'feature':[c.replace('date','const') for c in comb],\
                                'params':params,'lb':paramlb,'ub':paramub,'lag':lagc})],ignore_index=True)
                        aicdf = pd.concat([aicdf,pd.DataFrame({
                            'wwtp':catch,'prevind':useprev,'fit':mo+correction,'aic':reg.aic,\
                            'lag':lagc},index=[0])],ignore_index=True)
                        # This is for if we wanted to do train/test, and have an earlier end date for the fitting.
                        # Then this will compute a separate aicfulldf which uses all the data including out-of-sample data. 
                        if (myend!=altend):
                            # aic for all
#                             print('doing aic overall')
                            y = np.log10(mydf2[(mydf2.date<altend)][useprev])
                            testset = np.log10(mydf2[(mydf2.date<altend)][comb[1:]])
                            if model=='orig':
                                y = np.log10(mydf2[(mydf2.date<altend)][correction]/mydf2[(mydf2.date<altend)][useprev])
                                if catch=='comb':
                                    testset.replace(2,0,inplace=True)
                                    comb = ['date'] + newcomb
                                    testset = testset[comb[1:]].astype('float64')
                                else:
                                    testset = sm.add_constant(testset)
                                    if addparam:
                                        testset.replace(2,0,inplace=True)
                                        testset = testset[['const',boolfeat]]
                                    else:
                                        testset = testset[['const']]
                            else:
                                if catch=='comb':
                                    testset.replace(2,0,inplace=True)
                                else:
                                    if addparam:
                                        testset.replace(2,0,inplace=True)
                                    testset = sm.add_constant(testset)
#                             print(len(testset))
                            yhat = reg.predict(testset)
                            mse = mean_squared_error(y,yhat)
                            aicall = len(y)*np.log(mse*2*np.pi*np.exp(1)) + 2*len(params)
#                             print('aic overall completed: train=',reg.aic,', overall=',aicall)
                            # aic for test only
                            y = np.log10(mydf2[(mydf2.date<altend) & (mydf2.date>=myend)][useprev])
                            testset = np.log10(mydf2[(mydf2.date<altend) & (mydf2.date>=myend)][comb[1:]])
                            if model=='orig':
                                y = np.log10(mydf2[(mydf2.date<altend) & (mydf2.date>=myend)][correction]/
                                             mydf2[(mydf2.date<altend) & (mydf2.date>=myend)][useprev])
                                if catch=='comb':
                                    testset.replace(2,0,inplace=True)
                                    comb = ['date'] + newcomb
                                    testset = testset[comb[1:]].astype('float64')
                                else:
                                    testset = sm.add_constant(testset)
                                    if addparam:
                                        testset.replace(2,0,inplace=True)
                                        testset = testset[['const',boolfeat]]
                                    else:
                                        testset = testset[['const']]
                            else:
                                if catch=='comb':
                                    testset.replace(2,0,inplace=True)
                                else:
                                    if addparam:
                                        testset.replace(2,0,inplace=True)
                                    testset = sm.add_constant(testset)
#                             print(len(testset))
                            yhat = reg.predict(testset)
                            mse = mean_squared_error(y,yhat)
                            aictest = len(y)*np.log(mse*2*np.pi*np.exp(1)) + 2*len(params)
#                             print('now aic for test only completed: aic for test=',aictest)
                            aicfulldf = pd.concat([aicfulldf,pd.DataFrame({
                                'wwtp':catch,'prevind':useprev,'fit':mo+correction,'aic':reg.aic,'aicall':aicall,\
                                'aictest':aictest,'lag':lagc},index=[0])],ignore_index=True)
    return paramdf,aicdf,pdf,aicfulldf


def reformat_aic_params(aicdf,paramdf,trymodels):
    aic2 = aicdf.copy()
    param2 = paramdf.copy()
    test_dict = {'pmmov_norm':'new pmmov norm aic','bcov_flow_norm':'new bcov flow norm aic',\
                 'only_bcov_norm':'new only bcov norm aic','raw_data':'new no correction aic',\
                 'only_flow_norm':'new only flow aic','pmmov_bcov_norm':'new pmmov bcov norm aic',\
                 'pmmov_bcov_flow_norm':'new pm bc fl aic','orig-pmmov_norm':'dim pmmov norm aic',\
                 'orig-bcov_flow_norm':'dim bcov flow norm aic','orig-only_bcov_norm':'dim only bcov norm aic',\
                 'orig-raw_data':'dim no correction aic','orig-only_flow_norm':'dim only flow aic',\
                 'orig-pmmov_bcov_norm':'dim pmmov bcov norm aic','orig-pmmov_bcov_flow_norm':'dim pm bc fl aic'}
    
    aic2.fit = aic2.fit.apply(lambda x :test_dict[x])
    aic2.rename(columns={'fit':'model'},inplace=True)
    
    param2.fit = param2.fit.apply(lambda x :test_dict[x])
    param2.rename(columns={'fit':'model'},inplace=True)
    
    models = [test_dict['orig-'+mod] for mod in trymodels if 'pmmov_bcov' not in mod] + [test_dict[mod] for mod in trymodels]
    return aic2,param2,models
