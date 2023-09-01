# qPCRnormalization

## qPCR_ModelFit.ipynb 
This notebook fits models using available wastewater and clinical data.  Note that due to restricted availability of sensitive data, the results of these fits will not be exactly the same as in the manuscript.  

1. Data is read in and some initial parameters are set.  To run this file, make sure wastewater.csv and clinical.csv are both in a subfolder called data.  
2. Consistent lags between clinical prevalence indicators are obtained.  If using other data, one may need to change corthresh.
3. Models are fit, dataframes of parameters and aic values (fitaic... and fitparams...) are generated and saved.
4. Models are fit using training set which ends in August 2021, dataframes of parameters and aic values are generated and saved, including fitfullaic... in which aic values are calculated using the full dataset.

Note that there are two versions of each aic and parameter dataframe files included in the data folder:  
1. A set generated using the included available data:  
  - fitparams_qpcr_fits_230825.csv  
  - fitaic_qpcr_fits_230825.csv  
  - fitparams_qpcr_fits_trainearly_230825.csv  
  - fitaic_qpcr_fits_trainearly_230825.csv  
  - fitfullaic_qpcr_fits_trainearly_230825.csv  
2. A set generated using the sensitive data:  
  - fitparams_qpcr_fits_sensitive_230825.csv  
  - fitaic_qpcr_fits_sensitive_230825.csv  
  - fitparams_qpcr_fits_trainearly_sensitive_230825.csv  
  - fitaic_qpcr_fits_trainearly_sensitive_230825.csv  
  - fitfullaic_qpcr_fits_trainearly_sensitive_230825.csv  

## qPCR_Figures.ipynb
This notebook generates the figures included in the manuscript as well as most of those in the supplement.  At the beginning of this file, one can specify if they want to use the aic and parameter files generated from the sensitive data or from the included public data. 
