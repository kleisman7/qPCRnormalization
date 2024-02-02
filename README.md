# qPCRnormalization

## qPCR_ModelFit.ipynb 
This notebook fits models using available wastewater and clinical data.  Note that due to restricted availability of sensitive data, the results of these fits will not be exactly the same as in the manuscript.  

1. Data is read in and some initial parameters are set.  To run this file, make sure wastewater.csv and clinical.csv are both in a subfolder called data.  
2. Consistent lags between clinical prevalence indicators are obtained.  If using other data, one may need to change corthresh.
3. Models are trained over a training set (through myend) and tested on validation set (between myend and altend), dataframes of parameters (fitparams...) and aic values for in-sample (fitaic...) are generated and saved.  Dataframe (fitaicfull... is also saved, including aic on training set (aic), on validation set (aictest) and on combination of training and validation set (aicall).
4. Models are also trained over windows of various time domains, from 24-52 weeks at a time, rolling through the full time domain.
5. Finally, models are trained on a training set that systematically omits 2 weeks at a time.

Note that there are two versions of each relevant aic and parameter dataframe files included in the data folder:  
1. A set generated using the included available representative public data:  
   - Trained through 8/28/21, validation through 10/23/21:  
     - fitparams_qpcr_fits_240201.csv  
     - fitaic_qpcr_fits_240201.csv  
     - fitfullaic_qpcr_fits_240201.csv  
   - Trained through 10/23/21:  
     - fitparams_qpcr_fits_altend_240201.csv  
   - Windows of 24-52 weeks, rolling through entire time domain:  
     - fitparams_qpcr_window_fits_240201.csv     
   - Trained through 8/28/21, but omitting 2 weeks at a time:  
     - fitparams_qpcr_window_fits_skipped_end8_28_240201.csv  
     - fitaic_qpcr_window_fits_skipped_end8_28_240201.csv  
2. A set generated using the sensitive data:    
   - Trained through 8/28/21, validation through 10/23/21:  
     - fitparams_qpcr_fits_sensitive.csv  
     - fitaic_qpcr_fits_sensitive.csv  
     - fitfullaic_qpcr_fits_sensitive.csv  
   - Trained through 10/23/21:  
     - fitparams_qpcr_fits_altend_sensitive.csv  
   - Windows of 24-52 weeks, rolling through entire time domain:  
     - fitparams_qpcr_window_fits_sensitive.csv     
   - Trained through 8/28/21, but omitting 2 weeks at a time:  
     - fitparams_qpcr_window_fits_skipped_end8_28_sensitive.csv  
     - fitaic_qpcr_window_fits_skipped_end8_28_sensitive.csv  

## qPCR_Figures.ipynb
This notebook generates the figures included in the manuscript as well as most of those in the supplement.  At the beginning of this file, one can specify if they want to use the aic and parameter files generated from the sensitive data or from the included public data. 
