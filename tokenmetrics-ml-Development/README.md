# tokenmetrics-ml
 
 On Boot(Run Everything):
 Use `tokenmetrics.sh` script or run `my_flask_app/main.py` file after reboot.
 
 Run Only Api:
 - sudo fuser -k 8000/tcp   -> to kill past processes running on this port
 - gunicorn api:app --daemon &    -> to host the api
 
 Run Predictions: `price_prediction/training.py`
 
 Run Fundamentals: `Fundamental/fundamental_roi.py`
 
 Run Technology: `Technology/technology_roi.py`
 
 Run Technical: `technical_analysis/technical.py`
 
 Run Final Grade: `final_grade/final_grade/py`
 
 Run Quantstats: `my_flask_app/quant_data.py`
 
 Run Correlation: `my_flask_app/correlation.py`


**Apis**:

https://analytics.tokenmetrics.com/   -> http://10.25.11.158/

- Investment Stats: https://analytics.tokenmetrics.com/api/investment_stats/?date=2019-12-10&ico_id=0000,3375&price=125
- Stats: https://analytics.tokenmetrics.com/api/stats/?date=2019-12-10&ico_id=3375
- Correlation: https://analytics.tokenmetrics.com/api/correlation/?token=ETH
- Index Stats: https://analytics.tokenmetrics.com/api/index_stats/?investor_type=DT&investment_style=W&time_horizon=W

** ** 
- Predictions: https://analytics.tokenmetrics.com/api/predictions/?token=BTC&date=2019-12-10
- Prediction Metrics: https://analytics.tokenmetrics.com/api/metrics/?symbol=BTC
- Growth: https://analytics.tokenmetrics.com/api/growth/

** ** 
- Quantstats: https://analytics.tokenmetrics.com/api/quantstats/?token=ETH
- Quantstats data: https://analytics.tokenmetrics.com/api/quantstats/data/?token=ETH
- Quantstats Performance: https://analytics.tokenmetrics.com/api/quantstats_performance/?token=ETH&date=2020-05-25
- Quant Grade: https://analytics.tokenmetrics.com/api/quant_grade/
- Quant Final Grade: https://analytics.tokenmetrics.com/api/quant_final_grade/
** ** 

- Grades: https://analytics.tokenmetrics.com/api/grades
- Percentile Grades: https://analytics.tokenmetrics.com/api/percentile_grades/
- Final Grade: https://analytics.tokenmetrics.com/api/finalgrade/
- Fundamental Grade: https://analytics.tokenmetrics.com/api/fundamental/
- Technical Grade: https://analytics.tokenmetrics.com/api/technical/
- Technology Grade: https://analytics.tokenmetrics.com/api/technology/


**Index Api**
 
 - Monthly Index: https://analytics.tokenmetrics.com/api/monthly_index/?grade_type=fundamental_grade&percentile=FALSE
 - Weekly Index: https://analytics.tokenmetrics.com/api/weekly_index/?grade_type=fundamental_grade&percentile=FALSE
 - Quarterly Index: https://analytics.tokenmetrics.com/api/quarterly_index/?grade_type=fundamental_grade&percentile=FALSE
 - Yearly Index: https://analytics.tokenmetrics.com/api/yearly_index/?grade_type=fundamental_grade&percentile=FALSE
 - Daily Index:  https://analytics.tokenmetrics.com/api/yearly_index/?grade_type=fundamental_grade&percentile=FALSE
 
 
** ** 
 
 
 - Predicted Monthly Index: https://analytics.tokenmetrics.com/api/predicted_monthly_index/
 - Predicted Weekly Index: https://analytics.tokenmetrics.com/api/predicted_weekly_index/


 **Index Parameters**
 * grade_type:
   - fundamental_grade
   - technology_grade
   - technical_grade
   - final_grade
   
 * percentile:
   - TRUE
   - FALSE


**Apis**

Daily Indices
TM Trader Weighted Daily - https://analytics.tokenmetrics.com/api/daily_index/?grade_type=final_grade&percentile=TRUE

TM Trader Technical Analysis Daily - https://analytics.tokenmetrics.com/api/daily_index/?grade_type=technical_grade&percentile=TRUE

Weekly Indices
TM Trader Weighted Weekly -  https://analytics.tokenmetrics.com/api/weekly_index/?grade_type=final_grade&percentile=TRUE

TM Trader Technical Analysis Weekly -  https://analytics.tokenmetrics.com/api/weekly_index/?grade_type=technical_grade&percentile=TRUE

TM Trader Price Predictions Weekly -  https://analytics.tokenmetrics.com/api/predicted_weekly_index/

Monthly Indices
TM Trader Weighted Monthly -  https://analytics.tokenmetrics.com/api/monthly_index/?grade_type=final_grade&percentile=TRUE

TM Trader Technical Analysis Monthly -  https://analytics.tokenmetrics.com/api/monthly_index/?grade_type=technical_grade&percentile=TRUE

TM Trader Price Predictions Monthly -  https://analytics.tokenmetrics.com/api/predicted_monthly_index/

Quarterly Indices
TM Value Investor Balanced Quarterly - https://analytics.tokenmetrics.com/api/quarterly_index/?grade_type=final_grade&percentile=FALSE

TM Value Investor Fundamentals Quarterly -  https://analytics.tokenmetrics.com/api/quarterly_index/?grade_type=fundamental_grade&percentile=FALSE

TM Value Investor Technology Quarterly - https://analytics.tokenmetrics.com/api/quarterly_index/?grade_type=technology_grade&percentile=FALSE

Annual Indices
TM Value Investor Balanced Annually -  https://analytics.tokenmetrics.com/api/yearly_index/?grade_type=final_grade&percentile=FALSE

TM Value Investor Fundamentals Annually - https://analytics.tokenmetrics.com/api/yearly_index/?grade_type=fundamental_grade&percentile=FALSE

TM Value Investor Technology Annually - https://analytics.tokenmetrics.com/api/yearly_index/?grade_type=technology_grade&percentile=FALSE


** ** 

**Indices**

1. Monthly indexes rebalances only on the first date of every month. If index not get updated on the 1st date of month because of lack of data or some unforeseen errors then it will update on the next day.
2. Weekly index rebalance after `7 days` in the date mentioned in the weekly index. i.e. `29th July`  Next Update -> `5th August`
3. Quarterly index rebalance after `90 days` in the date mentioned in the quarterly index. i.e. `8th July` Next Update -> `6th October`
4. Yearly index rebalance after `365 days` in the date mentioned in the yearly index. i.e. `5th April 2020` Next Update -> `5th April 2021`

5. Indexes rebalance depend on grades so please check current date grades available in `ico_ml_grade_history` and `ico_ml_percentile_grade_history` and current date prices available in `ico_price_daily_summaries`. Please hit the apis after updating these tables.
