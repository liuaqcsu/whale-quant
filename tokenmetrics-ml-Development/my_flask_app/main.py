from datetime import datetime, timedelta
from threading import Timer
from subprocess import call
import time, os
import requests, json

x=datetime.today()
y = x.replace(day=x.day, hour=3, minute=0, second=0, microsecond=0) + timedelta(hours=6)
delta_t=y-x

secs=delta_t.total_seconds()

os.chdir("/Users/stella/Downloads/tokenmetrics-ml-Development/my_flask_app/")
call(['fuser','-k', '8000/tcp'])
os.system('GUNICORN_CMD_ARGS="--bind=0.0.0.0 " gunicorn api:app -t 300 -w 3 --threads 12 --daemon &')


def hello_world():
    os.chdir("/Users/stella/Downloads/tokenmetrics-ml-Development/Fundamental/")
    call(["python3","fundamental_roi.py"])
    os.chdir("/Users/stella/Downloads/tokenmetrics-ml-Development/Technology/")
    call(["python3","technology_roi.py"])
    os.chdir("/Users/stella/Downloads/tokenmetrics-ml-Development/final_grade/")
    call(["python3","final_grade.py"])
    os.chdir("/Users/stella/Downloads/tokenmetrics-ml-Development/my_flask_app/")
    call(["python3","correlation.py"])
    call(["python3","quant_data.py"])
    call(["rm","nohup.out"])
    #os.chdir("/home/tokenmetrics/price_prediction/")
    #call(["python3","training.py"])
    os.chdir("/Users/stella/Downloads/tokenmetrics-ml-Development/technical_analysis/")
    call(["python3","technical.py"])
    #call(["python3","my_flask_app/quant_data.py"])
    #call(["python3","price_prediction/training.py"])


def call_index():
    i = ['daily_index','weekly_index','monthly_index','quarterly_index','yearly_index']
    j = ['fundamental_grade','technology_grade','technical_grade','final_grade']
    k = ['TRUE','FALSE']
    urls = [('https://analytics.tokenmetrics.com/api/'+x+'/?grade_type='+y+'&percentile='+z) for x in i for y in j for z in k]
    url = ['https://analytics.tokenmetrics.com/api/predicted_monthly_index/',
      'https://analytics.tokenmetrics.com/api/predicted_weekly_index/']
    
    urls = urls + url
    for url in urls:
        res = requests.get(url)


while(True):
    hello_world()
    if datetime.now().hour >= 12:
        call_index()
    time.sleep(1*4)