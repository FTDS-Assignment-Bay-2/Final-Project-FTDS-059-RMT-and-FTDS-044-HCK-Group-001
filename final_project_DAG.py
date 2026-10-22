import pandas as pd
import datetime as dt
from datetime import timedelta
import psycopg2 as db
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import json

# Extract
def queryPostgresql():
    '''
    Fungsi ini bertujuan untuk mengambil data dari postgres SQL dan menyimpannya dalam format csv
    '''
    conn_string="dbname='hck_final_project' host='postgres' port='5432' user='airflow' password='airflow'"
    conn=db.connect(conn_string)

    df=pd.read_sql("SELECT * FROM produk_lipstick", conn)

    df.to_csv('/opt/airflow/dags/sqldata.csv', index=False)
    print("-------Data Saved------")

# Transform
def cleanData():
    '''
    Fungsi ini bertujuan untuk membaca data csv lalu melakukan cleaning tiap row dari data tersebut dan menyimpannya dalam file csv baru
    '''
    df=pd.read_csv('/opt/airflow/dags/sqldata.csv')

    df = df.drop_duplicates() # drop duplicates data

    #ubah data harga menjadi numerik
    df['harga'] = (df['harga'].astype(str).str.replace(r'[^\d]', '', regex=True))
    df['harga'] = pd.to_numeric(df['harga'])

    df.to_csv('/opt/airflow/dags/dataset_clean.csv', index=False) #clean di CSV baru
    print("-------Data Saved------")


#Load
def loadElastic():
    '''
    Fungsi ini bertujuan untuk load data CSV yang telah dilakukan cleaning ke dalam elasticsearch
    '''
    es = Elasticsearch('elasticsearch:9200') 
    df=pd.read_csv('/opt/airflow/dags/dataset_clean.csv')

    actions = (
        {
            "_index": "lipmatch",
            "_source": doc
        }
        for doc in json.loads(df.to_json(orient="records"))
    )

    response = helpers.bulk(es, actions)
    print(response, "Bulk Insert selesai")

default_args = {
    'owner': 'hck-044',
    'start_date': dt.datetime(2026, 10, 22),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=15),
}

with DAG('final_project',
         default_args=default_args,
         schedule_interval= '0 10 * * 7',
         catchup= False
         ) as dag:

    extractData = PythonOperator(task_id='select',
                                python_callable=queryPostgresql)

    transformData = PythonOperator(task_id='clean',
                                 python_callable=cleanData)
    
    loadData = PythonOperator(task_id='load',
                                 python_callable=loadElastic)


extractData >> transformData >> loadData