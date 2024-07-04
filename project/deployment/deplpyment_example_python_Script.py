import os
import sys
import pickle
import pandas as pd
import sklearn


year = int(sys.argv[1])
month = int(sys.argv[2])

input_file = f'/workspaces/mle-zoomcamp-prj/project/src_data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

MODEL_FILE = os.getenv('MODEL_FILE', '/workspaces/mle-zoomcamp-prj/project/models/lin_reg.bin')



with open('MODEL_FILE', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(input_file)


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)



y_pred_mean = y_pred.mean()

print('predicted mean duration:', y_pred_mean)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

os.makedirs('output', exist_ok=True)

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)