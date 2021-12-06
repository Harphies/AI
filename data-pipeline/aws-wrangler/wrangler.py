import awswrangler as wr
import pandas as pd
from datetime import datetime


df = pd.DataFrame({"id": [1, 2], "value": ["foo", "boo"]})

# Storing data on Data Lake
wr.s3.to_parquet(
    df=df,
    path="s3://bucket/dataset",
    dataset=True,
    database="my_db",
    table="my_table"
)

# Retreiveing data directly from S3
df = wr.s3.read_parquet(
    "s3://bucket/dataset",
    dataset=True
)

# Retrieving data from Amazon athena
df = wr.athena.read_sql_query(
    "SELECT * FROM my_table",
    database="my_db"
)

# Get a Redshift connection from Glue Catalog and retrieving data from Redshift spectrum
con = wr.redshift.connect("my-glue-connection")
df = wr.redshift.read_sql_query(
    "SELECT * FROM external_schema.my_table", con=con)
con.close()

# Amazon TimeStream Write
df = pd.DataFrame({
    "time": [datetime.now(), datetime.now()],
    "my_dimension": ["foo", "boo"],
    "measure": [0.1, 1.1]
})

rejected_records = wr.timestream.write(
    df,
    database="sampleTable",
    time_col="time",
    measure_col="measure",
    dimensions_cols=["my_dimension"]
)

# Amazon TimeStream Query
wr.timestream.query(
    """
    SELECT time, measure_value::double, my_dimension
    FROM "sample"."sampleTable" ORDER BY time DESC LIMIT 3
    """)
