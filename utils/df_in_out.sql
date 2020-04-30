USE [Akademikernes_MSCRM]
GO

DECLARE @input_query NVARCHAR(MAX)
    = N'SELECT TOP 9
     [AuditID]
   , [AttributeMask] = SUBSTRING([A].[AttributeMask], 2, LEN([A].[AttributeMask]) - 2)
   , [ChangeData]
FROM [Akademikernes_MSCRM].[dbo].[AuditBase] [A]';



EXEC [sys].[sp_execute_external_script]
@language = N'Python'
, @script = N'
import pandas as pd

# Get data from input query
df = my_input_data
    
df["AttributeMask"] = df["AttributeMask"].str.split(",")
df["ChangeData"] = df["ChangeData"].str.split("~")
rows = []
for _, row in df.iterrows():
    [rows.append([row["AuditID"], key, val]) for key, val in zip(row["AttributeMask"], row["ChangeData"])]

df_out = pd.DataFrame(rows)
df_out.columns = ["AuditID", "AttributeMask","ChangeData"] # name columns


OutputDataSet = df_out
'
, @input_data_1 = @input_query
, @input_data_1_name = N'my_input_data'
,@output_data_1_name = N'OutputDataSet'
WITH RESULT SETS
   ((
   AuditId NVARCHAR(25)
 , AttributeMask NVARCHAR(50)
 , ChangeData NVARCHAR(50)
   ))