CREATE PROCEDURE ConvertAuditLog( 
    @attribute_mask NVARCHAR(MAX),
    @changedata NVARCHAR(MAX)
)
AS

EXECUTE sp_execute_external_script 
@language = N'Python', 
@script = N'
import pandas as pd

df = pd.DataFrame([(item[0], item[1].split(",")) for item in list(zip(attribute_mask.split(","),changedata.split("~"))) if len(item[1])>0])
df.columns = ["entry", "text"] # name columns
df[["value","identifier"]] = pd.DataFrame(df.text.values.tolist(), index= df.index) # split text into two new columns
df = df[["entry","value","identifier"]] # only keep necessary

OutputDataSet = df
',
@output_data_1_name = N'df',
@params = N'@unseen_document NVARCHAR(MAX), @departmentteam NVARCHAR(MAX)', --Params the Python script uses
@attribute_mask_var = @attribute_mask,
@changedata_var = @changedata
WITH RESULT SETS
   ((
   [entry] NVARCHAR(25)
 , [value] NVARCHAR(50)
 , [identifier] NVARCHAR(50)
   ))