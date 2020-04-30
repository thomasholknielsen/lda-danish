EXEC [sys].[sp_execute_external_script]
@language = N'Python'
, @script = N'
import pandas as pd

attribute_mask = ''CRM Setup,5,28,18,71,87,91,96,78,19,23,17,35,82,105,73,20,11,21,85,74,109,80,10,90,77,16,81,13,107,101,7,8,37,75,79,36,34,104,6,24,9,103,100,22,2,99,72,30,12,15,83,88,76,14,93,86,84,''
changedata = ''False~0~10/15/2018 06:14:34~~~~team,c62bb782-6c90-e211-9da9-005056ad53c8~~1~~~~~~~contact,a78feeb2-f9e4-e611-b361-005056ad2d14~0~~~~~~10014~~~systemuser,a98b625a-3895-e611-8be5-0050569118b0~~~False~~~10/15/2018 06:14:34~False~~~False~~~systemuser,a98b625a-3895-e611-8be5-0050569118b0~systemuser,a98b625a-3895-e611-8be5-0050569118b0~~~~False~~~~True~~1~~False~~~~~''

p = pd.DataFrame([(item[0], item[1].split(",")) for item in list(zip(attribute_mask.split(","),changedata.split("~"))) if len(item[1])>0])
p.columns = [''entry'', ''text''] # name columns
p[[''value'',''identifier'']] = pd.DataFrame(p.text.values.tolist(), index= p.index) # split text into two new columns
p = p[[''entry'',''value'',''identifier'']] # only keep necessary
OutputDataSet = p

'
WITH RESULT SETS
   (
   (
   [entry] NVARCHAR(25)
 , [value] NVARCHAR(50)
 , [identifier] NVARCHAR(50)
   )
   )