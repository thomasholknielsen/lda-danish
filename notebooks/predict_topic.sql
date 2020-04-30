DROP PROCEDURE IF EXISTS PredictStringTopic;
GO

CREATE PROCEDURE PredictStringTopic( 
    @Message NVARCHAR(MAX),
    @ThreadResponsibleDepartmentTeam NVARCHAR(MAX)
)
AS

/*Test using the following*/
--DECLARE @Message NVARCHAR(MAX) = 'Hej, Så hvad skal jeg gøre nu, for at kunne få a-kasse? Eller vil de automatisk komme ind på min konto, med de ting jeg allerede har gjort?Venlig hilsenMads Thomsen';
--DECLARE @ThreadResponsibleDepartmentTeam NVARCHAR(MAX) = 'Udbetaling'

EXEC sp_execute_external_script
@language =N'Python',
@script=N'
# Load Modules
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
import pandas as pd

# Load Custom models
# Can be found in C:\Program Files\Microsoft SQL Server\MSSQL14.MSSQLSERVER\PYTHON_SERVICES\akademikernes_diagnostic
from lipht_lda import lda_predict_string

def load_topic_names(departmentteam):
    dicts_from_file = []
    with open("data/{}_lda_topic_names.txt".format(departmentteam),"r") as inf:
        for line in inf:
            dicts_from_file.append(eval(line))
    return dicts_from_file[0]

path = "data/model/"
model = departmentteam # Takes input
dictionary = model + "_dictionary.pkl"
lda_topic_names = load_topic_names(departmentteam)

LDAmodel_scope = LdaMulticore.load(path+model)
LDAmodel_dictionary = Dictionary.load(path+dictionary)

result = lda_predict_string(unseen_document, LDAmodel_scope, LDAmodel_dictionary, lda_topic_names)

df = pd.DataFrame({"Probability":result[0],
                   "Topic": result[1],
                   "Words": result[2]
                  }, index=[0])


# SQL SERVER SPECIFIC
OutputDataSet = InputDataSet

OutputDataSet = df
',
@output_data_1_name = N'df',
@params = N'@unseen_document NVARCHAR(MAX), @departmentteam NVARCHAR(MAX)', --Params the Python script uses
@unseen_document = @Message,
@departmentteam = @ThreadResponsibleDepartmentTeam   --LHS must match up with params inside Python script
WITH RESULT SETS (("Probability" FLOAT NULL, "Topic" NVARCHAR(150) NULL,"Words" NVARCHAR(350) NULL))
