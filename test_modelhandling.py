import pandas as pd
import sqlalchemy
import os
from datetime import datetime

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary


from utils.lipht_data import getEngine, save_model, get_model



# engine = getEngine('LIPHT-VM-01','Akademikernes_MSCRM_Addition')
engine = getEngine('LI-PH-01','Python')

query = """ 
    
"""


# df_scope = pd.read_sql(query, engine)
cur_dir = os.getcwd()

# model data
scope = 'AKAandMember-ThreadInitiatedBy_All_Topics-50_Model-c5aa52fa-8e0c-4f67-8f13-19a57a960022_model'
path = 'data'
model = '{}'.format(scope)
dictionary = model + '.id2word'

# load the model and data

model_dir = os.path.join(cur_dir,path,model)
# dictionary_dir = os.path.join(cur_dir,path,dictionary)

# LDAmodel_scope = LdaMulticore.load(model_dir)
# LDAmodel_dictionary = Dictionary.load(dictionary_dir)

test_doc = os.path.join(cur_dir, path, 'test.txt')
# print(test_doc)

# with open(test_doc) as f:
#     print(f.read())

import tempfile
io_file = tempfile.TemporaryFile()

data = "This is a simple string that we are testing"

# io_file.write(b'test tmp file')
io_file.write(bytes(data,'utf-8'))
io_file.seek(0)
print(io_file.read().decode())

# with open(io_file) as t:
#     print(t.read())

# io_file.seek(0)
# print(io_file.read())

# print(os.path.getsize(io_file))

# io_file.close()





# test
import uuid

model_guid = uuid.uuid4()
trained_model = LdaMulticore.load(model_dir)

# save_model(engine, 'model', 'repository',trained_model, model_guid, 'LDA_model', 'AKA - Requests Topic finder')

lda_model = get_model(engine, 'model', 'repository', '55644D1D-D187-4347-8DCD-C94A67F5D5A5')
print(lda_model)


print('COMPLETE')

def test():
    import pickle
    # from sqlalchemy.dialects.mssql import BINARY

    ## Create a semi-complex list to pickle
    listToPickle = LdaMulticore.load(model_dir)

    ## Pickle the list into a string
    pickledList = pickle.dumps(listToPickle, pickle.HIGHEST_PROTOCOL)

    connection = engine.connect()

    ## Create a cursor for interacting

    # cursor = connection.cursor()

    ## Add the information to the database table pickleTest
    connection.execute("""INSERT INTO dbo.model_test(id, binary_model) VALUES (?, ?)""", (1, pickledList))

    ## Select what we just added
    result = connection.execute("""SELECT binary_model FROM dbo.model_test WHERE id = 1""")

    ## Dump the results to a string
    rows = result.fetchall()

    ## Get the results
    for each in rows:
        ## The result is also in a tuple
        for pickledStoredList in each:
            ## Unpickle the stored string
            unpickledList = pickle.loads(pickledStoredList)
            print(unpickledList)



# # Make sure you store your pickled 
# # cur.execute("create table pickled(id integer primary key, data blob)")
# cur.execute("create table pickled(id integer primary key, data blob)")

# # Here we force pickle to use the efficient binary protocol
# # (protocol=2). This means you absolutely must use an SQLite BLOB field
# # and make sure you use sqlite3.Binary() to bind a BLOB parameter.
# p1 = Point(3, 4)
# cur.execute("insert into pickled(data) values (?)", (sqlite3.Binary(pickle.dumps(p1, protocol=2)),))


# # Fetch the BLOBs back from SQLite
# cur.execute("select data from pickled")
# for row in cur:
# serialized_point = row[0]

# # Deserialize the BLOB to a Python object - # pickle.loads() needs a
# # bytestring.
# point = pickle.loads(str(serialized_point))
# print "got point back from database", point


print('COMPLETE PICK')