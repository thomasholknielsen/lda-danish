def getEngine(server, database, user=None, password=None, instance=None):
    # import pyodbc
    from sqlalchemy import create_engine
    import urllib
    """ Returns a connection engine as a sqlalchemy object """
    if user is None:
        # driver = 'SQL Server'
        driver = 'ODBC Driver 13 for SQL Server'
        params = urllib.parse.quote_plus(r'DRIVER={0};SERVER={1};DATABASE={2};Trusted_Connection=yes'.format(driver, server, database))
        conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
    else: 
        driver = 'ODBC Driver 13 for SQL Server'
        params = urllib.parse.quote_plus(r'DRIVER={0};{1};DATABASE={2};UID={3};PWD={4}'.format(driver, server, database, user, password))
        conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)

    if instance:
        conn_str = 'mssql+pyodbc://{0}:{1}@{2}/{3}\\{5}:1433?driver={4}'.format(user, password, server, database, driver, instance)

    # params = urllib.parse.quote_plus(r'DRIVER={SQL Server};SERVER=LIPHT-VM-01;DATABASE=Akademikernes_MSCRM_addition;Trusted_Connection=yes')
    # params = urllib.parse.quote_plus(r'DRIVER={SQL Server};SERVER=LIPHT-VM-01;DATABASE=Akademikernes_MSCRM_addition;UID=sa;PWD=password')
    # conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
    engine = create_engine(conn_str)
    return engine



### Above this line is necessary







def list_to_stringlist(test):
    return ', '.join(test)

def stringlist_to_list(test):
    return list(test.split(','))

def df_clean_sting(df, col_name):
    from utils.lipht_regex import re_accronyms, re_keyconcepts, re_lemma, re_remove, re_system
    from bs4 import BeautifulSoup
    import re
    """ Cleans 'col_name' in a variaty of ways.
        Specificly designed with danish e-mails in mind."""

    def _clean_html(text):
        return BeautifulSoup(text, 'html.parser').get_text()

    def remove_accronyms(text):
        re_accronyms_comp = {re.compile(k): v for k, v in re_accronyms.items()}
        for pattern, replacement in re_accronyms_comp.items():
            text = pattern.sub(replacement, text)
        return text

    def make_keyconcepts(text):
       re_keyconcepts_comp = {re.compile(k): v for k, v in re_keyconcepts.items()}
       for pattern, replacement in re_keyconcepts_comp.items():
           text = pattern.sub(replacement, text)
       return text

    def lemmatize(text):
        re_lemma_comp = {re.compile(k): v for k, v in re_lemma.items()}
        for pattern, replacement in re_lemma_comp.items():
            text = pattern.sub(replacement, text)
        return text
            
    df[col_name]=df[col_name].str.replace('</p><p>','</p>.\n<p>').astype(str) # Insert newline in paragraphs
    df[col_name]=df[col_name].apply(_clean_html) # parse any HTML
    df[col_name]=df[col_name].str.lower() # convert text to lower
    df[col_name]=df[col_name].replace(regex=re_accronyms) # replace anything found in the dictionary: re_accronyms
    df[col_name]=df[col_name].replace(regex=re_keyconcepts) # replace anything found in the dictionary: re_keyconcepts
    df[col_name]=df[col_name].replace(regex=re_lemma) # replace anything found in the dictionary: re_lemma
    # df[col_name]=df[col_name].apply(remove_accronyms)
    # df[col_name]=df[col_name].apply(make_keyconcepts)
    # df[col_name]=df[col_name].apply(lemmatize)
    # df[col_name]=df[col_name].apply(keyword_processor.replace_keywords) # Use FlashText KeywordProcesser -> https://github.com/vi3k6i5/flashtext -> Build dict of replacements
    df[col_name]=df[col_name].str.replace(r'^\n','') # remove newline in start of message
    df[col_name]=df[col_name].str.replace(r'^((\n*\s*|\n*\S*\s*)(\b|\b\w)(kære|hej|hejsa|hello|hi|dear|til)\s?(\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w.-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{2,}|\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w.-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{2,}|\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w-]{0,}|\w[a-zæøåA-ZÆØÅ\w.-]{2,})?(|\n))','') # remove greeting
    df[col_name]=df[col_name].str.replace(r'^\n','') # remove newline in start of message
    df[col_name]=df[col_name].str.replace(r'(?<=[^\d\W])\.','_') # replaces punctuation in URL's with a underscore
    df[col_name]=df[col_name].str.replace(r'(?<=[^\W]\d)\.(?=\d)','') # replaces punctuation in numbers with nothing, to avoide splitting of numbers
    df[col_name]=df[col_name].str.split(r'((med\svenlig\shilsen|de\sbedste\shilsner|bedste\shilsener|venlig\shilsen|hilsen|vh|best\sregards|with\skind\sregards|kind\sregards|with\sregards|regards|kh|sende\sfra\smin|sent\sfrom\smy\s).*)').str[0] # remove everything after polite ending
    df[col_name]=df[col_name].str.replace(r'^.*(tak\sfor\s(din\smail|en\sbehagelig\ssamtale|din\sbesked|svar|din\stilbagemelding|indsendte))\.?','')#.str[1] # remove everything before these statements 
    df[col_name]=df[col_name].str.replace(r'((tak\sfor\shjælpen|på\sforhånd\stak|på\sforhånd\smange\stak|jeg\sser\sfrem\stil\sat\shøre\sdig|håber\sdet\svar\ssvar\snok|mange tak).*)','')#.str[0] # Remove every match and everything after. # Mail ending
    df[col_name]=df[col_name].str.replace(r'((på\sforhånd\stak)|(((jeg|du)\sønske(s|r\s)(dig)?|(hav))\s)?(en\s)?((fortsat|rigtig)\s)?god(t)?\s(weekend|dag|ferie|jul|nytår|påske)|(held\sog\slykke\s(fremover)?)|(god\sarbejdslyst).*)','') # Remove every match and everything after. # Mail polite ending
    df[col_name]=df[col_name].replace(regex=re_remove) # replace anything found in the dictionary: re_remove
    df[col_name]=df[col_name].replace(regex=re_system) # replace anything found in the dictionary: re_system

    df[col_name]=df[col_name].str.replace(r'^\n','') # remove newline in start of message
    df[col_name]=df[col_name].str.strip()
    # df[col_name]=df[col_name].str.replace(r'[\w\.-]+@[\w\.-]+','') # remove e-mail

# Fjern 
# månedsnavne, dagnavne

def df_simple_clean_string(df, col_name):
    df[col_name]=df[col_name].str.lower() # convert text to lower
    df[col_name]=df[col_name].str.replace(r'^\n','') # remove newline in start of message
    df[col_name]=df[col_name].str.replace(r'^((\n*\s*|\n*\S*\s*)(\b|\b\w)(kære|hej|hejsa|hello|hi|dear|til)\s?(\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w.-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{2,}|\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w.-]{0,}\s\w[a-zæøåA-ZÆØÅ\w-]{2,}|\w[a-zæøåA-ZÆØÅ\w-]{2,}\s\w[a-zæøåA-ZÆØÅ\w-]{0,}|\w[a-zæøåA-ZÆØÅ\w.-]{2,})?(|\n))','') # remove greeting
    df[col_name]=df[col_name].str.replace(r'^\n','') # remove newline in start of message
    df[col_name]=df[col_name].str.split(r'((med\svenlig\shilsen|de\sbedste\shilsner|bedste\shilsener|venlig\shilsen|hilsen|vh|best\sregards|with\skind\sregards|kind\sregards|with\sregards|regards|kh|sende\sfra\smin|sent\sfrom\smy\s).*)').str[0] # remove everything after polite ending
    df[col_name]=df[col_name].str.replace(r'(?<=[^\d\W])\.','_') # replaces punctuation in URL's with a underscore
    df[col_name]=df[col_name].str.replace(r'(?<=[^\W]\d)\.(?=\d)','') # replaces punctuation in numbers with nothing, to avoide splitting of numbers
    df[col_name]=df[col_name].str.replace(r'[\.!]','') # Remove all punctuation and exclamationmarks
    

def save_model(engine, schema, table_name, model_file, model_guid, model_type=None, scope=None):
    """ 
    Pickles and saves a model to the specified server 

    :param sqlalchemy-engine: The connection object
    :param schema: the schema of the table
    :param table_name: the name of the table on the SQL Server
    :param model_file: the file that needs to be pickled
    :param model_guid: the unique GUID identifier of the model
    :param model_type: the type of model that is saved - e.g. Logistic Regression
    :param scope: where is this model applied and how
    :return: the response from the server
    
        A table with the following attributes should be created at the SQL server

        CREATE TABLE model.[repository](
            ID INT IDENTITY(1, 1) NOT NULL,
            CreatedDate datetime DEFAULT(GETDATE()),
            model_guid NVARCHAR(250) NULL,
            pickle_file [varbinary](MAX) NULL,
            model_type NVARCHAR(500) NULL,
            scope NVARCHAR(500) NULL
        ) 
    """


    import pickle
    connection = engine.connect()

    try:
        pickle_file = pickle.dumps(model_file, pickle.HIGHEST_PROTOCOL)

        query = """INSERT INTO {0}.{1}(pickle_file, model_guid, model_type, scope) VALUES (?, ?, ?, ?)
                """.format(schema, table_name)
        response = connection.execute(query, pickle_file, model_guid, model_type, scope)

    except Exception as e:
        print(e)
        response = None

    finally:
        return response

def get_model(engine, schema, table_name, model_guid):
    """
    Uploades a model to the server 
    
    :param sqlalchemy-engine: The connection object
    :param schema: the schema of the table
    :param table_name: the name of the table on the SQL Server
    :param model_guid: the unique GUID identifier of the model to be retrieved
    :return: the model that was requested

    """
    import pickle
    connection = engine.connect()
    try:       
        query = """SELECT pickle_file FROM {0}.{1} WHERE model_guid = '{2}'""".format(schema, table_name,model_guid)
        print(query)
        rows = connection.execute(query)
        for row in rows:
            ## The result is also in a tuple
            for pickle_file in row:
                model_file = pickle.loads(pickle_file)

    except Exception as e:
        model_file = None

    finally:
        return model_file