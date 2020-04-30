# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import pandas as pd
import os
from datetime import datetime
import uuid
import sqlalchemy


from utils.lipht_visualization import topic_distribution_barplot
from utils.lipht_lda_utils import PrepareDictionary, lda_predict_df, df_lda_features, get_topics_and_probability, get_lda_topics, df_lda_preprocessing, TrainLDAModel
from utils.lipht_data import getEngine, save_model, list_to_stringlist, stringlist_to_list

import sys
import warnings

import logging.config

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__=='__main__':
    logging.config.fileConfig('logging.conf')
    logging = logging.getLogger(__name__)

    time_start = datetime.now()
    print('START Training')
    logging.info('START')
    
    # TRAINING PIPELINE
    # ENABLES differentiated number of topics per team
    # Create a new pipeline and test it out
    from train_pipeline import pipe as train_pipeline

    # process column types
    for k in train_pipeline.keys():

        engine = getEngine('THN-P53','GRE')

        query = train_pipeline[k].get('query')
        df_scope = pd.read_sql(query, engine)
        cur_dir = os.getcwd()

        # Get data
        df_sql = pd.read_sql(query, engine)

        # Default values
        n_gram = 1
        no_above = 1
        no_below= 8 # filter out tokens that appear in less than X documents
        random_state=1
        lda_sample_size = 10000
        lda_num_topics_start = 10 # To only train on a X number of topics, _start = X, and _end = X+1
        lda_num_topics_end = 11
        lda_num_topics_increment = 1
        lda_chunksize = 500
        lda_passes = 50
        lda_alpha = 'auto'
        lda_eta = 'auto'
        lda_minimum_probability = 0.005
        target_column = 'ContentDesriptionText'
        target_column_id = 'ContentID'
        predict_by_column = 'stopwords_removed'
        preproces_remove_stopwords = True
        only_keep_best_per_row = True
        no_of_words_per_topic = 100
        create_distribution_plot = False # When True - Creates a distribution plot
        iteration_guid = uuid.uuid4() # Don't change
        add_column_features = False
        add_preprocessing_to_sql = True # Will replace existing
        model_comment = 'No comment found'

    # # process column types
    for k in train_pipeline.keys():
        log = {
                'target_column': train_pipeline[k].get('target_column', target_column),
                'target_column_id': train_pipeline[k].get('target_column_id', target_column_id),
                'preproces_remove_stopwords': train_pipeline[k].get('preproces_remove_stopwords',preproces_remove_stopwords),
                'iteration_guid': iteration_guid,
        }

        # Preprocess dataset
        if train_pipeline[k]['Tests']:

            df = df_sql.copy(deep=True) # Make a copy of the dataframe

            # Downsample for speed increase
            samplesize = 100
            df = df.sample(samplesize, random_state=random_state).copy(deep=True)

            # if add_preprocessing_to_sql:
                # Retrieve data from lda_target_column_clean, and get all cases already processed
                # Split the data into preprocessed and not-preprocessed
                # df = not-preprocessed


            # Process data, add necessary columns for predictions
            time_clean_start = datetime.now()
            logging.info('Cleaning {} rows'.format(df.shape[0]))
            
            df_lda_preprocessing(df, log['target_column'], remove_stopwords=log['preproces_remove_stopwords'], add_features=add_column_features)
            
            time_clean_end = datetime.now()
            logging.info('Cleaned {} rows, with time: {}'.format(df.shape[0], datetime.strptime(str(time_clean_end-time_clean_start), '%H:%M:%S.%f').time()))

            # Send to cleaned data to SQL Server
            if add_preprocessing_to_sql:
                time_sql_start = datetime.now()
                logging.info('Transfering data to SQL Server')

                preprocess_to_sql = df.copy(deep=True)
                preprocess_to_sql.rename(columns={
                    log['target_column_id']:'target_column_id',
                    log['target_column']:'target_column',
                    }, inplace = True)
                preprocess_to_sql['iteration_guid'] = pd.Series(log['iteration_guid'], index=preprocess_to_sql.index)

                # To be able to send lists to SQL SERVER, we need to convert them to strings as the pling will result in an error
                convert_to_string_list = [x for x in list(preprocess_to_sql.columns) if x in ['ngrams', 'bigrams','trigrams','tokenized_text','stopwords_removed']]
                for column in convert_to_string_list:
                    preprocess_to_sql[column] = preprocess_to_sql[column].apply(list_to_stringlist)

                preprocess_to_sql.to_sql(name='lda_target_column_clean',con=engine , schema='dbo', if_exists='append', index=False)
                
                time_sql_end = datetime.now()
                logging.info('Transfer complete with time: {}'.format(datetime.strptime(str(time_sql_end-time_sql_start), '%H:%M:%S.%f').time()))

                # df.append(already_preprocess_df)

            for i in train_pipeline[k]['Tests']:
                logging.info('Start Test')
                # print('\t{ThreadInitiatedBy}, {ThreadResponsibleDepartmentTeam}, {lda_num_topics_start}'.format(**i))
                # Create the model
                log_test = {
                    'GroupColumn': i.get('GroupingColumn'),
                    'GroupValue': i.get('GroupValue'),
                    'predict_by_column': i.get('predict_by_column', predict_by_column),
                    'n_gram': i.get('n_gram', n_gram),
                    'no_above': i.get('no_above', no_above),
                    'no_below': i.get('no_below', no_below),
                    'random_state': i.get('random_state', random_state),
                    'lda_sample_size': i.get('lda_sample_size', None),
                    'lda_num_topics_start': i.get('lda_num_topics_start', lda_num_topics_start),
                    'lda_num_topics_end': i.get('lda_num_topics_end', lda_num_topics_end),
                    'lda_num_topics_increment': i.get('lda_num_topics_increment', lda_num_topics_increment),
                    'lda_chunksize': i.get('lda_chunksize', lda_chunksize),
                    'lda_passes': i.get('lda_passes', lda_passes),
                    'lda_alpha': i.get('lda_alpha', lda_alpha),
                    'lda_eta': i.get('lda_eta', lda_eta),
                    'lda_minimum_probability': i.get('lda_minimum_probability', lda_minimum_probability),
                    'model_comment': i.get('model_comment', model_comment),
                }
                if log_test['lda_sample_size'] == 0 or log_test['lda_sample_size'] == None or log_test['lda_sample_size'] > df.shape[0]:
                    # Set the sample to the total no of rows in the dataframe
                    log_test['lda_sample_size'] = df.shape[0]

                # Update the log with the data for the current iteration.
                log.update(log_test)

                # Slice the data to the scope of the model training
                if i['GroupColumn'] == 'None':
                    df_scope = df[df[log['target_column']].isnull()==False].copy(deep=True)
                else:
                    df_scope = df[(df[log['target_column']].isnull()==False) & (df[log['GroupColumn']].str.contains(log['GroupValue']==True))].copy(deep=True) # NEEDS TESTING OF GROUPING FUNCTIONALITY

                # log row count
                log['rows'] = df_scope.shape[0]
                
                # Split into training and population 
                df_train = df_scope.sample(n=log['lda_sample_size'], random_state=log['random_state'])#.copy(deep=True)
                # df_population = df_scope.loc[~df_scope.index.isin(df_train.index)] # Use this to se the dataset for what is not used for training.
                
                # Creates BoW and WordVector for prediction
                dictionary, corpus = PrepareDictionary(df_train, log['predict_by_column'], log['no_above'], log['no_below'], log)
                
                time_delta = datetime.now()
                logging.info('NOW Training {}-{}-{}'.format(log['target_column'],log['GroupColumn'],log['GroupValue']))
                for noTopics in range(log['lda_num_topics_start'],log['lda_num_topics_end'],log['lda_num_topics_increment']):
                    
                    # All below chould be in its own function
                    log['lda_num_topics'] = noTopics
                    log['model_guid'] = uuid.uuid4() 

                    # Crreate scope
                    log['scope'] = 'Target_{0}_Group-{1}_Topics_{2}'.format(log['target_column'], log['GroupValue'], log['lda_num_topics'])
                    # Create identifier
                    log['identifier'] = '{0}_Model_{1}'.format(log['scope'], log['model_guid'])
                    

                    # Train model
                    train_lda_model = TrainLDAModel(corpus, dictionary, log['lda_num_topics'], 4, log['lda_chunksize'], log['lda_passes'], log['lda_alpha'], log['lda_eta'], log['lda_minimum_probability'], log['random_state'], log)

                    import os
                    cur_dir = os.getcwd()
                    directory=log['identifier']

                    if directory:
                        # This should save the model to the database instead of to a file
                        dictionary.save(os.path.join(cur_dir,'data','{0}_dictionary.pkl'.format(directory)))
                    
                    if train_lda_model:
                        # train_lda_model.save(os.path.join(cur_dir,'data','{0}_model'.format(directory)))
                        save_model(engine, 'dbo', 'lda_model_repository', train_lda_model, log['model_guid'], 'model_LDA', log['scope'])
                    
                    # Make model features
                    df_lda_features(train_lda_model, df_train)

                    # Create distribution plot
                    if create_distribution_plot:
                        title = '{0}_Distribution'.format(log['identifier'])
                        topic_distribution_barplot(train_lda_model, df_train, 2, title)

                    # Get topics and probability
                    topics = get_topics_and_probability(df_train, train_lda_model, log['lda_num_topics'], no_of_words_per_topic)
                    #df_topwords, topics_dist = get_topics_and_probability(df_train, train_lda_model, log['lda_num_topics'], no_of_words_per_topic)

                    

                    topics['Group'] = pd.Series(log['GroupValue'], index=topics.index)
                    topics['target_column'] = pd.Series(log['target_column'], index=topics.index)
                    topics['lda_num_topics'] = pd.Series(log['lda_num_topics'], index=topics.index)
                    topics['model_guid'] = pd.Series(log['model_guid'], index=topics.index)
                    topics['iteration_guid'] = pd.Series(log['iteration_guid'], index=topics.index)
                    topics['datetime'] = pd.Series(datetime.now(), index=topics.index)

                    # Save topics to server
                    topics.to_sql(name='lda_topics_index' ,con=engine , schema='dbo', if_exists='append', index=False)

                    # Create dataset for predictions by train dataframe
                    # df_prediction = df_train[['ThreadID',log['target_column_id'],'ThreadInitiatedBy','text', predict_by_column,'bow']].copy(deep=True)
                    
                    # Predict on the dataset
                    # df_prediction = lda_predict_df(df_prediction, predict_by_column , train_lda_model, dictionary, only_best_prediction=only_keep_best_per_row) # Changed to enable insertion of all predictions
                    df_prediction = lda_predict_df(df_scope, predict_by_column , train_lda_model, dictionary, only_best_prediction=only_keep_best_per_row) # Changed to enable insertion of all predictions

                    # Append with out_of_scope
                    # df_prediction.append(df_out_of_scope)

                    # Slim to necessary columns
                    df_prediction = df_prediction[['ThreadID',log['target_column_id'],'ThreadInitiatedBy','pred_probability','pred_index']]
                    df_prediction.rename(columns={log['target_column_id']:'MessageID'}, inplace = True)

                    # Add additional info to reference the model
                    df_prediction['ThreadResponsibleDepartmentTeam'] = pd.Series(log['ThreadResponsibleDepartmentTeam'], index=df_prediction.index)
                    df_prediction['target_column'] = pd.Series(log['target_column'], index=df_prediction.index)
                    df_prediction['lda_num_topics'] = pd.Series(log['lda_num_topics'], index=df_prediction.index)
                    df_prediction['model_guid'] = pd.Series(log['model_guid'], index=df_prediction.index)
                    df_prediction['iteration_guid'] = pd.Series(log['iteration_guid'], index=df_prediction.index)
                    df_prediction['datetime'] = pd.Series(datetime.now(), index=df_prediction.index)

                    # Save predictions to server
                    df_prediction.to_sql(name='topics_predictions',con=engine , schema='input', if_exists='append', index=False)

                    # Save log to server
                    # df_log.pop('target_column_id', None) # Drop id_column as it has not been implemented yet
                    df_log = pd.DataFrame(log, index=['0']) # create df from dict
                    df_log.to_sql(name='log_multi_model' ,con=engine , schema='input', if_exists='append', index=False) # Save log to server

    time_end = datetime.now()
    logging.info('END')
    logging.info('TOTAL TIME {}'.format( datetime.strptime(str(time_end-time_start), '%H:%M:%S.%f').time() ))
    print('DONE - see log for details.')