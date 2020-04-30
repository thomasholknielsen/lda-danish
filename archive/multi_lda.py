import pandas as pd
import os
from datetime import datetime
import uuid
import sqlalchemy

from utils.lipht_visualization import topic_distribution_barplot
from utils.lipht_lda_utils import PrepareDictionary, lda_predict_df, df_lda_features, get_topics_and_probability, get_lda_topics, df_lda_preprocessing, TrainLDAModel
from utils.lipht_data import getEngine

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__=='__main__':
    time_start = datetime.now()
    print('{}\t START'.format(time_start))
    # Create connection to SQL DB
    # engine = getEngine('LI-PH-01','Akademikernes_MSCRM')
    engine = getEngine('LIPHT-VM-01','Akademikernes_MSCRM_Addition')
    query="""
        WITH lang AS(
            SELECT 
                *,CASE 
					WHEN [pred_index] = 5 THEN 0
					WHEN [pred_index] = 7 THEN 0
					ELSE 1
				END AS 'Pred_Danish'
				,
                ROW_NUMBER() OVER (PARTITION BY FirstMemberMessageID ORDER BY [pred_probability],[pred_index]) as rn
                FROM [Akademikernes_MSCRM_Addition].[input].[language_predictions] l
        )
        SELECT 
			a.*,
			CASE 
				WHEN FirstMemberMessageBody LIKE 'Dear%' THEN 0
				WHEN FirstMemberMessageBody LIKE 'Hi%' THEN 0
				ELSE 1
			END AS 'Remove_English'
			,l.Pred_Danish
        FROM [Akademikernes_MSCRM_Addition].[out].[LDA_Messages_persisted] a
        LEFT JOIN lang l ON a.FirstMemberMessageID = l.FirstMemberMessageID
        WHERE l.rn = 1
        --and [pred_index] not in (5,7) 
        --and l.[pred_probability] > 0.1
        --AND FirstMemberMessageBody NOT LIKE 'Dear%'
        --AND FirstMemberMessageBody NOT LIKE 'Hi%'
        --AND LEN(FirstMessageBody)>1
    """
    df_scope = pd.read_sql(query, engine)
    cur_dir = os.getcwd()

    # Get data
    df_sql = pd.read_sql(query, engine)

    # List of teams to iterate over
    # ThreadResponsibleDepartmentTeam = ['All','Udbetalingsteam','Medlemskabsteam','Job','Logistik','Team Rådighed','Ikke Fordelt Team','Øvrige']
    ThreadResponsibleDepartmentTeam = ['Job','Logistik']
    ThreadInitiatedBy = ['Member', 'AKA','AKAandMember']


    # TRAINING PIPELINE
    # ENABLES differentiated number of topics per team
    # Create a new pipeline and test it out
    from train_pipeline import train_pipeline_1 as train_pipeline

    # Default values
    n_gram = 1
    sample_size= 10000
    no_above = 1
    no_below= 8 # filter out tokens that appear in less than X documents
    random_state=1
    lda_num_topics_start = 30 # To only train on a X number of topics, _start = X, and _end = X+1
    lda_num_topics_end = 31
    lda_num_topics_increment = 1
    lda_chunksize = 500
    lda_passes = 50
    target_column = 'FirstMemberMessageBody'
    target_column_id = 'FirstMemberMessageID'
    predict_by_column = 'stopwords_removed'
    preproces_remove_stopwords = True
    only_keep_best_per_row = True
    no_of_words_per_topic = 20
    create_distribution_plot = False # When True - Creates a distribution plot
    iteration_guid = uuid.uuid4() # Don't change

    for initiator in ThreadInitiatedBy:

        for team in ThreadResponsibleDepartmentTeam:
            log = {
                'ThreadResponsibleDepartmentTeam': team, # Sub Grouping
                'ThreadInitiatedBy': initiator, # Main Grouping
                'target_column': train_pipeline[initiator].get('target_column') or target_column,
                'target_column_id': train_pipeline[initiator].get('target_column_id') or target_column_id,
                'predict_by_column': predict_by_column,
                'n_gram': n_gram,
                'no_above': no_above,
                'no_below': no_below,
                'random_state': random_state,
                'lda_num_topics_start': train_pipeline[initiator][team].get('lda_num_topics_start') or lda_num_topics_start,
                'lda_num_topics_end': train_pipeline[initiator][team].get('lda_num_topics_end') or lda_num_topics_end,
                'lda_num_topics_increment': train_pipeline[initiator][team].get('lda_num_topics_increment') or lda_num_topics_increment,
                'lda_chunksize': lda_chunksize,
                'lda_passes': lda_passes,
                'iteration_guid': iteration_guid,
                'preproces_remove_stopwords': preproces_remove_stopwords
            }

            # Slice the data to the scope of the model training
            if initiator == 'AKAandMember':
                if team == 'All':
                    df_scope = df_sql[df_sql[log['target_column']].isnull()==False].copy(deep=True)
                else:
                    df_scope = df_sql[(df_sql[log['target_column']].isnull()==False) & (df_sql['ThreadResponsibleDepartmentTeamGroup'].str.contains(team)==True)].copy(deep=True)
            else:
                df_scope = df_sql[(df_sql[log['target_column']].isnull()==False) & (df_sql['ThreadResponsibleDepartmentTeamGroup'].str.contains(team)==True) & (df_sql['ThreadInitiatedBy']==log['ThreadInitiatedBy'])].copy(deep=True)
            # log row count
            log['rows'] = df_scope.shape[0]
            
            # Split into two df one for training the model (not English, and NaN), and one with all data (English and NaN)
            df_train = df_scope[(df_scope['Remove_English']==1) & (df_scope['Pred_Danish']==1)].copy(deep=True)         # For model training
            df_out_of_scope = df_scope[(df_scope['Remove_English']==0) & (df_scope['Pred_Danish']==0)].copy(deep=True)  # Out of scope data. To be appended

            # Process data, add necessary columns for predictions
            df_lda_preprocessing(df_train, log['target_column'], remove_stopwords=log['preproces_remove_stopwords'])
            
            # Creates BoW and WordVector for prediction
            dictionary, corpus = PrepareDictionary(df_train, log['predict_by_column'], log['no_above'], log['no_below'], log)
            
            time_delta = datetime.now()
            print('{}\t\tNOW Training {}-{}'.format(time_delta,log['ThreadInitiatedBy'],log['ThreadResponsibleDepartmentTeam']))
            for noTopics in range(log['lda_num_topics_start'],log['lda_num_topics_end'],log['lda_num_topics_increment']):
                # All below chould be in its own function
                log['lda_num_topics'] = noTopics
                log['model_guid'] = uuid.uuid4() 

                # Create identifier
                log['identifier'] = '{0}-ThreadInitiatedBy_{1}_Topics-{2}_Model-{3}'.format(log['ThreadInitiatedBy'], log['ThreadResponsibleDepartmentTeam'], log['lda_num_topics'], log['model_guid'])

                # Train model
                train_lda_model = TrainLDAModel(corpus, dictionary, log['lda_num_topics'], 4, log['lda_chunksize'], log['lda_passes'], log['random_state'], log, directory=log['identifier'])
                 
                # Make model features
                df_lda_features(train_lda_model, df_train)

                # Create distribution plot
                if create_distribution_plot:
                    title = '{0}_Distribution'.format(log['identifier'])
                    topic_distribution_barplot(train_lda_model, df_train, 2, title)

                # Get topics and probability
                topics = get_topics_and_probability(df_train, train_lda_model, log['lda_num_topics'], no_of_words_per_topic)
                topics['ThreadResponsibleDepartmentTeam'] = pd.Series(log['ThreadResponsibleDepartmentTeam'], index=topics.index)
                topics['ThreadInitiatedBy'] = pd.Series(log['ThreadInitiatedBy'], index=topics.index)
                topics['lda_num_topics'] = pd.Series(log['lda_num_topics'], index=topics.index)
                topics['model_guid'] = pd.Series(log['model_guid'], index=topics.index)
                topics['iteration_guid'] = pd.Series(log['iteration_guid'], index=topics.index)
                topics['datetime'] = pd.Series(datetime.now(), index=topics.index)

                # Save topics to server
                topics.to_sql(name='topics_index' ,con=engine , schema='input', if_exists='append', index=False)

                # Create dataset for predictions
                df_prediction = df_train[['ThreadID',log['target_column_id'],'ThreadInitiatedBy','text',predict_by_column,'bow']].copy(deep=True)
                
                # Predict on the dataset
                df_prediction = lda_predict_df(df_prediction, predict_by_column , train_lda_model, dictionary, only_best_prediction=only_keep_best_per_row) # Changed to enable insertion of all predictions

                # Append with out_of_scope
                df_prediction.append(df_out_of_scope)

                # Slim to necessary columns
                df_prediction = df_prediction[['ThreadID',log['target_column_id'],'ThreadInitiatedBy','pred_probability','pred_index']]
                df_prediction.rename(columns={log['target_column_id']:'MessageID'}, inplace = True)

                # Add additional info to reference the model
                df_prediction['lda_num_topics'] = pd.Series(log['lda_num_topics'], index=df_prediction.index)
                df_prediction['ThreadResponsibleDepartmentTeam'] = pd.Series(log['ThreadResponsibleDepartmentTeam'], index=df_prediction.index)
                df_prediction['datetime'] = pd.Series(datetime.now(), index=df_prediction.index)
                df_prediction['model_guid'] = pd.Series(log['model_guid'], index=df_prediction.index)
                df_prediction['iteration_guid'] = pd.Series(log['iteration_guid'], index=df_prediction.index)

                # Save predictions to server
                df_prediction.to_sql(name='topics_predictions',con=engine , schema='input', if_exists='append', index=False)

                # Save log to server
                # df_log.pop('target_column_id', None) # Drop id_column as it has not been implemented yet
                df_log = pd.DataFrame(log, index=['0']) # create df from dict
                df_log.to_sql(name='log_multi_model' ,con=engine , schema='input', if_exists='append', index=False) # Save log to server

    time_end = datetime.now()
    print('{}\t END'.format(time_end))
    print('\t\t\t\t TOTAL TIME {}'.format( datetime.strptime(str(time_end-time_start), '%H:%M:%S.%f').time() ))