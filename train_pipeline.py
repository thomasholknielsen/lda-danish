pipe = {
	1: {
        'sqlserver': 'THN-P53',
		'database': 'GRE',
		'query': '''SELECT [ContentID]
                            ,[ContentDesriptionText]
                        FROM [GRE].[dbo].[vw_LDA_Content]
                        WHERE 1 = 1
                    ''',
		'target_column': 'ContentDesriptionText', 			# Which column holds the text to be processed
		'target_column_id': 'ContentID',					# Name of ID column related to target column
		'models': [
            {
				'GroupColumn':'None', 						# Set None if no groupcolumn
				'GroupValue':'None', 						# Set None if no group column
                'lda_sample_size': 30000, 					# Setting max sample size for each subgroup
				'lda_num_topics_start': 3,					# To only train on a X number of topics, _start = X, and _end = X+1
				'lda_num_topics_end': 4,
				'lda_num_topics_increment': 1,
                'lda_alpha': 'asymmetric',					# a-priori belief for the each topics’ probability
                'lda_eta': 'auto',							# A-priori belief on word probabilit
                'lda_minimum_probability': 0.0,				# Topics with a probability lower than this threshold will be filtered out.
				'no_above': 0.5,                            # filter out tokens that appear in more than x percent of all documents
    			'no_below': 8,                              # filter out tokens that appear in less than X documents
				'lda_chunksize': 500,                       # Change for performance
    			'lda_passes': 50,                           # Higher --> better performance, but slower training
                'model_comment': 'This is a comment'
			# },
            # {
			# 	'GroupColumn':'None', 						# Set None if no groupcolumn
			# 	'GroupValue':'None', 						# Set None if no group column
            #     'lda_sample_size': 30000, 					# Setting max sample size for each subgroup
			# 	'lda_num_topics_start': 3,					# To only train on a X number of topics, _start = X, and _end = X+1
			# 	'lda_num_topics_end': 4,
			# 	'lda_num_topics_increment': 1,
            #     'lda_alpha': 'asymmetric',					# a-priori belief for the each topics’ probability
            #     'lda_eta': 'auto',							# A-priori belief on word probabilit
            #     'lda_minimum_probability': 0.0,				# Topics with a probability lower than this threshold will be filtered out.
			# 	'no_above': 0.5,                            # filter out tokens that appear in more than x percent of all documents
    		# 	'no_below': 8,                              # filter out tokens that appear in less than X documents
			# 	'lda_chunksize': 500,                       # Change for performance
    		# 	'lda_passes': 50,                           # Higher --> better performance, but slower training
            #     'model_comment': 'This is a comment'
			}
		]
	}
}
    
                 
                     

	
	

    

    

