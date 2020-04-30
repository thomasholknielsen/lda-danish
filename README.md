# Installation
## Create environment
conda create --name lda --file requirements_lda.txt

## Create requirements file
conda list --export > requirements_lda.txt





# Usage
The following can be used in a bat file:
:: go to the folder with the scripts
cd E:\Dev\lda_

:: activate the environment with the correct packages
activate lda

:: start running the model with the current pipeline
:: to change the pipeline make changes to the dict pipe in train_pipeline.py
C:/ProgramData/Anaconda3/envs/lda/python.exe e:/Dev/lda_aka/multi_lda.py






# Development
## ToDo
- Add Comment on log
- Add SpaCy Lemma
- Optimizing
    String handling - to increase speed
        - Switch to C wrapper where possible
    SQL
        - Optimize for larger scale
    Threading
        - Prepare parts to thread where possible
- Change current model logging to dimension based
- Enable saving of model in database
- Enable sample size of input
    0 -> Total size
    N -> N sample of the population. Should then predict on the data after training.

## Complete
- Add pipeline
- Add query to pipeline
- Add model parameters to pipeline
- Change pipeline to enable differen sources
- Change pipeline to make data processing in one step
- Add option of getting values from pipeline
- Add Word Rank
- Add log file