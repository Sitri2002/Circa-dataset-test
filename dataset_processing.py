from datasets import load_dataset
from pandas import DataFrame

'''
    80:10:10 ratio for train:val:test
'''
train_size = 0.8 
val_size = 0.5

def dataset_process():
    data = load_dataset("circa", split = "train")
    data = data.add_column("QA-pair", [""]*len(data))
    data = data.map(lambda x:{"QA-pair": x["question-X"] + " [SEP] " + x["answer-Y"]})
    '''
        have to filter out unnamed label (-1), since BCE can only take positive input ranges. Did not
        affect overall accuracy too much, but maybe did have an effect somewhat.
    '''
    data = data.filter(lambda x: x["goldstandard2"] != -1)
    question_only_data = data.remove_columns(["context", "canquestion-X", "answer-Y", "judgements", "QA-pair", "goldstandard1"])
    question_only_data = question_only_data.rename_columns({"question-X":"text", "goldstandard2": "label"})
    
    answer_only_data = data.remove_columns(["context", "canquestion-X", "question-X", "judgements", "QA-pair", "goldstandard1"])
    answer_only_data = answer_only_data.rename_columns({"answer-Y":"text", "goldstandard2": "label"})

    pair_data = data.remove_columns(["context", "canquestion-X", "question-X", "judgements", "answer-Y", "goldstandard1"])
    pair_data = pair_data.rename_columns({"QA-pair":"text", "goldstandard2": "label"})

    question_ds = question_only_data.train_test_split(train_size = train_size, shuffle = False)
    answer_ds = answer_only_data.train_test_split(train_size = train_size, shuffle = False)
    pair_ds = pair_data.train_test_split(train_size = train_size, shuffle = False)
    
    '''
        change seed for train data shuffling as needed. Doesnt find it neccesary to shuffle val and test data.
    '''
    question_ds["train"].shuffle(seed = 22)
    answer_ds["train"].shuffle(seed = 22)
    pair_ds["train"].shuffle(seed = 22)

    question_ds["test"] = question_ds["test"].train_test_split(train_size = val_size, shuffle = False)
    answer_ds["test"] = answer_ds["test"].train_test_split(train_size = val_size, shuffle = False)
    pair_ds["test"] = pair_ds["test"].train_test_split(train_size = val_size, shuffle = False)

    return question_ds, answer_ds, pair_ds













    


