import finetuning
import pandas as pd
import os

def test():
    question_ds, answer_ds, pair_ds = finetuning.dataset_processing.dataset_process()
    base_tokenizer, mnli_tokenizer, boolq_tokenizer = finetuning.tokenizer_init()
    target_name = ['Yes', 'No', 'In the middle, neither yes nor no', 'Yes, subject to some conditions', 'Other']

    question = finetuning.tokenize_func(base_tokenizer, question_ds)
    answer = finetuning.tokenize_func(base_tokenizer, answer_ds)
    pair = finetuning.tokenize_func(base_tokenizer, pair_ds)
    boolq = finetuning.tokenize_func(boolq_tokenizer, pair_ds)
    mnli = finetuning.tokenize_func(mnli_tokenizer, pair_ds)


    batch_size_args = finetuning.TrainingArguments(output_dir= "./tmp_trainer",per_device_eval_batch_size= 16)
    
    question_model = finetuning.BertForSequenceClassification.from_pretrained("./my_model/question")
    answer_model = finetuning.BertForSequenceClassification.from_pretrained("./my_model/answer")
    base_pair_model = finetuning.BertForSequenceClassification.from_pretrained("./my_model/base_pair")
    mnli_model = finetuning.BertForSequenceClassification.from_pretrained("./my_model/mnli")
    boolq_model = finetuning.BertForSequenceClassification.from_pretrained("./my_model/boolq")

    '''
        handle some subdirectories stuff
    '''
    outdir = './my_reports'
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    '''
        predict and export to csv
    '''
    question_infer = finetuning.Trainer(model = question_model, args=batch_size_args)
    question_pred = finetuning.np.argmax(question_infer.predict(question["test"]).predictions, axis = -1)
    question_report = finetuning.sklearn.metrics.classification_report(question["test"]["label"], question_pred, target_names = target_name, output_dict = True)
    question_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": question_report["accuracy"], "support": question_report['macro avg']['support']}})
    pd.DataFrame(question_report).transpose().to_csv(f"{outdir}/question_report.csv")

    answer_infer = finetuning.Trainer(model = answer_model, args=batch_size_args)
    answer_pred = finetuning.np.argmax(answer_infer.predict(answer["test"]).predictions, axis = -1)
    answer_report = finetuning.sklearn.metrics.classification_report(answer["test"]["label"], answer_pred, target_names = target_name, output_dict = True)
    answer_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": answer_report["accuracy"], "support": answer_report['macro avg']['support']}})
    pd.DataFrame(answer_report).transpose().to_csv(f"{outdir}/answer_report.csv")

    base_pair_infer = finetuning.Trainer(model = base_pair_model, args=batch_size_args)
    base_pair_pred = finetuning.np.argmax(base_pair_infer.predict(pair["test"]).predictions, axis = -1)
    base_pair_report = finetuning.sklearn.metrics.classification_report(pair["test"]["label"], base_pair_pred, target_names = target_name, output_dict = True)
    base_pair_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": base_pair_report["accuracy"], "support": base_pair_report['macro avg']['support']}})
    pd.DataFrame(base_pair_report).transpose().to_csv(f"{outdir}/base_pair_report.csv")

    boolq_infer = finetuning.Trainer(model = boolq_model, args=batch_size_args)
    boolq_pred = finetuning.np.argmax(boolq_infer.predict(boolq["test"]).predictions, axis = -1)
    boolq_report = finetuning.sklearn.metrics.classification_report(boolq["test"]["label"], boolq_pred, target_names = target_name, output_dict = True)
    boolq_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": boolq_report["accuracy"], "support": boolq_report['macro avg']['support']}})
    pd.DataFrame(boolq_report).transpose().to_csv(f"{outdir}/boolq_report.csv")

    mnli_infer = finetuning.Trainer(model = mnli_model, args=batch_size_args)
    mnli_pred = finetuning.np.argmax(mnli_infer.predict(mnli["test"]).predictions, axis = -1)
    mnli_report = finetuning.sklearn.metrics.classification_report(mnli["test"]["label"], mnli_pred, target_names = target_name, output_dict = True)
    mnli_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": mnli_report["accuracy"], "support": mnli_report['macro avg']['support']}})
    pd.DataFrame(mnli_report).transpose().to_csv(f"{outdir}/mnli_report.csv")

if __name__ == "__main__":
    test()









