from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import dataset_processing
import sklearn.metrics
import numpy as np
import evaluate

CUDA_LAUNCH_BLOCKING=1
#test
def tokenizer_init():
    base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    mnli_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    boolq_tokenizer = AutoTokenizer.from_pretrained("lewtun/bert-base-uncased-finetuned-boolq")
    return base_tokenizer, mnli_tokenizer, boolq_tokenizer

def model_init():
    base_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 5, ignore_mismatched_sizes=True)
    mnli_bert = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI", num_labels = 5, ignore_mismatched_sizes=True)
    boolq_bert = BertForSequenceClassification.from_pretrained("lewtun/bert-base-uncased-finetuned-boolq", num_labels = 5, ignore_mismatched_sizes=True)
    return base_bert, mnli_bert, boolq_bert

def tokenize_func(tokenizer, dataset):
    train = dataset["train"].map(lambda x:tokenizer(x["text"], padding = "max_length", truncation = True), batched = True)
    val = dataset["test"]["train"].map(lambda x:tokenizer(x["text"], padding = "max_length", truncation = True), batched = True)
    test = dataset["test"]["test"].map(lambda x:tokenizer(x["text"], padding = "max_length", truncation = True), batched = True)
    dataset = {"train": train, "val": val, "test": test}
    return dataset

'''
    Training Args are based on hyperparameters specified in the paper (difference in learning rates)
'''
def args_init():
    training_args_answer = TrainingArguments(
        output_dir = "./saved_models/answer_only",
        evaluation_strategy = "epoch",
        per_device_train_batch_size=32,
        learning_rate = 0.00002,
        num_train_epochs = 3)

    training_args_question = TrainingArguments(
        output_dir = "./saved_models/question_only",
        evaluation_strategy = "epoch",
        per_device_train_batch_size=32,
        learning_rate = 0.00003,
        num_train_epochs = 3)

    training_args_base_pair = TrainingArguments(
        output_dir = "./saved_models/base_pair",
        evaluation_strategy = "epoch",
        per_device_train_batch_size=32,
        learning_rate = 0.00003,
        num_train_epochs = 3)

    training_args_mnli = TrainingArguments(
        output_dir = "./saved_models/mnli",
        evaluation_strategy = "epoch",
        per_device_train_batch_size=32,
        learning_rate = 0.00005,
        num_train_epochs = 3)

    training_args_boolq = TrainingArguments(
        output_dir = "./saved_models/boolq",
        evaluation_strategy = "epoch",
        per_device_train_batch_size=32,
        learning_rate = 0.00005,
        num_train_epochs = 3)
    return training_args_answer, training_args_question, training_args_base_pair, training_args_boolq, training_args_mnli

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train():
    training_args_answer, training_args_question, training_args_base_pair, \
    training_args_boolq, training_args_mnli = args_init()
    base_bert, mnli_bert, boolq_bert = model_init()
    base_tokenizer, mnli_tokenizer, boolq_tokenizer = tokenizer_init()

    question_ds, answer_ds, pair_ds = dataset_processing.dataset_process()

    question = tokenize_func(base_tokenizer, question_ds)
    answer = tokenize_func(base_tokenizer, answer_ds)
    pair = tokenize_func(base_tokenizer, pair_ds)
    boolq = tokenize_func(boolq_tokenizer, pair_ds)
    mnli = tokenize_func(mnli_tokenizer, pair_ds)

    trainer_question = Trainer(
        model = base_bert,
        args = training_args_question,
        train_dataset = question["test"],
        eval_dataset = question["val"],
        compute_metrics = compute_metrics
    )

    trainer_answer = Trainer(
        model = base_bert,
        args = training_args_answer,
        train_dataset = answer["train"],
        eval_dataset = answer["val"],
        compute_metrics = compute_metrics
    )

    trainer_base_pair = Trainer(
        model = base_bert,
        args = training_args_base_pair,
        train_dataset = pair["train"],
        eval_dataset = pair["val"],
        compute_metrics = compute_metrics
    )

    trainer_mnli = Trainer(
        model = mnli_bert,
        args = training_args_mnli,
        train_dataset = mnli["train"],
        eval_dataset = mnli["val"],
        compute_metrics = compute_metrics
    )

    trainer_boolq = Trainer(
        model = boolq_bert,
        args = training_args_boolq,
        train_dataset = boolq["train"],
        eval_dataset = boolq["val"],
        compute_metrics = compute_metrics
    )

    trainer_answer.train()
    trainer_question.train()
    trainer_base_pair.train()
    trainer_boolq.train()
    trainer_mnli.train()


    trainer_answer.evaluate()
    trainer_question.evaluate()
    trainer_base_pair.evaluate()
    trainer_boolq.evaluate()
    trainer_mnli.evaluate()


    trainer_answer.save_model("./my_model/answer")
    trainer_question.save_model("./my_model/question")
    trainer_base_pair.save_model("./my_model/base_pair")
    trainer_boolq.save_model("./my_model/boolq")
    trainer_mnli.save_model("./my_model/mnli")


if __name__ == "__main__":
    train()
