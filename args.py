import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vilbert_bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--vilbert_from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--vilbert_output_dir",
        default="",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--vilbert_config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--vilbert_learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--vilbert_batch_size",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--vilbert_optimizer", default='BertAdam', type=str, help="Optimizer"
    )
    parser.add_argument(
        "--captions_path", default='', type=str, help="Train captions"
    )
    parser.add_argument(
        "--cider_path", default='', type=str, help="Train cider scores"
    )
    parser.add_argument(
        "--val_captions_path", default='', type=str, help="Val captions"
    )
    parser.add_argument(
        "--val_cider_path", default='', type=str, help="Val cider"
    )
    parser.add_argument(
        "--tsv_path", default='', type=str, help="Path to locate acc, box, height, and width files"
    )

    return parser