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

    # Captioning model arguments

    parser.add_argument('--cider_model', type=str, default='../vilbert_beta/checkpoints/coco_minus_8_no_random/pytorch_model_10.bin')
    parser.add_argument('--config_file', type=str, default='../vilbert_beta/config/bert_base_6layer_6conect.json')
    parser.add_argument('--use_cider', action='store_true')

    # Solr scores
    parser.add_argument('--unigram_prob_file', type=str, default='../GoogleConceptualCaptioning/data/unigram_prob_cc.json')
    parser.add_argument('--use_slor', action='store_true')

    # Vifidel scores
    parser.add_argument('--glove_vectors', type=str, default='../GoogleConceptualCaptioning/data/glove_vectors.pt')
    parser.add_argument('--glove_word_to_ix', type=str, default='../GoogleConceptualCaptioning/data/glove_stoi.json')
    parser.add_argument('--ground_truth_object_annotations', type=str, default='../GoogleConceptualCaptioning/data/coco_gt_objs_modified.json')
    parser.add_argument('--use_vifidel', action='store_true')

    # Other
    parser.add_argument('--use_ref_caps', action='store_true')
    parser.add_argument('--save_all_train_captions', action='store_true')
    parser.add_argument('--eval_split_during_train', type=str, default='val')
    parser.add_argument('--use_base_model_for_greedy', action='store_true')
    parser.add_argument('--id_language_eval', type=str, default='')
    parser.add_argument('--is_classification_cider_model', type=int, default=0)
    parser.add_argument('--classification_threshold', type=float, default=0.999)
    parser.add_argument('--do_not_generate_cider_plots', action='store_true')

    parser.add_argument('--train_only', type=int, default=1, help='if true then use 80k, else use 110k')
    
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--captioning_model_checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    parser.add_argument('--captioning_model_batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--captioning_model_grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    # parser.add_argument('--drop_prob_lm', type=float, default=0.5,
    #                 help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    
    # defaults
    parser.add_argument('--caption_model', type=str, default="att2in2",
                help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, transformer')
    parser.add_argument('--rnn_size', type=int, default=2400,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')

    parser.add_argument('--input_json', type=str, default='../GoogleConceptualCaptioning/data/cocotalk_with_cc_vocab_rm_8_same_order_for_self_critical.json')
    parser.add_argument('--input_fc_dir', type=str, default='/srv/share2/sgondala/data_for_captioning_models/trainval_resnet101_faster_rcnn_genome_36.tsv')
    parser.add_argument('--input_att_dir', type=str, default='/srv/share2/sgondala/data_for_captioning_models/trainval_resnet101_faster_rcnn_genome_36.tsv')
    parser.add_argument('--input_box_dir', type=str, default='')
    parser.add_argument('--input_label_h5', type=str, default='../GoogleConceptualCaptioning/data/cocotalk_with_cc_vocab.h5')
    parser.add_argument('--captioning_model_start_from', type=str, default=None)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs')

    return parser