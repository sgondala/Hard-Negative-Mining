import torch
import torch.nn as nn
import torch.optim as optim
import sys

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
import opts
import models
from dataloader import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper
from misc.utils import decode_sequence
from eval_utils import language_eval
import math

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def eval_cider_and_append_values(predictions, writer, key, start_iteration, batch_size, losses_log_every):
    # TODO - This hardcodes to coco val. Take care
    out = language_eval(None, predictions, None, {}, 'val')
    cider_array = np.array(out['CIDErArary'])
    # print("Length of cider array ", len(cider_array))

    for i in range(math.ceil(len(cider_array) * 1.0 /batch_size)):
        # print("Index in eval cider", i)
        start_index = i*batch_size
        end_index = i*batch_size + batch_size
        cider_avg = cider_array[start_index:end_index].mean()
        print(key, cider_avg, start_iteration + (i + 1)*losses_log_every)
        add_summary_value(writer, key, cider_avg, start_iteration + (i + 1)*losses_log_every)
    
    return start_iteration + (i + 1) * losses_log_every

def train_captioning_model(opt, tb_summary_writer):
    opt.use_fc, opt.use_att = True, False
    
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    infos = {}
    histories = {}
    if opt.captioning_model_start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.captioning_model_start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                print(vars(saved_model_opt)[checkme], vars(opt)[checkme])
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.captioning_model_start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.captioning_model_start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    
    infos['iter'] = 0
    infos['epoch'] = 0
    infos['iterators'] = loader.iterators
    infos['split_ix'] = loader.split_ix
    infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    
    model_greedy = None
    initial_greedy_model_weights = []
    if True:
        model_greedy = models.setup(opt).cuda()
        model_greedy.eval()
        initial_greedy_model_weights = list(model_greedy.parameters())
        for param in model_greedy.parameters():
            param.requires_grad = False

    dp_model = torch.nn.DataParallel(model)
    
    vocab = opt.vocab
   
    cider_model = None
    cider_dataset = None

    open_gpt_tokenizer = None
    open_gpt_model = None
    unigram_prob_dict = None

    glove_embedding = None
    glove_word_to_ix = None
    ground_truth_object_annotations = None

    initial_cider_model_weights = []
    final_cider_model_weights = []

    if opt.self_critical_after != -1 and not opt.use_ref_caps:
        # CIDEr
        if opt.use_cider:
            print("Using cider")
            from vilbert.vilbert import BertConfig
            from vilbert.vilbert import VILBertForVLTasks

            from CiderDataset import CiderDataset

            from pytorch_pretrained_bert.tokenization import BertTokenizer

            config = BertConfig.from_json_file(opt.captioning_model_config_file)
            cider_model = VILBertForVLTasks.from_pretrained(opt.cider_model, config, num_labels=1, default_gpu=True)
            cider_model.cuda()
            cider_model.eval()
            for param in cider_model.parameters():
                param.requires_grad = False

            initial_cider_model_weights = list(cider_model.parameters())

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            cider_dataset = CiderDataset(None, opt.input_fc_dir, tokenizer)

        # SLOR
        if opt.use_slor:
            print("Using slor")
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            open_gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            open_gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
            open_gpt_model.cuda()
            open_gpt_model.eval()
            for param in open_gpt_model.parameters():
                param.requires_grad = False

            unigram_prob_dict = json.load(open(opt.unigram_prob_file, 'r'))

        # VIFIDEL
        if opt.use_vifidel:
            print("Using vifidel")
            glove_embedding = nn.Embedding.from_pretrained(torch.load(opt.glove_vectors), freeze=True)
            glove_word_to_ix = json.load(open(opt.glove_word_to_ix, 'r'))
            ground_truth_object_annotations = json.load(open(opt.ground_truth_object_annotations, 'r'))

    lw_model = LossWrapper(model, opt, vocab, cider_dataset, cider_model, open_gpt_model, open_gpt_tokenizer, unigram_prob_dict, glove_embedding, glove_word_to_ix, ground_truth_object_annotations, model_greedy, opt.is_classification_cider_model, opt.classification_threshold).cuda()
    
    dp_lw_model = torch.nn.DataParallel(lw_model)

    epoch_done = True
    dp_lw_model.train()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    
    # Load the optimizer
    if vars(opt).get('captioning_model_start_from', None) is not None and os.path.isfile(os.path.join(opt.captioning_model_start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.captioning_model_start_from, 'optimizer.pth')))

    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if captioning_model_checkpoint_path doesn't exist
        if not os.path.isdir(opt.captioning_model_checkpoint_path):
            os.makedirs(opt.captioning_model_checkpoint_path)
        captioning_model_checkpoint_path = os.path.join(opt.captioning_model_checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), captioning_model_checkpoint_path)
        print("model saved to {}".format(captioning_model_checkpoint_path))
        optimizer_path = os.path.join(opt.captioning_model_checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.captioning_model_checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.captioning_model_checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)

    gen_captions_all = {}
    greedy_captions_all = {}

    greedy_captions_since_last_checkpoint = []
    gen_captions_since_last_checkpoint = []

    try:
        while True:
            if epoch_done:
                if True:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                
                # If start self critical training
                sc_flag = True
                if opt.use_ref_caps:
                    init_scorer(opt.cached_tokens)

                epoch_done = False
                    
            data = loader.get_batch('train')

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            image_ids = data['image_ids']
            
            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, False, False, image_ids)

            # if not drop_worst_flag:
            loss = model_out['loss'].mean()

            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()

            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean(), iteration)
                add_summary_value(tb_summary_writer, 'avg_greedy_cider', model_out.get('average_greedy_cider', 0), iteration)
                add_summary_value(tb_summary_writer, 'avg_gen_cider', model_out.get('average_gen_cider', 0), iteration)
                add_summary_value(tb_summary_writer, 'avg_greedy_slor', model_out.get('average_greedy_slor', 0), iteration)
                add_summary_value(tb_summary_writer, 'avg_gen_slor', model_out.get('average_gen_slor', 0), iteration)
                add_summary_value(tb_summary_writer, 'avg_greedy_vifidel', model_out.get('average_greedy_vifidel', 0), iteration)
                add_summary_value(tb_summary_writer, 'avg_gen_vifidel', model_out.get('average_gen_vifidel', 0), iteration)
                
                greedy_captions_since_last_checkpoint += model_out['greedy_captions']

                gen_captions_since_last_checkpoint += model_out['gen_captions']

                gen_captions_all[iteration] = model_out['gen_captions']
                greedy_captions_all[iteration] = model_out['greedy_captions']

                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            
            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0):
                print("Calculating validation score")
                
                eval_kwargs = {'split': 'val',
                                'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                # Rechecking on train data too
                train_lang_stats = None
                if not opt.do_not_generate_cider_plots:
                    eval_kwargs_train = {'split': 'train',
                                    'dataset': opt.input_json}
                    eval_kwargs_train.update(vars(opt))
                    _, _, train_lang_stats = eval_utils.eval_split(
                        dp_model, lw_model.crit, loader, eval_kwargs_train)
                
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    add_summary_value(tb_summary_writer, 'CIDEr', lang_stats['CIDEr'], iteration)
                
                if train_lang_stats is not None:
                    add_summary_value(tb_summary_writer, 'CIDEr_train', train_lang_stats['CIDEr'], iteration)

                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                if iteration != 1:
                    if not opt.do_not_generate_cider_plots:
                        # Calculate actual cider values of previous iterations
                        # Doing this here to take advantage of batching
                        
                        end_iteration_val = eval_cider_and_append_values(greedy_captions_since_last_checkpoint, tb_summary_writer, 'greedy_generated_captions_actual_cider_scores', iteration - opt.save_checkpoint_every, opt.batch_size, opt.losses_log_every)
                        assert end_iteration_val == iteration, str(end_iteration_val) + ',' + str(iteration)

                        end_iteration_val = eval_cider_and_append_values(gen_captions_since_last_checkpoint, tb_summary_writer, 'gen_generated_captions_actual_cider_scores', iteration - opt.save_checkpoint_every, opt.batch_size, opt.losses_log_every)
                        assert end_iteration_val == iteration

                greedy_captions_since_last_checkpoint = []
                gen_captions_since_last_checkpoint = []

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    save_checkpoint(model, infos, optimizer, append=str(iteration))

                if best_flag:
                    save_checkpoint(model, infos, optimizer, append='best')

                dict_to_save = {}
                dict_to_save['gen_captions'] = gen_captions_all
                dict_to_save['greedy_captions'] = greedy_captions_all
                gen_captions_all = {}
                greedy_captions_all = {}
                json.dump(dict_to_save, open(opt.captioning_model_checkpoint_path + '/captions_' + str(iteration) + '.json', 'w'))

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
        
        if cider_model is not None:
            final_cider_model_weights = list(cider_model.parameters())
            assert initial_cider_model_weights == final_cider_model_weights
        
        if model_greedy is not None:
            final_greedy_model_weights = list(model_greedy.parameters())
            assert initial_greedy_model_weights == final_greedy_model_weights

    except:
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)