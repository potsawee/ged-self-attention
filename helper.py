import random
import collections
import pdb

def load_vocab(path):
    with open(path, encoding="utf8") as file:
        vocab = file.readlines()

    word2id = collections.OrderedDict()

    for i, word in enumerate(vocab):
        word = word.strip() # remove \n
        word2id[word] = i

    # -------------- Special Tokens -------------- #
    # <unk> <s> </s> are defined in the vocab list
    # -------------------------------------------- #

    return word2id

def load_ged_data(path, word2id, max_sentence_length):
    word_ids = []
    label_ids = []

    with open(path, encoding="utf8") as file:
        word_sentence = []  # [123, 10005, 688]
        label_sentence = [] # [0, 0, 1]
        sent_lengths = []

        for line in file:
            if line == '\n':
                # end of sentence
                if len(word_sentence) > max_sentence_length:
                    word_sentence = []
                    label_sentence = []
                else:
                    assert (len(word_sentence) == len(label_sentence)), "word_sentence != label_sentence"

                    sent_len = len(word_sentence)
                    word_sentence += [word2id['</s>']] * (max_sentence_length - sent_len)
                    label_sentence += [0] * (max_sentence_length - sent_len)

                    word_ids.append(word_sentence)
                    label_ids.append(label_sentence)
                    sent_lengths.append(sent_len)
                    word_sentence = []
                    label_sentence = []

            else:
                line = line.strip()
                items = line.split()
                word = items[0]
                label = items[-1]

                # word mapping using word2id
                if word in word2id:
                    word_sentence.append(word2id[word])
                else:
                    word_sentence.append(word2id['<unk>'])

                # Label 'c' => 0 | 'i' => 1
                if label == 'c':
                    label_sentence.append(0)
                elif label == 'i':
                    label_sentence.append(1)
                else:
                    raise ValueError("label value error --- only 'c' and 'i' are allowed")

    assert (len(word_ids) == len(label_ids)), "word_ids != label_ids"

    return word_ids, label_ids, sent_lengths

def construct_training_data_batches(config):
    batch_size = config['batch_size']
    max_sentence_length = config['max_sentence_length']

    vocab_path = config['vocab_path']
    data_path = config['data_path']

    word2id = load_vocab(vocab_path)
    vocab_size = len(word2id)
    print("vocab_size: ", vocab_size)

    word_ids, label_ids, sent_lengths = load_ged_data(data_path, word2id, max_sentence_length)

    num_training_sentences = len(word_ids)
    print("num_training_sentences: ", num_training_sentences) # only those that are not too long

    # shuffle
    _x = list(zip(word_ids, label_ids, sent_lengths))
    random.shuffle(_x)
    word_ids, label_ids, sent_lengths = zip(*_x)

    batches = []

    for i in range(int(num_training_sentences/batch_size)):
        i_start = i * batch_size
        i_end = i_start + batch_size
        batch = {'word_ids': word_ids[i_start:i_end],
                'label_ids': label_ids[i_start:i_end],
                'sent_lengths': sent_lengths[i_start:i_end]}

        batches.append(batch)

    return batches, vocab_size, word2id
