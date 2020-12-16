import numpy as np
import pandas as pd
from collections import Counter
import re
import textstat
from sklearn.model_selection import StratifiedShuffleSplit 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
list_stopwords = stopwords.words('english')
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
discard_symbol = ': ? ! @ # $ % ^ & * ( ) ; \' [ ] , . / \" - _ ` ~'.split()
from nltk import pos_tag, word_tokenize, RegexpParser
chunker = RegexpParser(""" 
                       NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases 
                       P: {<IN>}               #To extract Prepositions 
                       V: {<V.*>}              #To extract Verbs 
                       PP: {<P> <NP>}          #To extract Prepostional Phrases 
                       VP: {<V> <NP|PP>*}      #To extarct Verb Phrases 
                       """) 
from sklearn.metrics.ranking import roc_auc_score

def calculate_metric(gtnp, pdnp):
    # input are numpy vector
    o_pdnp = np.copy(pdnp) # this is for AUROC score
    pdnp[pdnp>=0.5] = 1
    pdnp[pdnp!=1] = 0
    total_samples = len(gtnp)
    #print(f"Total sample: {total_samples}")
    total_correct = np.sum(gtnp == pdnp)
    accuracy = total_correct / total_samples
    gt_pos = np.where(gtnp == 1)[0]
    gt_neg = np.where(gtnp == 0)[0]
    TP = np.sum(pdnp[gt_pos])
    TN = np.sum(1 - pdnp[gt_neg])
    FP = np.sum(pdnp[gt_neg])
    FN = np.sum(1 - pdnp[gt_pos])
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['tp'] = int(TP)
    metrics['tn'] = int(TN)
    metrics['fp'] = int(FP)
    metrics['fn'] = int(FN)
    try:
        metrics['auc'] = roc_auc_score(gtnp, o_pdnp)
    except Exception as e:
        print(e)
        metrics['auc'] = -1
    return metrics
def count_word(s):
    ss = s.split()
    return len(ss)

def remove_symbol(s):
    for sym in discard_symbol:
        s = s.replace(sym, '')
    return s

def split_dataframe(data, test_size=0.2, seed=1509):
    X = np.asarray(data['headline'].tolist())
    y = np.asarray(data['is_sarcastic'].tolist())
    
    splitter=StratifiedShuffleSplit(n_splits=1,random_state=seed, test_size=test_size)
    for train_idx, test_idx in splitter.split(X,y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
    dt = {'is_sarcastic': list(y_train),
          'headline': list(X_train)}
    df_train = pd.DataFrame(dt, columns = ['is_sarcastic', 'headline'])

    dt = {'is_sarcastic': list(y_test),
          'headline': list(X_test)}
    df_test = pd.DataFrame(dt, columns = ['is_sarcastic', 'headline'])
    
    return df_train, df_test


def split_dataframe_3_subsets(data, test_size=0.2, validate_size=0.2, seed=1509):
    df_temp, df_test = split_dataframe(data, test_size=test_size, seed=seed)
    validate_size = validate_size / (1 - test_size)
    df_train, df_validate = split_dataframe(df_temp, test_size=validate_size, seed=seed+1309)
    return df_train, df_validate, df_test

def lemmatize_word(w, pos='v'):
    w_list = w.split()
    result = [lemmatizer.lemmatize(x, pos=pos) for x in w_list]
    result = ' '.join(result)
    return result

def preprocess_sent(sent, remove_stopwords=True):
    sent_remove_symbol = remove_symbol(sent)
    sent_lemmatize = lemmatize_word(sent_remove_symbol, 'v')
    sent_lemmatize = lemmatize_word(sent_lemmatize, 'n')
    if remove_stopwords:
        sent_remove_stopwords = remove_stop_words(sent_lemmatize)
        return sent_remove_stopwords
    else:
        return sent_lemmatize
    
def remove_stop_words(sent):
    list_words = sent.split()
    list_avai = [x for x in list_words if x not in list_stopwords]
    return ' '.join(list_avai)

def count_lemmatized_word(df, col1='headline', col2='lemmatized'):
    list_headline = list(df[col1])
    list_lemmatized = list(df[col2])
    result = []
    for head, lem in zip(list_headline, list_lemmatized):
        head_s = head.split()
        lem_s = lem.split()
        count = 0
        for h, l in zip(head_s, lem_s):
            if h != l:
                count += 1
        result.append(count)
    return result

def count_capitalized(sent):
    list_word = sent.split()
    count = 0
    for word in list_word:
        if word.isupper():
            count += 1
    return count

def min_len_word(sent):
    list_word = sent.split()
    min_len = 9999
    for word in list_word:
        lenw = len(word)
        if lenw < min_len:
            min_len = lenw
    return min_len

def max_len_word(sent):
    list_word = sent.split()
    max_len = -1
    for word in list_word:
        lenw = len(word)
        if lenw > max_len:
            max_len = lenw
    return max_len

def avg_len_word(sent):
    list_word = sent.split()
    sum_len = 0
    for word in list_word:
        lenw = len(word)
        sum_len += lenw
    sum_len /= len(list_word)
    return sum_len

def has_Number(inputString):
    return bool(re.search(r'\d', inputString))

def count_syllable(sent):
    # Count number of syllable of entire sentence
    return textstat.syllable_count(sent)

def min_syl_word(sent):
    list_word = sent.split()
    min_syl = 9999
    for word in list_word:
        sylw = textstat.syllable_count(word)
        if sylw < min_syl:
            min_syl = sylw
    return min_syl

def max_syl_word(sent):
    list_word = sent.split()
    max_len = -1
    for word in list_word:
        lenw = textstat.syllable_count(word)
        if lenw > max_len:
            max_len = lenw
    return max_len

def avg_syl_word(sent):
    list_word = sent.split()
    sum_len = 0
    for word in list_word:
        lenw = textstat.syllable_count(word)
        sum_len += lenw
    sum_len /= len(list_word)
    return sum_len

def count_lexicon(sent):
    # Count number of syllable of entire sentence
    return textstat.lexicon_count(sent, removepunct=True)

def count_noun(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if 'NN' in pos:
            count += 1
    return count

def count_verb_past(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['VBD', 'VBN']:
            count += 1
    return count

def count_verb_present(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['VBZ', 'VBP', 'VB']:
            count += 1
    return count

def count_verb_ing(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['VBG']:
            count += 1
    return count

def count_adj(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['JJ', 'JJR', 'JJS', 'IN']:
            count += 1
    return count

def count_adv(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['RB', 'RBR', 'RBS']:
            count += 1
    return count

def count_DT(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['DT']:
            count += 1
    return count

def count_CD(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['CD']:
            count += 1
    return count

def count_pronoun(sent):
    text = word_tokenize(sent)
    tag = nltk.pos_tag(text)
    count = 0
    for word, pos in tag:
        if pos in ['PRP', 'PRP$']:
            count += 1
    return count

def get_depth_syntax_tree(sent):
    tagged = pos_tag(word_tokenize(sent))
    tree = chunker.parse(tagged) 
    tree_sent = f"{tree}"
    t = nltk.Tree.fromstring(tree_sent)
    depth = t.height()
    return depth

def most_common_words(sent, numb_words=20):
    words = sent.split()
    #words = word_tokenize(sent)
    wordCount = Counter(words)
    wordCount = wordCount.most_common()
    if numb_words > len(wordCount) or numb_words < 0:
        numb_words = len(wordCount)
    top_words = [x[0] for x in wordCount[:numb_words]]
    count_words = [x[1] for x in wordCount[:numb_words]]
    return top_words, count_words

def numb_common_words(sent, list_common_words, norm=True):
    sent_s = sent.split()
    count = 0
    for w in sent_s:
        if w in list_common_words:
            count += 1
    if norm:
        count = count / len(sent_s)
    return count

def normalize_data(X, list_mean=[], list_std=[]):
    nrow, ncol = X.shape
    if len(list_mean) == 0 or len(list_std) == 0:
        x_mean = np.asarray(np.mean(X, axis=0))
        x_std = np.std(X, axis=0)
        list_mean = list(x_mean)
        list_std = list(x_std)
    else:
        x_mean = np.asarray(list_mean)
        x_std = np.asarray(list_std)
    x_mean = x_mean.reshape(1,-1)
    x_std = x_std.reshape(1, -1)
    x_mean = np.repeat(x_mean, nrow, axis=0)
    x_std = np.repeat(x_std, nrow, axis=0)
    X_norm = (X - x_mean) / x_std
    return X_norm, list_mean, list_std