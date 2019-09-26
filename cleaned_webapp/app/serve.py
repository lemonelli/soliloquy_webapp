from os.path import join, dirname, realpath
from soliloquy_variation import sentalter
from soliloquy_variation import tokenizer
from soliloquy_variation.eval import awer
from soliloquy_2019 import lm_evaluation
from app import app

def get_model_api():
    #Initialize the AlterSent object
    lv = sentalter.AlterSent(
        vecfname="app/static/models/en_vec.txt",
        lmfname="app/static/models/wiki_other_en.o4.h10m.fst",
        onmt_dir='',
        model_dir='',
        kenlm_loc='',
        maxtypes=50000)

    def model_api(input_data):
        ##Implement FST paraphrase
        words = tokenizer.word_tokenize(input_data)
        lines = lv.fst_alter_sent(words, 100)

        return lines

    return model_api

def get_model_evaluator_api(model_file):
    #Initialize LM class
    lm = lm_evaluation.LM(model_file=model_file, train_file='', order='')

    def evaluator_api(test_file):
        #Evaluates based on test file
        test_data = lm_evaluation.process_test_data(test_file)

        #calculate perplexity ad output
        ppl = lm_evaluation.calc_perplexity(test_data, lm)

        return ppl

    return evaluator_api

def get_train_evaluator_api(train_file):
    #Initialize LM class
    lm = lm_evaluation.LM(train_file=train_file, model_file='', order=3)

    def evaluator_api(test_file):
        #Evaluates based on test file
        test_data = lm_evaluation.process_test_data(test_file)

        #calculate perplexity ad output
        ppl = lm_evaluation.calc_perplexity(test_data, lm)

        return ppl

    return evaluator_api

def get_awer_model_api():
    #initialize model
    UNIGRAM_FILEPATH = join(dirname(realpath(__file__)), app.config['MODEL_FOLDER'], 'tr.unigrams')
    FST_FILEPATH = join(dirname(realpath(__file__)), app.config['MODEL_FOLDER'], 'my_model.fst')
    lv = awer.AlterSent(UNIGRAM_FILEPATH, FST_FILEPATH, 50000)

    def awer_model_api(test_file):
        #initializes variable
        totalerr = 0
        linecnt = 0

        with open(test_file) as f:
            for line in f.readlines():
                linecnt += 1
                words = tokenizer.word_tokenize(line)
                lines = lv.fst_alter_sent(words,1)
                toks = lines[0][1].split()
                err = 0
                for i in range(len(words)):
                    if words[i] != toks[i]:
                        err += 1

                if len(words) > 0:
                    totalerr += err / len(words)
            if linecnt > 0:
                totalerr = totalerr / linecnt

        return totalerr

    return awer_model_api
