from soliloquy_variation import sentalter
from soliloquy_variation import tokenizer
from soliloquy_2019 import lm_evaluation

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
