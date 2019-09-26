from os.path import join, dirname, realpath
import subprocess
import pickle
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from soliloquy_2019 import tokenizer

from app import app
from app.forms import EnterForm
from app.serve import get_model_api
from app.serve import get_model_evaluator_api
from app.serve import get_train_evaluator_api
from app.serve import get_awer_model_api
from werkzeug.utils import secure_filename

model_api = get_model_api()

MODEL_ALLOWED_EXTENSIONS = {'arpa'}
TEXT_ALLOWED_EXTENSIONS = {'txt'}
ENDFILE = False
LOC = 0

TRAIN_FILEPATH = join(dirname(realpath(__file__)), app.config['DATA_FOLDER'], 'train_file.txt')
AUG_FILEPATH = join(dirname(realpath(__file__)), app.config['DATA_FOLDER'], 'aug_train_file.txt')
NEW_FILEPATH = join(dirname(realpath(__file__)), app.config['DATA_FOLDER'], 'new_train_file.txt')
SAVE_FILEPATH = join(dirname(realpath(__file__)), app.config['DATA_FOLDER'], 'savedata.txt')

class SaveObject:
    def __init__(self, trainfile, augfile, location):
        self.train = trainfile.readlines()
        self.aug = augfile.readlines()
        self.loc = location

    def save(self):
        with open(SAVE_FILEPATH, "wb") as out:
            pickle.dump(self, out)


def allowed_model_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() not in MODEL_ALLOWED_EXTENSIONS

def allowed_text_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() not in TEXT_ALLOWED_EXTENSIONS

def vocab_coverage(train_filepath, test_filepath):
    train_v = {}
    test_v = {}
    oov = 0

    #collect vocab in train
    print('counting train vocab')
    with open(train_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            toks = tokenizer.word_tokenize(line)
            for t in toks:
                if t not in train_v:
                    train_v[t] = 1

    #compare to vocab in test
    print('counting test vocab')
    with open(test_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            toks = tokenizer.word_tokenize(line)
            for t in toks:
                if t not in test_v:
                    test_v[t] = 1
                if t not in train_v:
                    oov += 1
                    train_v[t] = 1

    return [len(test_v), oov]

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home')

@app.route('/paraphrase', methods=['GET', 'POST'])
def paraphrase():
    global LOC
    if request.method == "POST":
        if "next 100" in request.form:
            #check for selected utterances
            if request.form.getlist("paraphrases"):
                #add selected utterances to new train file
                selected_paraphrases = request.form.getlist("paraphrases")
                with open(AUG_FILEPATH, "a+") as f:
                    for utt in selected_paraphrases:
                        f.write(utt + '\n')

            #open train file and paraphrase next 20 utterances
            lines = []
            with open(TRAIN_FILEPATH) as f:
                try:
                    f_list = f.readlines()
                    for line in f_list[LOC:(LOC + 20)]:
                        paraphrases = model_api(line)
                        paraphrases.reverse()
                        topfive = paraphrases[0:5]
                        lines.extend(topfive)
                        LOC += 1
                    endfile = False
                except:
                    endfile = True

            return render_template("paraphrase.html", title='Paraphrase Augmentation', lines=enumerate(lines), endfile=endfile)


        elif "train new" in request.form:
            #check for selected utterances
            if request.form.getlist("paraphrases"):
                #add selected utterances to new train file
                selected_paraphrases = request.form.getlist("paraphrases")
                with open(AUG_FILEPATH, "a+") as f:
                    for utt in selected_paraphrases:
                        f.write(utt + '\n')

            #add original and augmented train data to file
            with open(TRAIN_FILEPATH, "r") as f_old, open(AUG_FILEPATH, "r") as f_aug, open(NEW_FILEPATH, "a+") as f_new:
                #copy original train data
                for utt in f_old.readlines():
                    f_new.write(utt)
                #copy augmented train data
                for utt in f_aug.readlines():
                    f_new.write(utt)

            return redirect(url_for('aug_eval'))

        elif ("get para" in request.form and 'upload' in request.files):
            #reset file location iterator and filename paths
            LOC = 0
            #extract train file
            train_file = request.files['upload']
            #save train file
            train_file.save(TRAIN_FILEPATH)
            #clear new train file
            open(NEW_FILEPATH, "w").close()
            open(AUG_FILEPATH, "w").close()
            #open train file and paraphrase first 20 utterances
            lines = []
            with open(TRAIN_FILEPATH) as f:

                #
                for line in f.readlines()[0:20]:
                    paraphrases = model_api(line)
                    paraphrases.reverse()
                    topfive = paraphrases[0:5]
                    lines.extend(topfive)
                    LOC += 1

            return render_template("paraphrase.html", title='Paraphrase Augmentation', lines=enumerate(lines), endfile=False)

        elif ("load state" in request.form and 'upload' in request.files):
            #extract and save pickle file
            pickle_file = request.files['upload']
            pickle_file.save(SAVE_FILEPATH)

            #load state from pickle
            with open(SAVE_FILEPATH, 'rb') as f:
                load_object = pickle.load(f)

            #load location
            LOC = load_object.loc

            #write file objects to appropriate files
            with open(TRAIN_FILEPATH, 'w') as f:
                for l in load_object.train:
                    f.write(l)

            with open(AUG_FILEPATH, 'w') as f:
                for l in load_object.aug:
                    f.write(l)

            #clear new train file
            open(NEW_FILEPATH, "w").close()

            #open train file and paraphrase next 20 utterances
            lines = []
            with open(TRAIN_FILEPATH) as f:
                try:
                    f_list = f.readlines()
                    for line in f_list[LOC:(LOC + 20)]:
                        paraphrases = model_api(line)
                        paraphrases.reverse()
                        topfive = paraphrases[0:5]
                        lines.extend(topfive)
                        LOC += 1
                    endfile = False
                except:
                    endfile = True

            return render_template("paraphrase.html", title='Paraphrase Augmentation', lines=enumerate(lines), endfile=endfile)

        elif "save progress" in request.form:
            #check for selected utterances
            if request.form.getlist("paraphrases"):
                #add selected utterances to new train file
                selected_paraphrases = request.form.getlist("paraphrases")
                with open(AUG_FILEPATH, "a+") as f:
                    for utt in selected_paraphrases:
                        f.write(utt + '\n')

            #create file objects
            f_aug = open(AUG_FILEPATH, 'r')
            f_train = open(TRAIN_FILEPATH, 'r')

            #instantiate state object and save
            newsave = SaveObject(f_train, f_aug, LOC)
            newsave.save()

            return redirect(url_for('save_data'))

    return render_template("paraphrase.html", title='Sentence Entry', lines=[])

@app.route('/modeleval', methods=['GET', 'POST'])
def model_eval():
    if request.method == "POST":
        if "upload" in request.form:
            #check if post request has a file part
            if ('training' or 'test') not in request.files:
                flash('No files given')
                return redirect(request.url)
            #parse files
            train_file = request.files['training']
            test_file = request.files['test']
            #check that files aren't null
            if train_file.filename == '':
                flash('No training file selected')
                return redirect(request.url)
            if test_file.filename == '':
                flash('No test file selected')
                return redirect(request.url)
            #validate and save test file
            if not (test_file):
                flash('Test file invalid')
                return redirect(request.url)

            #check for model or training set
            if train_file:
                #save train file
                train_filename = secure_filename(train_file.filename)
                train_filepath = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], train_filename)
                train_file.save(train_filepath)

                #save test file
                test_filename = secure_filename(test_file.filename)
                test_filepath = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], test_filename)
                test_file.save(test_filepath)

                #build and evaluate model
                evaluator_api = get_train_evaluator_api(train_filepath)
                perplexity = evaluator_api(test_filepath)

                #convert ARPA to FST
                subprocess.call("ngramread --ARPA --epsilon_symbol='<eps>' app/static/models/my_model.arpa app/static/models/my_model.fst", shell=True)

                #evaluate AWER
                awer_model_api = get_awer_model_api()
                awer = awer_model_api(test_filepath)

                #calculate vocab coverage
                vocab = vocab_coverage(train_filepath, test_filepath)

                return render_template("score.html", title='Evaluation of Model based on Training Set',
                plxy=perplexity, awer=awer, coverage=vocab, type="train")

        return render_template('modelinput.html', title='Model Evaluation')

    return render_template('modelinput.html', title='Model Evaluation')

@app.route('/augeval', methods=['GET', 'POST'])
def aug_eval():
    if request.method == "POST":
        if "upload" in request.form:
            #check if post request has a file part
            if 'test' not in request.files:
                flash('No file given')
                return redirect(request.url)
            #parse file
            test_file = request.files['test']
            #save test file
            test_filename = secure_filename(test_file.filename)
            test_filepath = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], test_filename)
            test_file.save(test_filepath)
            #build and evaluate model by perplexity
            try:
                evaluator_api = get_train_evaluator_api(NEW_FILEPATH)
            except:
                render_template("smallerror.html")
            perplexity = evaluator_api(test_filepath)

            #convert ARPA to FST
            subprocess.call("ngramread --ARPA --epsilon_symbol='<eps>' app/static/models/my_model.arpa app/static/models/my_model.fst", shell=True)

            #evaluate AWER
            awer_model_api = get_awer_model_api()
            awer = awer_model_api(test_filepath)

            #calculate vocab coverage
            vocab = vocab_coverage(NEW_FILEPATH, test_filepath)

            return render_template("score.html", title='Perplexity of Model based on Augmented Training Set',
            plxy=perplexity, awer=awer, coverage=vocab, type="train")

    return render_template("testinput.html", title='Augmented Model Evaluation')

#@app.route('/score', methods=['GET', 'POST'])
#def score():
#    return render_template("score.html", title="Perplexity Score")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download')
def download_model():
    return send_from_directory(app.config['MODEL_FOLDER'], 'my_model.arpa')

@app.route('/paraphrase_savedata')
def save_data():
    return send_from_directory(app.config['DATA_FOLDER'], 'savedata.txt')
