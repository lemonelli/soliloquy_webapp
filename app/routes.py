from os.path import join, dirname, realpath
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from app import app
from app.forms import EnterForm
from werkzeug.utils import secure_filename

from app.serve import get_model_api
from app.serve import get_model_evaluator_api
from app.serve import get_train_evaluator_api

model_api = get_model_api()

MODEL_ALLOWED_EXTENSIONS = {'arpa'}
TEXT_ALLOWED_EXTENSIONS = {'txt'}

def allowed_model_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() not in MODEL_ALLOWED_EXTENSIONS

def allowed_text_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() not in TEXT_ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home')

@app.route('/paraphrase', methods=['GET', 'POST'])
def paraphrase():
    form = EnterForm()
    if form.submit_para.data:
        if form.validate_on_submit():
            lines = model_api(form.sentence.data)
            lines.reverse()
            lines.insert(0, [0, 0, form.sentence.data])
            return render_template("paraphrase.html", title='Sentence Entry', form=form, lines=enumerate(lines))

    elif request.form.getlist("paraphrases"):
        selected_paraphrases = request.form.getlist("paraphrases")
        filepath = join(dirname(realpath(__file__)), app.config['DATA_FOLDER'], "bootstrap_train.txt")
        with open(filepath, "a+") as f:
            for utt in selected_paraphrases:
                f.write(utt + '\n')
        return render_template("paraphrase.html", title='Sentence Entry', form=form, lines=[])

    ##implement retrain()
    ##clear bootstrap_train.txt

    return render_template("paraphrase.html", title='Sentence Entry', form=form, lines=[])

@app.route('/modeleval', methods=['GET', 'POST'])
def model_eval():
    if request.method == 'POST':
        #check if post request has a file part
        if ('model' or 'training' or 'test') not in request.files:
            flash('No files given')
            return redirect(request.url)
        #parse files
        model_file = request.files['model']
        train_file = request.files['training']
        test_file = request.files['test']
        #check that files aren't null
        if (model_file.filename == '' and train_file.filename == ''):
            flash('No model or training file selected')
            return redirect(request.url)
        if test_file.filename == '':
            flash('No test file selected')
            return redirect(request.url)
        #validate and save test file
        if not (test_file):
            #filename = secure_filename(test_file.filename)
            #filepath = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], filename)
            #test_file.save(filepath)
            flash('Test file invalid')
            return redirect(request.url)

        #check for model or training set
        if model_file:
            #save model file
            model_filename = secure_filename(model_file.filename)
            model_filepath = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], model_filename)
            model_file.save(model_filepath)
            #save test file
            test_filename = secure_filename(test_file.filename)
            test_filepath = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], test_filename)
            test_file.save(test_filepath)
            #build and evaluate model
            evaluator_api = get_model_evaluator_api(model_filepath)
            perplexity = evaluator_api(test_filepath)
            return render_template("score.html", title='Perplexity of Model', plxy=perplexity, type="model")
        elif train_file:
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
            return render_template("score.html", title='Perplexity of Model based on Training Set', plxy=perplexity, type="train")
    return render_template('modelinput.html', title='Model Evaluation')

#@app.route('/score', methods=['GET', 'POST'])
#def score():
#    return render_template("score.html", title="Perplexity Score")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download')
def download_model():
    return send_from_directory(app.config['MODEL_FOLDER'], 'my_model.arpa')
