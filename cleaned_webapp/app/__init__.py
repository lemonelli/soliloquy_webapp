from flask import Flask
from config import Config

UPLOAD_FOLDER = './static/uploads'
MODEL_FOLDER = './static/models'
DATA_FOLDER = './static/data'

#Initialize the app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

#Load the config file
app.config.from_object(Config)

#Load the routes
from app import routes
