from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class EnterForm(FlaskForm):
    sentence = StringField('Input Sentence Below:', validators=[DataRequired(), Length(min=2, max=200)])
    submit_para = SubmitField('Get Paraphrases')
    submit_train = SubmitField('Add Paraphrases to Train')
