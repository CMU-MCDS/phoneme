from flask_wtf import FlaskForm
from urllib.parse import urlparse, urljoin
from flask import request, url_for, redirect
from wtforms import StringField,PasswordField,BooleanField,SubmitField
from wtforms.validators import DataRequired,EqualTo,Length




class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[Length(min=4, max=25),DataRequired()])
    email = StringField('Email Address', validators=[Length(min=6, max=35),DataRequired()])
    password = PasswordField('New Password', validators=[
        DataRequired(),
        EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password',validators=[DataRequired()])
    submit = SubmitField('Sign Up')
