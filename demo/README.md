#Setup:

Make sure python 3 is installed. Persephone will work on python3.6 but there will be a warning. (The default python version required by Tensor flow is python 3.5.)

#Activate the virtual environment: 
	source demo-env/bin/activate
 
#Clone this branch

#Install package:
	pip install flask
	pip install persephone 

#Serve:
	python server.py

#In browser, go to
	http://127.0.0.1:5000


#When done, exit virtual environment with
	deactivate


#Potential next steps: 
1) Integrate the interface with MOSS
2) Backend API for transcribing new wav audio
3) Improve the design/implementation API: 
	TODO: output the model performance and progress back to the interface
	TODO: allow user to specify batch size, num_train, num_layers, and hidden size 
	TODO: implement functions to check the format of the uploaded data. 
	TODO: Consider what kind of files are supported and what if users uploaded multiple files
	TODO: add some functions to organize the uploaded data and manage different experiements. (
	if user uploaded multiple batches of data, the interface needs to allow user select a 
	particular training batch)