1. 	Language used: Python 3.7

	Additional packages used: 
	
	numpy 1.18.1
	pandas 1.0.1
	scikit-learn 0.22.1.post1
	scipy 1.4.1		(comes with scikit-learn 0.22.1)
	tensorflow 2.1.0
	keras 2.3.1
	matplotlib 3.2.0    	(optional)


2. 	How to run in Window command line:
	
	_ Make folder dataset in the same directory of the .py file. Inside dataset folder are optdigits.tes, optdigits.tra.
	_ Make sure the Python 3 folder is present in the PATH environment variable.
	_ Locate the "python.exe" file in the Python 3 folder.
	_ Copy and Paste the "python.exe" file within the Python 3 folder.
	_ Rename the copied file to "python3" (or whatever you want the command to be).
	_ Navigate Window prompt to the directory containig the .py file and enter:
		python3 file_name.py


3. 	How to change the hyper-paramenters:

3.1.a. 	In the MLPs.py file:

		|	hyper parameter				| line	|
		|_______________________________________________|_______|	
		|	loss function				| 38/41	(sum-of-squares error function => loss='mean_squared_error''mean_squared_error',
		|_______________________________________________|_______cross-entropy error function => loss='categorical_crossentropy') (and line 46 in MLPs_result.py)
		|	number of epochs			|  68	|
		|	batch size				|  69	|
		|	number of hidden layers			|  70	|
		|	number of hidden units in each layer	|  71	|
		|	learning rates				|  72	|
		|	momentum rates				|  73	|
		|	whether or not scaling input		| 23-24	|
		

3.1.b. 	In the MLPs.py file

		|	hyper parameter				| line	| 
		|_______________________________________________|_______|
		|	activation hidden unit			|  38	(tanh => activation='tanh', 	(and line 54 in MLPs_result.py)
		|_______________________________________________|_______ReLu => activation='relu')       
		|	number of epochs			|  68	|
		|	batch size				|  69	|
		|	number of hidden layers			|  70	|
		|	number of hidden units in each layer	|  71	|
		|	learning rates				|  72	|
		|	momentum rates				|  73	|
		|	whether or not scaling input		| 23-24	|
		


3.2.	In the CNNs.py	

		|	hyper parameter				| line	|
		|_______________________________________________|_______|
		|	number of epochs			|  80	|
		|	batch size				|  80	|
		|	number of hidden layers			| 49-57	|
		|	learning rates				|  	|
		|	momentum rates				|   	|
		|	whether or not scaling input		| 30-31	|
		|	filter size				|  47	|
		|	kernel size				|  47  	|
		| 	pooling size				|  50   |
		|	strides					|  50 	|
		|	dropout rate				|  52 	|
		|	number of hidden units in Dense layer	| 56-57	|




