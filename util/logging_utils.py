# logging_utils.py
from __future__ import print_function
import logging
import sys 
import time 

def logging_config(name, logfile=None):
	"""
	Configure stream and file output logging. 
	If we don't provide a logfile (discouraged) then we don't do 
	file logging.
	args:
		- name (str): The name of the module from which we log.
	kwargs:
		- logfile (str): The name of the logfile to use for file logging.
	"""
	logger = logging.getLogger(name)
	logger.propagate = False
	logger.setLevel(logging.INFO)
	logging.Formatter.converter = time.gmtime
	formatter_file = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s')
	formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')

	# configure stream logging (the output to stdout)
	sh = logging.StreamHandler(sys.stdout)
	sh.setLevel(logging.INFO)
	sh.setFormatter(formatter)

	logger.addHandler(sh)      

	# if we have a logfile, configure file logging.
	if logfile:
		fh = logging.FileHandler(logfile)
		fh.setLevel(logging.INFO)
		fh.setFormatter(formatter_file)
		logger.addHandler(fh)      

	return logger