import os
import logging.config
SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]
permissionFilename = os.path.join(SCRIPT_PATH, r"Data/all_permission.txt")
api_file = os.path.join(SCRIPT_PATH, r"Data/APIs/API_all.csv")
smaliOpcodeFilename = os.path.join(SCRIPT_PATH, r"Data/smaliOpcode.txt")
cppermissiontxt = os.path.join(SCRIPT_PATH, r"Data/APIs/all_cp.txt")
headerfile = os.path.join(SCRIPT_PATH, r"Data/head.txt")
FILE_LOGGING = os.path.join(SCRIPT_PATH + '/Data/logging.conf')

verbose = 0
logging.config.fileConfig(FILE_LOGGING)
# create logger
logger = logging.getLogger('fileLogger')