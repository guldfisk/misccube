import os

from appdirs import AppDirs


APP_DATA_PATH = AppDirs('misccube', 'misccube').user_data_dir

OUT_DIR = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	'out',
)