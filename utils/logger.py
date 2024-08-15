import logging, os, json
from utils.misc import mkdir_p, MyEncoder

class Logger(object):
    def __init__(self, logs_dir, file_name='log.txt', filemode='w'):
        self.logs_dir = logs_dir
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=f'{logs_dir}{file_name}',
            filemode=filemode
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # logger.setLevel(logging.DEBUG)
        self.debug = logger.debug
        self.info = logger.info
        self.warning = logger.warning
        self.error = logger.error
        self.critical = logger.critical

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=4, sort_keys=True, cls=MyEncoder)