import logging
from logging.handlers import RotatingFileHandler


class TrainingLogger:
    """
    A logger utility class for managing and formatting training logs.

    Attributes:
        logger (logging.Logger): The logger instance for ModularTrainer.

    Methods:
        info(message): Logs an informational message.
        warning(message): Logs a warning message.
        error(message): Logs an error message.
        debug(message): Logs a debug message.
        log_training_resume(epoch, global_step, total_epochs): Logs a message indicating training has resumed.
    """

    def __init__(self, 
                 log_path: str = './logs/training.log', 
                 level: int = logging.INFO,
                 max_log_size: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        """
        Initializes the TrainingLogger with file-based and console-based logging.

        Args:
            log_path (str): Path to the log file. Default is 'training.log'.
            level (int): Logging level (e.g., logging.DEBUG, logging.INFO). Default is logging.INFO.
            max_log_size (int): Maximum size (in bytes) of a single log file before rotation. Default is 10 MB.
            backup_count (int): Number of backup log files to retain during rotation. Default is 5.
        """
        self.logger = logging.getLogger('ModularTrainer')
        self.logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()
        
        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Rotating file handler setup
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_log_size, 
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Adding handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """
        Logs an informational message.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        Logs a warning message.

        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        Logs an error message.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)
    
    def debug(self, message: str):
        """
        Logs a debug message.

        Args:
            message (str): The message to log.
        """
        self.logger.debug(message)
    
    def log_training_resume(self, 
                            epoch: int, 
                            global_step: int, 
                            total_epochs: int):
        """
        Logs a message indicating that training has resumed.

        Args:
            epoch (int): The current epoch at which training is resuming.
            global_step (int): The global step count when training is resuming.
            total_epochs (int): The total number of epochs for the training process.
        """
        resume_message = (
            f"Training Resumed:\n"
            f"   Current Epoch: {epoch}\n"
            f"   Global Step: {global_step}\n"
            f"   Total Epochs: {total_epochs}\n"
            f"   Remaining Epochs: {total_epochs - epoch}"
        )
        self.info(resume_message)
