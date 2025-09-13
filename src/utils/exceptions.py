import sys
from src.utils.logger import logging

def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    fileName = exc_tb.tb_frame.f_code.co_filename
    lineNo = exc_tb.tb_lineno
    error_message = f"Error occured in {fileName} at line {lineNo} with error message {error}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message