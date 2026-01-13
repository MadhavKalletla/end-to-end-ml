import sys
import logging
import logger
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    line_no=exc_tb.tb_lineno
    msg=str(error)
    err_message="The error occured in the file name[{0}] and at line number[{1}] and the message is [{2}]".format(filename,line_no,msg)
    return err_message

class Customexception(Exception):
    def __init__(self,err_message,error_detail:sys):
        super().__init__(err_message)
        self.err_message=error_message_detail(err_message,error_detail=error_detail)
    def __str__(self):
        return self.err_message  

    
    
          