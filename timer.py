import time


class timer:

    """
    Timing procedure

    """
    def __init__(self, name):
     
       self.name = name
       self.acc_time = 0.0
       self.measure = False
      
    def start(self):

       self.time = time.time()
       self.measure = True

       return

    def end(self):

       self.measure = False
       self.acc_time = self.acc_time + time.time() - self.time

       return

    def prnt(self):

       print(self.name, ': accumulated time', self.acc_time)

       return
