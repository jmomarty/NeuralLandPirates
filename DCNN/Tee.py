import sys
import codecs


class Tee(object):
    def __init__(self, name, mode='w'):
        self.file = codecs.open(name, mode, 'utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
