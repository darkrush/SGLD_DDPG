import pickle
import os

def Debug(func):
    def wrapper(*args):
        if __debug__ :
            return func(*args)
    return wrapper

class Checker(object):
    def __init__(self):
        self.enable = False
    @Debug
    def set_up(self, log_dir):
        if __debug__ :
            self.enable = True
            self.log_dir = log_dir
            self.flag_dict = {}
    @Debug    
    def set_flag(self, flag_tag, flag_value):
        self.flag_dict[flag_tag] =flag_value
    
    @Debug
    def get_flag(self, flag_tag, flag_value):
        if not flag_tag in self.flag_dict:
            print('**** Fatal Error: Empty Flag! ****')
            return False
        if not self.flag_dict[flag_tag] == flag_value:
            print('**** Fatal Error: Flag Mismatch! ****')
            return False
        return True
        
    @Debug
    def store_flag(self):
        with open(os.path.join(self.log_dir,'check_dict.pkl'),'wb') as f:
            pickle.dump(self.flag_dict,f)
    
Singleton_checker = Checker()


if __name__ == "__main__":
    
    Singleton_checker.set_up('foo')
    Singleton_checker.set_flag('int_flag', 0)
    Singleton_checker.set_flag('str_flag', '123')
    Singleton_checker.set_flag('bool_flag', True)
    
    assert Singleton_checker.get_flag('int_flag', 0)
    assert Singleton_checker.get_flag('str_flag', '123')
    assert Singleton_checker.get_flag('bool_flag', True)
    
    assert not Singleton_checker.get_flag('int_flag', 1)
    assert not Singleton_checker.get_flag('str_flag', '123 ')
    assert not Singleton_checker.get_flag('bool_flag', False)
    assert not Singleton_checker.get_flag('no_flag', 1)
    
    