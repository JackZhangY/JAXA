import abc

class BaseBuffer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_sample(self, data):
        """
        Different buffers (SimpleBuffer, PointerBuffer et al.) require different ways to add samples
        """
        pass

    @abc.abstractmethod
    def add_batch(self, batch_data):
        """
        Add a batch of samples using '.add_sample()', 
        and thus 'batch_data' should be adaptive according to the buffer types (e.g., a List of trajectory data or a Dict of multiple lists) 
        """
        pass
    
    @abc.abstractmethod
    def get_random_batch(self, batch_size: int):
        pass


    

