from base_buffer import BaseBuffer
import numpy as np
import warnings



class SimpleBuffer(BaseBuffer):
    """
    maintain a Dict that includes different keys ('obs', 'act', 'next_obs', 'reward', 'done') and the corresponding List data.
    """
    def __init__(self,
                 max_buffer_size: int,
                 obs_dim: tuple,
                 act_dim: tuple,
                 replacement: bool =True) -> None:
        super().__init__()
        self.max_buffer_size = max_buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replacement = replacement

        # init buffer Dict
        self.data = {}
        self.data['obs'] = np.empty((self.max_buffer_size, *self.obs_dim))
        self.data['act'] = np.empty((self.max_buffer_size, *self.act_dim))
        self.data['next_obs'] = np.empty((self.max_buffer_size, *self.obs_dim))
        self.data['reward'] = np.empty((self.max_buffer_size, ))
        self.data['done'] = np.empty((self.max_buffer_size, ))
        self.keys = list(self.data)

        self.pos = 0
        self.size = 0 

    def add_sample(self, data):
        """ 'data': a Dict in default """
        for k, v in data.items():
            self.data[k][self.pos] = v
        self.pos = (self.pos + 1) % self.max_buffer_size
        self.size += 1
        self.size = min(self.size, self.max_buffer_size)

    def get_random_batch(self, batch_size: int):
        idx = np.random.choice(self.size, size=batch_size, replace=self.replacement or self.size < batch_size)
        if not self.replacement and self.size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = {k: self.data[k][idx] for k in self.keys}
        return batch

    def add_batch(self, batch_data):
        """
        'batch_data': a Dict consists of a batch of samples, 
        like: {'obs':[np.array, ..., np.array], ... , 'done':[bool, ..., bool]} 
        """
        assert list(batch_data) == self.keys, "Mismatch between buffer keys and batch keys."
        batch_size = len(batch_data['obs'])
        for i in range(batch_size):
            data = {k: batch_data[k][i] for k in self.keys}
            self.add_sample(data)

    def load_d4rl_dataset(self, env_name):
        pass


if __name__ == '__main__':
    buffer = SimpleBuffer(3, (6,), (4,))
    for i in range(10):
        synthetic_data = {
            'obs': np.array([i] * 6),
            'act': np.array([-i] * 4),
            'next_obs': np.array([i+1] * 6),
            'reward': np.array(0, dtype=int),
            'done': np.array(True, dtype=bool)
        }

        buffer.add_sample(synthetic_data)
    print(buffer.size)
    for k, v in buffer.data.items():
        print('Key value: {}'.format(k))
        print('Item value:')
        print(v)
    
