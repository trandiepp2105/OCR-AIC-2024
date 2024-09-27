import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.current_index = 0
        self.num_workers = num_workers

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_index = 0
        return self

    def load_data(self, idx):
        image, filename = self.dataset[idx]
        return image, filename

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        
        # Dùng ThreadPoolExecutor để tải dữ liệu song song
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            batch_data = list(executor.map(self.load_data, batch_indices))

        batch_images, batch_filenames = zip(*batch_data)
        self.current_index += self.batch_size
        
        return np.array(batch_images), list(batch_filenames)
