import numpy as np
import pandas as pd

class EEGDataGenerator:
    def __init__(self, num_channels=8, sequence_length=100, sample_rate=256):
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        # Expanded vocabulary
        self.vocabulary = ['hello', 'world', 'thank', 'you', 'yes', 'no', 
                           'goodbye', 'please', 'sorry', 'help', 
                           'maybe', 'love', 'friend', 'happy', 'sad']
        
        # Base frequencies for different words (Hz)
        self.base_frequencies = {
            0: [8, 12, 16],   # hello
            1: [10, 14, 18],  # world
            2: [9, 13, 17],   # thank
            3: [11, 15, 19],  # you
            4: [7, 11, 15],   # yes
            5: [6, 10, 14],   # no
            6: [5, 9, 13],    # goodbye
            7: [12, 16, 20],  # please
            8: [13, 17, 21],  # sorry
            9: [4, 8, 12],    # help
            10: [14, 18, 22], # maybe
            11: [3, 7, 11],   # love
            12: [15, 19, 23], # friend
            13: [2, 6, 10],   # happy
            14: [1, 5, 9],    # sad
        }
        
    def generate_word_pattern(self, word_idx):
        """Generate a unique pattern for each word"""
        pattern = np.zeros((self.sequence_length, self.num_channels))
        
        # Select base frequencies for the current word
        frequencies = self.base_frequencies[word_idx]
        
        time = np.linspace(0, self.sequence_length/self.sample_rate, self.sequence_length)
        
        for channel in range(self.num_channels):
            # Create composite signal with different frequencies
            signal = np.zeros(self.sequence_length)
            for freq in frequencies:
                # Create a more complex wave with different harmonics
                signal += (np.sin(2 * np.pi * freq * time) +
                            np.sin(2 * np.pi * (freq / 2) * time) * 0.5 +
                            np.sin(2 * np.pi * (freq * 2) * time) * 0.3)
            
            # Add channel-specific phase shift and noise
            phase_shift = np.random.uniform(0, 2 * np.pi)
            signal = np.roll(signal, int(phase_shift * self.sequence_length / (2 * np.pi)))
            noise = np.random.normal(0, np.random.uniform(0.05, 0.2), self.sequence_length)  # Varied noise
            pattern[:, channel] = signal + noise
            
        return pattern
        
    def generate_dataset(self, num_samples=1000):
        """Generate complete dataset with labels"""
        X = np.zeros((num_samples, self.sequence_length, self.num_channels))
        y = np.zeros(num_samples, dtype=int)
        words = []
        
        for i in range(num_samples):
            # Select random word
            word_idx = np.random.randint(0, len(self.vocabulary))
            y[i] = word_idx
            words.append(self.vocabulary[word_idx])
            
            # Generate pattern
            X[i] = self.generate_word_pattern(word_idx)
            
            # Add random amplitude variation
            X[i] *= np.random.uniform(0.5, 1.5)  # More significant amplitude variations
        
        # Create DataFrame with metadata
        metadata = pd.DataFrame({
            'sample_id': range(num_samples),
            'word': words,
            'word_idx': y
        })
        
        return X, metadata
    
    def save_dataset(self, X, metadata, base_filename):
        """Save dataset to files"""
        np.save(f'{base_filename}_signals.npy', X)
        metadata.to_csv(f'{base_filename}_metadata.csv', index=False)
        
    def load_dataset(self, base_filename):
        """Load dataset from files"""
        X = np.load(f'{base_filename}_signals.npy')
        metadata = pd.read_csv(f'{base_filename}_metadata.csv')
        return X, metadata

if __name__ == "__main__":
    # Example usage
    generator = EEGDataGenerator()
    X, metadata = generator.generate_dataset(num_samples=1000)
    generator.save_dataset(X, metadata, "eeg_dataset")
    print("Dataset generated and saved successfully!")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {len(metadata)}")
