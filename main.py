# main.py
from dataset_generator import EEGDataGenerator
from eeg_processor import EEGProcessor
from gui_interface import EEGGUI

def main():
    # Initialize components
    data_generator = EEGDataGenerator(
        num_channels=8,
        sequence_length=100,
        sample_rate=256
    )
    
    processor = EEGProcessor(
        num_channels=8,
        sequence_length=100
    )
    
    # Create and run GUI
    gui = EEGGUI(data_generator, processor)
    gui.run()

if __name__ == "__main__":
    main()