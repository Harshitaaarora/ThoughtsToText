# gui_interface.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class EEGGUI:
    def __init__(self, data_generator, processor):
        self.root = tk.Tk()
        self.root.title("EEG to Text Converter")
        self.root.geometry("800x600")
        
        self.data_generator = data_generator
        self.processor = processor
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.graph_frame = ttk.Frame(self.root, padding="10")
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Generate Dataset", 
                  command=self.generate_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Test Prediction", 
                  command=self.test_prediction).pack(side=tk.LEFT, padx=5)
        
        # Setup matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results display
        self.result_var = tk.StringVar()
        ttk.Label(self.graph_frame, textvariable=self.result_var, 
                 font=('Arial', 14)).pack(pady=10)
        
    def generate_dataset(self):
        """Generate new dataset"""
        try:
            self.X, self.metadata = self.data_generator.generate_dataset(num_samples=1000)
            messagebox.showinfo("Success", "Dataset generated successfully!")
            self.plot_sample_signal(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dataset: {str(e)}")
    
    def train_model(self):
        """Train the model"""
        try:
            if not hasattr(self, 'X'):
                messagebox.showerror("Error", "Please generate dataset first!")
                return
                
            history = self.processor.train(self.X, self.metadata['word_idx'].values)
            self.processor.plot_training_history(history)
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def test_prediction(self):
        """Test model prediction"""
        try:
            if not hasattr(self, 'X'):
                messagebox.showerror("Error", "Please generate dataset first!")
                return
                
            # Randomly select a sample
            idx = np.random.randint(0, len(self.X))
            signal = self.X[idx]
            true_word = self.metadata.iloc[idx]['word']
            
            # Make prediction
            word_idx, confidence = self.processor.predict(signal, return_confidence=True)
            predicted_word = self.data_generator.vocabulary[word_idx]
            
            # Update display
            self.plot_sample_signal(idx)
            self.result_var.set(
                f"Predicted: {predicted_word} (Confidence: {confidence:.2f})\n"
                f"Actual: {true_word}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
    
    def plot_sample_signal(self, idx):
        """Plot EEG signal"""
        self.ax.clear()
        signal = self.X[idx]
        for channel in range(signal.shape[1]):
            self.ax.plot(signal[:, channel] + channel*4, 
                        label=f'Channel {channel+1}')
        
        self.ax.set_title('EEG Signal')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend()
        self.canvas.draw()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()