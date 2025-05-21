import React, { useEffect, useRef } from 'react';
import { Box, Typography, Paper } from '@mui/material';

const EEGVisualizer = ({ eegData, channelNames = [] }) => {
  const canvasRef = useRef(null);
  
  // Draw EEG data on canvas
  useEffect(() => {
    if (!eegData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Get a random trial if eegData has multiple trials
    let signalData;
    if (Array.isArray(eegData.eeg_data) && eegData.eeg_data.length > 0) {
      // Get a random trial
      const trialIndex = Math.floor(Math.random() * eegData.eeg_data.length);
      signalData = eegData.eeg_data[trialIndex];
    } else {
      signalData = eegData;
    }
    
    // If no channel names provided, generate them
    const channels = channelNames.length > 0 ? channelNames : 
      (eegData.channels || Array.from({ length: signalData.length }, (_, i) => `Ch ${i+1}`));
    
    // Number of channels to display
    const numChannels = Math.min(signalData.length, 8); // Limit to 8 channels for visibility
    
    // Calculate channel height
    const channelHeight = height / numChannels;
    
    // Draw each channel
    for (let i = 0; i < numChannels; i++) {
      const channelData = signalData[i];
      
      // Skip if channel data is not available
      if (!channelData || !Array.isArray(channelData)) continue;
      
      // Calculate scaling factors
      const yCenter = i * channelHeight + channelHeight / 2;
      const yScale = channelHeight * 0.4; // Scale to 40% of channel height
      
      // Normalize data to [-1, 1] range
      const min = Math.min(...channelData);
      const max = Math.max(...channelData);
      const range = max - min;
      
      // Draw channel label
      ctx.fillStyle = '#555';
      ctx.font = '10px Arial';
      ctx.fillText(channels[i] || `Ch ${i+1}`, 5, yCenter - channelHeight * 0.4);
      
      // Draw baseline
      ctx.strokeStyle = '#ddd';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, yCenter);
      ctx.lineTo(width, yCenter);
      ctx.stroke();
      
      // Draw EEG signal
      ctx.strokeStyle = '#4285f4';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      
      const xStep = width / channelData.length;
      
      for (let j = 0; j < channelData.length; j++) {
        const x = j * xStep;
        const normalizedValue = range === 0 ? 0 : (channelData[j] - min) / range * 2 - 1;
        const y = yCenter - normalizedValue * yScale;
        
        if (j === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
    }
  }, [eegData, channelNames]);
  
  if (!eegData) {
    return null;
  }
  
  return (
    <Paper elevation={2} sx={{ p: 2, mb: 3, bgcolor: '#f9f9f9' }}>
      <Typography variant="h6" gutterBottom>
        EEG Signal Visualization
      </Typography>
      <Box sx={{ width: '100%', height: '300px', position: 'relative' }}>
        <canvas 
          ref={canvasRef} 
          width={800} 
          height={300} 
          style={{ width: '100%', height: '100%' }}
        />
      </Box>
      <Typography variant="caption" color="text.secondary">
        Showing {Array.isArray(eegData.eeg_data) ? `1 of ${eegData.eeg_data.length} trials` : 'EEG data'} 
        with {eegData.channels ? eegData.channels.length : 'multiple'} channels
      </Typography>
    </Paper>
  );
};

export default EEGVisualizer;
