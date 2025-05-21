import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Button, CircularProgress, Alert, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { getAvailableSubjects, getSampleEEG, predictFromEEG } from '../../services/api';
import EEGVisualizer from './EEGVisualizer';

const SimpleEEGProcessor = () => {
  // State for data
  const [subjects, setSubjects] = useState([]);
  const [currentSubject, setCurrentSubject] = useState('');
  const [eegData, setEegData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sentence, setSentence] = useState('');
  
  // State for UI
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Load available subjects on component mount
  useEffect(() => {
    loadSubjects();
  }, []);
  
  // Load available subjects
  const loadSubjects = async () => {
    try {
      setLoading(true);
      const availableSubjects = await getAvailableSubjects();
      // Make sure we have an array of subjects, even if empty
      const subjectList = Array.isArray(availableSubjects) ? availableSubjects : [];
      console.log('Loaded subjects:', subjectList);
      setSubjects(subjectList);
      
      if (subjectList.length > 0) {
        setCurrentSubject(subjectList[0]);
      } else {
        // If no subjects are available, show a helpful error message
        setError('No subjects found. Please make sure KaraOne dataset files are in the data directory.');
      }
      setLoading(false);
    } catch (err) {
      console.error('Failed to load subjects:', err);
      setError('Failed to load subjects. Please make sure the server is running.');
      setSubjects([]); // Ensure subjects is always an array
      setLoading(false);
    }
  };
  
  // Handle subject change
  const handleSubjectChange = (event) => {
    setCurrentSubject(event.target.value);
    setEegData(null);
    setPrediction(null);
    setSentence('');
  };
  
  // Load subject data
  const loadSubjectData = async (subjectId) => {
    if (!subjectId) {
      setError('Please select a subject');
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      const data = await getSampleEEG(subjectId);
      setEegData(data);
      setLoading(false);
    } catch (err) {
      console.error('Failed to load EEG data:', err);
      setError(`Failed to load EEG data: ${err.message}`);
      setEegData(null);
      setLoading(false);
    }
  };
  
  // Process EEG signal and generate sentence
  const processEEG = async () => {
    if (!eegData) {
      setError('No EEG data available. Please load data first.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setPrediction(null);
    setSentence('');
    
    try {
      console.log('EEG data structure:', eegData);
      
      // Get a random trial for demonstration
      const trialIndex = Math.floor(Math.random() * eegData.eeg_data.length);
      
      // Ensure we're working with a proper array structure
      let eegSignal = eegData.eeg_data[trialIndex];
      
      console.log(`Selected trial ${trialIndex} for prediction`);
      
      // Check if the EEG signal is a dictionary/object instead of an array
      if (eegSignal && typeof eegSignal === 'object' && !Array.isArray(eegSignal)) {
        console.log('EEG signal is an object, converting to array');
        // Convert object to array of arrays
        const channels = Object.keys(eegSignal).sort();
        eegSignal = channels.map(channel => eegSignal[channel]);
      }
      
      // Ensure it's a 2D array (channels x samples)
      if (Array.isArray(eegSignal)) {
        console.log('EEG signal shape:', eegSignal.length, 'channels with', 
          Array.isArray(eegSignal[0]) ? eegSignal[0].length : 'unknown', 'samples');
      }
      
      // Process the EEG signal - ensure it's properly formatted as an array
      const result = await predictFromEEG(eegSignal, currentSubject);
      console.log('Prediction result:', result);
      setPrediction(result);
      
      // Set the predicted syllable as the sentence
      if (result && result.prediction) {
        setSentence(result.prediction);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Failed to process EEG signal:', err);
      setError(`Failed to process EEG signal: ${err.message}`);
      setLoading(false);
    }
  };
  
  return (
    <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4, p: 2 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom align="center">
          EEG to Text Converter
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ mb: 3 }}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="subject-select-label">Select Subject</InputLabel>
            <Select
              labelId="subject-select-label"
              value={currentSubject || ''}
              label="Select Subject"
              onChange={handleSubjectChange}
              disabled={loading}
            >
              {Array.isArray(subjects) ? (
                subjects.length > 0 ? (
                  subjects.map((subject) => (
                    <MenuItem key={subject} value={subject}>{subject}</MenuItem>
                  ))
                ) : (
                  <MenuItem value="">No subjects available</MenuItem>
                )
              ) : (
                <MenuItem value="">No subjects available</MenuItem>
              )}
            </Select>
          </FormControl>
          
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={() => loadSubjectData(currentSubject)}
            disabled={!currentSubject || loading}
            sx={{ mb: 2 }}
          >
            Load EEG Data
          </Button>
          
          <Button
            variant="contained"
            color="secondary"
            fullWidth
            onClick={processEEG}
            disabled={!eegData || loading}
          >
            Process EEG Signal
          </Button>
        </Box>
        
        {/* Display EEG visualization when data is loaded */}
        {eegData && !loading && (
          <EEGVisualizer 
            eegData={eegData} 
            channelNames={eegData.channels || []}
          />
        )}
        
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
            <CircularProgress />
          </Box>
        )}
        
        {sentence && (
          <Paper elevation={2} sx={{ p: 3, bgcolor: 'primary.light', color: 'white', borderRadius: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom align="center">
              Predicted Text:
            </Typography>
            <Typography variant="h3" align="center" sx={{ fontWeight: 'bold' }}>
              {sentence}
            </Typography>
            {prediction && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body1" align="center" sx={{ mt: 1 }}>
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </Typography>
                
                {prediction.actual_label && (
                  <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                    <Typography variant="body1">
                      Actual: <strong>{prediction.actual_label}</strong>
                    </Typography>
                    {prediction.is_correct ? (
                      <Box sx={{ color: '#34a853', display: 'flex', alignItems: 'center' }}>
                        ✓ Correct
                      </Box>
                    ) : (
                      <Box sx={{ color: '#ea4335', display: 'flex', alignItems: 'center' }}>
                        ✗ Incorrect
                      </Box>
                    )}
                  </Box>
                )}
              </Box>
            )}
          </Paper>
        )}
        
        {prediction && prediction.model_metrics && (
          <Paper elevation={2} sx={{ p: 3, bgcolor: '#f5f5f5', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom align="center">
              Model Metrics
            </Typography>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', mb: 3 }}>
              <Box sx={{ textAlign: 'center', p: 1 }}>
                <Typography variant="h5" color="primary">
                  {(prediction.model_metrics.accuracy * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Accuracy
                </Typography>
              </Box>
              
              <Box sx={{ textAlign: 'center', p: 1 }}>
                <Typography variant="h5" color="primary">
                  {(prediction.model_metrics.precision * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Precision
                </Typography>
              </Box>
              
              <Box sx={{ textAlign: 'center', p: 1 }}>
                <Typography variant="h5" color="primary">
                  {(prediction.model_metrics.recall * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Recall
                </Typography>
              </Box>
              
              <Box sx={{ textAlign: 'center', p: 1 }}>
                <Typography variant="h5" color="primary">
                  {(prediction.model_metrics.f1_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  F1 Score
                </Typography>
              </Box>
            </Box>
            
            {prediction.model_metrics.confusion_matrix && (
              <Box>
                <Typography variant="subtitle1" align="center" gutterBottom>
                  Confusion Matrix
                </Typography>
                <Box sx={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '8px', border: '1px solid #ddd' }}>Actual \ Predicted</th>
                        {prediction.model_metrics.class_names.map(name => (
                          <th key={name} style={{ padding: '8px', border: '1px solid #ddd' }}>{name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {prediction.model_metrics.confusion_matrix.map((row, i) => (
                        <tr key={i}>
                          <td style={{ padding: '8px', border: '1px solid #ddd', fontWeight: 'bold' }}>
                            {prediction.model_metrics.class_names[i]}
                          </td>
                          {row.map((cell, j) => (
                            <td 
                              key={j} 
                              style={{ 
                                padding: '8px', 
                                border: '1px solid #ddd',
                                backgroundColor: i === j ? '#e6f4ea' : 'transparent'
                              }}
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              </Box>
            )}
          </Paper>
        )}
      </Paper>
    </Box>
  );
};

export default SimpleEEGProcessor;
