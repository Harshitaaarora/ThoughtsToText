import React from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import SimpleEEGProcessor from './components/SimpleEEG/SimpleEEGProcessor';
import './App.css'

// Create a theme for our EEG application
const theme = createTheme({
  palette: {
    primary: {
      main: '#4285f4',
    },
    secondary: {
      main: '#34a853',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SimpleEEGProcessor />
    </ThemeProvider>
  )
}

export default App
