// ToggleTask.jsx
import React from 'react';
import { Typography, Box } from '@mui/material';
import ToggleTaskDefinition from './ToggleTaskDefinition';

const ToggleTask = () => {
  return (
    <Box sx={{ maxWidth: '800px', margin: '0 auto', padding: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Toggle Design Task
      </Typography>

      <ToggleTaskDefinition />
    </Box>
  );
};

export default ToggleTask;
