import React from 'react';
import {Typography, Box} from '@mui/material';

const ChartReaderTask = ({chartTask}) => {
  const getTaskDescription = () => {
    switch (chartTask) {
      case 'theme_parks':
        return (
          <Typography variant="body1">
            Imagine being a graphic designer at{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              Magic Kingdom Walt Disney
            </Typography>
            . They hire you to create a chart displaying the{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              top 20 theme parks worldwide
            </Typography>
            .
            <br />
            <br />
            They instruct you that the chart should present data{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              clearly and professionally
            </Typography>{' '}
            while adhering to the{' '}
            <Typography
              component="strong"
              sx={{
                color: '#2196f3',
                fontWeight: 'bold',
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                padding: '2px 6px',
                borderRadius: '4px',
              }}>
              blue color scheme
            </Typography>
            . The visual design should ensure{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              maximum readability
            </Typography>
            .
            <br />
            <br />
            Additionally, they have a specific task for you: the audience should
            quickly identify{' '}
            <Typography
              component="strong"
              sx={{
                color: '#f44336',
                fontWeight: 'bold',
                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                padding: '2px 6px',
                borderRadius: '4px',
              }}>
              which theme park had the highest attendance
            </Typography>
            .
          </Typography>
        );

      case 'paralympic':
        return (
          <Typography variant="body1">
            Imagine being a graphic designer for the{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              Paralympics
            </Typography>
            . They hire you to create a chart that displays the{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              ranking of Paralympic competitors per million population
            </Typography>
            .
            <br />
            <br />
            They instruct you that the chart should present data{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              clearly and professionally
            </Typography>{' '}
            while adhering to the{' '}
            <Typography
              component="strong"
              sx={{
                color: '#d32f2f',
                fontWeight: 'bold',
                backgroundColor: 'rgba(211, 47, 47, 0.1)',
                padding: '2px 6px',
                borderRadius: '4px',
              }}>
              red color scheme
            </Typography>
            . The visual design must ensure{' '}
            <Typography
              component="strong"
              sx={{color: '#1976d2', fontWeight: 'bold'}}>
              maximum readability
            </Typography>
            .
            <br />
            <br />
            Additionally, they have a specific task for you: the audience should
            be able to quickly identify{' '}
            <Typography
              component="strong"
              sx={{
                color: '#f44336',
                fontWeight: 'bold',
                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                padding: '2px 6px',
                borderRadius: '4px',
              }}>
              which country has the most Paralympic competitors per million
              population
            </Typography>
            .
          </Typography>
        );

      default:
        return '';
    }
  };

  const getTaskTitle = () => {
    switch (chartTask) {
      case 'theme_parks':
        return 'Theme Parks Chart Design';
      case 'paralympic':
        return 'Paralympic Competitors Chart Design';
      default:
        return 'Chart Design Task';
    }
  };

  // Don't render anything if no chart task is specified
  if (!chartTask) {
    return null;
  }

  return (
    <Box sx={{maxWidth: '800px', margin: '0 auto', padding: 4}}>
      <Typography variant="h4" gutterBottom align="center">
        {getTaskTitle()}
      </Typography>

      {/* Task Description */}
      <Box
        sx={{
          border: '1px solid #bbdefb',
          borderRadius: 3,
          padding: 3,
          marginBottom: 4,
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}>
        {getTaskDescription()}
      </Box>
    </Box>
  );
};

export default ChartReaderTask;
