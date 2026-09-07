// src/ToggleTaskDefinition.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';
import duolingoLogo from '/duolingo-logo.png'; // <-- put image in /public

const ToggleTaskDefinition = () => (
  <Box
    sx={{
      border: '1px solid #bbdefb',
      borderRadius: 3,
      padding: 3,
      marginBottom: 4,
      backgroundColor: 'rgba(255, 255, 255, 0.7)',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    }}
  >
    <Box sx={{ display: 'flex', alignItems: 'center', marginBottom: 2 }}>
      <Box
        component="img"
        src={duolingoLogo}
        alt="Duolingo Logo"
        sx={{
          height: 48,
          width: 48,
          borderRadius: 1,
          marginRight: 2,
          backgroundColor: '#58cc02',
        }}
      />
      <Typography variant="body1">
        <Typography component="strong" sx={{ color: '#1976d2', fontWeight: 'bold' }}>
          Duolingo
        </Typography>{' '}
        is a language-learning app with millions of users worldwide. It offers bite-sized lessons,
        progress tracking, and gamified challenges to help people learn languages on the go. The brand
        is friendly, playful, and highly accessible, appealing to learners of all ages and backgrounds.
        It uses bright colors and a cheerful tone to make learning fun and motivating.
        <br />
        <br />
        <Typography component="strong" sx={{ color: '#1976d2', fontWeight: 'bold' }}>
          Goal:{' '}
        </Typography>
        Design a toggle for Duolingo's settings screen where users can enable or disable daily
        reminder notifications. Choose the toggle design that fits naturally into Duolingo's settings
        screen and supports a clear, inviting, and easy-to-use experience for managing daily reminders.
      </Typography>
    </Box>

    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#c8e6c9',
        padding: 2,
        borderRadius: 4,
        border: '2px solid #81c784',
        marginTop: 2,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Box
          sx={{
            width: 40,
            height: 40,
            backgroundColor: 'white',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '18px',
          }}
        >
          🗓️
        </Box>
        <Box>
          <Typography variant="body2" sx={{ color: '#2e7d32', fontWeight: 'bold', fontSize: '14px' }}>
            Daily Reminder
          </Typography>
          <Typography variant="caption" sx={{ color: '#4caf50', fontSize: '12px' }}>
            Notifications
          </Typography>
        </Box>
      </Box>
      <Box
        sx={{
          backgroundColor: '#e0e0e0',
          border: '2px dashed #9e9e9e',
          borderRadius: 3,
          padding: '8px 16px',
          textAlign: 'center',
        }}
      >
        <Typography variant="caption" sx={{ color: '#757575', fontWeight: 'bold', fontSize: '12px' }}>
          [ Toggle Goes Here ]
        </Typography>
      </Box>
    </Box>
  </Box>
);

export default ToggleTaskDefinition;
