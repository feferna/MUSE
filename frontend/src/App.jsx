// App.jsx
import {useState, useEffect} from 'react';
import {Routes, Route, useLocation, useNavigate} from 'react-router-dom';
import yaml from 'js-yaml';

import AddIcon from '@mui/icons-material/Add';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import TuneIcon from '@mui/icons-material/Tune';

import './App.css';
import {useAppContext} from './AppContext';
import OptimizerForm from './OptimizerForm';
import ResultView from './ResultView';
import TaskSelection from './TaskSelection';
import ToggleTask from './ToggleTask';
import {BACKEND_URL} from './config';

import {logEvent, getUserId} from './logger';

const App = () => {
  const [files, setFiles] = useState({environment: null, evaluation: null});
  const {formValues, setFormValues} = useAppContext(); //useState(null)
  const [isOptimizing, setIsOptimizing] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();

  /*useEffect(() => {
    console.log('form values:', formValues)
    const loadDefaultConfig = async () => {
        let configUrl = '';
        if (location.pathname.startsWith('/chartreader')) {
          configUrl = '/files/chartreader_default.yaml';
        } else {
          configUrl = '/files/tutorial_default.yaml';
        }
        try {
          const response = await fetch(configUrl);
          const text = await response.text();
          const config = yaml.load(text);
          setFormValues(config);
        } catch (error) {
          console.error('Error loading YAML config:', error);
      }
    };
    if (!formValues) {
      console.log('form values:', formValues)
      loadDefaultConfig();
    } 
  }, [location.pathname]);

  // Load environment/evaluation files once on mount
  useEffect(() => {
    const loadFile = async (url, filename, type) => {
      const response = await fetch(url)
      const blob = await response.blob()
      return new File([blob], filename, { type })
    }

    const fetchFile = async () => {
      try {
        const envFile = await loadFile(
          '/files/EnvMDPLunarLander.py',
          'environment.py',
          'text/plain'
        )
        const evalFile = await loadFile(
          '/files/EvaluationPixelLunarLander.py',
          'user_evaluation.py',
          'text/plain'
        )
        setFiles({ environment: envFile, evaluation: evalFile })
      } catch (error) {
        console.error('Error loading files:', error)
      }
    }

    fetchFile()
  }, [location.pathname])*/

  const resetFormValues = async () => {
    try {
      const userId = getUserId();
      setFiles({environment: null, evaluation: null});
      setFormValues(null);

      /*const configUrl = location.pathname.startsWith('/chartreader')
        ? '/files/chartreader_default.yaml'
        : '/files/tutorial_default.yaml'
      const response = await fetch(configUrl)
      const text = await response.text()
      const config = yaml.load(text)
      setFormValues(config)*/

      // Also stop any ongoing optimization
      console.log('Stopping current optimization before reset...');
      const stopResponse = await fetch(`${BACKEND_URL}/stop_optimization`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userId, // <-- add this
        },
      });
      if (!stopResponse.ok) {
        throw new Error('Failed to stop current optimization.');
      }
      setIsOptimizing(false);
    } catch (error) {
      console.error('Error resetting form:', error);
    }

    // Navigate to the current task route
    /*if (location.pathname.startsWith('/chartreader')) {
      navigate('/chartreader')
    } else {
      navigate('/tutorial')
    }*/
    navigate('/?uid=' + encodeURIComponent(getUserId()));
  };

  return (
    <>
      <AppBar position="static">
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            <TuneIcon sx={{display: {xs: 'none', md: 'flex'}, mr: 1}} />
            <Typography
              variant="h5"
              noWrap
              component="a"
              sx={{
                mr: 2,
                display: {xs: 'none', md: 'flex'},
                fontFamily: 'monospace',
                fontWeight: 700,
                letterSpacing: '.3rem',
                color: 'inherit',
                textDecoration: 'none',
              }}>
              MUSE
            </Typography>
            <Box sx={{flexGrow: 1, display: {xs: 'flex', md: 'none'}}} />
            <TuneIcon sx={{display: {xs: 'flex', md: 'none'}, mr: 1}} />
            <Typography
              variant="h5"
              noWrap
              component="a"
              sx={{
                mr: 2,
                display: {xs: 'flex', md: 'none'},
                flexGrow: 1,
                fontFamily: 'monospace',
                fontWeight: 700,
                letterSpacing: '.3rem',
                color: 'inherit',
                textDecoration: 'none',
              }}>
              MUSE
            </Typography>
            <Box sx={{flexGrow: 1, display: {xs: 'none', md: 'flex'}}} />
            {location.pathname !== '/' && (
              <>
                <Box sx={{display: {xs: 'none', md: 'flex'}}}>
                  <Button
                    onClick={() => {
                      logEvent('buttonClick', 'New optimization clicked');
                      resetFormValues();
                    }}
                    color="secondary"
                    variant="contained"
                    startIcon={<AddIcon />}
                    disabled={true}>
                    New optimization
                  </Button>
                </Box>
                <Box sx={{display: {xs: 'flex', md: 'none'}}}>
                  <Tooltip title="New optimization">
                    <div>
                      <IconButton
                        onClick={() => {
                          logEvent(
                            'buttonClick',
                            'New optimization clicked (icon)',
                          );
                          resetFormValues();
                        }}
                        sx={{
                          p: 1,
                          size: 'large',
                          color: 'white',
                          background: '#9c27b0',
                        }}
                        disabled={true} //isOptimizing
                      >
                        <AddIcon fontSize="inherit" />
                      </IconButton>
                    </div>
                  </Tooltip>
                </Box>
              </>
            )}
          </Toolbar>
        </Container>
      </AppBar>

      <Routes>
        <Route
          path="/"
          element={<TaskSelection setFormValues={setFormValues} />}
        />
        {/* Tutorial routes */}
        <Route
          path="/tutorial"
          element={
            <OptimizerForm
              isResultView={false}
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        <Route
          path="/tutorial/results"
          element={
            <ResultView
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        {/* ChartReader routes */}
        <Route
          path="/chartreader"
          element={
            <OptimizerForm
              isResultView={false}
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        <Route
          path="/chartreader/results"
          element={
            <ResultView
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        <Route
          path="/byos"
          element={
            <OptimizerForm
              isResultView={false} // start at the first step
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        <Route
          path="/byos/results"
          element={
            <ResultView
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        {/* Toggle Design routes */}
        {/* <Route path="/toggle" element={<TogglePage />} /> */}
        <Route
          path="/toggle"
          element={
            <OptimizerForm
              formValues={formValues}
              setFormValues={setFormValues}
              files={files}
              setFiles={setFiles}
              isOptimizing={isOptimizing}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
        <Route
          path="/toggle/results"
          element={
            <ResultView
              formValues={formValues}
              files={files}
              isOptimizing={isOptimizing}
              setFormValues={setFormValues}
              setFiles={setFiles}
              setIsOptimizing={setIsOptimizing}
            />
          }
        />
      </Routes>
    </>
  );
};

export default App;
