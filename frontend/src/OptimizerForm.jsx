// OptimizerForm.jsx
import yaml from 'js-yaml';
import * as React from 'react';
import {useState, useEffect} from 'react';
import {useLocation, useNavigate} from 'react-router-dom';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

import DesignSpace from './DesignSpace';
import OptimizerConfiguration from './OptimizerConfiguration';
import {BACKEND_URL} from './config';

// Import the logger
import {logEvent, getUserId, setLogContext} from './logger';

const steps = ['', ''];

const OptimizerForm = ({
  isResultView,
  formValues,
  files,
  isOptimizing,
  setFormValues,
  setFiles,
  setIsOptimizing,
}) => {
  const location = useLocation();
  const navigate = useNavigate();
  const userIdFromQuery = new URLSearchParams(location.search).get('uid');
  const userId = userIdFromQuery || getUserId();

  // Tell the logger the study folder as soon as we can derive it
  useEffect(() => {
    if (location.pathname.startsWith('/chartreader')) {
      setLogContext({
        task_name: 'chart_reader',
        chart_mode: formValues?.chart_mode || 'tool',
        chart_task: formValues?.chart_task || 'theme_parks',
      });
    } else if (location.pathname.startsWith('/toggle')) {
      const s1 = formValues?.source1 || '';
      const s2 = formValues?.source2 || '';
      const s3 = formValues?.source3 || '';
      const toggleMode = [s1, s2, s3].filter(Boolean).join('+') || undefined; // <-- no fallback

      setLogContext({
        task_name: 'toggle_task',
        // pass sources explicitly so backend /log can derive the folder
        source1: s1,
        source2: s2,
        source3: s3,
        // optional: include derived mode (no fallback)
        toggle_mode: toggleMode,
      });
    } else if (location.pathname.startsWith('/tutorial')) {
      setLogContext({ task_name: 'tutorial' });
    } else if (location.pathname.startsWith('/byos')) {
      setLogContext({ task_name: 'byos' });
    }
  }, [
    formValues?.study_name,
    formValues?.chart_mode,
    formValues?.chart_task,
    // IMPORTANT: include sources so context updates when user changes them
    formValues?.source1,
    formValues?.source2,
    formValues?.source3,
    location.pathname,
  ]);

  const log = (eventType, detail, extras = {}) => {
    if (location.pathname.startsWith('/toggle')) {
      logEvent(eventType, detail, {
        task_name: 'toggle_task',
        source1: formValues?.source1 ?? '',
        source2: formValues?.source2 ?? '',
        source3: formValues?.source3 ?? '',
        ...extras,
      });
    } else {
      logEvent(eventType, detail, extras);
    }
  };

  // Decide which step to show initially
  const getInitialStep = () => {
    // If user is in results view, skip straight to step 1
    if (isResultView) {
      return 1;
    }
    // If route is /byos, show step 0 (the hidden "OptimizerConfiguration")
    if (location.pathname.startsWith('/byos')) {
      return 0;
    }
    // Otherwise (tutorial, chartreader, toggle), skip step 0 and start at step 1
    return 1;
  };
  const [activeStep, setActiveStep] = useState(getInitialStep());
  const [errors, setErrors] = useState({});

  // Track if the user has made changes in the design space that aren’t yet applied
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const lambdaFromToggleMode = mode => {
    switch (mode) {
      case 'human_only':
        return 0.0;
      case 'llm_only':
        return 1.0;
      case 'tool':
      default:
        return 0.5;
    }
  };

  // Reload default config on every route change
  useEffect(() => {
    const loadDefaultConfig = async () => {
      let configUrl = '';
      if (location.pathname.startsWith('/chartreader')) {
        configUrl = '/files/chartreader_default.yaml';
      } else if (location.pathname.startsWith('/tutorial')) {
        configUrl = '/files/tutorial_default.yaml';
      } else if (location.pathname.startsWith('/byos')) {
        configUrl = '/files/byos_default.yaml';
      } else if (location.pathname.startsWith('/toggle')) {
        configUrl = '/files/toggle_default.yaml';
      }
      try {
        const response = await fetch(configUrl);
        const text = await response.text();
        const config = yaml.load(text);

        setFormValues(prev => ({
          ...prev,
          ...config,
        }));
      } catch (error) {
        console.error('Error loading YAML config:', error);
      }
    };
    if (!formValues || !formValues.task_name) {
      console.log('form values:', formValues);
      loadDefaultConfig();
    }
  }, [location.pathname]);

  // Pre-load environment & evaluation files
  useEffect(() => {
    // For BYOS, do not preload files so that the user must upload them manually.
    if (location.pathname.startsWith('/byos')) {
      return;
    }
    const loadFile = async (url, filename, type) => {
      const response = await fetch(url);
      const blob = await response.blob();
      return new File([blob], filename, {type});
    };

    const fetchFile = async () => {
      try {
        let envUrl = '/files/EnvMDPLunarLander.py';
        let evalUrl = '/files/EvaluationPixelLunarLander.py';

        const envFile = await loadFile(envUrl, 'environment.py', 'text/plain');
        const evalFile = await loadFile(
          evalUrl,
          'user_evaluation.py',
          'text/plain',
        );
        setFiles({environment: envFile, evaluation: evalFile});
      } catch (error) {
        console.error('Error loading files:', error);
      }
    };
    fetchFile();
  }, [location.pathname]);

  const validateForm = () => {
    const errors = {};

    // Step 1
    if (!formValues.study_name) {
      errors.study_name = 'Study name is required';
    }
    if (!formValues.database_name) {
      errors.database_name = 'Database name is required';
    }
    if (!formValues.num_startup_trials) {
      errors.num_startup_trials = 'Number of startup trials is required';
    } else if (formValues.num_startup_trials < 1) {
      errors.num_startup_trials = 'Number of startup trials must be at least 1';
    }
    if (!formValues.num_ei_candidates) {
      errors.num_ei_candidates = 'Number of EI candidates is required';
    } else if (formValues.num_ei_candidates < 1) {
      errors.num_ei_candidates = 'Number of EI candidates must be at least 1';
    }
    if (!formValues.n_processes) {
      errors.n_processes = 'Number of parallel evaluations is required';
    } else if (formValues.n_processes < 1) {
      errors.n_processes = 'Number of parallel evaluations must be at least 1';
    }
    if (!formValues.n_gpus) {
      errors.n_gpus = 'Number of GPUs is required';
    } else if (formValues.n_gpus < 0) {
      errors.n_gpus = 'Number of GPUs cannot be negative';
    }
    if (!formValues.max_number_steps_per_episode) {
      errors.max_number_steps_per_episode = 'Max number of steps is required';
    } else if (formValues.max_number_steps_per_episode < 1) {
      errors.max_number_steps_per_episode =
        'Max number of steps must be at least 1';
    }
    if (!formValues.number_environments_for_training) {
      errors.number_environments_for_training =
        'Number of vectorized environments is required';
    } else if (formValues.number_environments_for_training < 1) {
      errors.number_environments_for_training =
        'Number of vectorized environments must be at least 1';
    }
    if (!formValues.training_batch_size) {
      errors.training_batch_size = 'Batch size is required';
    } else if (formValues.training_batch_size < 1) {
      errors.training_batch_size = 'Batch size must be at least 1';
    }
    if (!formValues.random_seed) {
      errors.random_seed = 'Random seed is required';
    }

    if (!files.environment) {
      errors.environment = 'Environment file is required';
    }
    if (!files.evaluation) {
      errors.evaluation = 'Evaluation file is required';
    }
    // Step 3
    if (!formValues.n_designs) {
      errors.n_designs = 'Number of design suggestions is required';
    } else if (formValues.n_designs < 1) {
      errors.n_designs = 'Number of design suggestions must be at least 1';
    }
    if (!formValues.number_training_timesteps) {
      errors.number_training_timesteps =
        'Number of training timesteps is required';
    } else if (formValues.number_training_timesteps < 1) {
      errors.number_training_timesteps =
        'Number of training timesteps must be at least 1';
    }

    return errors;
  };

  // ---------------------------
  // Track "last applied" values and changed params
  // ---------------------------
  // We'll store a snapshot of the entire form when user last clicked "Apply changes"
  // so we can compare new edits against that snapshot.
  const [lastAppliedValues, setLastAppliedValues] = useState(null);

  // “changedParams” will store booleans like changedParams['training_parameters']['some_param'] = true
  const [changedParams, setChangedParams] = useState({});

  // We only want to do comparisons if we’re in the results screen (isResultView).
  // If lastAppliedValues is null (i.e. first time?), we store them once on mount:
  useEffect(() => {
    if (isResultView && lastAppliedValues === null && formValues) {
      setLastAppliedValues(JSON.parse(JSON.stringify(formValues))); // deep clone
    }
  }, [isResultView, formValues]);

  // This helper toggles changedParams for a single parameter if we detect a difference
  const markParamChanged = (index, paramKey, oldVal, newVal) => {
    // Are they different? If so, set changed=true
    const isDifferent = oldVal !== newVal;
    if (isDifferent) {
      setChangedParams(prev => ({
        ...prev,
        [index]: {
          ...(prev[index] || {}),
          [paramKey]: true,
        },
      }));
    }
  };

  // Call this inside each of the change handlers to mark that user has unsaved changes
  // only if we’re on the results screen (isResultView).
  const markUnsavedChanges = () => {
    if (isResultView) {
      setHasUnsavedChanges(true);
    }
  };

  const handleInputChange = (e, key) => {
    markUnsavedChanges();
    if (isResultView && lastAppliedValues) {
      const oldVal = lastAppliedValues[key];
      const {type, value} = e.target;
      const newValue =
        type === 'number'
          ? Number(value)
          : type === 'radio'
          ? value === 'true'
          : value;
      if (oldVal !== newValue) {
        // Mark changed at top-level, e.g. changedParams['study_name'] = true
        setChangedParams(prev => ({
          ...prev,
          [key]: true,
        }));
      } else {
        setChangedParams(prev => {
          const copy = {...prev};
          delete copy[key];
          return copy;
        });
      }
    }

    const {type, value} = e.target;
    const newValue =
      type === 'number'
        ? Number(value)
        : type === 'radio'
        ? value === 'true'
        : value;
    log('inputChange', `${key} changed to ${newValue}`);
    setFormValues(prev => ({
      ...prev,
      [key]: newValue,
    }));
  };

  const handleFileChange = (e, key) => {
    const file = e.target.files[0];
    setFiles(prev => ({
      ...prev,
      [key]: file,
    }));
  };

  const handleCheckboxChange = (e, index, paramKey) => {
    markUnsavedChanges();
    if (isResultView && lastAppliedValues) {
      const oldVal = lastAppliedValues[index]?.[paramKey]?.searchable;
      const newVal = e.target.checked;
      markParamChanged(index, paramKey, oldVal, newVal);
    }
    const {checked} = e.target;
    setFormValues(prev => ({
      ...prev,
      [index]: {
        ...prev[index],
        [paramKey]: {
          ...prev[index][paramKey],
          searchable: e.target.checked,
        },
      },
    }));
  };

  const handleRangeChange = (e, newValue, index, paramKey) => {
    markUnsavedChanges();
    if (isResultView && lastAppliedValues) {
      // Compare old range to new range
      const oldStart = lastAppliedValues[index]?.[paramKey]?.start;
      const oldStop = lastAppliedValues[index]?.[paramKey]?.stop;
      const oldRange = JSON.stringify([oldStart, oldStop]);
      const newRange = JSON.stringify(newValue); // e.g. [min, max]
      markParamChanged(index, paramKey, oldRange, newRange);
    }

    const [min, max] = newValue;
    setFormValues(prev => ({
      ...prev,
      [index]: {
        ...prev[index],
        [paramKey]: {
          ...prev[index][paramKey],
          start: Number(min),
          stop: Number(max),
          integer: Number.isInteger(min) && Number.isInteger(max),
        },
      },
    }));
  };

  const handleDeleteParam = (e, index, paramKey) => {
    markUnsavedChanges();
    if (isResultView && lastAppliedValues) {
      // Deleting definitely changes it
      markParamChanged(index, paramKey, 'exists', 'deleted');
    }
    setFormValues(prev => {
      const updated = {...prev};
      delete updated[index][paramKey];
      return updated;
    });
  };

  const handleAddParam = (e, index, paramKey, newObj) => {
    markUnsavedChanges();
    if (isResultView && lastAppliedValues) {
      // Adding also changes it
      markParamChanged(index, paramKey, 'missing', 'added');
    }
    setFormValues(prev => ({
      ...prev,
      [index]: {
        ...prev[index],
        [paramKey]: newObj,
      },
    }));
  };

  // Replace your entire generateYaml with this
  const generateYaml = () => {
    const isChart = location.pathname.startsWith('/chartreader');
    const isToggle = location.pathname.startsWith('/toggle');

    // derive lambda WITHOUT mutating formValues
    let multiLambda = formValues.multi_surrogate_lambda;
    if (isChart) {
      multiLambda = formValues.chart_mode === 'baseline' ? 0.0 : 0.5;
    } else if (isToggle) {
      const params = new URLSearchParams(location.search);
      const toggleMode =
        formValues?.toggle_mode ?? params.get('mode') ?? 'tool';
      const lambdaFromToggle = lambdaFromToggleMode(toggleMode);
      multiLambda = lambdaFromToggle;
    }

    const configuration = {
      task_name: formValues.task_name,
      study_name: formValues.study_name,
      database_name: formValues.database_name,
      n_designs: formValues.n_designs,
      participant_id: userId,
      ...(isChart && {
        chart_task: formValues.chart_task || 'theme_parks',
        chart_mode: formValues.chart_mode || 'tool',
      }),
      ...(isToggle && {
        toggle_mode: formValues.toggle_mode || 'tool',
        // include explicit source selections so backend can detect which
        // metrics to compute/display (llm, wave, human, etc.)
        source1: formValues.source1 || '',
        source2: formValues.source2 || '',
        source3: formValues.source3 || '',
      }),
    };

    const yamlDoc = {
      configuration,
      agent_training_configuration: {
        continue_from_existing_database:
          formValues.continue_from_existing_database,
        number_training_timesteps: formValues.number_training_timesteps,
        max_number_steps_per_episode: formValues.max_number_steps_per_episode,
        number_environments_for_training:
          formValues.number_environments_for_training,
        training_batch_size: formValues.training_batch_size,
        stable_baselines_algorithm: formValues.stable_baselines_algorithm,
        stable_baselines_policy: formValues.stable_baselines_policy,
        random_seed: formValues.random_seed,
        device: formValues.device,
      },
      parallellization: {
        n_processes: formValues.n_processes,
        n_gpus: formValues.n_gpus,
      },
      parameter_optimization_config: {
        optuna: {
          num_startup_trials: formValues.num_startup_trials,
          num_ei_candidates: formValues.num_ei_candidates,
          multivariate: formValues.multivariate,
        },
      },
      parameters_bounds: {},
      multi_surrogate_lambda: multiLambda,
      y_min: formValues.y_min,
      y_max: formValues.y_max,
    };

    // copy bounds without UI-only fields
    const trainingParameters = formValues.training_parameters || {};
    const addedParameters = formValues.added_parameters || {};
    for (const [key, value] of [
      ...Object.entries(trainingParameters),
      ...Object.entries(addedParameters),
    ]) {
      const backendValue = {...value};
      delete backendValue.label;
      delete backendValue.user_preference_min;
      delete backendValue.user_preference_max;
      yamlDoc.parameters_bounds[key] = backendValue;
    }

    return yaml.dump(yamlDoc);
  };

  const handleStartOptimization = async () => {
    try {
      // If an optimization is already running, stop it first.
      if (isOptimizing) {
        console.log('Stopping current optimization before resuming...');
        const stopResponse = await fetch(`${BACKEND_URL}/stop_optimization`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-User-Id': userId,
          },
        });
        if (!stopResponse.ok) {
          throw new Error('Failed to stop current optimization.');
        }
        setIsOptimizing(false);
        await new Promise(resolve => setTimeout(resolve, 5000));
      }

      if (isResultView) {
        formValues.continue_from_existing_database = true;
      }

      const yamlContent = generateYaml();
      const envFile = new File([files['environment']], 'environment.py', {
        type: 'text/plain',
      });
      const evalFile = new File([files['evaluation']], 'user_evaluation.py', {
        type: 'text/plain',
      });

      // Prepare FormData to send the files
      const formData = new FormData();
      formData.append('config_yaml', yamlContent);
      formData.append('environment', envFile);
      formData.append('user_evaluation', evalFile);
      formData.append('user_id', userId);

      const response_upload_files = await fetch(
        `${BACKEND_URL}/upload_user_files`,
        {
          method: 'POST',
          body: formData,
          headers: {'X-User-Id': userId},
        },
      );

      if (!response_upload_files.ok) {
        throw new Error(
          'Failed to send the Config, Environment and Evaluation files',
        );
      }

      console.log('Config, Environment and Evaluation files sent successfully');

      const params = new URLSearchParams(location.search);
      params.set('uid', userId); // preserve existing params, just ensure uid is present
      const resultsQS = `?${params.toString()}`;
      if (location.pathname.startsWith('/tutorial')) {
        navigate('/tutorial/results' + resultsQS);
      } else if (location.pathname.startsWith('/byos')) {
        navigate('/byos/results' + resultsQS);
      } else if (location.pathname.startsWith('/toggle')) {
        navigate('/toggle/results' + resultsQS);
      } else {
        navigate('/chartreader/results' + resultsQS);
      }
      setIsOptimizing(true);

      const response = await fetch(`${BACKEND_URL}/start_optimization`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userId,
        },
      });
      if (!response.ok) {
        throw new Error('Failed to start optimization.');
      }
    } catch (error) {
      console.error('Error in handleStartOptimization:', error);
    }
  };

  const handleNext = () => {
    const newErrors = validateForm();
    setErrors(newErrors);
    if (Object.keys(newErrors).length === 0) {
      setActiveStep(prev => prev + 1);
    } else {
      console.log('Form failed to proceed due to validation errors');
    }
  };

  const handleBack = () => {
    setActiveStep(prev => prev - 1);
  };

  const [isCooldownActive, setIsCooldownActive] = useState(false);

  const handleFinish = async () => {
    const newErrors = validateForm();
    setErrors(newErrors);
    if (Object.keys(newErrors).length === 0) {
      // If we are in the result view, disable (cool down) this button for 30 seconds
      if (isResultView) {
        setIsCooldownActive(true);
        setTimeout(() => setIsCooldownActive(false), 30000);
        // After applying changes, the unsaved changes are “official,” so reset
        setHasUnsavedChanges(false);
        // 1) Copy current formValues => lastAppliedValues
        setLastAppliedValues(JSON.parse(JSON.stringify(formValues)));
        // 2) Clear out changedParams
        setChangedParams({});
      }
      await handleStartOptimization();
    }
  };

  const stepComponents = [
    <OptimizerConfiguration
      key={0}
      formValues={formValues}
      handleInputChange={handleInputChange}
      files={files}
      handleFileChange={handleFileChange}
      errors={errors}
    />,
    <DesignSpace
      key={1}
      formValues={formValues}
      handleInputChange={handleInputChange}
      setFormValues={setFormValues}
      handleCheckboxChange={handleCheckboxChange}
      handleRangeChange={handleRangeChange}
      handleDeleteParam={handleDeleteParam}
      handleAddParam={handleAddParam}
      isOptimizing={isOptimizing}
      errors={errors}
      changedParams={changedParams}
      markParamChanged={markParamChanged}
      markUnsavedChanges={markUnsavedChanges}
      lastAppliedValues={lastAppliedValues}
      isResultView={isResultView}
    />,
  ];

  if (!formValues || !formValues.task_name)
    return <div>Loading configuration...</div>;

  return (
    <div className={isResultView ? 'smallFormContainer' : 'formContainer'}>
      <Box sx={{width: '100%'}}>
        <Stepper activeStep={activeStep}>
          {steps.map((label, index) => (
            <Step key={index}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        {activeStep === steps.length ? (
          <React.Fragment>
            <Typography sx={{mt: 2, mb: 1}}>
              All steps completed - you&apos;re finished
            </Typography>
          </React.Fragment>
        ) : (
          <React.Fragment>
            <Box
              sx={{
                pt: 3,
                width: {xs: '100%', sm: '720px', md: '900px'},
                maxWidth: '100%',
                px: {xs: 2, sm: 0},
                margin: '30px auto',
                textAlign: 'left',
              }}>
              {/* If in results screen and we have unsaved changes, show a banner */}
              {isResultView && hasUnsavedChanges && (
                <Box
                  sx={{
                    mb: 2,
                    p: 1,
                    backgroundColor: '#fff3cd',
                    border: '1px solid #ffeeba',
                    borderRadius: '4px',
                  }}>
                  <Typography variant="body2" color="textSecondary">
                    <strong>Note:</strong> You have unsaved changes. Click
                    "Apply changes" to use them.
                  </Typography>
                </Box>
              )}
              {stepComponents[activeStep]}
            </Box>
            <Box
              sx={{display: 'flex', flexDirection: 'row', pt: 2, pb: 2, mt: 5}}>
              <Box sx={{flex: '1 1 auto'}} />
              {activeStep === steps.length - 1 ? (
                <Button
                  disabled={isResultView && isCooldownActive}
                  onClick={() => {
                    log(
                      'buttonClick',
                      isResultView
                        ? 'Apply changes clicked'
                        : 'Start optimization clicked',
                    );
                    handleFinish();
                  }}>
                  {isResultView ? 'Apply changes' : 'Start optimization'}
                </Button>
              ) : (
                <Button
                  onClick={() => {
                    log('buttonClick', 'Next clicked');
                    handleNext();
                  }}>
                  Next
                </Button>
              )}
            </Box>
          </React.Fragment>
        )}
      </Box>
    </div>
  );
};

export default OptimizerForm;
