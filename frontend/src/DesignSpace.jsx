// DesignSpace.jsx
import {useState, useRef, useEffect} from 'react';
import {useLocation} from 'react-router-dom';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Checkbox from '@mui/material/Checkbox';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormLabel from '@mui/material/FormLabel';
import FormHelperText from '@mui/material/FormHelperText';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';

import CheckRange from './components/CheckRange';

// 1) Import your logEvent helper
import {logEvent} from './logger';
import Input from './components/Input';
import Range from './components/Range';

import ToggleTaskDefinition from './ToggleTaskDefinition';
import ChartReaderTask from './ChartReaderTask';

const DesignSpace = ({
  formValues,
  handleCheckboxChange,
  handleRangeChange,
  handleInputChange,
  handleDeleteParam,
  handleAddParam,
  isOptimizing,
  setFormValues,
  changedParams,
  markParamChanged,
  markUnsavedChanges,
  lastAppliedValues,
  isResultView,
}) => {
  const location = useLocation();
  const isToggleRoute = location.pathname.startsWith('/toggle');

  const log = (eventType, detail, extras = {}) => {
    if (isToggleRoute) {
      // STRICT: only the toggle_task needs sources for the backend to pick the folder
      logEvent(eventType, detail, {
        task_name: 'toggle_task',
        source1: formValues?.source1 ?? null,
        source2: formValues?.source2 ?? null,
        source3: formValues?.source3 ?? null,
        ...extras,
      });
    } else {
      // other tasks unchanged
      logEvent(eventType, detail, extras);
    }
  };

  const [toggle, setToggle] = useState(false);
  const parameterNameRef = useRef();
  const dataTypeRef = useRef();
  const parameterTypeRef = useRef();
  const minRef = useRef();
  const maxRef = useRef();
  const preferenceRef = useRef();

  const [optimizeParam, setOptimizeParam] = useState(true);
  const [errors, setErrors] = useState({});
  const [showParams, setShowParams] = useState(false);

  // multi_surrogate_lambda slider
  const [lambdaValue, setLambdaValue] = useState(
    formValues['multi_surrogate_lambda'] ?? 0.0,
  );

  // If chart_mode is "baseline", lock slider at 0.0
  useEffect(() => {
    if (formValues.chart_mode === 'baseline') {
      setLambdaValue(0.0);
    } else if (formValues.chart_mode === 'tool') {
      setLambdaValue(0.5);
    }
  }, [formValues.chart_mode]);

  useEffect(() => {
    if (formValues['multi_surrogate_lambda'] !== undefined) {
      setLambdaValue(formValues['multi_surrogate_lambda']);
    }
  }, [formValues]);

  // For toggle routes, ensure n_designs is always 4
  useEffect(() => {
    if (isToggleRoute && formValues['n_designs'] !== 4) {
      if (typeof setFormValues === 'function') {
        setFormValues(prevFormValues => ({
          ...prevFormValues,
          n_designs: 4,
        }));
      }
    }
  }, [isToggleRoute, formValues['n_designs'], setFormValues]);

  const handleLambdaChange = (event, newValue) => {
    markUnsavedChanges();
    setLambdaValue(newValue);
    log('sliderChange', `multi_surrogate_lambda changed to ${newValue}`);

    if (typeof setFormValues === 'function') {
      setFormValues(prevFormValues => ({
        ...prevFormValues,
        multi_surrogate_lambda: newValue,
      }));
    }
    // If in the results screen, compare old vs new
    if (isResultView && lastAppliedValues) {
      const oldVal = lastAppliedValues['multi_surrogate_lambda'];
      // "root" is the 'index' we use, "multi_surrogate_lambda" is paramKey
      markParamChanged('root', 'multi_surrogate_lambda', oldVal, newValue);
    }
  };

  const handleUserPrefChange = (e, newVal, index, paramKey) => {
    // 1) Log it (like you do in handleRangeChangeWrapper)
    log(
      'rangeChange',
      `User changed param '${paramKey}' in ${index} user_preference to ${newVal}`,
    );

    // 2) Mark unsaved changes
    markUnsavedChanges();

    // 3) If in results screen, compare old vs new, just like handleRangeChange
    if (isResultView && lastAppliedValues) {
      const oldVal = lastAppliedValues[index]?.[paramKey]?.user_preference;
      markParamChanged(index, paramKey, oldVal, newVal);
    }

    setFormValues(prev => ({
      ...prev,
      [index]: {
        ...prev[index],
        [paramKey]: {
          ...prev[index][paramKey],
          user_preference: Number(newVal),
        },
      },
    }));
  };

  // -----------------------------
  // Wrappers to log everything
  // -----------------------------
  const handleCheckboxChangeWrapper = (e, index, key) => {
    handleCheckboxChange(e, index, key);
  };

  const handleRangeChangeWrapper = (e, newValue, index, key) => {
    log(
      'rangeChange',
      `User changed param '${key}' in ${index} range to [${newValue[0]}, ${newValue[1]}]`,
    );
    handleRangeChange(e, newValue, index, key);
  };

  // A wrapper to log the trash icon / delete
  const handleDeleteParamWrapper = (e, index, key) => {
    log('buttonClick', `User deleted param '${key}' from ${index}`);
    handleDeleteParam(e, index, key);
  };

  // -----------------------------
  // Validation & new param code
  // -----------------------------
  const validateForm = () => {
    const newErrors = {};
    if (!parameterNameRef.current.value) {
      newErrors.param_name = 'Parameter name is required';
    }
    if (!minRef.current.value) {
      newErrors.min = 'Minimum value is required';
    } else if (
      maxRef.current.value &&
      Number(minRef.current.value) >= Number(maxRef.current.value)
    ) {
      newErrors.min = 'Minimum value must be less than maximum value';
    }
    if (!maxRef.current.value) {
      newErrors.max = 'Maximum value is required';
    } else if (
      minRef.current.value &&
      Number(maxRef.current.value) <= Number(minRef.current.value)
    ) {
      newErrors.max = 'Maximum value must be greater than minimum value';
    }
    if (!preferenceRef.current.value) {
      newErrors.preference = 'Preferred value is required';
    } else if (
      (minRef.current.value &&
        Number(preferenceRef.current.value) < Number(minRef.current.value)) ||
      (maxRef.current.value &&
        Number(preferenceRef.current.value) > Number(maxRef.current.value))
    ) {
      newErrors.preference =
        'Preferred value must be between minimum and maximum value';
    }
    return newErrors;
  };

  const onAddParam = e => {
    e.preventDefault();
    const newErrors = validateForm();
    setErrors(newErrors);

    if (Object.keys(newErrors).length === 0) {
      // Create an ID from the parameter name
      const id = parameterNameRef.current.value
        .toLowerCase()
        .replace(/\s+/g, '_')
        .replace(/[^a-z0-9_]/g, '');

      const minVal = Number(minRef.current.value);
      const maxVal = Number(maxRef.current.value);
      const prefVal = Number(preferenceRef.current.value);

      const newObj = {
        label: parameterNameRef.current.value || 'Label',
        type: parameterTypeRef.current.value || 'environment_kwargs',
        searchable: optimizeParam,
        integer:
          Number.isInteger(minVal) &&
          Number.isInteger(maxVal) &&
          Number.isInteger(prefVal),
        user_preference: prefVal || 5,
        user_preference_min: minVal || 0,
        user_preference_max: maxVal || 10,
        start: minVal || 0,
        stop: maxVal || 10,
      };

      // *** Here is the important logEvent with all details ***
      log(
        'buttonClick',
        `User added new parameter: name=${parameterNameRef.current.value} range=[${minVal} ${maxVal}] preferred=${prefVal} optimizeParam=${optimizeParam}`,
      );

      handleAddParam(e, 'added_parameters', id, newObj);

      // Reset
      setToggle(false);
      setOptimizeParam(true);
      parameterNameRef.current.value = '';
      dataTypeRef.current.value = 'number';
      parameterTypeRef.current.value = 'environment_kwargs';
      minRef.current.value = '';
      maxRef.current.value = '';
      preferenceRef.current.value = '';
    } else {
      console.log('Failed to add parameter due to validation errors');
    }
  };

  const onCheckboxChange = e => {
    const {checked} = e.target;
    log(
      'checkboxChange',
      `User toggled 'Optimize parameter' checkbox in new-param form → ${checked}`,
    );
    setOptimizeParam(checked);
  };

  const handleToggleBtnClick = e => {
    e.preventDefault();
    log(
      'buttonClick',
      toggle
        ? 'User closed new parameter form'
        : 'User opened new parameter form',
    );
    setToggle(!toggle);
  };

  // Detect if this is the Toggle task
  let disableLambda = false;
  let lambdaDisplayValue = lambdaValue; // default

  if (location.pathname.startsWith('/toggle')) {
    const params = new URLSearchParams(location.search);
    const toggleMode = formValues?.toggle_mode ?? params.get('mode') ?? 'tool';

    if (toggleMode === 'human_only') {
      lambdaDisplayValue = 0.0;
      disableLambda = true;
    } else if (toggleMode === 'llm_only') {
      lambdaDisplayValue = 1.0;
      disableLambda = true;
    } else {
      // tool mode
      lambdaDisplayValue = lambdaValue;
      disableLambda = false;
    }
  } else {
    // Non-toggle pages: keep your original chart_mode behavior
    if (formValues.chart_mode === 'baseline') {
      lambdaDisplayValue = 0.0;
      disableLambda = true;
    } else if (formValues.chart_mode === 'tool') {
      lambdaDisplayValue = 0.5;
      disableLambda = true;
    } else {
      lambdaDisplayValue = lambdaValue;
      disableLambda =
        location.pathname.startsWith('/tutorial') ||
        formValues.chart_mode === 'baseline' ||
        formValues.chart_mode === 'tool';
    }
  }

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Design space
      </Typography>

      {location.pathname.startsWith('/toggle') && !isResultView && (
        <Box sx={{mt: 2, mb: 4, width: '100%', maxWidth: 900, mx: 'auto'}}>
          <ToggleTaskDefinition />
        </Box>
      )}

      {formValues.chart_task && !isResultView && (
        <Box sx={{mt: 2, mb: 4, width: '100%', maxWidth: 900, mx: 'auto'}}>
          <ChartReaderTask chartTask={formValues.chart_task} />
        </Box>
      )}

      {/* TRAINING PARAMETERS: hide this entire section if chart_task is set */}
      {!(formValues.chart_task || isToggleRoute) && (
        <Box sx={{mt: 5}}>
          <Typography variant="h5" gutterBottom>
            Training parameters
          </Typography>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
              justifyContent: 'center',
              mb: 2,
            }}
            onClick={() => {
              log(
                'buttonClick',
                showParams
                  ? 'Hide training parameters panel'
                  : 'Show training parameters panel',
              );
              setShowParams(!showParams);
            }}>
            <IconButton size="small">
              {showParams ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
            <Typography variant="body2" sx={{ml: 1}}>
              {showParams
                ? 'Hide training parameters'
                : 'See training parameters'}
            </Typography>
          </Box>

          {showParams &&
            formValues['training_parameters'] &&
            Object.entries(formValues['training_parameters']).map(
              ([key, value]) => {
                // Safety check to ensure value has required properties
                if (!value || typeof value !== 'object') return null;

                const isChanged =
                  changedParams &&
                  changedParams['training_parameters'] &&
                  changedParams['training_parameters'][key];

                return (
                  <CheckRange
                    key={key}
                    index="training_parameters"
                    id={key}
                    label={value['label']}
                    formObj={value}
                    //onCheckboxChange={handleCheckboxChangeWrapper}
                    onCheckboxChange={(e, i, paramKey) => {
                      const isChecked = e.target.checked;
                      const extraInfo = isChecked
                        ? `Current range = [${value.start}, ${value.stop}]`
                        : `Current user_preference = ${value.user_preference}`;
                      log(
                        'checkboxChange',
                        `User toggled 'Optimize' for param '${paramKey}' in ${i} → ${isChecked}. ${extraInfo}`,
                      );
                      handleCheckboxChangeWrapper(e, i, paramKey);
                    }}
                    onRangeChange={handleRangeChangeWrapper}
                    onDelete={handleDeleteParamWrapper}
                    isEditable={false}
                    highlight={isChanged}
                    onUserPrefChange={handleUserPrefChange}
                    disableCheckbox={isResultView}
                    disableSlider={isResultView && !value.searchable}
                    originalRangeMin={
                      isResultView &&
                      lastAppliedValues?.training_parameters?.[key]
                        ? lastAppliedValues.training_parameters[key]
                            .user_preference_min
                        : null
                    }
                    originalRangeMax={
                      isResultView &&
                      lastAppliedValues?.training_parameters?.[key]
                        ? lastAppliedValues.training_parameters[key]
                            .user_preference_max
                        : null
                    }
                  />
                );
              },
            )}
        </Box>
      )}

      {/* DESIGN PARAMETERS */}
      <Box sx={{mt: 10}}>
        <Typography variant="h5" gutterBottom>
          Design parameters
        </Typography>
        {formValues['added_parameters'] &&
          Object.entries(formValues['added_parameters']).map(([key, value]) => {
            // Safety check to ensure value has required properties
            if (!value || typeof value !== 'object') return null;

            // isChanged => do we have changedParams["added_parameters"]?.[key]?
            const isChanged =
              changedParams?.['added_parameters'] &&
              changedParams['added_parameters'][key];
            return (
              <CheckRange
                key={key}
                index="added_parameters"
                id={key}
                label={value['label']}
                formObj={value}
                //onCheckboxChange={handleCheckboxChangeWrapper}
                onCheckboxChange={(e, i, paramKey) => {
                  const isChecked = e.target.checked;
                  const extraInfo = isChecked
                    ? `Current range = [${value.start}, ${value.stop}]`
                    : `Current user_preference = ${value.user_preference}`;
                  log(
                    'checkboxChange',
                    `User toggled 'Optimize' for param '${paramKey}' in ${i} → ${isChecked}. ${extraInfo}`,
                  );
                  handleCheckboxChangeWrapper(e, i, paramKey);
                }}
                onRangeChange={handleRangeChangeWrapper}
                onDelete={handleDeleteParamWrapper}
                isEditable={true}
                highlight={isChanged}
                onUserPrefChange={handleUserPrefChange}
                disableCheckbox={isResultView}
                disableSlider={isResultView && !value.searchable}
                originalRangeMin={
                  isResultView && lastAppliedValues?.added_parameters?.[key]
                    ? lastAppliedValues.added_parameters[key]
                        .user_preference_min
                    : null
                }
                originalRangeMax={
                  isResultView && lastAppliedValues?.added_parameters?.[key]
                    ? lastAppliedValues.added_parameters[key]
                        .user_preference_max
                    : null
                }
              />
            );
          })}

        <Box sx={{mt: 5, mb: 5}}>
          {!toggle ? (
            <Button
              variant="contained"
              onClick={handleToggleBtnClick}
              disabled={!location.pathname.startsWith('/byos')}>
              New parameter
            </Button>
          ) : (
            <div>
              <Typography variant="h6" gutterBottom>
                New parameter
              </Typography>
              <Box
                component="form"
                sx={{mt: 3, mb: 3}}
                noValidate
                autoComplete="off">
                <TextField
                  label="Parameter name"
                  type="text"
                  variant="outlined"
                  size="small"
                  sx={{background: 'white'}}
                  inputRef={parameterNameRef}
                  error={!!errors.param_name}
                  fullWidth
                />
                {errors.param_name && (
                  <FormHelperText error>{errors.param_name}</FormHelperText>
                )}
              </Box>

              <Box sx={{mt: 3, mb: 3, background: 'white', textAlign: 'left'}}>
                <FormControl fullWidth>
                  <InputLabel id="kwarg_type">Kwarg type</InputLabel>
                  <Select
                    labelId="kwarg_type"
                    defaultValue="environment_kwargs"
                    label="Kwarg type"
                    disabled={true}
                    size="small"
                    inputRef={parameterTypeRef}>
                    <MenuItem value="environment_kwargs">
                      Environment kwargs
                    </MenuItem>
                    <MenuItem value="policy_kwargs">Policy kwargs</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              <Box sx={{mt: 3, mb: 3, background: 'white', textAlign: 'left'}}>
                <FormControl fullWidth>
                  <InputLabel id="param_type">Parameter type</InputLabel>
                  <Select
                    labelId="param_type"
                    defaultValue="number"
                    label="Parameter type"
                    disabled={true}
                    size="small"
                    inputRef={dataTypeRef}>
                    <MenuItem value="number">Number</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              <Box sx={{mt: 0, mb: 2}}>
                <Typography variant="body1" gutterBottom>
                  Parameter range
                </Typography>
                <Box sx={{display: 'flex', justifyContent: 'center'}}>
                  <Box sx={{mr: 1, mb: 1, mt: 1, width: '47%'}}>
                    <TextField
                      label="Minimum value"
                      type="number"
                      variant="outlined"
                      size="small"
                      inputRef={minRef}
                      sx={{background: 'white'}}
                      error={!!errors.min}
                    />
                    {errors.min && (
                      <FormHelperText error>{errors.min}</FormHelperText>
                    )}
                  </Box>
                  <Box sx={{ml: 1, mb: 1, mt: 1, width: '47%'}}>
                    <TextField
                      label="Maximum value"
                      type="number"
                      variant="outlined"
                      size="small"
                      inputRef={maxRef}
                      sx={{background: 'white'}}
                      error={!!errors.max}
                    />
                    {errors.max && (
                      <FormHelperText error>{errors.max}</FormHelperText>
                    )}
                  </Box>
                </Box>
                <TextField
                  label="Preferred value"
                  type="number"
                  variant="outlined"
                  size="small"
                  inputRef={preferenceRef}
                  sx={{mt: 1, background: 'white'}}
                  fullWidth
                  error={!!errors.preference}
                />
                {errors.preference && (
                  <FormHelperText error>{errors.preference}</FormHelperText>
                )}
              </Box>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={optimizeParam}
                    onChange={onCheckboxChange}
                  />
                }
                label="Optimize parameter"
              />

              <Box sx={{mt: 3}}>
                <Button color="inherit" onClick={handleToggleBtnClick}>
                  Close
                </Button>
                <Button variant="contained" onClick={onAddParam} sx={{ml: 1}}>
                  Add
                </Button>
              </Box>
            </div>
          )}
        </Box>
      </Box>

      {/* OPTIMIZATION GUIDANCE SLIDER - Hidden for toggle tasks */}
      {!isToggleRoute && (
        <Box sx={{mt: 10}}>
          <Typography variant="h5" gutterBottom>
            Optimization Guidance
            {changedParams &&
              changedParams['root'] &&
              changedParams['root']['multi_surrogate_lambda'] && (
                <span style={{color: 'orange', marginLeft: '8px'}}>*</span>
              )}
          </Typography>
          <Typography variant="body2" gutterBottom>
            0.0 → Guided by human preference only
            <br />
            1.0 → Guided by simulator-based metrics only
          </Typography>

          <Box display="flex" alignItems="center" sx={{width: '100%'}}>
            <FormLabel htmlFor={'slider'}>0</FormLabel>
            <Box sx={{flexGrow: 1, ml: 2, mr: 2}}>
              <Box>
                <Slider
                  id="slider"
                  value={lambdaDisplayValue}
                  min={0.0}
                  max={1.0}
                  step={0.1}
                  onChange={handleLambdaChange}
                  valueLabelDisplay="auto"
                  track={false}
                  sx={{mt: 1}}
                  // Disable slider if chart_mode is "baseline" or "tool" OR in result view
                  disabled={
                    location.pathname.startsWith('/tutorial') ||
                    formValues.chart_mode === 'baseline' ||
                    formValues.chart_mode === 'tool' ||
                    isResultView
                  }
                />
              </Box>
            </Box>
            <FormLabel htmlFor={'slider'}>1</FormLabel>
          </Box>
        </Box>
      )}

      <Box sx={{mt: 10}}>
        <Typography variant="h5" gutterBottom>
          Design suggestions
          {changedParams && changedParams['n_designs'] && (
            <span style={{color: 'orange', marginLeft: '8px'}}>*</span>
          )}
        </Typography>

        {isToggleRoute ? (
          <FormHelperText sx={{mt: 1, fontSize: '1rem'}}>
            For toggle tasks, the number of design suggestions is fixed at 4.
          </FormHelperText>
        ) : (
          <Input
            id="n_designs"
            label="Number of design suggestions"
            type="number"
            value={formValues['n_designs']}
            onChange={handleInputChange}
            placeholder="1"
            min={1}
            error={errors.n_designs}
            disabled={isResultView}
          />
        )}
      </Box>

      {/* PARAMETER MAPPINGS FOR TOGGLE TASK */}
      {isToggleRoute && (
        <Box sx={{mt: 10}}>
          <Typography variant="h5" gutterBottom>
            Parameter Reference
          </Typography>
          <Typography variant="body2" sx={{mb: 3, color: 'text.secondary'}}>
            Categorical parameters are mapped to numbers for optimization.
            Here's the reference:
          </Typography>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: 3,
            }}>
            {/* Thumb Shape Mapping */}
            <Box
              sx={{
                p: 2,
                border: '1px solid #e0e0e0',
                borderRadius: 2,
                backgroundColor: '#fafafa',
              }}>
              <Typography variant="h6" sx={{mb: 1, color: 'primary.main'}}>
                Thumb Shape
              </Typography>
              <Box sx={{fontSize: '0.875rem', lineHeight: 1.6}}>
                <Box>0 → Circle (Perfect round)</Box>
                <Box>1 → Square (Sharp corners)</Box>
                <Box>2 → Diamond (Rotated square)</Box>
                <Box>3 → Hexagon (Six-sided)</Box>
                <Box>4 → Star (Five-pointed)</Box>
                <Box>5 → Triangle (Equilateral)</Box>
                <Box>6 → Teardrop (Pointed drop)</Box>
                <Box>7 → Bean (Organic blob)</Box>
              </Box>
            </Box>

            {/* Mappings for removed parameters omitted */}

            {/* Visual Style Mapping */}
            <Box
              sx={{
                p: 2,
                border: '1px solid #e0e0e0',
                borderRadius: 2,
                backgroundColor: '#fafafa',
              }}>
              <Typography variant="h6" sx={{mb: 1, color: 'primary.main'}}>
                Visual Style
              </Typography>
              <Box sx={{fontSize: '0.875rem', lineHeight: 1.6}}>
                <Box>0 → Flat Solid (Clean minimal)</Box>
                <Box>1 → Shadow Gradient (Depth+gradient)</Box>
                <Box>2 → Glow Glass (Luminous glass)</Box>
                <Box>3 → Outline Style (Bordered)</Box>
                <Box>4 → Elevated Textured (Floating feel)</Box>
              </Box>
            </Box>

            {/* Color Scheme Mapping */}
            <Box
              sx={{
                p: 2,
                border: '1px solid #e0e0e0',
                borderRadius: 2,
                backgroundColor: '#fafafa',
              }}>
              <Typography variant="h6" sx={{mb: 1, color: 'primary.main'}}>
                Color Scheme
              </Typography>
              <Box sx={{fontSize: '0.875rem', lineHeight: 1.6}}>
                <Box>0 → Modern Blue</Box>
                <Box>1 → Nature Green</Box>
                <Box>2 → Sunset Orange</Box>
                <Box>3 → Ocean Teal</Box>
                <Box>4 → Dark Mode</Box>
                <Box>5 → Neon Purple</Box>
                <Box>6 → Enterprise Gray</Box>
              </Box>
            </Box>

            {/* Easing Mapping */}
            <Box
              sx={{
                p: 2,
                border: '1px solid #e0e0e0',
                borderRadius: 2,
                backgroundColor: '#fafafa',
              }}>
              <Typography variant="h6" sx={{mb: 1, color: 'primary.main'}}>
                Easing Curve
              </Typography>
              <Box sx={{fontSize: '0.875rem', lineHeight: 1.6}}>
                <Box>0 → Linear (Constant speed)</Box>
                <Box>1 → Ease (Default smooth)</Box>
                <Box>2 → Ease In (Slow start)</Box>
                <Box>3 → Ease Out (Slow end)</Box>
                <Box>4 → Ease In-Out (Smooth both ends)</Box>
              </Box>
            </Box>
          </Box>
        </Box>
      )}
    </div>
  );
};

export default DesignSpace;
