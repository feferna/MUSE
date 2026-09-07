// TaskSelection.jsx
import {useState, useEffect} from 'react';
import {Link} from 'react-router-dom';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';

import Select from './components/Select';

// import the logger
import {setLogContext, logEvent} from './logger';

import {BACKEND_URL} from './config';

const TaskSelection = ({setFormValues}) => {
  const [userId, setUserId] = useState(
    localStorage.getItem('participant_id') || '',
  );
  const [maxOptimizationSteps, setMaxOptimizationSteps] = useState(50);
  const [chartSelected, setChartSelected] = useState(false);
  const [chartTaskType, setChartTaskType] = useState('theme_parks');
  const [chartMode, setChartMode] = useState('tool');

  const [toggleSelected, setToggleSelected] = useState(false);
  const [toggleMode, setToggleMode] = useState('tool');
  const [source1, setSource1] = useState('');
  const [source2, setSource2] = useState('');
  const [source3, setSource3] = useState('');

  useEffect(() => {
    if (userId) {
      localStorage.setItem('participant_id', userId);
    }
  }, [userId]);

  // On "/" load, stop any running optimization for this participant.
  useEffect(() => {
    let uid = localStorage.getItem('participant_id') || userId;
    if (!uid) {
      uid = `u-${Math.random().toString(36).slice(2, 6)}${Date.now()
        .toString(36)
        .slice(-4)}`;
      setUserId(uid);
      localStorage.setItem('participant_id', uid);
    }
    fetch(`${BACKEND_URL}/stop_optimization`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-User-Id': uid,
      },
    }).catch(() => {});
  }, []); // once on landing at "/"

  const genUserId = () => {
    // short, unique-enough ID: u-<rand><time base36>
    const id = `u-${Math.random().toString(36).slice(2, 6)}${Date.now()
      .toString(36)
      .slice(-4)}`;
    setUserId(id);
    localStorage.setItem('participant_id', id);
    logEvent('userIdSet', id);
  };

  const handleToggleBtnClick = e => {
    e.preventDefault();
    logEvent(
      'buttonClick',
      !chartSelected ? 'ChartReader Task clicked' : 'ChartReader Task closed',
    );
    setChartSelected(!chartSelected);
  };

  const handleTaskTypeChange = e => {
    const {value} = e.target;
    logEvent('inputChange', `chart_task changed to ${value}`);
    setChartTaskType(value);
  };

  // New handler for the tool/baseline selection
  const handleModeChange = e => {
    const {value} = e.target;
    logEvent('inputChange', `chart_mode changed to ${value}`);
    setChartMode(value);
  };

  const handleSubmitTask = e => {
    // Tell logger the task context so the server can derive the folder
    setLogContext({
      task_name: 'chart_reader',
      chart_mode: chartMode || 'tool',
      chart_task: chartTaskType || 'theme_parks',
    });

    logEvent('taskSelected', 'ChartReader Task selected');
    // Include both chart_task and chart_mode (and study_name) in the form values
    if (!userId) genUserId();
    setChartSelected(!chartSelected);

    setFormValues({
      study_name: 'chart_reader',
      chart_task: chartTaskType,
      chart_mode: chartMode,
      user_id: userId || localStorage.getItem('participant_id'),
      max_optimization_steps: Number(maxOptimizationSteps),
    });
  };

  const openTogglePanel = () => {
    logEvent('buttonClick', 'Toggle Design Task clicked');
    setToggleSelected(true);
  };
  const closeTogglePanel = () => {
    logEvent('buttonClick', 'Toggle Design Task closed');
    setToggleSelected(false);
  };

  const handleToggleModeChange = e => {
    const {value} = e.target;
    logEvent('inputChange', `toggle_mode changed to ${value}`);
    setToggleMode(value);
  };

  const handleSource1Change = e => {
    const {value} = e.target;
    logEvent('inputChange', `source1 changed to ${value}`);
    setSource1(value);
  };

  const handleSource2Change = e => {
    const {value} = e.target;
    logEvent('inputChange', `source2 changed to ${value}`);
    setSource2(value);
  };

  const handleSource3Change = e => {
    const {value} = e.target;
    logEvent('inputChange', `source3 changed to ${value}`);
    setSource3(value);
  };

  const getAvailableOptions = (currentValue, otherValues) => {
    const allOptions = [
      {value: '', label: 'None'},
      {value: 'human', label: 'Human'},
      {value: 'llm', label: 'LLM'},
      {value: 'wave', label: 'Wave'},
    ];

    return allOptions.filter(
      option =>
        option.value === currentValue ||
        option.value === '' ||
        !otherValues.includes(option.value),
    );
  };

  const handleSubmitToggleTask = () => {
    const study = 'toggle_task';
    const sources = [source1, source2, source3].filter(s => s !== '');
    setLogContext({
      task_name: 'toggle_task',
      toggle_mode: toggleMode || 'tool',
      sources: sources,
    });
    logEvent('taskSelected', 'Toggle Task selected');
    setToggleSelected(false);
    if (!userId) genUserId();
    setFormValues({
      study_name: study,
      toggle_mode: toggleMode,
      source1: source1 || '',
      source2: source2 || '',
      source3: source3 || '',
      user_id: userId || localStorage.getItem('participant_id'),
      max_optimization_steps: Number(maxOptimizationSteps),
    });
  };

  const handleTutorialClick = () => {
    setLogContext({task_name: 'tutorial'});
    if (!userId) genUserId();
    setLogContext({task_name: 'tutorial'});
    logEvent('taskSelected', 'Tutorial Task selected');
  };

  const handleByosClick = () => {
    if (!userId) genUserId();
    setLogContext({task_name: 'byos'});
    logEvent('taskSelected', 'BYOS task selected');
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
      }}>
      <div style={{textAlign: 'center'}}>
        <Typography variant="h4" gutterBottom>
          Select a Task
        </Typography>

        {/* Participant ID */}
        <Box
          sx={{
            mt: 2,
            mb: 3,
            display: 'flex',
            gap: 1,
            justifyContent: 'center',
          }}>
          <TextField
            label="Participant ID"
            size="small"
            value={userId}
            onChange={e => setUserId(e.target.value.trim())}
            placeholder="e.g., P001"
          />
          <Button variant="outlined" onClick={genUserId}>
            Generate
          </Button>
        </Box>

        {/* Max optimization steps (applies to all tasks) */}
        <Box
          sx={{
            mt: 1,
            mb: 3,
            display: 'flex',
            gap: 1,
            justifyContent: 'center',
          }}>
          <TextField
            label="Max optimization steps"
            type="number"
            size="small"
            value={maxOptimizationSteps}
            inputProps={{min: 1}}
            onChange={e => {
              const n = Number(e.target.value);
              const clamped = Number.isFinite(n)
                ? Math.max(1, Math.floor(n))
                : 50;
              setMaxOptimizationSteps(clamped);
              logEvent(
                'inputChange',
                `max_optimization_steps changed to ${clamped}`,
              );
            }}
          />
        </Box>

        <div style={{margin: '20px'}}>
          <Link
            to={`/tutorial?uid=${encodeURIComponent(
              userId || localStorage.getItem('participant_id') || 'anon',
            )}&max=${encodeURIComponent(maxOptimizationSteps)}`}>
            <Button
              variant="contained"
              disabled={chartSelected}
              onClick={handleTutorialClick}>
              Tutorial Task
            </Button>
          </Link>
        </div>

        <div style={{margin: '20px'}}>
          <Button
            variant="contained"
            disabled={chartSelected}
            onClick={handleToggleBtnClick}>
            ChartReader Task
          </Button>

          {chartSelected && (
            <div>
              <Select
                id="chart_task"
                label="Chartreader task"
                value={chartTaskType}
                options={[
                  {value: 'theme_parks', label: 'Theme Parks'},
                  {value: 'paralympic', label: 'Paralympic Competitors'},
                ]}
                onChange={handleTaskTypeChange}
                disabled={false}
              />
              {/* New selection for tool vs baseline */}
              <Select
                id="chart_mode"
                label="Use tool or baseline?"
                value={chartMode}
                options={[
                  {value: 'tool', label: 'Tool'},
                  {value: 'baseline', label: 'Baseline'},
                ]}
                onChange={handleModeChange}
                disabled={false}
              />
              <Box sx={{mt: 3}}>
                <Button color="inherit" onClick={handleToggleBtnClick}>
                  Close
                </Button>
                <Link
                  to={`/chartreader?uid=${encodeURIComponent(
                    userId || localStorage.getItem('participant_id') || 'anon',
                  )}`}>
                  <Button
                    variant="contained"
                    onClick={handleSubmitTask}
                    sx={{ml: 1}}>
                    Select Task
                  </Button>
                </Link>
              </Box>
            </div>
          )}
        </div>

        {/* Toggle Design entry + panel (NEW) */}
        <div style={{margin: '20px'}}>
          <Button
            variant="contained"
            disabled={chartSelected}
            onClick={openTogglePanel}>
            Toggle Design Task
          </Button>

          {toggleSelected && (
            <div>
              <Select
                id="source1"
                label="Source 1"
                value={source1}
                options={getAvailableOptions(source1, [source2, source3])}
                onChange={handleSource1Change}
                disabled={false}
              />
              <Select
                id="source2"
                label="Source 2"
                value={source2}
                options={getAvailableOptions(source2, [source1, source3])}
                onChange={handleSource2Change}
                disabled={false}
              />
              <Select
                id="source3"
                label="Source 3"
                value={source3}
                options={getAvailableOptions(source3, [source1, source2])}
                onChange={handleSource3Change}
                disabled={false}
              />
              <Box sx={{mt: 3}}>
                <Button color="inherit" onClick={closeTogglePanel}>
                  Close
                </Button>
                <Link
                  to={`/toggle?uid=${encodeURIComponent(
                    userId || localStorage.getItem('participant_id') || 'anon',
                  )}`}>
                  <Button
                    variant="contained"
                    onClick={handleSubmitToggleTask}
                    sx={{ml: 1}}>
                    Select Task
                  </Button>
                </Link>
              </Box>
            </div>
          )}
        </div>

        <div style={{margin: '20px'}}>
          <Link
            to={`/byos?uid=${encodeURIComponent(
              userId || localStorage.getItem('participant_id') || 'anon',
            )}&max=${encodeURIComponent(maxOptimizationSteps)}`}>
            <Button
              variant="contained"
              disabled={chartSelected}
              onClick={handleByosClick}>
              Bring your own RL-based simulator
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default TaskSelection;
