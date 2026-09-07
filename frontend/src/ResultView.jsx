// ResultView.jsx
import {useState, useEffect, useRef} from 'react';

import {useLocation} from 'react-router-dom';

import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Checkbox from '@mui/material/Checkbox';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import IconButton from '@mui/material/IconButton';

import OptimizerForm from './OptimizerForm';
import AnimatedToggle from './AnimatedToggle';
import ToggleTask from './ToggleTask';
import ChartReaderTask from './ChartReaderTask';
import {BACKEND_URL} from './config';

import {logEvent, getUserId, setLogContext} from './logger';

const DEBUG_HIDE_BATCH_INFO = false; // set to false (or remove) for final release

const ResultView = ({
  formValues,
  files,
  isOptimizing,
  setFormValues,
  setFiles,
  setIsOptimizing,
}) => {
  const [isFormOpen, setIsFormOpen] = useState(true);
  const [batchWaitingUser, setBatchWaitingUser] = useState(false);
  const [bestTrial, setBestTrial] = useState(null);
  const [isStopping, setIsStopping] = useState(false);
  const [designOptions, setDesignOptions] = useState([]);
  const [selectedDesigns, setSelectedDesigns] = useState([]);
  const [showParams, setShowParams] = useState({});
  const [showBestParams, setShowBestParams] = useState(false);

  const [maxTrialNumberSeen, setMaxTrialNumberSeen] = useState(-1);
  const [autoStopTriggered, setAutoStopTriggered] = useState(false);
  const stopRequestedRef = useRef(false); // blocks UI updates after stop

  const location = useLocation();
  const userIdFromQuery = new URLSearchParams(location.search).get('uid');
  const userId = userIdFromQuery || getUserId();

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
      const toggle_mode = [s1, s2, s3].filter(Boolean).join('+') || undefined; // no fallback

      setLogContext({
        task_name: 'toggle_task',
        source1: s1,
        source2: s2,
        source3: s3,
        toggle_mode,
      });
    } else if (location.pathname.startsWith('/tutorial')) {
      setLogContext({task_name: 'tutorial'});
    } else if (location.pathname.startsWith('/byos')) {
      setLogContext({task_name: 'byos'});
    }
  }, [
    location.pathname,
    formValues?.chart_mode,
    formValues?.chart_task,
    formValues?.source1,
    formValues?.source2,
    formValues?.source3,
  ]);

  const params = new URLSearchParams(location.search);
  const maxParam = params.get('max');
  const fromQS = maxParam != null ? Number(maxParam) : undefined; // null -> undefined, not 0
  const fromForm = Number(formValues?.max_optimization_steps);
  const maxSteps =
    (Number.isFinite(fromForm) && fromForm > 0) ? fromForm
    : (Number.isFinite(fromQS) && fromQS > 0) ? fromQS
    : 50;

  const isToggleTask = formValues?.task_name === 'toggle_task';

  // Prefer formValues.toggle_mode if you ever set it; otherwise fall back to ?mode=
  const toggleMode = formValues?.toggle_mode ?? params.get('mode') ?? null;

  // Human-only when we're on the toggle task and mode is human_only
  // (fallback: if mode is missing, infer from lambda == 0.0 for toggle)
  const humanOnlyToggle =
    isToggleTask &&
    (toggleMode === 'human_only' ||
      (toggleMode === null && formValues?.multi_surrogate_lambda === 0.0));

  const llmOnlyToggle =
    isToggleTask &&
    (toggleMode === 'llm_only' ||
      (toggleMode === null && formValues?.multi_surrogate_lambda === 1.0));

  // Derive explicit selected sources from formValues.source1/2/3 when present.
  const _normSource = s => {
    if (!s) return null;
    const x = String(s).trim().toLowerCase();
    if (['llm', 'sim', 'simulation', 'model'].includes(x)) return 'llm';
    if (['human', 'pref', 'preference', 'user'].includes(x)) return 'human';
    if (['wave', 'color', 'colour'].includes(x)) return 'wave';
    return null;
  };

  const selectedSources = new Set();
  ['source1', 'source2', 'source3'].forEach(k => {
    const v = formValues?.[k] ?? null;
    const n = _normSource(v);
    if (n) selectedSources.add(n);
  });

  const nonHumanOnly =
    isToggleTask && selectedSources.size > 0 && !selectedSources.has('human');
  const humanOnlySelected =
    isToggleTask && selectedSources.size === 1 && selectedSources.has('human');
  const llmOnlySelected =
    isToggleTask && selectedSources.size === 1 && selectedSources.has('llm');
  // effective flags: from toggle_mode OR explicit sources
  const effectiveHumanOnly = humanOnlyToggle || humanOnlySelected;
  const effectiveLlmOnly   = llmOnlyToggle   || llmOnlySelected;

  // Reuse this everywhere we conditionally show the sim metric
  const hideSimMetric =
   (formValues.chart_task && formValues.chart_mode === 'baseline') ||
   effectiveHumanOnly;
  
  const hideDisagreement =
   (formValues.chart_task && formValues.chart_mode === 'baseline') ||
   effectiveHumanOnly ||
   effectiveLlmOnly;

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

  useEffect(() => {
    if (isOptimizing) {
      console.log('Resetting local state because isOptimizing = true');
      setBatchWaitingUser(false);
      setDesignOptions([]);
      setSelectedDesigns([]);
      setBestTrial(null);
      setShowParams({});
      setShowBestParams(false);

      setMaxTrialNumberSeen(-1);
      setAutoStopTriggered(false);
    }
  }, [isOptimizing]);

  const handleSelect = (e, trialId) => {
    const alreadySelected = selectedDesigns.includes(trialId);
    let newSelected;
    if (alreadySelected) {
      newSelected = selectedDesigns.filter(id => id !== trialId);
    } else {
      newSelected = [...selectedDesigns, trialId];
    }
    setSelectedDesigns(newSelected);
  };

  const handleToggleParams = trialNumber => {
    setShowParams(prev => ({
      ...prev,
      [trialNumber]: !prev[trialNumber],
    }));
  };

  const fetchBestTrial = async () => {
    console.log('fetchBestTrial called');
    try {
      const response = await fetch(`${BACKEND_URL}/best_trial`, {
        headers: {'X-User-Id': userId},
      });
      if (!response.ok) throw new Error('Failed to fetch best trial');
      const data = await response.json();
      console.log('fetchBestTrial response:', data);
      if (data.disagreement_data) {
        console.log('Disagreement data received:', data.disagreement_data);
      } else {
        console.log('No disagreement data received in response');
      }
      setBestTrial(data);
    } catch (error) {
      console.error('Error fetching best trial:', error.message);
    }
  };

  const fetchBatch = async () => {
    if (stopRequestedRef.current || !isOptimizing) return;
    console.log('fetchBatch called');
    try {
      const response = await fetch(`${BACKEND_URL}/current_batch_results`, {
        headers: {'X-User-Id': userId},
      });
      const batchData = await response.json();

      // Re-check after await to avoid racing a just-triggered stop
      if (stopRequestedRef.current || !isOptimizing) return;

      if (Array.isArray(batchData?.trials) && batchData.trials.length > 0) {
        const maxInBatch = Math.max(
          ...batchData.trials.map(t => t.trial_number),
        );
        setMaxTrialNumberSeen(prev => Math.max(prev, maxInBatch));
      }

      if (
        !batchData.batch_in_progress &&
        batchData.trials.length > 0 &&
        !stopRequestedRef.current
      ) {
        console.log('Batch trials data:', batchData.trials);

        // Log disagreement data for each trial
        batchData.trials.forEach(trial => {
          if (trial.disagreement_display) {
            console.log(
              `Trial ${trial.trial_number} disagreement: ${trial.disagreement_display}`,
            );
          }
        });

        try {
          const statusResponse = await fetch(
            `${BACKEND_URL}/optimization_status`,
            {
              headers: {'X-User-Id': userId},
            },
          );
          const statusData = await statusResponse.json();

          if (statusData.status === 'completed') {
            console.log('Optimization is completed, fetching best trial');
            await fetchBestTrial();
            setIsOptimizing(false);
          } else {
            console.log('Found new batch, user must pick');
            if (!stopRequestedRef.current) {
              setBatchWaitingUser(true);
              setDesignOptions(batchData.trials);
              setSelectedDesigns([]);
            }
          }
        } catch (error) {
          console.error('Error fetching optimization status:', error.message);
        }
      }
    } catch (error) {
      console.error('Error fetching batch data:', error.message);
    }
  };

  useEffect(() => {
    if (!isOptimizing || batchWaitingUser) return;
    console.log(
      'Starting polling interval: isOptimizing =',
      isOptimizing,
      ', batchWaitingUser =',
      batchWaitingUser,
    );
    const interval = setInterval(fetchBatch, 5000);
    return () => {
      console.log('Clearing polling interval.');
      clearInterval(interval);
    };
  }, [batchWaitingUser, isOptimizing]);

  const roundSignificantFigures = (number, precision) => {
    return parseFloat(number.toPrecision(precision));
  };

  const bestLlm = bestTrial?.disagreement_data?.llm_value ?? null;
  const bestWave = bestTrial?.disagreement_data?.wave_value ?? null;

  const confirmSelection = async chosen => {
    if (!Array.isArray(chosen) || chosen.length === 0) return;

    const allTrialNumbers = designOptions.map(d => d.trial_number);
    const notChosen = allTrialNumbers.filter(tn => !chosen.includes(tn));

    try {
      await fetch(`${BACKEND_URL}/choose_trial`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userId,
        },
        body: JSON.stringify({
          chosen_trials: chosen,
          not_chosen_trials: notChosen,
        }),
      });
      console.log('Auto-confirmed selection:', chosen);
    } catch (error) {
      console.error('Error sending selected designs:', error.message);
    }

    setBatchWaitingUser(false);
    setSelectedDesigns([]);
  };

  const handleConfirm = async () => {
    if (selectedDesigns.length === 0) {
      console.log('No picks selected, ignoring confirm.');
      return;
    }
    await confirmSelection(selectedDesigns);
  };

  const handleStopOptimization = async () => {
    console.log('handleStopOptimization -> stop requested.');
    // Immediately freeze UI and block any future batch UI updates
    stopRequestedRef.current = true;
    setAutoStopTriggered(true);
    setIsOptimizing(false); // kills polling on next render
    setBatchWaitingUser(false); // hide selection grid
    setDesignOptions([]); // clear any designs shown
    setSelectedDesigns([]);
    setIsStopping(true); // brief transition while fetching best
    try {
      const response = await fetch(`${BACKEND_URL}/stop_optimization`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userId,
        },
      });
      if (!response.ok) throw new Error('Failed to stop optimization');
      console.log('handleStopOptimization -> backend acknowledged stop');
    } catch (error) {
      console.error('Error stopping optimization:', error.message);
    } finally {
      // Move straight to best view
      await fetchBestTrial();
      setIsStopping(false);
    }
  };

  // Auto-select behavior when only non-human sources are used (llm, wave, or both).
  // Prefer the trial with highest average of available source metrics.
  useEffect(() => {
    if (
      !nonHumanOnly ||
      !batchWaitingUser ||
      designOptions.length === 0 ||
      stopRequestedRef.current
    )
      return;

    log(
      'autoSelect',
      'Non-human-only: will auto-confirm best predicted design in 5s',
    );

    const scoreFor = design => {
      const vals = [];
      if (
        selectedSources.has('llm') &&
        design.llm_value != null &&
        !Number.isNaN(design.llm_value)
      )
        vals.push(Number(design.llm_value));
      if (
        selectedSources.has('wave') &&
        design.wave_value != null &&
        !Number.isNaN(design.wave_value)
      )
        vals.push(Number(design.wave_value));
      if (vals.length === 0) return -Infinity;
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    };

    const timeout = setTimeout(async () => {
      if (stopRequestedRef.current) return;
      // pick best-scoring trial
      let best = null;
      let bestScore = -Infinity;
      for (const d of designOptions) {
        const s = scoreFor(d);
        if (s > bestScore) {
          bestScore = s;
          best = d;
        }
      }
      const pick = best ? best.trial_number : designOptions[0].trial_number;
      setSelectedDesigns([pick]);
      await confirmSelection([pick]);
    }, 5000);

    return () => clearTimeout(timeout);
  }, [nonHumanOnly, batchWaitingUser, designOptions, selectedSources]);

  // Global auto-stop for ANY task/mode based on maxSteps
  useEffect(() => {
    const totalSoFar = maxTrialNumberSeen + 1; // trial numbers start at 0
    if (isOptimizing && !autoStopTriggered && totalSoFar >= maxSteps + 1) {
      setAutoStopTriggered(true);
      log(
        'autoStop',
        `Stopping at ${totalSoFar} designs (limit ${maxSteps})`,
      );
      // no await needed; we don't depend on the promise here
      handleStopOptimization();
    }
  }, [isOptimizing, maxTrialNumberSeen, maxSteps, autoStopTriggered]);

  return (
    <div className="resultContainer">
      {isFormOpen && (
        <div className="formView">
          <OptimizerForm
            isResultView={true}
            formValues={formValues}
            files={files}
            isOptimizing={isOptimizing}
            setFormValues={setFormValues}
            setFiles={setFiles}
            setIsOptimizing={setIsOptimizing}
          />
        </div>
      )}
      <div>
        {isFormOpen ? (
          <button
            className="toggleButtonOpen"
            onClick={() => {
              log('buttonClick', 'Collapse form in result view');
              setIsFormOpen(!isFormOpen);
            }}>
            <ArrowBackIosIcon fontSize="small" />
          </button>
        ) : (
          <button
            className="toggleButtonClosed"
            onClick={() => {
              log('buttonClick', 'Expand form in result view');
              setIsFormOpen(!isFormOpen);
            }}>
            <ArrowForwardIosIcon fontSize="small" />
          </button>
        )}
      </div>

      <div className="resultView">
        <Box sx={{mt: 5, mb: 5, ml: 2, mr: 2}}>
          {formValues.chart_task && (
            <ChartReaderTask chartTask={formValues.chart_task} />
          )}

          {formValues?.task_name === 'toggle_task' && <ToggleTask />}

          <Typography variant="h4" sx={{mt: 0}} gutterBottom>
            Results
          </Typography>

          {batchWaitingUser ? (
            <Box sx={{mt: 5, mb: 5}}>
              <Typography variant="h5" gutterBottom>
                Select your preferred designs
              </Typography>
              <Grid container spacing={2}>
                {designOptions.map(design => (
                  <Grid item key={design.trial_number}>
                    <Card
                      variant="outlined"
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        padding: '10px',
                        minWidth: '300px',
                      }}>
                      <CardContent>
                        <Typography variant="h6">
                          Design {design.trial_number + 1}
                        </Typography>
                        <Box sx={{mt: 4, mb: 4}}>
                          {/* Show toggle component for toggle design task */}
                          {formValues?.task_name === 'toggle_task' ? (
                            <AnimatedToggle params={design.params} />
                          ) : (
                            /* Show video for other tasks */
                            <video width="300" autoPlay loop muted controls>
                              <source
                                src={`${BACKEND_URL}/${
                                  design.video_url
                                }`}
                                type="video/mp4"
                              />
                              Your browser does not support the video tag.
                            </video>
                          )}
                        </Box>

                        {/* Show metrics based on task type and sources */}
                        {!DEBUG_HIDE_BATCH_INFO && !hideSimMetric && (
                          <>
                            {/* For toggle tasks, show metrics based on selected sources */}
                            {isToggleTask && (
                              <>
                                {design.llm_value != null && (
                                  <Typography variant="subtitle2" sx={{mb: 1}}>
                                    LLM-based metric:{' '}
                                    {roundSignificantFigures(
                                      design.llm_value,
                                      5,
                                    )}
                                  </Typography>
                                )}
                                {design.wave_value != null && (
                                  <Typography variant="subtitle2" sx={{mb: 1}}>
                                    Wave-based metric:{' '}
                                    {roundSignificantFigures(
                                      design.wave_value,
                                      5,
                                    )}
                                  </Typography>
                                )}
                                {/* Fallback to simulated metric if no source-based values */}
                                {design.llm_value == null &&
                                  design.wave_value == null &&
                                  design.simulated_value != null && (
                                    <Typography
                                      variant="subtitle2"
                                      sx={{mb: 1}}>
                                      Simulated-based metric:{' '}
                                      {roundSignificantFigures(
                                        design.simulated_value,
                                        5,
                                      )}
                                    </Typography>
                                  )}
                              </>
                            )}

                            {/* For non-toggle tasks, show simulated metric */}
                            {!isToggleTask && (
                              <Typography variant="subtitle2" sx={{mb: 1}}>
                                Simulated-based metric:{' '}
                                {design.simulated_value != null
                                  ? roundSignificantFigures(
                                      design.simulated_value,
                                      5,
                                    )
                                  : 'N/A'}
                              </Typography>
                            )}
                          </>
                        )}

                        {/* Display disagreement information */}
                        {!DEBUG_HIDE_BATCH_INFO && !hideDisagreement && design.disagreement_display && (
                          <Typography
                            variant="subtitle2"
                            sx={{mb: 1, color: '#d93025'}}>
                            {design.disagreement_display}
                          </Typography>
                        )}

                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            cursor: 'pointer',
                            justifyContent: 'center',
                          }}
                          onClick={() => {
                            log(
                              'buttonClick',
                              `Toggle trial ${design.trial_number} parameters`,
                            );
                            handleToggleParams(design.trial_number);
                          }}>
                          <IconButton size="small">
                            {showParams[design.trial_number] ? (
                              <ExpandLessIcon />
                            ) : (
                              <ExpandMoreIcon />
                            )}
                          </IconButton>
                          <Typography variant="body2" sx={{ml: 1}}>
                            {showParams[design.trial_number]
                              ? 'Hide design parameters'
                              : 'See design parameters'}
                          </Typography>
                        </Box>

                        {showParams[design.trial_number] && (
                          <Box sx={{mt: 2, mb: 2}}>
                            <TableContainer>
                              <Table aria-label="params table">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Parameter</TableCell>
                                    <TableCell align="right">Value</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {Object.entries(design.params).map(
                                    ([key, value]) => (
                                      <TableRow key={key}>
                                        <TableCell component="th" scope="row">
                                          {key.replace('_', ' ')}
                                        </TableCell>
                                        <TableCell align="right">
                                          {roundSignificantFigures(value, 3)}
                                        </TableCell>
                                      </TableRow>
                                    ),
                                  )}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </Box>
                        )}
                      </CardContent>

                      <Checkbox
                        checked={selectedDesigns.includes(design.trial_number)}
                        onChange={e => {
                          log(
                            'buttonClick',
                            `Select design ${design.trial_number}`,
                          );
                          handleSelect(e, design.trial_number);
                        }}
                      />
                    </Card>
                  </Grid>
                ))}
              </Grid>
              <Button
                variant="contained"
                sx={{mt: 4}}
                disabled={selectedDesigns.length === 0}
                onClick={() => {
                  log('buttonClick', 'Confirm selection clicked');
                  handleConfirm();
                }}>
                Confirm Selection
              </Button>
            </Box>
          ) : isStopping ? (
            <Typography variant="subtitle1" gutterBottom>
              Stopping optimization
            </Typography>
          ) : isOptimizing ? (
            <Typography variant="subtitle1" gutterBottom>
              Optimizing...
            </Typography>
          ) : bestTrial ? (
            <div>
              <Box sx={{mt: 5}}>
                <Typography variant="h5" gutterBottom>
                  Best design information
                </Typography>
                <Typography variant="body1" gutterBottom>
                  Design number: {bestTrial['best_trial_number'] + 1}
                </Typography>
                {/* Primary metric line:
                  - Chart Reader:
                      • tool -> Combined
                      • baseline -> Human-Preference
                  - Toggle:
                      • human_only -> Human-Preference
                      • llm_only   -> Simulated-based metric
                      • tool       -> Combined
                  - Other tasks -> Combined
                */}
                {formValues.chart_task ? (
                  formValues.chart_mode === 'tool' ? (
                    <Typography variant="body1" gutterBottom>
                      Combined objective value:{' '}
                      {bestTrial['best_value'].toFixed(2)}
                    </Typography>
                  ) : (
                    <Typography variant="body1" gutterBottom>
                      Human-Preference Value:{' '}
                      {bestTrial['best_value'].toFixed(2)}
                    </Typography>
                  )
                ) : isToggleTask ? (
                  effectiveHumanOnly ? (
                    <Typography variant="body1" gutterBottom>
                      Human-Preference Value:{' '}
                      {bestTrial['best_value'].toFixed(2)}
                    </Typography>
                  ) : effectiveLlmOnly ? (
                    <>
                      {bestLlm != null ? (
                        <Typography variant="body1" gutterBottom>
                          LLM-based metric:{' '}
                          {roundSignificantFigures(bestLlm, 3)}
                        </Typography>
                      ) : (
                        <Typography variant="body1" gutterBottom>
                          Simulated-based metric:{' '}
                          {bestTrial['simulated_based_value'] != null
                            ? roundSignificantFigures(
                                bestTrial['simulated_based_value'],
                                3,
                              )
                            : 'N/A'}
                        </Typography>
                      )}
                    </>
                  ) : (
                    <Typography variant="body1" gutterBottom>
                      Combined objective value:{' '}
                      {bestTrial['best_value'].toFixed(2)}
                    </Typography>
                  )
                ) : (
                  <Typography variant="body1" gutterBottom>
                    Combined objective value:{' '}
                    {bestTrial['best_value'].toFixed(2)}
                  </Typography>
                )}

                {/* Secondary metrics for toggle tasks */}
                {!hideSimMetric && !llmOnlyToggle && isToggleTask && (
                  <>
                    {bestLlm != null && (
                      <Typography variant="body1" gutterBottom>
                        LLM-based metric:{' '}
                        {roundSignificantFigures(bestLlm, 3)}
                      </Typography>
                    )}
                    {bestWave != null && (
                      <Typography variant="body1" gutterBottom>
                        Wave-based metric:{' '}
                        {roundSignificantFigures(bestWave, 3)}
                      </Typography>
                    )}
                    {/* Prediction error removed per user request; we show Predicted disagreement instead */}
                    {/* Fallback to simulated metric if no source-based values */}
                    {bestLlm == null &&
                      bestWave == null && (
                        <Typography variant="body1" gutterBottom>
                          Simulated-based metric:{' '}
                          {bestTrial['simulated_based_value'] != null
                            ? roundSignificantFigures(
                                bestTrial['simulated_based_value'],
                                3,
                              )
                            : 'N/A'}
                        </Typography>
                      )}
                  </>
                )}

                {/* Secondary metric for non-toggle tasks */}
                {!hideSimMetric && !llmOnlyToggle && !isToggleTask && (
                  <Typography variant="body1" gutterBottom>
                    Simulated-based metric:{' '}
                    {bestTrial['simulated_based_value'] != null
                      ? roundSignificantFigures(
                          bestTrial['simulated_based_value'],
                          3,
                        )
                      : 'N/A'}
                  </Typography>
                )}

                {/* Display disagreement for best trial */}
                {!hideDisagreement && bestTrial.disagreement_data && (
                  <Typography
                    variant="body1"
                    gutterBottom
                    sx={{color: '#d93025'}}>
                    {bestTrial.disagreement_data.display}
                  </Typography>
                )}
              </Box>
              <Box sx={{mt: 10}}>
                <Typography variant="h6" gutterBottom>
                  {formValues?.task_name === 'toggle_task'
                    ? 'Best Toggle Design'
                    : 'Video'}
                </Typography>
                {formValues &&
                  (formValues.task_name === 'toggle_task' ? (
                    <AnimatedToggle params={bestTrial.best_params} />
                  ) : (
                        bestTrial.video_url ? (
                          <video
                            key={bestTrial.video_url}     // force reload if URL changes
                            width="640"
                            autoPlay
                            loop
                            muted
                            controls
                            onError={(e) => {
                              console.error(
                                'Best-trial video failed to load:',
                                `${BACKEND_URL}/${bestTrial.video_url}`,
                                e
                              );
                            }}
                          >
                            <source
                              src={`${BACKEND_URL}/${bestTrial.video_url}`}
                              type="video/mp4"
                            />
                            {/* Fallback: show a clickable link */}
                            <a
                              href={`${BACKEND_URL}/${bestTrial.video_url}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Open video
                            </a>
                          </video>
                        ) : (
                          <Typography variant="body2">No video for this task.</Typography>
                        )
                      )
                    )}
              </Box>
              <Box sx={{mt: 10, mb: 5}}>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    cursor: 'pointer',
                    justifyContent: 'flex-start',
                    mb: 2,
                  }}
                  onClick={() => {
                    log('buttonClick', 'Toggle best trial parameters');
                    setShowBestParams(!showBestParams);
                  }}>
                  <IconButton size="small">
                    {showBestParams ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </IconButton>
                  <Typography variant="body2" sx={{ml: 1}}>
                    {showBestParams
                      ? 'Hide design parameters'
                      : 'See design parameters'}
                  </Typography>
                </Box>
                {showBestParams && (
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Parameter</TableCell>
                          <TableCell align="right">Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(bestTrial['best_params']).map(
                          ([key, value]) => (
                            <TableRow key={key}>
                              <TableCell component="th" scope="row">
                                {key.replace('_', ' ')}
                              </TableCell>
                              <TableCell align="right">
                                {roundSignificantFigures(value, 3)}
                              </TableCell>
                            </TableRow>
                          ),
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Box>
            </div>
          ) : (
            <Typography variant="subtitle1" gutterBottom>
              No best trial available
            </Typography>
          )}
        </Box>
      </div>
    </div>
  );
};

export default ResultView;
