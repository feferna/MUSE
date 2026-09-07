//OptimizerConfiguration.jsx

import { useState } from 'react'

import Box from '@mui/material/Box'
import FormHelperText from '@mui/material/FormHelperText'
import Typography from '@mui/material/Typography'

import Input from "./components/Input"
import Select from "./components/Select"
import RadioInput from "./components/RadioInput"

const OptimizerConfiguration = ({formValues, handleInputChange, files, handleFileChange, errors}) => {

  // Customized environment & Evaluation method
  //const [envContent, setEnvContent] = useState('');
  //const [evalContent, setEvalContent] = useState('');

  //const handleFileRead = (e, setContent) => {
  //  const content = e.target.result;
  //  console.log(content)
  //  setContent(content)
  //};
  
  //const handleFileChange = (e, setContent) => {
  //  const file = e.target.files[0]
  //  const fileReader = new FileReader()
  //  fileReader.onloadend = (e) => handleFileRead(e, setContent)
  //  fileReader.readAsText(file) 
  //}
  console.log(files)

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Optimizer setup
      </Typography>
      <Box sx={{ mt: 5 }}>
        <Typography variant="h5" gutterBottom>
          Optimizer configuration
        </Typography>

        <Input
              id="study_name" 
              label="Study name" 
              type="text" 
              value={formValues['study_name']}
              onChange={handleInputChange}
              placeholder="Enter study name"
              error={errors.study_name}
        />

        <Input
              id="database_name" 
              label="Database Name" 
              type="text" 
              value={formValues['database_name']}
              onChange={handleInputChange}
              placeholder="Enter database name"
              error={errors.database_name}
        />

        <Input
              id="num_startup_trials" 
              label="Number of startup trials" 
              type="number" 
              value={formValues['num_startup_trials']}
              onChange={handleInputChange}
              placeholder='1'
              min={1}
              error={errors.num_startup_trials}
        />

        <Input
              id="num_ei_candidates" 
              label="Number of expected improvement candidates" 
              type="number" 
              value={formValues['num_ei_candidates']}
              onChange={handleInputChange}
              placeholder='1'
              min={1}
              error={errors.num_ei_candidates}
        />
        
        <RadioInput 
          id="multivariate" 
          label="Multivariate:" 
          value={formValues['multivariate']}
          options={
            [
              {
                value: true,
                label: 'Yes',
              },
              {
                value: false,
                label: 'No',
              }
            ]
          }
          onChange={handleInputChange}
          disabled={true}
        />

        <Input
              id="n_processes" 
              label="Number of parallel evaluations" 
              type="number" 
              value={formValues['n_processes']}
              onChange={handleInputChange}
              placeholder='1'
              min={1}
              error={errors.n_processes}
        />

        <Input
              id="n_gpus" 
              label="Number of GPUs" 
              type="number" 
              value={formValues['n_gpus']}
              onChange={handleInputChange}
              placeholder='0'
              min={0}
              error={errors.n_gpus}
        />

        <Input
              id="y_min" 
              label="Simulation-based objective lower bound (y_min)" 
              type="number" 
              value={formValues['y_min']}
              onChange={handleInputChange}
              placeholder="e.g., -300.0"
              error={errors.y_min}
        />

        <Input
              id="y_max" 
              label="Simulation-based objective upper bound (y_max)" 
              type="number" 
              value={formValues['y_max']}
              onChange={handleInputChange}
              placeholder="e.g., 250.0"
              error={errors.y_max}
        />
        
        <RadioInput 
          id="continue_from_existing_database" 
          label="Continue from existing database:" 
          value={formValues['continue_from_existing_database']}
          options={
            [
              {
                value: true,
                label: 'Yes',
              },
              {
                value: false,
                label: 'No',
              }
            ]
          }
          onChange={handleInputChange}
          disabled={false}
        />

      </Box> 
      
      <Box sx={{ mt: 10 }}>
        <Typography variant="h5" gutterBottom>
          Agent training
        </Typography>

        <Input
          id="number_training_timesteps" 
          label="Number of training steps" 
          type="number" 
          value={formValues['number_training_timesteps']}
          onChange={handleInputChange}
          placeholder='1'
          min={1}
          error={errors.number_training_timesteps}
        />

        <Input
              id="max_number_steps_per_episode" 
              label="Max number of steps per episode" 
              type="number" 
              value={formValues['max_number_steps_per_episode']}
              onChange={handleInputChange}
              placeholder='1'
              min={1}
              error={errors.max_number_steps_per_episode}
        />

        <Input
              id="number_environments_for_training" 
              label="Number of vectorized environments" 
              type="number" 
              value={formValues['number_environments_for_training']}
              onChange={handleInputChange}
              placeholder='1'
              min={1}
              error={errors.number_environments_for_training}
        />
      
        <Input
              id="training_batch_size" 
              label="Batch size" 
              type="number" 
              value={formValues['training_batch_size']}
              onChange={handleInputChange}
              placeholder='1'
              min={1}
              error={errors.training_batch_size}
        />

        <Select 
          id="stable_baselines_algorithm" 
          label="Reinforcement learning algorithm:" 
          value={formValues['stable_baselines_algorithm']}
          options={
            [
              {
                value: 'PPO',
                label: 'PPO',
              },
              {
                value: 'A2C',
                label: 'A2C',
              },
              {
                value: 'LSTM-PPO',
                label: 'LSTM-PPO',
              }
            ]
          }
          onChange={handleInputChange}
          disabled={true}
        />

        <Select 
          id="stable_baselines_policy" 
          label="Policy type:" 
          value={formValues['stable_baselines_policy']}
          options={
            [
              {
                value: 'CnnPolicy',
                label: 'CNN Policy',
              },
              {
                value: 'MlpPolicy',
                label: 'MLP Policy',
              },
              {
                value: 'MultiInputPolicy',
                label: 'Multi-Input Policy',
              }
            ]
          }
          onChange={handleInputChange}
          disabled={false}
        />

        <Input
              id="random_seed" 
              label="Fixed random seed" 
              type="number" 
              value={formValues['random_seed']}
              onChange={handleInputChange}
              placeholder='0'
              error={errors.random_seed}
        />

        <Select 
          id="device" 
          label="Device:" 
          value={formValues['device']}
          options={
            [
              {
                value: 'cuda',
                label: 'Cuda',
              },
              {
                value: 'cpu',
                label: 'CPU',
              },
              {
                value: 'mps',
                label: 'MPS',
              }
            ]
          }
          onChange={handleInputChange}
          disabled={true}
        />
      </Box>

      <Box sx={{ mt: 10 }}>
        <Typography variant="h5" gutterBottom>
          Customized environment
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Upload customized environment file:
        </Typography>
        <input required type="file" onChange={(e) => handleFileChange(e, 'environment')} />
        {errors.environment && (
          <FormHelperText error={errors.environment !== null}>{errors.environment}</FormHelperText>
        )}
      </Box>

      <Box sx={{ mt: 10 }}>
        <Typography variant="h5" gutterBottom>
          Evaluation method
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Upload evaluation method file:
        </Typography>
        <input type="file" onChange={(e) => handleFileChange(e, 'evaluation')} />
        {errors.evaluation && (
          <FormHelperText error={errors.evaluation !== null}>{errors.evaluation}</FormHelperText>
        )}
      </Box>
    </div>
  )
}
//value={files['environment'] ? files['environment'].name : 'test.py'}
export default OptimizerConfiguration;
