// Resources.jsx

import Box from '@mui/material/Box'
import FormHelperText from '@mui/material/FormHelperText'
import Typography from '@mui/material/Typography'

import Input from "./components/Input"
import RadioInput from "./components/RadioInput"

const Resources = ({formValues, handleInputChange, errors}) => {

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Iteration settings
      </Typography>
      <Box sx={{ mt: 5 }}>
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
          id="n_designs" 
          label="Number of design suggestions" 
          type="number" 
          value={formValues['n_designs']}
          onChange={handleInputChange}
          placeholder='1'
          min={1}
          error={errors.n_designs}
        />

      </Box>
    </div>
  )
}

export default Resources