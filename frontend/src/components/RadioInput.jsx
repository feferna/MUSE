import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';

const RadioInput = ({id, label, value, options, onChange, disabled}) => {
    
    return (
      <FormControl>
        <FormLabel id={id}>{label}</FormLabel>
        <RadioGroup
          row
          aria-labelledby={id}
          name={id}
          value={value}
          onChange={(e) => onChange(e, id)}
          sx={{margin: '0 auto'}}
        >
          {options.map((obj, idx) => (
            <div key={idx} >
              <FormControlLabel
                value={obj.value}
                disabled={disabled}
                control={<Radio />}
                label={obj.label}
                sx={{margin: '0 auto'}}
              />
              
            </div>
          ))}
        </RadioGroup>
      </FormControl>
    )
  }
  
  export default RadioInput    