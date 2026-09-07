import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

const SelectInput = ({id, label, value, options, onChange, disabled}) => {
  
    return (
      <Box sx={{ mt: 5, mb: 5, background: 'white', textAlign: 'left' }}>
        <FormControl fullWidth>
          <InputLabel id={id}>{label}</InputLabel>
          <Select
            labelId={id}
            id={id}
            value={value}
            label={label}
            onChange={(e) => onChange(e, id)}
            disabled={disabled}
            size='small'
          >
            {options.map((obj, idx) => (
              <MenuItem key={idx} value={obj.value}>{obj.label}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
    )
  }

  /**
   * <label htmlFor={id} className="inputLabel">{label}</label>
        <select id={id} defaultValue={value} onChange={(e) => onChange(e, id)} disabled={disabled}>
          {options.map((obj, idx) => (
            <option key={idx} value={obj.value}>{obj.label}</option>
          ))}
        </select>
   */
  
  export default SelectInput