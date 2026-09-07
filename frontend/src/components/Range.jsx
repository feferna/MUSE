import DeleteIcon from '@mui/icons-material/Delete'
import Box from '@mui/material/Box'
import FormGroup from '@mui/material/FormGroup'
import FormLabel from '@mui/material/FormLabel'
import Checkbox from '@mui/material/Checkbox'
import IconButton from '@mui/material/IconButton'
import Slider from '@mui/material/Slider'

const Range = ({ id, label, value, onRangeChange }) => {

    return (
      <FormGroup sx={{ mb: 2, mt: 2, textAlign: 'left' }}>
        <FormLabel htmlFor={"slider"}>{label}:</FormLabel>
        <Box display="flex" alignItems="center" sx={{ width: '100%' }}>
          <FormLabel htmlFor={"slider"}>0</FormLabel>
          <Box sx={{ flexGrow: 1, ml: 2, mr: 2 }}>
            <Box>
                  <Slider
                    id="slider"
                    //getAriaLabel={() => 'Parameter range'}
                    //getAriaValueText={valueText}
                    value={value}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={(e, newValue) => onRangeChange(e, newValue, "", id)}
                    valueLabelDisplay="auto"
                    sx={{ mt: 1 }}
                  />
                </Box>
          </Box>
          <FormLabel htmlFor={"slider"}>1</FormLabel>
        </Box>
      </FormGroup>
    )
  }

  /**
   * <Checkbox 
                aria-labelledby={id}
                checked={formObj['searchable']}
                onChange={(e) => onCheckboxChange(e, index, id)} 
              />
              <div className='rangeInput'>
                <FormLabel id={id}>{label}: {formObj['start']}, {formObj['stop']}</FormLabel>
                <Slider
                  getAriaLabel={() => 'Parameter range'}
                  getAriaValueText={valueText}
                  value={[formObj['start'], formObj['stop']]}
                  min={min}
                  max={max}
                  step={step}
                  onChange={(e, newValue) => onRangeChange(e, newValue, index, id)}
                  valueLabelDisplay="auto"
                />
                  
              </div>
   * 
   * 
   * <div className='rangeInput'>
        <label htmlFor={id} className="paramLabel">{label}: {formObj['start']}, {formObj['stop']} </label>
   * 
   * <input id={id} className='checkboxInput' type="checkbox" checked={formObj['searchable']} onChange={(e) => onCheckboxChange(e, index, id)} />
            <label htmlFor={id} className="paramLabel">{label}: {formObj['user_preference']}</label>
   */
  
  export default Range 