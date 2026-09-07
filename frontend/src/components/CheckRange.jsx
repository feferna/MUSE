import {useRef} from 'react';

import DeleteIcon from '@mui/icons-material/Delete';
import Box from '@mui/material/Box';
import FormGroup from '@mui/material/FormGroup';
import FormLabel from '@mui/material/FormLabel';
import Checkbox from '@mui/material/Checkbox';
import IconButton from '@mui/material/IconButton';
import Slider from '@mui/material/Slider';

const CheckRange = ({
  index,
  id,
  label,
  formObj,
  onCheckboxChange,
  onRangeChange,
  onUserPrefChange,
  isEditable,
  highlight,
  onDelete,
  disableCheckbox = false,
  disableSlider = false,
  originalRangeMin = null,
  originalRangeMax = null,
}) => {
  const valueText = value => {
    return `${value}°C`;
  };

  /**
   * <input 
            type="range"
            id={id}
            defaultValue={formObj['user_preference']}
            min={formObj['start']}
            max={formObj['stop']}
            step="any"
            onChange={(e) => onRangeChange(e, index, id)} 
          />
   */
  const calculateStepSize = number => {
    // Safety check for undefined/null values
    if (number === undefined || number === null || isNaN(number)) {
      return 1;
    }

    const numArr = number.toString().split('.');
    const length = numArr.length;
    if (length === 1) return 1;
    const numStr = '0.' + '0'.repeat(numArr[length - 1].length) + '1';
    //if (number === 0) return 1
    //let power = Math.floor(Math.log10(Math.abs(number)))
    //if (power !== 0) {power -= 1}
    //const magnitude = Math.pow(10, power)
    //return magnitude / 10;
    //return parseFloat(magnitude.toPrecision(1))
    return parseFloat(numStr);
  };

  // Safety checks for formObj properties
  if (!formObj || typeof formObj !== 'object') {
    return <div>Error: Invalid form object</div>;
  }

  const scaleMin =
    formObj['user_preference_min'] !== undefined
      ? formObj['user_preference_min']
      : 0;
  const scaleMax =
    formObj['user_preference_max'] !== undefined
      ? formObj['user_preference_max']
      : 10;
  const scalePreference =
    formObj['user_preference'] !== undefined ? formObj['user_preference'] : 5;
  const buffer = 0.5; // Extend range by 50%
  const constantMin = scaleMin - (scaleMax - scaleMin) * buffer;
  const constantMax = scaleMax + (scaleMax - scaleMin) * buffer;

  const step = Math.min(
    calculateStepSize(scaleMin),
    calculateStepSize(scaleMax),
    calculateStepSize(scalePreference),
  );
  const min = scaleMin; //Math.round(1.0*constantMin / step)*step
  const max = scaleMax; //Math.round(1.0*constantMax / step)*step

  // Additional safety checks
  const startValue = formObj['start'] || scaleMin;
  const stopValue = formObj['stop'] || scaleMax;
  const searchable =
    formObj['searchable'] !== undefined ? formObj['searchable'] : true;

  // Use original range if provided, otherwise use current values
  const actualMin = originalRangeMin !== null ? originalRangeMin : min;
  const actualMax = originalRangeMax !== null ? originalRangeMax : max;

  return (
    <FormGroup sx={{mb: 2, mt: 2, textAlign: 'left'}}>
      <Box display="flex" alignItems="center" sx={{width: '100%'}}>
        <Checkbox
          id={id}
          checked={searchable}
          onChange={e => onCheckboxChange(e, index, id)}
          disabled={disableCheckbox}
        />
        <Box sx={{flexGrow: 1, ml: 2, mr: 2}}>
          {searchable ? (
            <Box>
              <FormLabel htmlFor={id}>
                {label}: {startValue}, {stopValue}
              </FormLabel>
              {highlight && (
                <span style={{color: 'orange', marginLeft: '8px'}}>*</span>
              )}
              <Slider
                getAriaLabel={() => 'Parameter range'}
                getAriaValueText={valueText}
                value={[startValue, stopValue]}
                min={actualMin}
                max={actualMax}
                step={step}
                onChange={(e, newValue) =>
                  onRangeChange(e, newValue, index, id)
                }
                valueLabelDisplay="auto"
                sx={{mt: 1}}
                disabled={disableSlider}
              />
            </Box>
          ) : (
            // 2) When not searchable, show a single-value slider for user_preference
            <Box>
              <FormLabel htmlFor={id}>
                {label}: {scalePreference}
                {highlight && (
                  <span style={{color: 'orange', marginLeft: '8px'}}>*</span>
                )}
              </FormLabel>
              <Slider
                value={scalePreference}
                min={actualMin}
                max={actualMax}
                step={step}
                onChange={(e, newVal) => {
                  // 3) call onUserPrefChange to update user_preference in parent
                  onUserPrefChange(e, newVal, index, id);
                }}
                valueLabelDisplay="auto"
                track={false}
                sx={{mt: 1}}
                disabled={disableSlider || !searchable}
              />
            </Box>
          )}
        </Box>
        {/* <IconButton 
            color='error' 
            size='small' 
            aria-label="delete"
            sx={isEditable ? {display: "initial"} : {display: "none"}} 
            onClick={(e) => onDelete(e, index, id)}
            disabled={true}
          >
            <DeleteIcon />
          </IconButton> */}
      </Box>
    </FormGroup>
  );
};

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

export default CheckRange;
