import Box from '@mui/material/Box'
import FormHelperText from '@mui/material/FormHelperText'
import TextField from '@mui/material/TextField'

const Input = ({id, label, type, value, onChange, placeholder, step, min, error, disabled}) => {

  return (
    <div>

    <Box
      component="form"
      sx={{  mt: 5, mb: 5 }}
      noValidate
      autoComplete="off"
    >
      <TextField 
        id={id} 
        label={label}
        type={type || "text"}  
        value={value || ""} 
        onChange={(e) => onChange(e, id)}
        placeholder={placeholder}
        {...(type === "number" && { step, min })}
        variant='outlined' 
        size='small'
        sx={{ background: 'white'  }}
        error={!!error}
        disabled={disabled}
        //helperText={error}
        fullWidth
      />
      {error && (
          <FormHelperText sx={{}} error={error !== null}>{error}</FormHelperText>
        )}
    </Box>

      
     
    </div>
  )
}

/**
 * <label htmlFor={id} className="inputLabel">{label}:</label>
      <input 
        type={type || "text"} 
        id={id} 
        value={value || ""} 
        onChange={(e) => onChange(e, id)} 
        placeholder={placeholder}
        {...(type === "number" && { step, min })}
      />
 */

export default Input