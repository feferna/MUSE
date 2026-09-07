import { createContext, useContext, useState } from "react";

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [formValues, setFormValues] = useState(null)
  return (
    <AppContext.Provider value={{ formValues, setFormValues }}>
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);