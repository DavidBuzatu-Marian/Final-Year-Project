import React from "react";
import { Snackbar, Alert } from "@mui/material";

const SnackbarAlert = ({ message }) => {
  const [state, setState] = React.useState({
    open: true,
    vertical: "bottom",
    horizontal: "center",
  });

  const { vertical, horizontal, open } = state;

  const handleClose = () => {
    setState({ ...state, open: false });
  };
  return (
    <Snackbar
      anchorOrigin={{ vertical, horizontal }}
      open={open}
      autoHideDuration={6000}
      key={crypto.randomUUID()}
    >
      <Alert
        onClose={handleClose}
        severity="error"
        sx={{
          width: "100%",
          "& .MuiAlert-message": { wordWrap: "break-word" },
        }}
      >
        {message}
      </Alert>
    </Snackbar>
  );
};

export default SnackbarAlert;
