import React, { useEffect } from "react";
import Box from "@mui/material/Box";
import Alert from "@mui/material/Alert";
import IconButton from "@mui/material/IconButton";
import Collapse from "@mui/material/Collapse";
export default function ClosableAlert({ severity, alertMessage }) {
  const [open, setOpen] = React.useState(true);
  return (
    <Box sx={{ width: "35ch" }}>
      <Collapse in={open}>
        <Alert
          action={
            <IconButton
              aria-label="close"
              color="inherit"
              size="small"
              onClick={() => {
                setOpen(false);
              }}
            >
              <span className="material-icons">close</span>
            </IconButton>
          }
          sx={{ mb: 2 }}
          severity={severity}
        >
          {alertMessage}
        </Alert>
      </Collapse>
    </Box>
  );
}
