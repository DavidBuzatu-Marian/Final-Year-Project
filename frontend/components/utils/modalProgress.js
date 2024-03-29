import React, { useEffect } from "react";
import {
  Modal,
  Box,
  Typography,
  Button,
  CircularProgress,
} from "@mui/material";
import Link from "next/link";
import { getTask } from "../../hooks/environment";
import ClosableAlert from "../alert/closableAlert";
import { useEnvironment } from "../../hooks/environment";
import { getConfig } from "../../config/defaultConfig";

const ModalProgress = ({
  isOpen,
  modalButtonText,
  modalTitle,
  modalContent,
  modalAlertMessage,
  redirectUrl,
  jobLink,
}) => {
  const [environments, { mutate }] = useEnvironment();
  const [open, setOpen] = React.useState(isOpen);
  const handleClose = () => setOpen(false);
  const [modalState, setModalState] = React.useState({
    redirectDisabled: true,
    errorMessage: null,
    loading: true,
    successMessage: null,
  });

  useEffect(() => {
    setOpen(isOpen);
  }, [isOpen]);

  useEffect(() => {
    if (jobLink) {
      const scheduledRequest = setInterval(async () => {
        const task = await getTask(jobLink);
        if (task.jobState === "failed" || task.jobState === "active") {
          clearInterval(scheduledRequest);
          mutate(getConfig()["environmentAddressesUrl"]);
          setModalState({
            redirectDisabled: false,
            errorMessage:
              task.jobState === "failed" ? task.jobFailReason : null,
            loading: false,
            successMessage:
              task.jobState === "active" ? modalAlertMessage : null,
            alertId: task.id,
          });
        }
      }, 1000);
    }
    return () => {
      clearInterval(scheduledRequest);
    };
  }, [jobLink]);
  return (
    <>
      <Modal
        open={open}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
        sx={{ overflow: "hidden", display: "flex", justifyContent: "center" }}
      >
        <Box
          sx={{
            position: "absolute",
            top: "40%",
            m: 1,
            mx: "auto",
            minWidth: "20%",
            bgcolor: "background.paper",
            boxShadow: 24,
            alignItems: "center",
            display: "flex",
            flexDirection: "column",
            p: 4,
          }}
        >
          {modalState.errorMessage && (
            <ClosableAlert
              key={modalState.alertId}
              severity={"error"}
              alertMessage={modalState.errorMessage}
            />
          )}
          {modalState.successMessage && (
            <ClosableAlert
              key={modalState.alertId}
              severity={"success"}
              alertMessage={modalState.successMessage}
            />
          )}
          <Typography id="modal-modal-title" variant="h6" component="h2">
            {modalTitle}
          </Typography>
          <Box
            id="modal-modal-description"
            sx={{
              mt: 2,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            {modalState.loading && (
              <>
                <CircularProgress />
                <Typography variant="p">{modalContent}</Typography>
              </>
            )}
          </Box>
          {redirectUrl ? (
            <Link href={redirectUrl}>
              <Button
                variant="outlined"
                onClick={handleClose}
                disabled={modalState.redirectDisabled}
                sx={{ mt: 1 }}
              >
                {modalButtonText}
              </Button>
            </Link>
          ) : (
            <Button variant="outlined" onClick={handleClose} sx={{ mt: 1 }}>
              {modalButtonText}
            </Button>
          )}
        </Box>
      </Modal>
    </>
  );
};

export default ModalProgress;
