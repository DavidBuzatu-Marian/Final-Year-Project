import React, { useEffect } from "react";
import {
  Modal,
  Box,
  Typography,
  Button,
  CircularProgress,
} from "@mui/material";
import axios from "axios";
import { getConfig } from "../../config/defaultConfig";
import { getTask } from "../../hooks/environment";
import ClosableAlert from "../alert/closableAlert";

const ModalDistributionForm = ({
  isOpen,
  modalTitle,
  modalButtonText,
  modalContent,
  modalForm,
  initialFormValues,
  headerModals,
  setHeaderModalsState,
  activeHeaderModal,
}) => {
  const [open, setOpen] = React.useState(isOpen);
  const [modalState, setModalState] = React.useState({
    errorMessage: null,
    loading: false,
    successMessage: null,
  });
  const [formValues, setFormValues] = React.useState(initialFormValues);

  const handleClose = () => {
    setOpen(false);
    setHeaderModalsState({
      ...headerModals,
      [activeHeaderModal]: { isVisible: false },
    });
  };

  useEffect(() => {
    setOpen(isOpen);
  }, [isOpen]);

  useEffect(() => {
    setFormValues(initialFormValues);
  }, [initialFormValues]);

  const onSubmit = async () => {
    try {
      setModalState({ ...modalState, loading: true });
      const res = await axios.post(
        getConfig()[headerModals[activeHeaderModal].url],
        {
          ...formValues,
        },
        { withCredentials: true }
      );
      const jobLink = res.data.jobLink;
      const scheduledRequest = setInterval(async () => {
        const task = await getTask(jobLink);
        if (task.jobState === "completed" || task.jobState === "failed") {
          clearInterval(scheduledRequest);
          setModalState({
            errorMessage:
              task.jobState === "failed" ? task.jobFailReason : null,
            loading: false,
            successMessage:
              task.jobState === "completed"
                ? "Environment training data distribution set!"
                : null,
            alertId: task.id,
          });
        }
      }, 1000);
    } catch (error) {
      console.log(error);
      setModalState({
        ...modalState,
        errorMessage: "Request could not be made",
        loading: false,
        alertId: crypto.randomUUID(),
      });
    }
  };
  const ModalForm = modalForm;
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
            m: 1,
            mx: "auto",
            minWidth: "30%",
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
            <ModalForm formValues={formValues} setFormValues={setFormValues} />
            {modalState.loading && (
              <>
                <CircularProgress />
                <Typography variant="p">{modalContent}</Typography>
              </>
            )}
          </Box>
          <Button
            variant="contained"
            onClick={(event) => onSubmit()}
            sx={{ mt: 1, width: "35ch" }}
            disabled={modalState.loading}
          >
            Save
          </Button>
          <Button
            variant="outlined"
            color="error"
            onClick={handleClose}
            sx={{ mt: 1, width: "35ch" }}
          >
            {modalButtonText}
          </Button>
        </Box>
      </Modal>
    </>
  );
};

export default ModalDistributionForm;
