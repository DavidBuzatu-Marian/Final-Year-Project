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
import TrainModelForm from "../model/trainModelForm";

const ModalTrainModel = ({
  isOpen,
  initialFormValues,
  setHeaderModalsState,
  headerModals,
  activeHeaderModal,
  selectedRow,
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
      [activeHeaderModal]: {
        ...headerModals[activeHeaderModal],
        isVisible: false,
      },
    });
  };

  useEffect(() => {
    setFormValues(initialFormValues);
  }, [selectedRow, activeHeaderModal]);

  useEffect(() => {
    setOpen(isOpen);
  }, [isOpen]);

  const performRequest = async (formValues) => {
    return await axios.post(
      getConfig()["environmentModelTrainUrl"],
      {
        ...formValues,
      },
      { withCredentials: true }
    );
  };

  const onSubmit = async () => {
    try {
      setModalState({ ...modalState, loading: true });
      console.log(formValues);
      const res = await performRequest(formValues);
      const jobLink = res.data.jobLink;

      const scheduledRequest = setInterval(async () => {
        try {
          const task = await getTask(jobLink);
          if (task.jobState === "active" || task.jobState === "failed") {
            setModalState({
              errorMessage:
                task.jobState === "failed" ? task.jobFailReason : null,
              loading: false,
              successMessage: task.jobState === "active" ? "Task active" : null,
              alertId: task.id,
            });
            clearInterval(scheduledRequest);
          }
        } catch (error) {
          setModalState({
            errorMessage: error,
            loading: false,
            alertId: crypto.randomUUID(),
          });
          clearInterval(scheduledRequest);
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

  return (
    <>
      <Modal
        open={open}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
        sx={{ display: "flex", justifyContent: "center", overflow: "scroll" }}
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
            Train model on environment
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
            <TrainModelForm
              formValues={formValues}
              setFormValues={setFormValues}
            />
            {modalState.loading && (
              <>
                <CircularProgress />
                <Typography variant="p">
                  Starting training process...
                </Typography>
              </>
            )}
          </Box>
          <Button
            variant="contained"
            onClick={(event) => onSubmit()}
            sx={{ mt: 1, width: "35ch" }}
            disabled={modalState.loading}
          >
            Start training
          </Button>
          <Button
            variant="outlined"
            color="error"
            onClick={handleClose}
            sx={{ mt: 1, width: "35ch" }}
          >
            Close
          </Button>
        </Box>
      </Modal>
    </>
  );
};

export default ModalTrainModel;
