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
import { useDatasetDataDistribution } from "../../hooks/dataset";

const ModalCompletedStatusForm = ({
  isOpen,
  modalTitle,
  modalButtonText,
  modalContent,
  modalForm,
  initialFormValues,
  headerModals,
  setHeaderModalsState,
  activeHeaderModal,
  selectedRow,
}) => {
  const [environmentsDataDistribution, { mutateDataDistribution: mutate }] =
    useDatasetDataDistribution();
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
    setOpen(isOpen);
  }, [isOpen]);

  useEffect(() => {
    setFormValues(initialFormValues);
  }, [selectedRow]);

  const performRequest = async (activeModal, formValues) => {
    if (activeModal.hasOwnProperty("isMultipartForm")) {
      const formData = new FormData();
      for (const file of formValues[formValues.dataName]) {
        formData.append(formValues.dataName, file);
      }
      for (const file of formValues[formValues.labelsName]) {
        formData.append(formValues.labelsName, file);
      }
      return await axios.post(
        `${getConfig()[activeModal.url]}?user_id=${
          formValues.user.user_id
        }&environment_id=${formValues.environment_id}`,
        formData,
        {
          withCredentials: true,
          maxContentLength: Infinity,
        }
      );
    } else {
      return await axios.post(
        getConfig()[activeModal.url],
        {
          ...formValues,
        },
        { withCredentials: true }
      );
    }
  };

  const onSubmit = async () => {
    try {
      setModalState({ ...modalState, loading: true });
      const res = await performRequest(
        headerModals[activeHeaderModal],
        formValues
      );

      const jobLink = res.data.jobLink;
      const scheduledRequest = setInterval(async () => {
        const task = await getTask(jobLink);
        if (task.jobState === "completed" || task.jobState === "failed") {
          clearInterval(scheduledRequest);
          mutate(null);
          setModalState({
            errorMessage:
              task.jobState === "failed" ? task.jobFailReason : null,
            loading: false,
            successMessage:
              task.jobState === "completed" ? "Task completed" : null,
            alertId: task.id,
          });
          setFormValues(initialFormValues);
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
            <ModalForm
              formValues={formValues}
              setFormValues={setFormValues}
              headerModals={headerModals}
              setHeaderModalsState={setHeaderModalsState}
              activeHeaderModal={activeHeaderModal}
            />
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

export default ModalCompletedStatusForm;
