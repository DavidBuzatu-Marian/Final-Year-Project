import React, { useEffect } from 'react';
import {
  Modal,
  Box,
  Typography,
  Button,
  CircularProgress,
} from '@mui/material';
import axios from 'axios';
import { getConfig } from '../../config/defaultConfig';
import { getTask } from '../../hooks/environment';
import ClosableAlert from '../alert/closableAlert';

const ModalTrainingDistribution = ({
  isOpen,
  modalTitle,
  modalButtonText,
  modalContent,
  modalForm,
  environment_id,
}) => {
  const [open, setOpen] = React.useState(isOpen);
  const [modalState, setModalState] = React.useState({
    errorMessage: null,
    loading: false,
    successMessage: null,
  });
  const [formValues, setFormValues] = React.useState({
    environment_id: environment_id,
    data_distribution: {},
    dataset_length: 0,
  });

  const handleClose = () => setOpen(false);

  useEffect(() => {
    setOpen(isOpen);
  }, [isOpen]);

  const onSubmit = async () => {
    try {
      setModalState({ ...modalState, loading: true });
      const res = await axios.post(
        getConfig()['environmentTrainingDistributionAddUrl'],
        {
          ...formValues,
        },
        { withCredentials: true }
      );
      const jobLink = res.data;
      const scheduledRequest = setInterval(async () => {
        const task = await getTask(jobLink);
        console.log(task);
        if (task.jobState === 'completed' || task.jobState === 'failed') {
          clearInterval(scheduledRequest);
          setModalState({
            redirectDisabled: false,
            errorMessage:
              task.jobState === 'failed' ? task.jobFailReason : null,
            loading: false,
            successMessage:
              task.jobState === 'completed'
                ? 'Environment training data distribution set!'
                : null,
          });
        }
      }, 1000);
    } catch (error) {
      console.log(error);
    }
  };
  return (
    <>
      <Modal
        open={open}
        aria-labelledby='modal-modal-title'
        aria-describedby='modal-modal-description'
        sx={{ overflow: 'hidden', display: 'flex', justifyContent: 'center' }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: '40%',
            m: 1,
            mx: 'auto',
            minWidth: '30%',
            bgcolor: 'background.paper',
            boxShadow: 24,
            alignItems: 'center',
            display: 'flex',
            flexDirection: 'column',
            p: 4,
          }}
        >
          {modalState.errorMessage && (
            <ClosableAlert
              severity={'error'}
              alertMessage={modalState.errorMessage}
            />
          )}
          {modalState.successMessage && (
            <ClosableAlert
              severity={'success'}
              alertMessage={modalState.successMessage}
            />
          )}
          <Typography id='modal-modal-title' variant='h6' component='h2'>
            {modalTitle}
          </Typography>
          <Box
            id='modal-modal-description'
            sx={{
              mt: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            {modalForm}
            {modalState.loading && (
              <>
                <CircularProgress />
                <Typography variant='p'>{modalContent}</Typography>
              </>
            )}
          </Box>
          <Button variant='outlined' onClick={handleClose} sx={{ mt: 1 }}>
            {modalButtonText}
          </Button>
        </Box>
      </Modal>
    </>
  );
};

export default ModalTrainingDistribution;
