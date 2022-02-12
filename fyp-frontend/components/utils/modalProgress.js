import React, { useEffect } from 'react';
import {
  Modal,
  Box,
  Typography,
  Button,
  CircularProgress,
} from '@mui/material';
import Link from 'next/link';
import { getTask } from '../../hooks/environment';

const ModalProgress = ({
  isOpen,
  modalButtonText,
  modalTitle,
  modalContent,
  redirectUrl,
  jobLink,
}) => {
  const [open, setOpen] = React.useState(isOpen);
  const handleClose = () => setOpen(false);
  const [redirectDisabled, setRedirectDisabled] = React.useState(true);

  useEffect(() => {
    setOpen(isOpen);
  }, [isOpen]);

  useEffect(() => {
    if (jobLink) {
      const scheduledRequest = setInterval(async () => {
        const task = await getTask(jobLink);
        if (task.jobState === 'failed' || task.jobState === 'active') {
          clearInterval(scheduledRequest);
          setRedirectDisabled(false);
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
        onClose={handleClose}
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
            minWidth: '20%',
            bgcolor: 'background.paper',
            boxShadow: 24,
            alignItems: 'center',
            display: 'flex',
            flexDirection: 'column',
            p: 4,
          }}
        >
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
            <CircularProgress />
            <Typography variant='p'>{modalContent}</Typography>
          </Box>
          <Link href={redirectUrl}>
            <Button
              variant='outlined'
              onClick={handleClose}
              disabled={redirectDisabled}
              sx={{ mt: 1 }}
            >
              {modalButtonText}
            </Button>
          </Link>
        </Box>
      </Modal>
    </>
  );
};

export default ModalProgress;
