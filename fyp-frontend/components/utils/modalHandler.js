import React from 'react';
import { Modal, Box, Typography, Button } from '@mui/material';
import SyntaxHighlighter from 'react-syntax-highlighter';

const ModalHandler = ({ modalButtonText, modalTitle, modalContent }) => {
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);
  return (
    <>
      <Button onClick={handleOpen}>{modalButtonText}</Button>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby='modal-modal-title'
        aria-describedby='modal-modal-description'
        sx={{ overflow: 'scroll', display: 'flex', justifyContent: 'center' }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: '10%',
            m: 1,
            mx: 'auto',
            minWidth: '50%',
            bgcolor: 'background.paper',
            boxShadow: 24,
            overflowX: 'scroll',
            p: 4,
          }}
        >
          <Typography id='modal-modal-title' variant='h6' component='h2'>
            {modalTitle}
          </Typography>
          <Box id='modal-modal-description' sx={{ mt: 2 }}>
            <SyntaxHighlighter language={'json'}>
              {JSON.stringify(modalContent, null, ' ')}
            </SyntaxHighlighter>
          </Box>
        </Box>
      </Modal>
    </>
  );
};

export default ModalHandler;
