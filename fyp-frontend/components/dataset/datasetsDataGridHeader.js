import React from 'react';
import Box from '@mui/material/Box';
import { Divider } from '@mui/material';
import Toolbar from '@mui/material/Toolbar';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import ModalTrainingDistribution from './modalTrainingDistribution';
import AddTrainingDataDistributionForm from './addTrainingDataDistributionForm';

const DatasetsDataGridHeader = ({ selectedRow }) => {
  const [progressModal, setProgressModal] = React.useState({
    isVisible: false,
    jobLink: null,
  });

  return (
    <Box
      component='main'
      sx={{
        bgcolor: 'background.default',
        p: 3,
      }}
    >
      <Toolbar sx={{ justifyContent: 'start' }}>
        <Typography variant='h6' noWrap component='div'>
          Data distribution
        </Typography>
        <Stack direction='row' spacing={2} sx={{ ml: 4 }}>
          <Button
            variant='outlined'
            startIcon={<span className='material-icons'>add</span>}
            disabled={Object.keys(selectedRow).length === 0}
          >
            Add data
          </Button>
          <Button
            variant='contained'
            startIcon={<span className='material-icons'>add</span>}
            disabled={Object.keys(selectedRow).length === 0}
            onClick={(event) => setProgressModal({ isVisible: true })}
          >
            Add training distribution
          </Button>
        </Stack>
      </Toolbar>
      <ModalTrainingDistribution
        isOpen={progressModal.isVisible}
        modalTitle={'Add environment training data distribution'}
        modalContent={'Saving environment training data distribution...'}
        modalForm={<AddTrainingDataDistributionForm />}
        environment_id={selectedRow.environment_id}
        modalButtonText={'Close'}
      />
      <Divider />
    </Box>
  );
};

export default DatasetsDataGridHeader;
