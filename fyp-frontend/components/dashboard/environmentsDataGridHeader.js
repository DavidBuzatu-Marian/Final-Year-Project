import React from 'react';
import Box from '@mui/material/Box';
import { Divider } from '@mui/material';
import Toolbar from '@mui/material/Toolbar';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Link from 'next/link';
import ModalProgress from '../utils/modalProgress';
import axios from 'axios';
import { getConfig } from '../../config/defaultConfig';

const EnvironmentsDataGridHeader = ({ selectedRow }) => {
  const [progressModal, setProgressModal] = React.useState({
    isVisible: false,
    jobLink: null,
  });

  const deleteEnvironments = async () => {
    setProgressModal({ isVisible: true });
    try {
      const res = await axios.delete(getConfig()['environmentDeleteUrl'], {
        withCredentials: true,
        data: { environment_id: selectedRow._id },
      });
      setProgressModal({ isVisible: true, ...res.data });
    } catch (error) {
      console.log(error);
    }
  };
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
          Environments
        </Typography>
        <Stack direction='row' spacing={2} sx={{ ml: 4 }}>
          <Link href='/environment/createEnvironment'>
            <Button
              variant='outlined'
              startIcon={<span className='material-icons'>add</span>}
            >
              Create environment
            </Button>
          </Link>
          <Button
            variant='contained'
            startIcon={<span className='material-icons'>delete</span>}
            onClick={() => deleteEnvironments()}
            disabled={Object.keys(selectedRow).length === 0}
          >
            Delete environment
          </Button>
        </Stack>
        <ModalProgress
          isOpen={progressModal.isVisible}
          modalButtonText={'Close'}
          modalTitle={'Environment deletion progress'}
          modalContent={'Deleting environment...'}
          jobLink={progressModal.jobLink}
        />
      </Toolbar>
      <Divider />
    </Box>
  );
};

export default EnvironmentsDataGridHeader;
