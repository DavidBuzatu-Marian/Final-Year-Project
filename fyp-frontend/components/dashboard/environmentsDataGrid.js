import * as React from 'react';
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import { useEnvironment } from '../../hooks/environment';
import {
  CircularProgress,
  Modal,
  Box,
  Typography,
  Button,
} from '@mui/material';

const EnvironmentsDataGrid = () => {
  const [environments, { loading, mutate }] = useEnvironment();
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const columns = [
    { field: 'id', headerName: 'ID', width: 220 },
    {
      field: 'environment_ips',
      headerName: 'Environment IP addresses',
      width: 250,
    },
    {
      field: 'environment_options',
      headerName: 'Environment options',
      renderCell: (params) => {
        return (
          <>
            <Button onClick={handleOpen}>Open environment options</Button>
            <Modal
              open={open}
              onClose={handleClose}
              aria-labelledby='modal-modal-title'
              aria-describedby='modal-modal-description'
            >
              <Box
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 400,
                  bgcolor: 'background.paper',
                  border: '2px solid #000',
                  boxShadow: 24,
                  p: 4,
                }}
              >
                <Typography id='modal-modal-title' variant='h6' component='h2'>
                  Environment options
                </Typography>
                <Typography id='modal-modal-description' sx={{ mt: 2 }}>
                  {params.value}
                </Typography>
              </Box>
            </Modal>
          </>
        );
      },
      width: 250,
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 150,
    },
    { field: 'machine_type', headerName: 'Instances type', width: 200 },
    {
      field: 'date',
      headerName: 'Date created',
      width: 200,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
  ];

  return (
    <div style={{ height: 480, width: '100%' }}>
      {loading ? (
        <CircularProgress />
      ) : (
        <DataGrid
          rows={environments ? environments : []}
          columns={columns}
          pageSize={5}
          rowsPerPageOptions={[5]}
          checkboxSelection
        />
      )}
    </div>
  );
};

export default EnvironmentsDataGrid;
