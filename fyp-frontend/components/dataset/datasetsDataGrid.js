import React, { useEffect, useState } from 'react';
import { DataGrid } from '@mui/x-data-grid';
import {
  useDatasetDataDistribution,
  useDatasetTrainingDistribution,
} from '../../hooks/dataset';
import { CircularProgress, Box } from '@mui/material';
import ModalHandler from '../utils/modalHandler';

const DatasetsDataGrid = () => {
  const [environmentsDataDistribution, { loadingDataDistribution }] =
    useDatasetDataDistribution();
  const [environmentsTrainingDistribution, { loadingTrainingDistribution }] =
    useDatasetTrainingDistribution();
  const [environmentsDataDistributions, setEnvironmentsDataDistributions] =
    useState([]);

  useEffect(() => {
    if (
      !loadingDataDistribution &&
      !loadingTrainingDistribution &&
      environmentsDataDistribution &&
      environmentsTrainingDistribution
    ) {
      const distributionMap = new Map();
      environmentsDataDistribution.map((distribution) => {
        distributionMap.set(
          distribution.user_id + distribution.environment_id,
          distribution
        );
      });
      environmentsTrainingDistribution.map((distribution) => {
        const key = distribution.user_id + distribution.environment_id;
        if (distributionMap.has(key)) {
          distributionMap.set(key, {
            ...distributionMap.get(key),
            ...distribution,
          });
        }
      });
      setEnvironmentsDataDistributions([...distributionMap.values()]);
    }
  }, [loadingDataDistribution, loadingTrainingDistribution]);

  const columns = [
    { field: 'id', headerName: 'ID', width: 220 },
    {
      field: 'environment_id',
      headerName: 'Environment ID',
      width: 250,
    },
    {
      field: 'train_data_distribution',
      headerName: 'Environment training data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment training data distribution'}
            modalContent={params.value}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
    {
      field: 'train_labels_data_distribution',
      headerName: 'Environment training labels data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment training labels data distribution'}
            modalContent={params.value}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
    {
      field: 'validation_data_distribution',
      headerName: 'Environment validation data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment validation data distribution'}
            modalContent={params.value}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
    {
      field: 'validation_labels_data_distribution',
      headerName: 'Environment validation labels data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment validation labels data distribution'}
            modalContent={params.value}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
    {
      field: 'test_data_distribution',
      headerName: 'Environment test data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment test data distribution'}
            modalContent={params.value}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
    {
      field: 'test_labels_data_distribution',
      headerName: 'Environment test labels data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment test labels data distribution'}
            modalContent={params.value}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
    {
      field: 'distributions',
      headerName: 'Environment training data distribution split',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment training data distribution split'}
            modalContent={params.value ? params.value : []}
            modalButtonText={'Open'}
          />
        );
      },
      width: 300,
    },
  ];

  return (
    <div style={{ height: 480, width: '100%' }}>
      {loadingTrainingDistribution || loadingDataDistribution ? (
        <CircularProgress />
      ) : (
        <DataGrid
          rows={environmentsDataDistributions}
          columns={columns}
          pageSize={5}
          rowsPerPageOptions={[5]}
          // onSelectionModelChange={(ids) => {
          //   const selectedIDs = new Set(ids);
          //   const selectedRowData = datasets.filter((row) =>
          //     selectedIDs.has(row._id)
          //   );
          // }}
        />
      )}
    </div>
  );
};

export default DatasetsDataGrid;
